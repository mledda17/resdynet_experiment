from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from main import (
    LinearStateSpace,
    ResDyNet,
    RolloutWindowDataset,
    apply_linear_informed_initialization,
    fit_n4sid_state_space,
    reconstructability_matrix,
    weighted_multistep_loss,
)


@dataclass(frozen=True)
class Normalization:
    u_mean: torch.Tensor
    u_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


def standardize_from_train(
    u_train: torch.Tensor,
    y_train: torch.Tensor,
    *pairs: tuple[torch.Tensor, torch.Tensor],
):
    eps = torch.finfo(u_train.dtype).eps
    norm = Normalization(
        u_mean=u_train.mean(dim=0, keepdim=True),
        u_std=u_train.std(dim=0, keepdim=True).clamp_min(eps),
        y_mean=y_train.mean(dim=0, keepdim=True),
        y_std=y_train.std(dim=0, keepdim=True).clamp_min(eps),
    )

    out = [((u_train - norm.u_mean) / norm.u_std, (y_train - norm.y_mean) / norm.y_std)]
    for u, y in pairs:
        out.append(((u - norm.u_mean) / norm.u_std, (y - norm.y_mean) / norm.y_std))
    return norm, out


def tensors_from_benchmark(dataset, dtype: torch.dtype):
    try:
        u, y = dataset
    except Exception:
        u, y = dataset.u, dataset.y
    u_tensor = torch.as_tensor(u, dtype=dtype)
    y_tensor = torch.as_tensor(y, dtype=dtype)
    if u_tensor.ndim == 1:
        u_tensor = u_tensor.unsqueeze(-1)
    if y_tensor.ndim == 1:
        y_tensor = y_tensor.unsqueeze(-1)
    return u_tensor, y_tensor


def dataset_to_tensors(dataset: RolloutWindowDataset):
    y_past, u_past, u_future, y_future = zip(*(dataset[i] for i in range(len(dataset))))
    return (
        torch.stack(y_past, dim=0),
        torch.stack(u_past, dim=0),
        torch.stack(u_future, dim=0),
        torch.stack(y_future, dim=0),
    )


def make_batch_schedule(num_samples: int, batch_size: int, epochs: int, seed: int) -> list[list[torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    schedule = []
    for _ in range(epochs):
        perm = torch.randperm(num_samples, generator=generator)
        schedule.append([perm[start : start + batch_size] for start in range(0, num_samples, batch_size)])
    return schedule


def make_model(
    ny: int,
    nu: int,
    nx: int,
    n: int,
    stream_dim: int,
    num_blocks: int,
    hidden_width: int,
    hidden_layers: int,
    evolver_hidden_width: int,
    linear: LinearStateSpace | None,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> ResDyNet:
    torch.manual_seed(seed)
    hidden = (hidden_width,) * hidden_layers
    model = ResDyNet(
        ny=ny,
        nu=nu,
        nx=nx,
        na=n,
        nb=n,
        encoder_hidden_dims=hidden,
        decoder_hidden_dims=hidden,
        stream_dim=stream_dim,
        num_blocks=num_blocks,
        evolver_hidden_dims=(evolver_hidden_width,),
        activation=nn.Tanh,
        zero_nonlinear_outputs=False,
    ).to(device=device, dtype=dtype)
    if linear is not None:
        apply_linear_informed_initialization(model, linear, n=n)
    return model


def scale_linear_realization_for_encoder(linear: LinearStateSpace, n: int, eps: float = 1e-12) -> LinearStateSpace:
    """Apply a diagonal similarity transform so rows of R_L are O(1).

    This preserves the input-output linear predictor exactly while improving the
    numerical scale of S_e, S_f and S_g used to initialize ResDyNet.
    """

    R = reconstructability_matrix(linear, n)
    row_scale = torch.max(torch.abs(R), dim=1).values.clamp_min(eps)
    P = torch.diag(1.0 / row_scale)
    P_inv = torch.diag(row_scale)
    return LinearStateSpace(
        A=P @ linear.A @ P_inv,
        B=P @ linear.B,
        C=linear.C @ P_inv,
        D=linear.D,
    )


@torch.no_grad()
def nrmse(model: ResDyNet, tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_size: int) -> float:
    y_past, u_past, u_future, y_future = tensors
    param = next(model.parameters())
    model_was_training = model.training
    model.eval()
    sq_error_sum = 0.0
    target_values = []
    for start in range(0, y_future.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        yp = y_past[sl].to(device=param.device, dtype=param.dtype)
        up = u_past[sl].to(device=param.device, dtype=param.dtype)
        uf = u_future[sl].to(device=param.device, dtype=param.dtype)
        yf = y_future[sl].to(device=param.device, dtype=param.dtype)
        pred = model(yp, up, uf)
        sq_error_sum += torch.sum((yf - pred) ** 2).item()
        target_values.append(yf.detach().cpu().reshape(-1, yf.shape[-1]))
    target = torch.cat(target_values, dim=0)
    rmse = math.sqrt(sq_error_sum / target.numel())
    denom = torch.std(target).item()
    if model_was_training:
        model.train()
    return rmse / max(denom, torch.finfo(target.dtype).eps)


def train_one_method(
    method: str,
    model: ResDyNet,
    train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    val_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    test_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    schedule: list[list[torch.Tensor]],
    learning_rate: float,
    eval_batch_size: int,
    gamma: torch.Tensor | None,
    clip_grad_norm: float | None,
) -> tuple[list[dict[str, float | int | str]], float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    y_past, u_past, u_future, y_future = train_tensors
    param = next(model.parameters())

    history: list[dict[str, float | int | str]] = []
    start_time = time.perf_counter()
    history.append(
        {
            "method": method,
            "epoch": 0,
            "updates": 0,
            "wall_time_sec": 0.0,
            "val_nrmse": nrmse(model, val_tensors, eval_batch_size),
        }
    )

    updates = 0
    model.train()
    for epoch, batches in enumerate(schedule, start=1):
        for idx in batches:
            yp = y_past[idx].to(device=param.device, dtype=param.dtype)
            up = u_past[idx].to(device=param.device, dtype=param.dtype)
            uf = u_future[idx].to(device=param.device, dtype=param.dtype)
            yf = y_future[idx].to(device=param.device, dtype=param.dtype)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_multistep_loss(model(yp, up, uf), yf, gamma)
            loss.backward()
            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            updates += 1
        history.append(
            {
                "method": method,
                "epoch": epoch,
                "updates": updates,
                "wall_time_sec": time.perf_counter() - start_time,
                "val_nrmse": nrmse(model, val_tensors, eval_batch_size),
            }
        )

    test_nrmse = nrmse(model, test_tensors, eval_batch_size)
    return history, test_nrmse


def threshold_rows(rows: list[dict[str, float | int | str]], thresholds: list[float]):
    out = []
    for threshold in thresholds:
        hit = next((row for row in rows if float(row["val_nrmse"]) <= threshold), None)
        out.append(
            {
                "threshold": threshold,
                "hit": hit is not None,
                "epoch": "" if hit is None else hit["epoch"],
                "updates": "" if hit is None else hit["updates"],
                "wall_time_sec": "" if hit is None else hit["wall_time_sec"],
            }
        )
    return out


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    return torch.quantile(tensor, q).item()


def save_plots(output_dir: Path, trajectory_rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row["method"]) for row in trajectory_rows})

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in methods:
        method_rows = [row for row in trajectory_rows if row["method"] == method]
        xs = sorted({int(row["epoch"]) for row in method_rows})
        medians, lows, highs = [], [], []
        for x in xs:
            vals = [float(row["val_nrmse"]) for row in method_rows if int(row["epoch"]) == x]
            medians.append(percentile(vals, 0.5))
            lows.append(percentile(vals, 0.25))
            highs.append(percentile(vals, 0.75))
        ax.plot(xs, medians, label=method)
        ax.fill_between(xs, lows, highs, alpha=0.18)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NRMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "val_nrmse_vs_epoch.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in methods:
        method_rows = [row for row in trajectory_rows if row["method"] == method]
        seeds = sorted({row["seed"] for row in method_rows})
        seed_curves = []
        max_common_time = min(
            max(float(row["wall_time_sec"]) for row in method_rows if row["seed"] == seed)
            for seed in seeds
        )
        grid = torch.linspace(0.0, max_common_time, 100)
        for seed in seeds:
            rows = sorted(
                [row for row in method_rows if row["seed"] == seed],
                key=lambda row: float(row["wall_time_sec"]),
            )
            times = torch.tensor([float(row["wall_time_sec"]) for row in rows], dtype=torch.float64)
            vals = torch.tensor([float(row["val_nrmse"]) for row in rows], dtype=torch.float64)
            curve = torch.empty_like(grid)
            for idx, t in enumerate(grid):
                right = torch.searchsorted(times, t, right=False).item()
                if right == 0:
                    curve[idx] = vals[0]
                elif right >= len(times):
                    curve[idx] = vals[-1]
                else:
                    t0, t1 = times[right - 1], times[right]
                    v0, v1 = vals[right - 1], vals[right]
                    alpha = (t - t0) / (t1 - t0).clamp_min(torch.finfo(torch.float64).eps)
                    curve[idx] = v0 + alpha * (v1 - v0)
            seed_curves.append(curve)
        curves = torch.stack(seed_curves, dim=0)
        median = torch.quantile(curves, 0.5, dim=0)
        low = torch.quantile(curves, 0.25, dim=0)
        high = torch.quantile(curves, 0.75, dim=0)
        ax.plot(grid.tolist(), median.tolist(), label=method)
        ax.fill_between(grid.tolist(), low.tolist(), high.tolist(), alpha=0.18)
    ax.set_xlabel("wall-clock time [s]")
    ax.set_ylabel("validation NRMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "val_nrmse_vs_time.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cascaded Tanks ResDyNet initialization ablation.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(7)))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--linear-order", type=int, default=6)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--history-length", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--stream-dim", type=int, default=None)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--evolver-hidden-width", type=int, default=64)
    parser.add_argument("--n4sid-block-rows", type=int, default=10)
    parser.add_argument("--no-linear-state-scaling", action="store_true")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.03, 0.02])
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/cascaded_tanks_ablation")
    args = parser.parse_args()

    if args.latent_dim < args.linear_order:
        raise ValueError("--latent-dim must be >= --linear-order.")

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    import nonlinear_benchmarks

    train_val, test = nonlinear_benchmarks.Cascaded_Tanks(atleast_2d=True)
    train_val_u, train_val_y = tensors_from_benchmark(train_val, dtype=dtype)
    test_u, test_y = tensors_from_benchmark(test, dtype=dtype)
    split_idx = int(train_val_u.shape[0] * (1.0 - args.val_fraction))
    train_u_raw, train_y_raw = train_val_u[:split_idx], train_val_y[:split_idx]
    val_u_raw, val_y_raw = train_val_u[split_idx:], train_val_y[split_idx:]
    _, normalized = standardize_from_train(train_u_raw, train_y_raw, (val_u_raw, val_y_raw), (test_u, test_y))
    (train_u, train_y), (val_u, val_y), (test_u_norm, test_y_norm) = normalized

    n = args.history_length
    stream_dim = args.stream_dim or 2 * (args.latent_dim + train_u.shape[1]) + 1
    train_dataset = RolloutWindowDataset(train_u, train_y, n, n, args.horizon)
    val_dataset = RolloutWindowDataset(val_u, val_y, n, n, args.horizon)
    test_dataset = RolloutWindowDataset(test_u_norm, test_y_norm, n, n, args.horizon)
    train_tensors = dataset_to_tensors(train_dataset)
    val_tensors = dataset_to_tensors(val_dataset)
    test_tensors = dataset_to_tensors(test_dataset)

    linear = fit_n4sid_state_space(
        train_u,
        train_y,
        order=args.linear_order,
        num_block_rows=args.n4sid_block_rows,
        zero_direct_feedthrough=True,
    )
    if not args.no_linear_state_scaling:
        linear = scale_linear_realization_for_encoder(linear, args.history_length)

    output_dir = Path(args.output_dir)
    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    threshold_output_rows: list[dict[str, object]] = []
    ny, nu = train_y.shape[1], train_u.shape[1]

    print("Cascaded Tanks ablation")
    print(f"train/val/test samples: {len(train_u)}/{len(val_u)}/{len(test_u_norm)}")
    print(f"rollout windows train/val/test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    print(f"n={n}, H={args.horizon}, n_L={args.linear_order}, n_x={args.latent_dim}, B={args.num_blocks}, n_h={stream_dim}")

    for seed in args.seeds:
        schedule = make_batch_schedule(len(train_dataset), args.batch_size, args.epochs, seed=10_000 + seed)
        seed_rows = []
        for method, linear_for_method in [
            ("ResDyNet-Random", None),
            ("ResDyNet-LinearInit", linear),
        ]:
            model = make_model(
                ny=ny,
                nu=nu,
                nx=args.latent_dim,
                n=n,
                stream_dim=stream_dim,
                num_blocks=args.num_blocks,
                hidden_width=args.hidden_width,
                hidden_layers=args.hidden_layers,
                evolver_hidden_width=args.evolver_hidden_width,
                linear=linear_for_method,
                seed=seed,
                dtype=dtype,
                device=device,
            )
            history, test_score = train_one_method(
                method,
                model,
                train_tensors,
                val_tensors,
                test_tensors,
                schedule,
                learning_rate=args.learning_rate,
                eval_batch_size=args.eval_batch_size,
                gamma=None,
                clip_grad_norm=args.clip_grad_norm,
            )
            for row in history:
                full_row = {"seed": seed, **row}
                all_rows.append(full_row)
                seed_rows.append(full_row)
            best_val = min(float(row["val_nrmse"]) for row in history)
            final_val = float(history[-1]["val_nrmse"])
            summary_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "best_val_nrmse": best_val,
                    "final_val_nrmse": final_val,
                    "final_test_nrmse": test_score,
                }
            )
            for threshold_row in threshold_rows(history, args.thresholds):
                threshold_output_rows.append({"seed": seed, "method": method, **threshold_row})
            print(
                f"seed={seed} {method}: "
                f"best val={best_val:.5f}, final val={final_val:.5f}, test={test_score:.5f}"
            )

    save_csv(output_dir / "trajectories.csv", all_rows)
    save_csv(output_dir / "summary.csv", summary_rows)
    save_csv(output_dir / "time_to_threshold.csv", threshold_output_rows)
    save_plots(output_dir, all_rows)
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
