from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from main import (
    LinearStateSpace,
    ResDyNet,
    RolloutWindowDataset,
    apply_linear_informed_initialization,
    fit_n4sid_state_space,
    reconstructability_matrix,
    weighted_multistep_loss,
)
from run_cascaded_tanks_ablation import dataset_to_tensors, make_batch_schedule, scale_linear_realization_for_encoder
from test_duffing_n4sid import DuffingSplit, load_duffing_dataset, make_normalizer


def make_resdynet(args, ny: int, nu: int, seed: int, dtype: torch.dtype, device: torch.device) -> ResDyNet:
    torch.manual_seed(seed)
    hidden = (args.hidden_width,) * args.hidden_layers
    stream_dim = args.stream_dim or 2 * (args.latent_dim + nu) + 1
    return ResDyNet(
        ny=ny,
        nu=nu,
        nx=args.latent_dim,
        na=args.history_length,
        nb=args.history_length,
        encoder_hidden_dims=hidden,
        decoder_hidden_dims=hidden,
        stream_dim=stream_dim,
        num_blocks=args.num_blocks,
        evolver_hidden_dims=(args.evolver_hidden_width,),
        activation=nn.Tanh,
        zero_nonlinear_outputs=False,
    ).to(device=device, dtype=dtype)


@torch.no_grad()
def nrmse_model(model: ResDyNet, tensors: tuple[Tensor, Tensor, Tensor, Tensor], batch_size: int) -> float:
    y_past, u_past, u_future, y_future = tensors
    param = next(model.parameters())
    was_training = model.training
    model.eval()
    sq = 0.0
    targets = []
    for start in range(0, y_future.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        yp = y_past[sl].to(device=param.device, dtype=param.dtype)
        up = u_past[sl].to(device=param.device, dtype=param.dtype)
        uf = u_future[sl].to(device=param.device, dtype=param.dtype)
        yf = y_future[sl].to(device=param.device, dtype=param.dtype)
        pred = model(yp, up, uf)
        sq += torch.sum((pred - yf) ** 2).item()
        targets.append(yf.cpu().reshape(-1, yf.shape[-1]))
    target = torch.cat(targets, dim=0)
    if was_training:
        model.train()
    return math.sqrt(sq / target.numel()) / torch.std(target).clamp_min(torch.finfo(target.dtype).eps).item()


@torch.no_grad()
def loss_model(model: ResDyNet, tensors: tuple[Tensor, Tensor, Tensor, Tensor], batch_size: int) -> float:
    y_past, u_past, u_future, y_future = tensors
    param = next(model.parameters())
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, y_future.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        yp = y_past[sl].to(device=param.device, dtype=param.dtype)
        up = u_past[sl].to(device=param.device, dtype=param.dtype)
        uf = u_future[sl].to(device=param.device, dtype=param.dtype)
        yf = y_future[sl].to(device=param.device, dtype=param.dtype)
        loss = weighted_multistep_loss(model(yp, up, uf), yf)
        total += loss.item() * yf.shape[0]
        count += yf.shape[0]
    if was_training:
        model.train()
    return total / count


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_data(args, dtype: torch.dtype):
    splits, cfg = load_duffing_dataset(Path(args.dataset), dtype)
    normalizer = make_normalizer(splits["train"])
    train_u, train_y = normalizer.uy(splits["train"])
    val_u, val_y = normalizer.uy(splits["val"])
    test_u, test_y = normalizer.uy(splits["test"])
    train_ds = RolloutWindowDataset(train_u, train_y, args.history_length, args.history_length, args.horizon)
    val_ds = RolloutWindowDataset(val_u, val_y, args.history_length, args.history_length, args.horizon)
    test_ds = RolloutWindowDataset(test_u, test_y, args.history_length, args.history_length, args.horizon)
    tensors = {
        "train": dataset_to_tensors(train_ds),
        "val": dataset_to_tensors(val_ds),
        "test": dataset_to_tensors(test_ds),
    }
    return (train_u, train_y, val_u, val_y, test_u, test_y), tensors, cfg


@torch.no_grad()
def resdynet_rollout_outputs(model: ResDyNet, y_past: Tensor, u_past: Tensor, u_future: Tensor) -> Tensor:
    return model(y_past, u_past, u_future)


@torch.no_grad()
def n4sid_linear_rollout(linear: LinearStateSpace, y_past: Tensor, u_past: Tensor, u_future: Tensor, history: int) -> Tensor:
    R = reconstructability_matrix(linear, history)
    outs = []
    for b in range(y_past.shape[0]):
        reg = torch.cat((y_past[b].reshape(-1), u_past[b].reshape(-1)))
        x = R @ reg
        pred = []
        for h in range(u_future.shape[1]):
            pred.append(linear.C @ x + linear.D @ u_future[b, h])
            if h < u_future.shape[1] - 1:
                x = linear.A @ x + linear.B @ u_future[b, h]
        outs.append(torch.stack(pred, dim=0))
    return torch.stack(outs, dim=0)


def verify_linear_initialization(model: ResDyNet, linear: LinearStateSpace, tensors, history: int, num_windows: int) -> dict[str, float]:
    y_past, u_past, u_future, _ = tensors
    n = min(num_windows, y_past.shape[0])
    yp = y_past[:n].to(dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
    up = u_past[:n].to(dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
    uf = u_future[:n].to(dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
    lin = LinearStateSpace(
        A=linear.A.to(device=yp.device, dtype=yp.dtype),
        B=linear.B.to(device=yp.device, dtype=yp.dtype),
        C=linear.C.to(device=yp.device, dtype=yp.dtype),
        D=linear.D.to(device=yp.device, dtype=yp.dtype),
    )
    y_model = resdynet_rollout_outputs(model, yp, up, uf)
    y_linear = n4sid_linear_rollout(lin, yp, up, uf, history)
    return {
        "max_abs_output_error": torch.max(torch.abs(y_model - y_linear)).item(),
        "rmse_output_error": torch.sqrt(torch.mean((y_model - y_linear) ** 2)).item(),
    }


def train_one(args, method: str, seed: int, linear: LinearStateSpace, tensors, dtype: torch.dtype, device: torch.device, output_dir: Path):
    y_train, u_train, uf_train, yf_train = tensors["train"]
    ny, nu = yf_train.shape[-1], uf_train.shape[-1]
    model = make_resdynet(args, ny=ny, nu=nu, seed=seed, dtype=dtype, device=device)
    if method == "n4sid_init":
        apply_linear_informed_initialization(model, linear, n=args.history_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    schedule = make_batch_schedule(y_train.shape[0], args.batch_size, args.epochs, seed=args.shuffle_seed_base + seed)
    history = []
    best_val = float("inf")
    best_epoch = 0
    best_time = 0.0
    best_path = output_dir / "checkpoints" / f"{method}_seed_{seed}_best.pt"
    bad_epochs = 0
    start = time.perf_counter()

    for epoch in range(args.epochs + 1):
        val_loss = loss_model(model, tensors["val"], args.eval_batch_size)
        val_nrmse = nrmse_model(model, tensors["val"], args.eval_batch_size)
        train_loss = float("nan") if epoch == 0 else last_train_loss
        elapsed = time.perf_counter() - start
        history.append(
            {
                "method": method,
                "seed": seed,
                "epoch": epoch,
                "training_loss": train_loss,
                "validation_loss": val_loss,
                "validation_nrmse": val_nrmse,
                "wall_time_sec": elapsed,
            }
        )
        if val_nrmse < best_val:
            best_val = val_nrmse
            best_epoch = epoch
            best_time = elapsed
            bad_epochs = 0
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "method": method, "seed": seed, "epoch": epoch, "val_nrmse": val_nrmse}, best_path)
        else:
            bad_epochs += 1
        if epoch == args.epochs or (args.patience and bad_epochs >= args.patience):
            break

        model.train()
        total = 0.0
        count = 0
        param = next(model.parameters())
        for idx in schedule[epoch]:
            yp = y_train[idx].to(device=param.device, dtype=param.dtype)
            up = u_train[idx].to(device=param.device, dtype=param.dtype)
            uf = uf_train[idx].to(device=param.device, dtype=param.dtype)
            yf = yf_train[idx].to(device=param.device, dtype=param.dtype)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_multistep_loss(model(yp, up, uf), yf)
            loss.backward()
            if args.clip_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            total += loss.item() * yf.shape[0]
            count += yf.shape[0]
        last_train_loss = total / count

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    test_nrmse = nrmse_model(model, tensors["test"], args.eval_batch_size)
    summary = {
        "method": method,
        "seed": seed,
        "best_val_nrmse": best_val,
        "best_epoch": best_epoch,
        "time_to_best_sec": best_time,
        "final_test_nrmse": test_nrmse,
    }
    return history, summary


def mean_std(values: list[float]) -> tuple[float, float]:
    t = torch.tensor(values, dtype=torch.float64)
    return torch.mean(t).item(), torch.std(t, unbiased=True).item() if len(values) > 1 else 0.0


def aggregate_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for method in sorted({str(row["method"]) for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        agg = {"method": method, "num_seeds": len(subset)}
        for key in ("best_val_nrmse", "best_epoch", "time_to_best_sec", "final_test_nrmse"):
            mean, std = mean_std([float(row[key]) for row in subset])
            agg[f"{key}_mean"] = mean
            agg[f"{key}_std"] = std
        out.append(agg)
    return out


def save_plots(output_dir: Path, histories: list[dict[str, object]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {"n4sid_init": "N4SID init", "random_init": "Random init"}
    colors = {"n4sid_init": "tab:blue", "random_init": "tab:orange"}

    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    for method in ("n4sid_init", "random_init"):
        rows = [row for row in histories if row["method"] == method]
        epochs = sorted({int(row["epoch"]) for row in rows})
        xs, means, stds = [], [], []
        for epoch in epochs:
            vals = [float(row["validation_nrmse"]) for row in rows if int(row["epoch"]) == epoch]
            if len(vals) == len({int(row["seed"]) for row in rows}):
                mean, std = mean_std(vals)
                xs.append(epoch)
                means.append(mean)
                stds.append(std)
        x = torch.tensor(xs, dtype=torch.float64)
        m = torch.tensor(means, dtype=torch.float64)
        s = torch.tensor(stds, dtype=torch.float64)
        ax.plot(x, m, label=labels[method], color=colors[method])
        ax.fill_between(x, m - s, m + s, color=colors[method], alpha=0.18)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NRMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "val_nrmse_vs_epoch_mean_std.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    for method in ("n4sid_init", "random_init"):
        rows = [row for row in histories if row["method"] == method]
        seeds = sorted({int(row["seed"]) for row in rows})
        max_common = min(max(float(row["wall_time_sec"]) for row in rows if int(row["seed"]) == seed) for seed in seeds)
        grid = torch.linspace(0.0, max_common, 100)
        curves = []
        for seed in seeds:
            seed_rows = sorted([row for row in rows if int(row["seed"]) == seed], key=lambda r: float(r["wall_time_sec"]))
            times = torch.tensor([float(row["wall_time_sec"]) for row in seed_rows], dtype=torch.float64)
            vals = torch.tensor([float(row["validation_nrmse"]) for row in seed_rows], dtype=torch.float64)
            curve = torch.empty_like(grid)
            for i, t in enumerate(grid):
                idx = torch.searchsorted(times, t, right=False).item()
                if idx == 0:
                    curve[i] = vals[0]
                elif idx >= len(times):
                    curve[i] = vals[-1]
                else:
                    alpha = (t - times[idx - 1]) / (times[idx] - times[idx - 1]).clamp_min(torch.finfo(torch.float64).eps)
                    curve[i] = vals[idx - 1] + alpha * (vals[idx] - vals[idx - 1])
            curves.append(curve)
        stack = torch.stack(curves)
        mean = torch.mean(stack, dim=0)
        std = torch.std(stack, dim=0, unbiased=True) if len(curves) > 1 else torch.zeros_like(mean)
        ax.plot(grid, mean, label=labels[method], color=colors[method])
        ax.fill_between(grid, mean - std, mean + std, color=colors[method], alpha=0.18)
    ax.set_xlabel("wall-clock time [s]")
    ax.set_ylabel("validation NRMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "val_nrmse_vs_time_mean_std.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Duffing ResDyNet initialization ablation: N4SID-informed vs random.")
    p.add_argument("--dataset", default="duffing_resdynet_dataset.mat")
    p.add_argument("--output-dir", default="results/duffing_init_ablation")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--latent-dim", type=int, default=6)
    p.add_argument("--linear-order", type=int, default=6)
    p.add_argument("--history-length", type=int, default=50)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--stream-dim", type=int, default=None)
    p.add_argument("--hidden-width", type=int, default=64)
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--evolver-hidden-width", type=int, default=64)
    p.add_argument("--n4sid-block-rows", type=int, default=10)
    p.add_argument("--max-n4sid-samples", type=int, default=10000)
    p.add_argument("--max-train-windows", type=int, default=None)
    p.add_argument("--verify-windows", type=int, default=128)
    p.add_argument("--shuffle-seed-base", type=int, default=12345)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.latent_dim != 6 or args.linear_order != 6:
        raise ValueError("This ablation is requested with n_x=6 and n_L=6.")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    (train_u, train_y, _val_u, _val_y, _test_u, _test_y), tensors, cfg = prepare_data(args, dtype)
    if args.max_train_windows is not None:
        tensors["train"] = tuple(t[: args.max_train_windows] for t in tensors["train"])
    n4sid_u = train_u[: args.max_n4sid_samples] if args.max_n4sid_samples else train_u
    n4sid_y = train_y[: args.max_n4sid_samples] if args.max_n4sid_samples else train_y
    linear = fit_n4sid_state_space(
        n4sid_u,
        n4sid_y,
        order=args.linear_order,
        num_block_rows=args.n4sid_block_rows,
        zero_direct_feedthrough=True,
    )
    linear = scale_linear_realization_for_encoder(linear, args.history_length)

    verify_model = make_resdynet(args, ny=train_y.shape[1], nu=train_u.shape[1], seed=args.seeds[0], dtype=dtype, device=device)
    apply_linear_informed_initialization(verify_model, linear, n=args.history_length)
    verify = verify_linear_initialization(verify_model, linear, tensors["val"], args.history_length, args.verify_windows)
    (output_dir / "linear_initialization_verification.json").write_text(json.dumps(verify, indent=2))
    print(f"linear init verification: {verify}")

    all_history = []
    seed_summaries = []
    for seed in args.seeds:
        for method in ("n4sid_init", "random_init"):
            history, summary = train_one(args, method, seed, linear, tensors, dtype, device, output_dir)
            all_history.extend(history)
            seed_summaries.append(summary)
            print(
                f"seed={seed} {method}: best_val={summary['best_val_nrmse']:.6g} "
                f"epoch={summary['best_epoch']} test={summary['final_test_nrmse']:.6g}"
            )

    aggregate = aggregate_summary(seed_summaries)
    write_csv(output_dir / "epoch_history.csv", all_history)
    write_csv(output_dir / "seed_summary.csv", seed_summaries)
    write_csv(output_dir / "summary_mean_std.csv", aggregate)
    save_plots(output_dir, all_history)
    print(f"saved results to {output_dir}")
    for row in aggregate:
        print(row)


if __name__ == "__main__":
    main()
