from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from main import (
    LinearPlusMLPCorrectionEvolver,
    ResDyNet,
    ResDyNetEvolver,
    RolloutWindowDataset,
    apply_linear_informed_initialization,
    count_parameters,
    fit_n4sid_state_space,
)
from run_cascaded_tanks_ablation import (
    dataset_to_tensors,
    make_batch_schedule,
    save_csv,
    scale_linear_realization_for_encoder,
    standardize_from_train,
    tensors_from_benchmark,
    threshold_rows,
    train_one_method,
)


def res_nonlinear_parameter_count(evolver: ResDyNetEvolver) -> int:
    return count_parameters(evolver.W_i) + count_parameters(evolver.W_o) + count_parameters(evolver.blocks)


def mlp_correction_parameter_count(evolver: LinearPlusMLPCorrectionEvolver) -> int:
    return count_parameters(evolver.correction)


def mlp_count(nx: int, nu: int, hidden_dims: tuple[int, ...]) -> int:
    dims = (nx + nu, *hidden_dims, nx)
    return sum(dims[i + 1] * (dims[i] + 1) for i in range(len(dims) - 1))


def matched_mlp_hidden_dims(
    target_params: int,
    nx: int,
    nu: int,
    min_depth: int = 1,
    max_depth: int = 12,
    max_width: int = 512,
) -> tuple[int, ...]:
    best_dims = (1,)
    best_gap = float("inf")
    for depth in range(min_depth, max_depth + 1):
        for width in range(1, max_width + 1):
            dims = (width,) * depth
            gap = abs(mlp_count(nx, nu, dims) - target_params)
            if gap < best_gap:
                best_gap = gap
                best_dims = dims
    return best_dims


def make_res_model(
    ny: int,
    nu: int,
    nx: int,
    n: int,
    stream_dim: int,
    num_blocks: int,
    hidden_width: int,
    hidden_layers: int,
    evolver_hidden_width: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> ResDyNet:
    torch.manual_seed(seed)
    hidden = (hidden_width,) * hidden_layers
    return ResDyNet(
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


def make_mlp_model(
    ny: int,
    nu: int,
    nx: int,
    n: int,
    stream_dim: int,
    num_blocks: int,
    hidden_width: int,
    hidden_layers: int,
    mlp_hidden_dims: tuple[int, ...],
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> ResDyNet:
    torch.manual_seed(seed)
    hidden = (hidden_width,) * hidden_layers
    evolver = LinearPlusMLPCorrectionEvolver(
        nx=nx,
        nu=nu,
        hidden_dims=mlp_hidden_dims,
        activation=nn.Tanh,
        zero_output=False,
    )
    return ResDyNet(
        ny=ny,
        nu=nu,
        nx=nx,
        na=n,
        nb=n,
        encoder_hidden_dims=hidden,
        decoder_hidden_dims=hidden,
        stream_dim=stream_dim,
        num_blocks=num_blocks,
        evolver_hidden_dims=(hidden_width,),
        activation=nn.Tanh,
        zero_nonlinear_outputs=False,
        evolver=evolver,
    ).to(device=device, dtype=dtype)


def save_mean_std_plots(output_dir: Path, trajectory_rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({str(row["method"]) for row in trajectory_rows})
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in methods:
        rows = [row for row in trajectory_rows if row["method"] == method]
        epochs = sorted({int(row["epoch"]) for row in rows})
        means, stds = [], []
        for epoch in epochs:
            vals = torch.tensor([float(row["val_nrmse"]) for row in rows if int(row["epoch"]) == epoch])
            means.append(torch.mean(vals).item())
            stds.append(torch.std(vals, unbiased=False).item())
        mean = torch.tensor(means)
        std = torch.tensor(stds)
        x = torch.tensor(epochs)
        ax.plot(x, mean, label=method)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NRMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "val_nrmse_vs_epoch_mean_std.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in methods:
        rows = [row for row in trajectory_rows if row["method"] == method]
        seeds = sorted({row["seed"] for row in rows})
        max_common_time = min(
            max(float(row["wall_time_sec"]) for row in rows if row["seed"] == seed)
            for seed in seeds
        )
        grid = torch.linspace(0.0, max_common_time, 100)
        curves = []
        for seed in seeds:
            seed_rows = sorted(
                [row for row in rows if row["seed"] == seed],
                key=lambda row: float(row["wall_time_sec"]),
            )
            times = torch.tensor([float(row["wall_time_sec"]) for row in seed_rows], dtype=torch.float64)
            vals = torch.tensor([float(row["val_nrmse"]) for row in seed_rows], dtype=torch.float64)
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
            curves.append(curve)
        stacked = torch.stack(curves, dim=0)
        mean = torch.mean(stacked, dim=0)
        std = torch.std(stacked, dim=0, unbiased=False)
        ax.plot(grid, mean, label=method)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.18)
    ax.set_xlabel("wall-clock time [s]")
    ax.set_ylabel("validation NRMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "val_nrmse_vs_time_mean_std.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="E2: residual evolver correction vs MLP correction.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(7)))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
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
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.3, 0.25, 0.2])
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/E2")
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
    ny, nu = train_y.shape[1], train_u.shape[1]
    stream_dim = args.stream_dim or 2 * (args.latent_dim + nu) + 1

    reference_res_evolver = ResDyNetEvolver(
        nx=args.latent_dim,
        nu=nu,
        stream_dim=stream_dim,
        num_blocks=args.num_blocks,
        block_hidden_dims=(args.evolver_hidden_width,),
    )
    n_theta_res = res_nonlinear_parameter_count(reference_res_evolver)
    mlp_hidden_dims = matched_mlp_hidden_dims(
        n_theta_res,
        nx=args.latent_dim,
        nu=nu,
        min_depth=1,
        max_depth=max(2, args.num_blocks * 2),
    )
    reference_mlp_evolver = LinearPlusMLPCorrectionEvolver(
        nx=args.latent_dim,
        nu=nu,
        hidden_dims=mlp_hidden_dims,
    )
    n_theta_mlp = mlp_correction_parameter_count(reference_mlp_evolver)
    rel_gap = abs(n_theta_mlp - n_theta_res) / n_theta_res
    if rel_gap > 0.05:
        raise ValueError(f"MLP correction parameter gap is {rel_gap:.2%}, expected <= 5%.")

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
    linear = scale_linear_realization_for_encoder(linear, n)

    output_dir = Path(args.output_dir)
    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    threshold_output_rows: list[dict[str, object]] = []

    print("E2 Cascaded Tanks: residual correction vs MLP correction")
    print(f"train/val/test samples: {len(train_u)}/{len(val_u)}/{len(test_u_norm)}")
    print(f"rollout windows train/val/test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    print(f"n={n}, H={args.horizon}, n_L={args.linear_order}, n_x={args.latent_dim}, B={args.num_blocks}, n_h={stream_dim}")
    print(f"N_theta_Res={n_theta_res}, N_theta_MLP={n_theta_mlp}, gap={rel_gap:.2%}, mlp_hidden_dims={mlp_hidden_dims}")

    for seed in args.seeds:
        schedule = make_batch_schedule(len(train_dataset), args.batch_size, args.epochs, seed=20_000 + seed)
        for method in ["ResDyNet-ResidualEvolver", "ResDyNet-MLPEvolver"]:
            if method == "ResDyNet-ResidualEvolver":
                model = make_res_model(
                    ny,
                    nu,
                    args.latent_dim,
                    n,
                    stream_dim,
                    args.num_blocks,
                    args.hidden_width,
                    args.hidden_layers,
                    args.evolver_hidden_width,
                    seed,
                    dtype,
                    device,
                )
            else:
                model = make_mlp_model(
                    ny,
                    nu,
                    args.latent_dim,
                    n,
                    stream_dim,
                    args.num_blocks,
                    args.hidden_width,
                    args.hidden_layers,
                    mlp_hidden_dims,
                    seed,
                    dtype,
                    device,
                )
            apply_linear_informed_initialization(model, linear, n=n)
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
                all_rows.append({"seed": seed, **row})
            best_val = min(float(row["val_nrmse"]) for row in history)
            final_val = float(history[-1]["val_nrmse"])
            summary_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "best_val_nrmse": best_val,
                    "final_val_nrmse": final_val,
                    "final_test_nrmse": test_score,
                    "nonlinear_parameters": n_theta_res if method == "ResDyNet-ResidualEvolver" else n_theta_mlp,
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
    save_csv(
        output_dir / "parameter_counts.csv",
        [
            {
                "model": "ResDyNet-ResidualEvolver",
                "nonlinear_parameters": n_theta_res,
                "relative_gap": 0.0,
                "hidden_dims": f"stream_dim={stream_dim}, blocks={args.num_blocks}, block_hidden={args.evolver_hidden_width}",
            },
            {
                "model": "ResDyNet-MLPEvolver",
                "nonlinear_parameters": n_theta_mlp,
                "relative_gap": rel_gap,
                "hidden_dims": str(mlp_hidden_dims),
            },
        ],
    )
    save_mean_std_plots(output_dir, all_rows)
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
