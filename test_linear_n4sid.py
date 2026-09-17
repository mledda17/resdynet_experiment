from __future__ import annotations

import argparse
from pathlib import Path

import torch

from main import (
    fit_n4sid_state_space,
    official_benchmark_split,
)


def simulate_linear(linear, u: torch.Tensor, x0: torch.Tensor | None = None) -> torch.Tensor:
    if x0 is None:
        x = torch.zeros(linear.A.shape[0], dtype=u.dtype, device=u.device)
    else:
        x = x0.to(dtype=u.dtype, device=u.device)

    y_hat = []
    for k in range(u.shape[0]):
        y_hat.append(linear.C @ x + linear.D @ u[k])
        x = linear.A @ x + linear.B @ u[k]
    return torch.stack(y_hat, dim=0)


def relative_mse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    mse = torch.mean((y - y_hat) ** 2)
    variance = torch.mean((y - torch.mean(y, dim=0, keepdim=True)) ** 2)
    return (mse / variance.clamp_min(torch.finfo(y.dtype).eps)).item()


def save_performance_plot(
    train_y: torch.Tensor,
    train_y_hat: torch.Tensor,
    val_y: torch.Tensor,
    val_y_hat: torch.Tensor,
    output_path: Path,
    max_points: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_train = min(max_points, train_y.shape[0])
    n_val = min(max_points, val_y.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
    axes[0, 0].plot(train_y[:n_train, 0].cpu(), label="measured", linewidth=1.2)
    axes[0, 0].plot(train_y_hat[:n_train, 0].cpu(), label="N4SID", linewidth=1.0, alpha=0.85)
    axes[0, 0].set_title("Train: output")
    axes[0, 0].set_xlabel("sample")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[1, 0].plot((train_y[:n_train, 0] - train_y_hat[:n_train, 0]).cpu(), color="tab:red", linewidth=1.0)
    axes[1, 0].set_title("Train: residual")
    axes[1, 0].set_xlabel("sample")
    axes[1, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(val_y[:n_val, 0].cpu(), label="measured", linewidth=1.2)
    axes[0, 1].plot(val_y_hat[:n_val, 0].cpu(), label="N4SID", linewidth=1.0, alpha=0.85)
    axes[0, 1].set_title("Validation: output")
    axes[0, 1].set_xlabel("sample")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 1].plot((val_y[:n_val, 0] - val_y_hat[:n_val, 0]).cpu(), color="tab:red", linewidth=1.0)
    axes[1, 1].set_title("Validation: residual")
    axes[1, 1].set_xlabel("sample")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("N4SID linear model performance")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def standardize(train_u: torch.Tensor, train_y: torch.Tensor, *pairs: tuple[torch.Tensor, torch.Tensor]):
    u_mean = train_u.mean(dim=0, keepdim=True)
    u_std = train_u.std(dim=0, keepdim=True).clamp_min(torch.finfo(train_u.dtype).eps)
    y_mean = train_y.mean(dim=0, keepdim=True)
    y_std = train_y.std(dim=0, keepdim=True).clamp_min(torch.finfo(train_y.dtype).eps)
    out = [((train_u - u_mean) / u_std, (train_y - y_mean) / y_std)]
    for u, y in pairs:
        out.append(((u - u_mean) / u_std, (y - y_mean) / y_std))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and inspect a linear N4SID state-space model.")
    parser.add_argument("--benchmark", default="WienerHammerBenchMark")
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--num-block-rows", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--plot-path", default="outputs/n4sid_performance.png")
    parser.add_argument("--plot-points", type=int, default=1500)
    parser.add_argument("--keep-d", action="store_true", help="Keep N4SID estimated D instead of forcing D=0.")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    train_val_splits, _ = official_benchmark_split(
        args.benchmark,
        val_fraction=args.val_fraction,
        dtype=dtype,
    )
    if len(train_val_splits) != 1:
        raise ValueError("This inspection script expects one continuous train_val trajectory.")

    split = train_val_splits[0]
    train_u = split.u_train[: args.max_train_samples] if args.max_train_samples else split.u_train
    train_y = split.y_train[: args.max_train_samples] if args.max_train_samples else split.y_train
    val_u = split.u_val[: args.max_val_samples] if args.max_val_samples else split.u_val
    val_y = split.y_val[: args.max_val_samples] if args.max_val_samples else split.y_val
    (u_train, y_train), (u_val, y_val) = standardize(
        train_u,
        train_y,
        (val_u, val_y),
    )

    linear = fit_n4sid_state_space(
        u_train,
        y_train,
        order=args.order,
        num_block_rows=args.num_block_rows,
        zero_direct_feedthrough=not args.keep_d,
    )

    print(f"benchmark: {args.benchmark}")
    print(f"order n_L: {linear.order}")
    print(f"num_block_rows: {args.num_block_rows or max(args.order + 1, 2 * args.order)}")
    print(f"D forced to zero: {not args.keep_d}")
    print(f"A shape: {tuple(linear.A.shape)}")
    print(linear.A)
    print(f"B shape: {tuple(linear.B.shape)}")
    print(linear.B)
    print(f"C shape: {tuple(linear.C.shape)}")
    print(linear.C)
    print(f"D shape: {tuple(linear.D.shape)}")
    print(linear.D)

    eigenvalues = torch.linalg.eigvals(linear.A)
    print(f"A eigenvalues: {eigenvalues}")
    print(f"A spectral radius: {torch.max(torch.abs(eigenvalues)).item():.6g}")
    y_train_hat = simulate_linear(linear, u_train)
    y_val_hat = simulate_linear(linear, u_val)
    print(f"train relative MSE: {relative_mse(y_train_hat, y_train):.6g}")
    print(f"validation relative MSE: {relative_mse(y_val_hat, y_val):.6g}")
    save_performance_plot(
        y_train,
        y_train_hat,
        y_val,
        y_val_hat,
        Path(args.plot_path),
        max_points=args.plot_points,
    )
    print(f"saved plot: {args.plot_path}")


if __name__ == "__main__":
    main()
