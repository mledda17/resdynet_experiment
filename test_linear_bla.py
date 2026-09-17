from __future__ import annotations

import argparse

import torch

from main import (
    estimate_markov_parameters_bla_from_trajectories,
    fit_bla_state_space_from_trajectories,
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


def predict_fir(markov: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    horizon = markov.shape[0] - 1
    y_hat = []
    for k in range(horizon, u.shape[0]):
        terms = [markov[lag] @ u[k - lag] for lag in range(horizon + 1)]
        y_hat.append(torch.stack(terms, dim=0).sum(dim=0))
    return torch.stack(y_hat, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and inspect the linear BLA state-space model.")
    parser.add_argument("--benchmark", default="WienerHammerBenchMark")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--fir-horizon", type=int, default=80)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    train_val_splits, _ = official_benchmark_split(
        args.benchmark,
        val_fraction=args.val_fraction,
        dtype=dtype,
    )
    train_trajectories = [(split.u_train, split.y_train) for split in train_val_splits]
    val_trajectories = [(split.u_val, split.y_val) for split in train_val_splits]

    markov = estimate_markov_parameters_bla_from_trajectories(
        train_trajectories,
        fir_horizon=args.fir_horizon,
        ridge=args.ridge,
    )
    linear = fit_bla_state_space_from_trajectories(
        train_trajectories,
        order=args.order,
        fir_horizon=args.fir_horizon,
        ridge=args.ridge,
    )

    print(f"benchmark: {args.benchmark}")
    print(f"order n_L: {linear.order}")
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

    fir_train_scores = []
    for u, y in train_trajectories:
        y_hat = predict_fir(markov, u)
        fir_train_scores.append(relative_mse(y_hat, y[args.fir_horizon :]))
    fir_val_scores = []
    for u, y in val_trajectories:
        y_hat = predict_fir(markov, u)
        fir_val_scores.append(relative_mse(y_hat, y[args.fir_horizon :]))

    train_scores = []
    for u, y in train_trajectories:
        train_scores.append(relative_mse(simulate_linear(linear, u), y))
    val_scores = []
    for u, y in val_trajectories:
        val_scores.append(relative_mse(simulate_linear(linear, u), y))

    print(f"FIR/BLA train relative MSE per trajectory: {fir_train_scores}")
    print(f"FIR/BLA validation relative MSE per trajectory: {fir_val_scores}")
    print(f"train relative MSE per trajectory: {train_scores}")
    print(f"validation relative MSE per trajectory: {val_scores}")


if __name__ == "__main__":
    main()
