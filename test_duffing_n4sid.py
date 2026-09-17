from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import torch
from torch import Tensor

from main import LinearStateSpace, fit_n4sid_state_space, observability_matrix, reconstructability_matrix


@dataclass(frozen=True)
class DuffingSplit:
    u: Tensor
    y: Tensor
    x: Tensor | None = None


@dataclass(frozen=True)
class Normalizer:
    u_mean: Tensor
    u_std: Tensor
    y_mean: Tensor
    y_std: Tensor

    def uy(self, split: DuffingSplit) -> tuple[Tensor, Tensor]:
        return (split.u - self.u_mean) / self.u_std, (split.y - self.y_mean) / self.y_std

    def y_from_norm(self, y: Tensor) -> Tensor:
        return y * self.y_std + self.y_mean


def read_trajectory(group, dtype: torch.dtype) -> DuffingSplit:
    def read(key: str) -> Tensor:
        arr = torch.as_tensor(group[key][()], dtype=dtype)
        if arr.ndim == 2:
            arr = arr.T
        if arr.ndim == 1:
            arr = arr.unsqueeze(-1)
        return arr.contiguous()

    return DuffingSplit(u=read("u"), y=read("y"), x=read("x") if "x" in group else None)


def load_duffing_dataset(path: Path, dtype: torch.dtype) -> tuple[dict[str, DuffingSplit], dict[str, float]]:
    with h5py.File(path, "r") as f:
        splits = {name: read_trajectory(f[name], dtype) for name in ("train", "val", "test")}
        cfg = {key: float(f["cfg"][key][()][0, 0]) for key in f["cfg"].keys() if f["cfg"][key].shape == (1, 1)}
    return splits, cfg


def make_normalizer(train: DuffingSplit) -> Normalizer:
    eps = torch.finfo(train.u.dtype).eps
    return Normalizer(
        u_mean=train.u.mean(dim=0, keepdim=True),
        u_std=train.u.std(dim=0, keepdim=True).clamp_min(eps),
        y_mean=train.y.mean(dim=0, keepdim=True),
        y_std=train.y.std(dim=0, keepdim=True).clamp_min(eps),
    )


def rmse(y_hat: Tensor, y: Tensor) -> float:
    return torch.sqrt(torch.mean((y_hat - y) ** 2)).item()


def nrmse(y_hat: Tensor, y: Tensor) -> float:
    return rmse(y_hat, y) / torch.std(y).clamp_min(torch.finfo(y.dtype).eps).item()


def bfr(y_hat: Tensor, y: Tensor) -> float:
    numerator = torch.linalg.norm((y - y_hat).reshape(-1))
    denominator = torch.linalg.norm((y - y.mean(dim=0, keepdim=True)).reshape(-1)).clamp_min(torch.finfo(y.dtype).eps)
    return 100.0 * (1.0 - numerator / denominator).item()


def controllability_matrix(A: Tensor, B: Tensor) -> Tensor:
    blocks = []
    Apow = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    for _ in range(A.shape[0]):
        blocks.append(Apow @ B)
        Apow = A @ Apow
    return torch.cat(blocks, dim=1)


def matrix_rank(M: Tensor) -> int:
    return int(torch.linalg.matrix_rank(M).item())


def reconstruct_states(linear: LinearStateSpace, u: Tensor, y: Tensor, history: int) -> tuple[Tensor, Tensor]:
    R = reconstructability_matrix(linear, history)
    states = []
    indices = []
    for k in range(history, u.shape[0]):
        regressor = torch.cat((y[k - history : k].reshape(-1), u[k - history : k].reshape(-1)))
        states.append(R @ regressor)
        indices.append(k)
    return torch.stack(states, dim=0), torch.tensor(indices, dtype=torch.long)


def reconstructed_one_step_prediction(linear: LinearStateSpace, u: Tensor, y: Tensor, history: int) -> tuple[Tensor, Tensor]:
    x_hat, idx = reconstruct_states(linear, u, y, history)
    y_hat = torch.stack([linear.C @ x_hat[i] + linear.D @ u[int(k)] for i, k in enumerate(idx)], dim=0)
    return y_hat, y[idx]


def open_loop_simulation(linear: LinearStateSpace, u: Tensor, y: Tensor, history: int) -> tuple[Tensor, Tensor]:
    x0, idx = reconstruct_states(linear, u, y, history)
    start = int(idx[0])
    x = x0[0]
    y_hat = []
    for k in range(start, u.shape[0]):
        y_hat.append(linear.C @ x + linear.D @ u[k])
        x = linear.A @ x + linear.B @ u[k]
    return torch.stack(y_hat, dim=0), y[start:]


def finite_metrics(y_hat: Tensor, y: Tensor) -> tuple[float, float]:
    if not torch.isfinite(y_hat).all():
        return float("inf"), float("inf")
    return rmse(y_hat, y), nrmse(y_hat, y)


def format_complex(values: Tensor) -> str:
    out = []
    for z in values.detach().cpu():
        out.append(f"{z.real.item():.8g}{z.imag.item():+.8g}j")
    return ";".join(out)


def evaluate_order(
    order: int,
    train_u: Tensor,
    train_y: Tensor,
    val_u: Tensor,
    val_y: Tensor,
    normalizer: Normalizer,
    history: int,
    num_block_rows: int | None,
    keep_d: bool,
) -> tuple[LinearStateSpace, dict[str, object], Tensor, Tensor]:
    linear = fit_n4sid_state_space(
        train_u,
        train_y,
        order=order,
        num_block_rows=num_block_rows,
        zero_direct_feedthrough=not keep_d,
    )
    pred_norm, pred_target_norm = reconstructed_one_step_prediction(linear, val_u, val_y, history)
    sim_norm, sim_target_norm = open_loop_simulation(linear, val_u, val_y, history)
    pred = normalizer.y_from_norm(pred_norm)
    pred_target = normalizer.y_from_norm(pred_target_norm)
    sim = normalizer.y_from_norm(sim_norm)
    sim_target = normalizer.y_from_norm(sim_target_norm)
    pred_rmse, pred_nrmse = finite_metrics(pred, pred_target)
    sim_rmse, sim_nrmse = finite_metrics(sim, sim_target)
    eig = torch.linalg.eigvals(linear.A)
    spectral_radius = torch.max(torch.abs(eig)).item()
    obs_rank = matrix_rank(observability_matrix(linear.A, linear.C, order))
    ctrb_rank = matrix_rank(controllability_matrix(linear.A, linear.B))
    row = {
        "order": order,
        "validation_rmse": pred_rmse,
        "validation_nrmse": pred_nrmse,
        "validation_open_loop_rmse": sim_rmse,
        "validation_open_loop_nrmse": sim_nrmse,
        "spectral_radius": spectral_radius,
        "stable": spectral_radius < 1.0,
        "observability_rank": obs_rank,
        "controllability_rank": ctrb_rank,
        "eigenvalues": format_complex(eig),
    }
    return linear, row, sim, sim_target


def save_matrix(path: Path, linear: LinearStateSpace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "A_L": linear.A.detach().cpu().tolist(),
        "B_L": linear.B.detach().cpu().tolist(),
        "C_L": linear.C.detach().cpu().tolist(),
        "D_L": linear.D.detach().cpu().tolist(),
        "eigenvalues": format_complex(torch.linalg.eigvals(linear.A)),
    }
    path.write_text(json.dumps(payload, indent=2))


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(
    output_dir: Path,
    val_y: Tensor,
    val_y_hat: Tensor,
    test_y: Tensor,
    test_y_hat: Tensor,
    rows: list[dict[str, object]],
    max_points: int,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    n_val = min(max_points, val_y.shape[0])
    n_test = min(max_points, test_y.shape[0])

    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    ax.plot(val_y[:n_val, 0].cpu(), label="measured", linewidth=1.2)
    ax.plot(val_y_hat[:n_val, 0].cpu(), label="N4SID open-loop", linewidth=1.0)
    ax.set_xlabel("sample")
    ax.set_ylabel("output")
    ax.set_title("Validation open-loop output")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "validation_open_loop_output.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    ax.plot(test_y[:n_test, 0].cpu(), label="measured", linewidth=1.2)
    ax.plot(test_y_hat[:n_test, 0].cpu(), label="N4SID open-loop", linewidth=1.0)
    ax.set_xlabel("sample")
    ax.set_ylabel("output")
    ax.set_title("Test open-loop output")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "test_open_loop_output.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    ax.plot((test_y[:n_test, 0] - test_y_hat[:n_test, 0]).cpu(), color="tab:red", linewidth=1.0)
    ax.set_xlabel("sample")
    ax.set_ylabel("prediction error")
    ax.set_title("Test open-loop prediction error")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "test_prediction_error.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot([int(row["order"]) for row in rows], [float(row["validation_open_loop_nrmse"]) for row in rows], marker="o")
    ax.set_xlabel("model order")
    ax.set_ylabel("validation open-loop NRMSE")
    ax.set_title("N4SID model-order selection")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "validation_nrmse_vs_order.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate N4SID linear identification on the Duffing dataset.")
    parser.add_argument("--dataset", default="duffing_resdynet_dataset.mat")
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8, 10])
    parser.add_argument("--history", type=int, default=50)
    parser.add_argument("--num-block-rows", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="results/duffing_n4sid")
    parser.add_argument("--plot-points", type=int, default=3000)
    parser.add_argument("--keep-d", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser.parse_args()


def maybe_truncate(split: DuffingSplit, n: int | None) -> DuffingSplit:
    if n is None:
        return split
    return DuffingSplit(
        u=split.u[:n],
        y=split.y[:n],
        x=None if split.x is None else split.x[:n],
    )


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    splits, cfg = load_duffing_dataset(Path(args.dataset), dtype)
    train = maybe_truncate(splits["train"], args.max_train_samples)
    val = maybe_truncate(splits["val"], args.max_val_samples)
    test = maybe_truncate(splits["test"], args.max_test_samples)
    if min(train.u.shape[0], val.u.shape[0], test.u.shape[0]) <= args.history:
        raise ValueError("--history must be smaller than every split length.")

    normalizer = make_normalizer(train)
    train_u, train_y = normalizer.uy(train)
    val_u, val_y = normalizer.uy(val)
    test_u, test_y = normalizer.uy(test)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    models: dict[int, LinearStateSpace] = {}
    val_sims: dict[int, tuple[Tensor, Tensor]] = {}
    print(f"dataset: {args.dataset}")
    print(f"samples train/val/test: {train.u.shape[0]}/{val.u.shape[0]}/{test.u.shape[0]}")
    print(f"Ts: {cfg.get('Ts', 'unknown')}")
    for order in args.orders:
        linear, row, val_sim, val_target = evaluate_order(
            order=order,
            train_u=train_u,
            train_y=train_y,
            val_u=val_u,
            val_y=val_y,
            normalizer=normalizer,
            history=args.history,
            num_block_rows=args.num_block_rows,
            keep_d=args.keep_d,
        )
        rows.append(row)
        models[order] = linear
        val_sims[order] = (val_sim, val_target)
        print(
            f"order={order}: val NRMSE={row['validation_nrmse']:.6g}, "
            f"open-loop NRMSE={row['validation_open_loop_nrmse']:.6g}, "
            f"rho(A)={row['spectral_radius']:.6g}, stable={row['stable']}"
        )

    save_csv(output_dir / "order_sweep.csv", rows)
    selected = min(rows, key=lambda row: float(row["validation_open_loop_nrmse"]))
    selected_order = int(selected["order"])
    selected_linear = models[selected_order]
    test_sim_norm, test_target_norm = open_loop_simulation(selected_linear, test_u, test_y, args.history)
    test_pred = normalizer.y_from_norm(test_sim_norm)
    test_target = normalizer.y_from_norm(test_target_norm)
    test_rmse = rmse(test_pred, test_target)
    test_nrmse = nrmse(test_pred, test_target)
    test_bfr = bfr(test_pred, test_target)
    eig = torch.linalg.eigvals(selected_linear.A)
    test_summary = {
        "selected_order": selected_order,
        "selection_metric": "validation_open_loop_nrmse",
        "test_open_loop_rmse": test_rmse,
        "test_open_loop_nrmse": test_nrmse,
        "test_bfr": test_bfr,
        "spectral_radius": torch.max(torch.abs(eig)).item(),
        "stable": bool(torch.max(torch.abs(eig)).item() < 1.0),
        "eigenvalues": format_complex(eig),
    }
    (output_dir / "selected_test_summary.json").write_text(json.dumps(test_summary, indent=2))
    save_matrix(output_dir / "selected_linear_model.json", selected_linear)

    val_pred, val_target = val_sims[selected_order]
    save_plots(output_dir, val_target, val_pred, test_target, test_pred, rows, args.plot_points)

    print(f"selected order: {selected_order}")
    print(f"test RMSE: {test_rmse:.6g}")
    print(f"test NRMSE: {test_nrmse:.6g}")
    print(f"test BFR: {test_bfr:.6g}")
    print(f"A_L:\n{selected_linear.A}")
    print(f"B_L:\n{selected_linear.B}")
    print(f"C_L:\n{selected_linear.C}")
    print(f"D_L:\n{selected_linear.D}")
    print(f"eig(A_L): {eig}")
    print(f"saved results to {output_dir}")


if __name__ == "__main__":
    main()
