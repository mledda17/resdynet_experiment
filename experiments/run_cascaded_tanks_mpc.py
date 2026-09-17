from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import LinearStateSpace, RolloutWindowDataset, fit_n4sid_state_space, reconstructability_matrix, weighted_multistep_loss
from run_cascaded_tanks_ablation import dataset_to_tensors, make_batch_schedule, make_model, nrmse, scale_linear_realization_for_encoder, standardize_from_train
from systems.cascaded_tanks import (
    CascadedTanks,
    CascadedTanksConfig,
    Trajectory,
    generate_piecewise_constant_inputs,
    save_dataset_npz,
    shifted_sine_sweep,
    sinusoidal_reference,
    steady_state_for_input,
)


def require_cvxpy():
    try:
        import cvxpy as cp
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install CVXPY and OSQP with `python3 -m pip install cvxpy osqp`.") from exc
    return cp


def as_np(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


class NumpyNormalizer:
    def __init__(self, norm_torch) -> None:
        self.u_mean = as_np(norm_torch.u_mean).reshape(1)
        self.u_std = as_np(norm_torch.u_std).reshape(1)
        self.y_mean = as_np(norm_torch.y_mean).reshape(1)
        self.y_std = as_np(norm_torch.y_std).reshape(1)

    def u_to_norm(self, u: np.ndarray) -> np.ndarray:
        return (u - self.u_mean) / self.u_std

    def u_from_norm(self, u: np.ndarray) -> np.ndarray:
        return u * self.u_std + self.u_mean

    def y_to_norm(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def y_from_norm(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


class CheckpointNormalizer:
    def __init__(self, state: dict[str, Tensor], dtype: torch.dtype) -> None:
        self.u_mean = state["u_mean"].to(dtype=dtype)
        self.u_std = state["u_std"].to(dtype=dtype)
        self.y_mean = state["y_mean"].to(dtype=dtype)
        self.y_std = state["y_std"].to(dtype=dtype)


def generate_dataset(args, plant_cfg: CascadedTanksConfig) -> dict[str, Trajectory]:
    plant = CascadedTanks(plant_cfg)
    x0 = steady_state_for_input(args.input_mean, plant_cfg)
    specs = {
        "train": (args.train_samples, args.seed + 1),
        "val": (args.val_samples, args.seed + 2),
        "test": (args.test_samples, args.seed + 3),
    }
    out = {}
    for name, (n_samples, seed) in specs.items():
        u = generate_piecewise_constant_inputs(
            n_samples=n_samples,
            hold_length=args.input_hold,
            mean=args.input_mean,
            std=args.input_std,
            u_min=args.u_min,
            u_max=args.u_max,
            seed=seed,
        )
        out[name] = plant.simulate(u, x0=x0)
    return out


def load_dataset_npz(path: Path) -> dict[str, Trajectory]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    data = np.load(path, allow_pickle=True)
    splits = {}
    for name in ("train", "val", "test"):
        required = [f"{name}_{suffix}" for suffix in ("t", "u", "y", "x")]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Dataset {path} is missing keys: {missing}")
        splits[name] = Trajectory(
            t=data[f"{name}_t"],
            u=data[f"{name}_u"],
            y=data[f"{name}_y"],
            x=data[f"{name}_x"],
        )
    return splits


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_resdynet(args, splits: dict[str, Trajectory], output_dir: Path, dtype: torch.dtype, device: torch.device):
    train_u_raw = torch.as_tensor(splits["train"].u, dtype=dtype)
    train_y_raw = torch.as_tensor(splits["train"].y, dtype=dtype)
    val_u_raw = torch.as_tensor(splits["val"].u, dtype=dtype)
    val_y_raw = torch.as_tensor(splits["val"].y, dtype=dtype)
    test_u_raw = torch.as_tensor(splits["test"].u, dtype=dtype)
    test_y_raw = torch.as_tensor(splits["test"].y, dtype=dtype)
    norm_torch, normalized = standardize_from_train(train_u_raw, train_y_raw, (val_u_raw, val_y_raw), (test_u_raw, test_y_raw))
    (train_u, train_y), (val_u, val_y), (test_u, test_y) = normalized

    linear = fit_n4sid_state_space(train_u, train_y, order=args.linear_order, num_block_rows=args.n4sid_block_rows, zero_direct_feedthrough=True)
    linear = scale_linear_realization_for_encoder(linear, args.history_length)
    stream_dim = args.stream_dim or 2 * (args.latent_dim + train_u.shape[1]) + 1
    model = make_model(
        ny=train_y.shape[1],
        nu=train_u.shape[1],
        nx=args.latent_dim,
        n=args.history_length,
        stream_dim=stream_dim,
        num_blocks=args.num_blocks,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        evolver_hidden_width=args.evolver_hidden_width,
        linear=linear,
        seed=args.seed,
        dtype=dtype,
        device=device,
    )

    train_ds = RolloutWindowDataset(train_u, train_y, args.history_length, args.history_length, args.identification_horizon)
    val_ds = RolloutWindowDataset(val_u, val_y, args.history_length, args.history_length, args.identification_horizon)
    test_ds = RolloutWindowDataset(test_u, test_y, args.history_length, args.history_length, args.identification_horizon)
    train_t = dataset_to_tensors(train_ds)
    val_t = dataset_to_tensors(val_ds)
    test_t = dataset_to_tensors(test_ds)
    schedule = make_batch_schedule(len(train_ds), args.batch_size, args.train_epochs, args.seed + 10_000)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    rows = []
    ckpt_path = output_dir / "checkpoints" / "resdynet_best.pt"
    start = time.perf_counter()

    for epoch in range(args.train_epochs + 1):
        val_score = nrmse(model, val_t, args.eval_batch_size)
        rows.append({"epoch": epoch, "val_nrmse": val_score, "wall_time_sec": time.perf_counter() - start})
        if val_score < best_val:
            best_val = val_score
            best_epoch = epoch
            bad_epochs = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(args),
                    "normalizer": {k: getattr(norm_torch, k).cpu() for k in ("u_mean", "u_std", "y_mean", "y_std")},
                    "linear": {k: getattr(linear, k).cpu() for k in ("A", "B", "C", "D")},
                    "epoch": epoch,
                    "val_nrmse": val_score,
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1
        if epoch == args.train_epochs or (args.patience and bad_epochs >= args.patience):
            break
        model.train()
        yp_all, up_all, uf_all, yf_all = train_t
        param = next(model.parameters())
        for idx in schedule[epoch]:
            yp = yp_all[idx].to(device=param.device, dtype=param.dtype)
            up = up_all[idx].to(device=param.device, dtype=param.dtype)
            uf = uf_all[idx].to(device=param.device, dtype=param.dtype)
            yf = yf_all[idx].to(device=param.device, dtype=param.dtype)
            opt.zero_grad(set_to_none=True)
            loss = weighted_multistep_loss(model(yp, up, uf), yf)
            loss.backward()
            if args.clip_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            opt.step()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    test_score = nrmse(model, test_t, args.eval_batch_size)
    save_csv(output_dir / "training_history.csv", rows)
    return model, linear, NumpyNormalizer(norm_torch), {"best_val_nrmse": best_val, "best_epoch": best_epoch, "test_open_loop_nrmse": test_score}


def load_trained_artifacts(checkpoint_path: Path, dtype: torch.dtype, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = argparse.Namespace(**ckpt["config"])
    nx = int(cfg.latent_dim)
    nu = int(ckpt["linear"]["B"].shape[1])
    ny = int(ckpt["linear"]["C"].shape[0])
    stream_dim = cfg.stream_dim or 2 * (nx + nu) + 1
    model = make_model(
        ny=ny,
        nu=nu,
        nx=nx,
        n=cfg.history_length,
        stream_dim=stream_dim,
        num_blocks=cfg.num_blocks,
        hidden_width=cfg.hidden_width,
        hidden_layers=cfg.hidden_layers,
        evolver_hidden_width=cfg.evolver_hidden_width,
        linear=None,
        seed=cfg.seed,
        dtype=dtype,
        device=device,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    linear = LinearStateSpace(**{k: v.to(dtype=dtype, device=device) for k, v in ckpt["linear"].items()})
    norm = NumpyNormalizer(CheckpointNormalizer(ckpt["normalizer"], dtype))
    open_loop = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch", ""),
        "checkpoint_val_nrmse": ckpt.get("val_nrmse", ""),
    }
    return model, linear, norm, open_loop, cfg


def tensor1(a: np.ndarray, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.as_tensor(a.reshape(1, -1), dtype=dtype, device=device)


def encode_resdynet(model, y_hist, u_hist, dtype, device):
    with torch.no_grad():
        return as_np(model.encode_regressor(torch.as_tensor(y_hist[None], dtype=dtype, device=device), torch.as_tensor(u_hist[None], dtype=dtype, device=device))).reshape(-1)


def encode_linear(linear: LinearStateSpace, y_hist: np.ndarray, u_hist: np.ndarray) -> np.ndarray:
    r = np.concatenate([y_hist.reshape(-1), u_hist.reshape(-1)])
    return as_np(reconstructability_matrix(linear, y_hist.shape[0]) @ torch.as_tensor(r, dtype=linear.A.dtype, device=linear.A.device)).reshape(-1)


def model_f(model, x, u, dtype, device):
    with torch.no_grad():
        return as_np(model.evolver(tensor1(x, dtype, device), tensor1(u, dtype, device))).reshape(-1)


def model_g(model, x, u, dtype, device):
    with torch.no_grad():
        return as_np(model.decoder(tensor1(np.concatenate([x, u]), dtype, device))).reshape(-1)


def nominal_rollout(model, x0, u_seq, dtype, device):
    xs = [x0.reshape(-1)]
    x = xs[0]
    for u in u_seq:
        x = model_f(model, x, u, dtype, device)
        xs.append(x)
    return np.asarray(xs)


def ltv_terms_resdynet(model, x_bar, u_bar, dtype, device):
    nx = model.nx
    out = [[] for _ in range(6)]
    for x_np, u_np in zip(x_bar[:-1], u_bar):
        xu0 = torch.as_tensor(np.concatenate([x_np, u_np]), dtype=dtype, device=device)

        def f_xu(xu: Tensor) -> Tensor:
            return model.evolver(xu[:nx].unsqueeze(0), xu[nx:].unsqueeze(0)).squeeze(0)

        def g_xu(xu: Tensor) -> Tensor:
            return model.decoder(xu.unsqueeze(0)).squeeze(0)

        jf = torch.autograd.functional.jacobian(f_xu, xu0).detach()
        jg = torch.autograd.functional.jacobian(g_xu, xu0).detach()
        f0, g0 = f_xu(xu0).detach(), g_xu(xu0).detach()
        A, B = as_np(jf[:, :nx]), as_np(jf[:, nx:])
        C, D = as_np(jg[:, :nx]), as_np(jg[:, nx:])
        c = as_np(f0) - A @ x_np - B @ u_np
        d = as_np(g0) - C @ x_np - D @ u_np
        for bucket, value in zip(out, (A, B, C, D, c, d)):
            bucket.append(value)
    return tuple(out)


def lti_terms(linear: LinearStateSpace, horizon: int):
    A, B, C, D = (as_np(v) for v in (linear.A, linear.B, linear.C, linear.D))
    return [A] * horizon, [B] * horizon, [C] * horizon, [D] * horizon, [np.zeros(A.shape[0])] * horizon, [np.zeros(C.shape[0])] * horizon


def solve_mpc(x0, u_prev, refs, u_ref, terms, args, norm, warm_u):
    cp = require_cvxpy()
    A, B, C, D, c, d = terms
    H, nx, nu, ny = len(A), x0.size, B[0].shape[1], C[0].shape[0]
    x, u, y = cp.Variable((nx, H + 1)), cp.Variable((nu, H)), cp.Variable((ny, H))
    cons = [x[:, 0] == x0]
    obj = 0.0
    u_min_n = float(norm.u_to_norm(np.array([args.u_min]))[0])
    u_max_n = float(norm.u_to_norm(np.array([args.u_max]))[0])
    u_ref_n = float(norm.u_to_norm(np.array([u_ref]))[0])
    for j in range(H):
        cons += [
            x[:, j + 1] == A[j] @ x[:, j] + B[j] @ u[:, j] + c[j],
            y[:, j] == C[j] @ x[:, j] + D[j] @ u[:, j] + d[j],
            u[:, j] >= u_min_n,
            u[:, j] <= u_max_n,
        ]
        du = u[:, j] - (u_prev if j == 0 else u[:, j - 1])
        obj += args.Wy * cp.sum_squares(y[:, j] - refs[j].reshape(ny))
        obj += args.Wu * cp.sum_squares(u[:, j] - u_ref_n)
        obj += args.Wdu * cp.sum_squares(du)
    prob = cp.Problem(cp.Minimize(obj), cons)
    u.value = warm_u.T
    start = time.perf_counter()
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, eps_abs=args.osqp_eps_abs, eps_rel=args.osqp_eps_rel, max_iter=args.osqp_max_iter)
    elapsed = time.perf_counter() - start
    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or u.value is None:
        raise RuntimeError(f"MPC QP failed with status {prob.status}")
    return np.asarray(u.value.T), float(prob.value), prob.status, elapsed


def build_reference(args, n_total: int) -> np.ndarray:
    if args.reference == "sine":
        return sinusoidal_reference(n_total, args.sampling_time, args.ref_offset, args.ref_amplitude, args.ref_frequency)
    return shifted_sine_sweep(n_total, args.sampling_time, args.ref_offset, args.ref_amplitude, args.ref_f_start, args.ref_f_end)


def run_closed_loop(args, model, linear, norm, dtype, device):
    n_total = args.mpc_steps + args.prediction_horizon + 1
    ref = build_reference(args, n_total)
    ref_n = norm.y_to_norm(ref)
    summaries, rows = [], []
    for controller, seed_add in [("linear_mpc", 0), ("resdynet_ltv_mpc", 1000)]:
        cfg = CascadedTanksConfig(sampling_time=args.sampling_time, rk4_substeps=args.rk4_substeps, seed=args.seed + seed_add)
        plant = CascadedTanks(cfg)
        plant.reset(steady_state_for_input(args.warmup_input, cfg))
        u_hist, y_hist = [], []
        for _ in range(args.history_length):
            y = plant.step(args.warmup_input)
            u_hist.append([args.warmup_input])
            y_hist.append([y])
        u_hist_n, y_hist_n = norm.u_to_norm(np.asarray(u_hist)), norm.y_to_norm(np.asarray(y_hist))
        u_prev = norm.u_to_norm(np.array([args.warmup_input]))
        warm_u = np.repeat(u_prev.reshape(1, 1), args.prediction_horizon, axis=0)
        costs, times, abs_errs, sq_errs, track_errs = [], [], [], [], []
        u_viol = 0
        for k in range(args.mpc_steps):
            refs = ref_n[k : k + args.prediction_horizon]
            if controller == "linear_mpc":
                x0 = encode_linear(linear, y_hist_n, u_hist_n)
                terms = lti_terms(linear, args.prediction_horizon)
            else:
                x0 = encode_resdynet(model, y_hist_n, u_hist_n, dtype, device)
                terms = ltv_terms_resdynet(model, nominal_rollout(model, x0, warm_u, dtype, device), warm_u, dtype, device)
            u_seq, cost, status, solve_time = solve_mpc(x0, u_prev, refs, args.u_ref, terms, args, norm, warm_u)
            u_phys = float(norm.u_from_norm(u_seq[0])[0])
            y = plant.step(u_phys)
            err = y - float(ref[k, 0])
            costs.append(cost)
            times.append(solve_time)
            abs_errs.append(abs(err))
            sq_errs.append(err * err)
            track_errs.append(err)
            u_viol += int(u_phys < args.u_min - 1e-8 or u_phys > args.u_max + 1e-8)
            rows.append({"controller": controller, "k": k, "t": k * args.sampling_time, "reference": float(ref[k, 0]), "y": y, "u": u_phys, "tracking_error": err, "x1_simulator_only": plant.x[0], "x2_simulator_only": plant.x[1], "mpc_cost": cost, "qp_status": status, "solve_time_sec": solve_time})
            y_hist_n = np.vstack([y_hist_n[1:], norm.y_to_norm(np.array([[y]])).reshape(1, 1)])
            u_hist_n = np.vstack([u_hist_n[1:], norm.u_to_norm(np.array([[u_phys]])).reshape(1, 1)])
            u_prev = u_seq[0]
            warm_u = np.vstack([u_seq[1:], u_seq[-1:]])
        summaries.append({"controller": controller, "tracking_rmse": math.sqrt(float(np.mean(sq_errs))), "accumulated_tracking_error": float(np.sum(np.square(track_errs))), "accumulated_mpc_cost": float(np.sum(costs)), "max_abs_tracking_error": float(np.max(abs_errs)), "actuator_constraint_violations": u_viol, "avg_qp_solve_time_sec": float(np.mean(times)), "max_qp_solve_time_sec": float(np.max(times))})
    return rows, summaries


def save_plots(output_dir: Path, rows: list[dict[str, object]], args) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    controller = "resdynet_ltv_mpc"
    color = "tab:blue"
    label = "ResDyNet LTV-MPC"
    base = [r for r in rows if r["controller"] == controller]
    fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
    ax.plot([r["t"] for r in base], [r["reference"] for r in base], "k--", linewidth=1.5, label="reference")
    ax.plot([r["t"] for r in base], [r["y"] for r in base], color=color, linewidth=1.5, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("output")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.savefig(output_dir / "reference_vs_output.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.0), constrained_layout=True)
    ax.step([r["t"] for r in base], [r["u"] for r in base], where="post", color=color, linewidth=1.4, label=label)
    ax.axhline(args.u_min, color="k", linestyle="--", linewidth=1.0, label="bounds")
    ax.axhline(args.u_max, color="k", linestyle="--", linewidth=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("input")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.savefig(output_dir / "control_inputs.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 3.8), constrained_layout=True)
    ax.plot([r["t"] for r in base], [r["tracking_error"] for r in base], color=color, linewidth=1.2, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("tracking error")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.savefig(output_dir / "tracking_error.png", dpi=220)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end Cascaded Tanks ResDyNet identification and MPC.")
    p.add_argument("--output-dir", default="results/end_to_end_cascaded_tanks")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampling-time", type=float, default=1.0)
    p.add_argument("--rk4-substeps", type=int, default=20)
    p.add_argument("--train-samples", type=int, default=20000)
    p.add_argument("--val-samples", type=int, default=4000)
    p.add_argument("--test-samples", type=int, default=2000)
    p.add_argument("--input-hold", type=int, default=5)
    p.add_argument("--input-mean", type=float, default=0.5)
    p.add_argument("--input-std", type=float, default=0.25)
    p.add_argument("--u-min", type=float, default=0.0)
    p.add_argument("--u-max", type=float, default=0.8)
    p.add_argument("--u-ref", type=float, default=0.5)
    p.add_argument("--warmup-input", type=float, default=0.5)
    p.add_argument("--history-length", type=int, default=50)
    p.add_argument("--identification-horizon", type=int, default=50)
    p.add_argument("--linear-order", type=int, default=6)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--stream-dim", type=int, default=None)
    p.add_argument("--hidden-width", type=int, default=64)
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--evolver-hidden-width", type=int, default=64)
    p.add_argument("--n4sid-block-rows", type=int, default=10)
    p.add_argument("--train-epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--mpc-steps", type=int, default=160)
    p.add_argument("--prediction-horizon", type=int, default=5)
    p.add_argument("--Wy", type=float, default=1.0)
    p.add_argument("--Wu", type=float, default=0.001)
    p.add_argument("--Wdu", type=float, default=0.01)
    p.add_argument("--reference", choices=("sweep", "sine"), default="sweep")
    p.add_argument("--ref-offset", type=float, default=0.45)
    p.add_argument("--ref-amplitude", type=float, default=0.25)
    p.add_argument("--ref-f-start", type=float, default=0.002)
    p.add_argument("--ref-f-end", type=float, default=0.02)
    p.add_argument("--ref-frequency", type=float, default=0.01)
    p.add_argument("--osqp-eps-abs", type=float, default=1e-5)
    p.add_argument("--osqp-eps-rel", type=float, default=1e-5)
    p.add_argument("--osqp-max-iter", type=int, default=10000)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.latent_dim < args.linear_order and not args.skip_training:
        raise ValueError("--latent-dim must be >= --linear-order.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    plant_cfg = CascadedTanksConfig(sampling_time=args.sampling_time, rk4_substeps=args.rk4_substeps, seed=args.seed)
    if args.skip_training:
        if args.checkpoint is None:
            raise ValueError("--skip-training requires --checkpoint.")
        if args.dataset is None:
            raise ValueError("--skip-training requires --dataset.")
        load_dataset_npz(Path(args.dataset))
        model, linear, norm, open_loop, train_cfg = load_trained_artifacts(Path(args.checkpoint), dtype, device)
        args.history_length = int(train_cfg.history_length)
        args.identification_horizon = int(train_cfg.identification_horizon)
        args.latent_dim = int(train_cfg.latent_dim)
        args.linear_order = int(train_cfg.linear_order)
        metrics_path = Path(args.checkpoint).parents[1] / "open_loop_metrics.json"
        if metrics_path.exists():
            open_loop = {**json.loads(metrics_path.read_text()), **open_loop}
    else:
        splits = generate_dataset(args, plant_cfg)
        save_dataset_npz(output_dir / "dataset.npz", plant_cfg, splits, {"args": vars(args)})
        model, linear, norm, open_loop = train_resdynet(args, splits, output_dir, dtype, device)
    rows, summary = run_closed_loop(args, model, linear, norm, dtype, device)
    save_csv(output_dir / "closed_loop_trajectories.csv", rows)
    save_csv(output_dir / "summary.csv", summary)
    (output_dir / "open_loop_metrics.json").write_text(json.dumps(open_loop, indent=2))
    (output_dir / "config.json").write_text(json.dumps({"args": vars(args), "plant": asdict(plant_cfg)}, indent=2))
    save_plots(output_dir, rows, args)
    print(f"open-loop test NRMSE: {open_loop['test_open_loop_nrmse']:.6g}")
    print(f"saved end-to-end Cascaded Tanks results to {output_dir}")
    for row in summary:
        print(
            f"{row['controller']}: RMSE={row['tracking_rmse']:.6g}, "
            f"acc_track={row['accumulated_tracking_error']:.6g}, cost={row['accumulated_mpc_cost']:.6g}, "
            f"max_abs={row['max_abs_tracking_error']:.6g}, u_viol={row['actuator_constraint_violations']}, "
            f"avg_qp={row['avg_qp_solve_time_sec']:.4g}s"
        )


if __name__ == "__main__":
    main()
