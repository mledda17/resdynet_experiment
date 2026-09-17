from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from main import (
    LinearStateSpace,
    ResDyNet,
    RolloutWindowDataset,
    apply_linear_informed_initialization,
    fit_n4sid_state_space,
    reconstructability_matrix,
)
from run_cascaded_tanks_ablation import (
    dataset_to_tensors,
    make_batch_schedule,
    make_model,
    nrmse,
    scale_linear_realization_for_encoder,
    standardize_from_train,
    tensors_from_benchmark,
)


DEFAULT_CONFIG = {
    "checkpoint": None,
    "output_dir": "results/mpc/cascaded_tanks",
    "seed": 0,
    "train_epochs": 60,
    "batch_size": 128,
    "eval_batch_size": 256,
    "learning_rate": 1e-3,
    "clip_grad_norm": 1.0,
    "val_fraction": 0.2,
    "linear_order": 6,
    "latent_dim": 8,
    "history_length": 50,
    "identification_horizon": 50,
    "num_blocks": 4,
    "stream_dim": None,
    "hidden_width": 64,
    "hidden_layers": 2,
    "evolver_hidden_width": 64,
    "n4sid_block_rows": 10,
    "simulation_steps": 80,
    "prediction_horizon": 20,
    "q_y": 20.0,
    "r_u": 0.02,
    "s_du": 0.2,
    "u_min": 0.4,
    "u_max": 6.5,
    "y_min": 0.0,
    "y_max": 10.0,
    "output_slack_weight": 1e3,
    "reference_value": 4.0,
    "u_ref": 1.7,
    "initial_state": [2.25, 3.24],
    "warmup_input": 1.5,
    "rk4_substeps": 20,
    "process_noise_std": 0.0,
    "measurement_noise_std": 0.0,
    "osqp_eps_abs": 1e-5,
    "osqp_eps_rel": 1e-5,
    "osqp_max_iter": 10000,
    "dtype": "float64",
    "device": "cpu",
}

TANK_PARAMETERS = {"k1": 0.5, "k2": 0.4, "k3": 0.2, "k4": 0.3}


@dataclass(frozen=True)
class Normalizer:
    u_mean: np.ndarray
    u_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def u_to_norm(self, u: np.ndarray) -> np.ndarray:
        return (u - self.u_mean) / self.u_std

    def u_from_norm(self, u: np.ndarray) -> np.ndarray:
        return u * self.u_std + self.u_mean

    def y_to_norm(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def y_from_norm(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


@dataclass
class CascadedTanksPlant:
    Ts: float
    x: np.ndarray
    rk4_substeps: int
    process_noise_std: float
    measurement_noise_std: float
    rng: np.random.Generator
    state_projection_count: int = 0

    def rhs(self, x: np.ndarray, u: float, w: np.ndarray) -> np.ndarray:
        if np.any(x < -1e-9):
            raise RuntimeError(f"Plant state left physical domain before sqrt: {x}")
        x_safe = np.maximum(x, 0.0)
        k1, k2, k3, k4 = (TANK_PARAMETERS[k] for k in ("k1", "k2", "k3", "k4"))
        return np.array(
            [
                -k1 * math.sqrt(x_safe[0]) + k4 * u + w[0],
                k2 * math.sqrt(x_safe[0]) - k3 * math.sqrt(x_safe[1]) + w[1],
            ],
            dtype=float,
        )

    def measured_output(self) -> float:
        return float(self.x[1] + self.rng.normal(0.0, self.measurement_noise_std))

    def step(self, u: float) -> float:
        y = self.measured_output()
        w = self.rng.normal(0.0, self.process_noise_std, size=2)
        dt = self.Ts / self.rk4_substeps
        for _ in range(self.rk4_substeps):
            x0 = self.x
            k1 = self.rhs(x0, u, w)
            k2 = self.rhs(x0 + 0.5 * dt * k1, u, w)
            k3 = self.rhs(x0 + 0.5 * dt * k2, u, w)
            k4 = self.rhs(x0 + dt * k3, u, w)
            self.x = x0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if np.any(self.x < 0.0):
                if np.any(self.x < -1e-7):
                    self.state_projection_count += 1
                self.x = np.maximum(self.x, 0.0)
        return y


def require_cvxpy():
    try:
        import cvxpy as cp
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install CVXPY and OSQP with `python3 -m pip install cvxpy osqp`.") from exc
    return cp


def as_np(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def load_benchmark(dtype: torch.dtype, val_fraction: float):
    import nonlinear_benchmarks

    train_val, test = nonlinear_benchmarks.Cascaded_Tanks(atleast_2d=True)
    train_val_u, train_val_y = tensors_from_benchmark(train_val, dtype=dtype)
    test_u, test_y = tensors_from_benchmark(test, dtype=dtype)
    split_idx = int(train_val_u.shape[0] * (1.0 - val_fraction))
    train_u_raw, train_y_raw = train_val_u[:split_idx], train_val_y[:split_idx]
    val_u_raw, val_y_raw = train_val_u[split_idx:], train_val_y[split_idx:]
    norm_torch, normalized = standardize_from_train(train_u_raw, train_y_raw, (val_u_raw, val_y_raw), (test_u, test_y))
    (train_u, train_y), (val_u, val_y), (test_u_norm, test_y_norm) = normalized
    norm = Normalizer(
        u_mean=as_np(norm_torch.u_mean).reshape(1),
        u_std=as_np(norm_torch.u_std).reshape(1),
        y_mean=as_np(norm_torch.y_mean).reshape(1),
        y_std=as_np(norm_torch.y_std).reshape(1),
    )
    return train_val, train_u, train_y, val_u, val_y, test_u_norm, test_y_norm, norm, norm_torch


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    model: ResDyNet,
    normalizer_torch,
    linear: LinearStateSpace,
    epoch: int,
    val_nrmse: float,
    sampling_time: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
            "normalizer": {
                "u_mean": normalizer_torch.u_mean.cpu(),
                "u_std": normalizer_torch.u_std.cpu(),
                "y_mean": normalizer_torch.y_mean.cpu(),
                "y_std": normalizer_torch.y_std.cpu(),
            },
            "linear": {k: getattr(linear, k).cpu() for k in ("A", "B", "C", "D")},
            "sampling_time": sampling_time,
            "tank_parameters": TANK_PARAMETERS,
            "epoch": epoch,
            "val_nrmse": val_nrmse,
        },
        path,
    )


def train_or_load_model(args: argparse.Namespace, dtype: torch.dtype, device: torch.device):
    train_meta, train_u, train_y, val_u, val_y, test_u, test_y, norm, norm_torch = load_benchmark(dtype, args.val_fraction)
    sampling_time = float(train_meta.sampling_time)
    n = args.history_length
    stream_dim = args.stream_dim or 2 * (args.latent_dim + train_u.shape[1]) + 1
    linear = fit_n4sid_state_space(
        train_u,
        train_y,
        order=args.linear_order,
        num_block_rows=args.n4sid_block_rows,
        zero_direct_feedthrough=True,
    )
    linear = scale_linear_realization_for_encoder(linear, n)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = argparse.Namespace(**ckpt["config"])
        model = make_model(
            ny=train_y.shape[1],
            nu=train_u.shape[1],
            nx=cfg.latent_dim,
            n=cfg.history_length,
            stream_dim=cfg.stream_dim or 2 * (cfg.latent_dim + train_u.shape[1]) + 1,
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
        ckpt_norm = Normalizer(
            u_mean=as_np(ckpt["normalizer"]["u_mean"].to(dtype=dtype)).reshape(1),
            u_std=as_np(ckpt["normalizer"]["u_std"].to(dtype=dtype)).reshape(1),
            y_mean=as_np(ckpt["normalizer"]["y_mean"].to(dtype=dtype)).reshape(1),
            y_std=as_np(ckpt["normalizer"]["y_std"].to(dtype=dtype)).reshape(1),
        )
        ckpt_linear = LinearStateSpace(**{k: v.to(dtype=dtype, device=device) for k, v in ckpt["linear"].items()})
        return model, ckpt_linear, ckpt_norm, float(ckpt.get("sampling_time", sampling_time)), ckpt

    model = make_model(
        ny=train_y.shape[1],
        nu=train_u.shape[1],
        nx=args.latent_dim,
        n=n,
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
    train_dataset = RolloutWindowDataset(train_u, train_y, n, n, args.identification_horizon)
    val_dataset = RolloutWindowDataset(val_u, val_y, n, n, args.identification_horizon)
    test_dataset = RolloutWindowDataset(test_u, test_y, n, n, args.identification_horizon)
    train_tensors = dataset_to_tensors(train_dataset)
    val_tensors = dataset_to_tensors(val_dataset)
    test_tensors = dataset_to_tensors(test_dataset)
    schedule = make_batch_schedule(len(train_dataset), args.batch_size, args.train_epochs, seed=30_000 + args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val = float("inf")
    best_epoch = 0
    ckpt = {}
    checkpoint_path = Path(args.output_dir) / "cascaded_tanks_checkpoints" / f"seed_{args.seed}_best.pt"
    start = time.perf_counter()
    for epoch in range(args.train_epochs + 1):
        val_score = nrmse(model, val_tensors, args.eval_batch_size)
        if val_score < best_val:
            best_val = val_score
            best_epoch = epoch
            save_checkpoint(checkpoint_path, args, model, norm_torch, linear, epoch, val_score, sampling_time)
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        if epoch == args.train_epochs:
            break
        model.train()
        param = next(model.parameters())
        y_past, u_past, u_future, y_future = train_tensors
        for idx in schedule[epoch]:
            yp = y_past[idx].to(device=param.device, dtype=param.dtype)
            up = u_past[idx].to(device=param.device, dtype=param.dtype)
            uf = u_future[idx].to(device=param.device, dtype=param.dtype)
            yf = y_future[idx].to(device=param.device, dtype=param.dtype)
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(yp, up, uf) - yf) ** 2).mean()
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
    test_score = nrmse(model, test_tensors, args.eval_batch_size)
    print(
        f"identified ResDyNet: best val NRMSE={best_val:.5g} at epoch {best_epoch}, "
        f"final test NRMSE={test_score:.5g}, train time={time.perf_counter() - start:.2f}s"
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, linear, norm, sampling_time, ckpt


def tensor1(array: np.ndarray, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.as_tensor(array.reshape(1, -1), dtype=dtype, device=device)


def encode_state(model: ResDyNet, y_hist: np.ndarray, u_hist: np.ndarray, dtype: torch.dtype, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        y_past = torch.as_tensor(y_hist[None, :, :], dtype=dtype, device=device)
        u_past = torch.as_tensor(u_hist[None, :, :], dtype=dtype, device=device)
        return as_np(model.encode_regressor(y_past, u_past)).reshape(-1)


def model_f(model: ResDyNet, x: np.ndarray, u: np.ndarray, dtype: torch.dtype, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        return as_np(model.evolver(tensor1(x, dtype, device), tensor1(u, dtype, device))).reshape(-1)


def model_g(model: ResDyNet, x: np.ndarray, u: np.ndarray, dtype: torch.dtype, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        xu = np.concatenate([x.reshape(-1), u.reshape(-1)])
        return as_np(model.decoder(tensor1(xu, dtype, device))).reshape(-1)


def nominal_rollout(model: ResDyNet, x0: np.ndarray, u_seq: np.ndarray, dtype: torch.dtype, device: torch.device):
    xs = [x0.reshape(-1)]
    ys = []
    x = xs[0]
    for u in u_seq:
        ys.append(model_g(model, x, u, dtype, device))
        x = model_f(model, x, u, dtype, device)
        xs.append(x)
    return np.asarray(xs), np.asarray(ys)


def linearize_resdynet(model: ResDyNet, x_bar: np.ndarray, u_bar: np.ndarray, dtype: torch.dtype, device: torch.device):
    A_list, B_list, C_list, D_list, c_list, d_list = [], [], [], [], [], []
    nx = model.nx
    for x_np, u_np in zip(x_bar[:-1], u_bar):
        xu0 = torch.as_tensor(np.concatenate([x_np, u_np]), dtype=dtype, device=device)

        def f_xu(xu: Tensor) -> Tensor:
            return model.evolver(xu[:nx].unsqueeze(0), xu[nx:].unsqueeze(0)).squeeze(0)

        def g_xu(xu: Tensor) -> Tensor:
            return model.decoder(xu.unsqueeze(0)).squeeze(0)

        jac_f = torch.autograd.functional.jacobian(f_xu, xu0, create_graph=False).detach()
        jac_g = torch.autograd.functional.jacobian(g_xu, xu0, create_graph=False).detach()
        f0 = f_xu(xu0).detach()
        g0 = g_xu(xu0).detach()
        A = as_np(jac_f[:, :nx])
        B = as_np(jac_f[:, nx:])
        C = as_np(jac_g[:, :nx])
        D = as_np(jac_g[:, nx:])
        c = as_np(f0) - A @ x_np - B @ u_np
        d = as_np(g0) - C @ x_np - D @ u_np
        A_list.append(A)
        B_list.append(B)
        C_list.append(C)
        D_list.append(D)
        c_list.append(c)
        d_list.append(d)
    return A_list, B_list, C_list, D_list, c_list, d_list


def encode_linear_state(linear: LinearStateSpace, y_hist: np.ndarray, u_hist: np.ndarray) -> np.ndarray:
    dtype = linear.A.dtype
    device = linear.A.device
    r = np.concatenate([y_hist.reshape(-1), u_hist.reshape(-1)])
    r_t = torch.as_tensor(r, dtype=dtype, device=device)
    return as_np(reconstructability_matrix(linear, y_hist.shape[0]) @ r_t).reshape(-1)


def identified_linear_model(linear: LinearStateSpace, horizon: int):
    A = as_np(linear.A)
    B = as_np(linear.B)
    C = as_np(linear.C)
    D = as_np(linear.D)
    nx = A.shape[0]
    ny = C.shape[0]
    return (
        [A] * horizon,
        [B] * horizon,
        [C] * horizon,
        [D] * horizon,
        [np.zeros(nx)] * horizon,
        [np.zeros(ny)] * horizon,
    )


def solve_ltv_mpc(x0, u_prev, refs, u_refs, model_terms, weights, bounds, solver_opts, warm_u):
    cp = require_cvxpy()
    A_list, B_list, C_list, D_list, c_list, d_list = model_terms
    horizon = len(A_list)
    nx = x0.size
    nu = B_list[0].shape[1]
    ny = C_list[0].shape[0]
    x = cp.Variable((nx, horizon + 1))
    u = cp.Variable((nu, horizon))
    y = cp.Variable((ny, horizon))
    y_slack_low = cp.Variable((ny, horizon), nonneg=True)
    y_slack_high = cp.Variable((ny, horizon), nonneg=True)
    constraints = [x[:, 0] == x0]
    objective = 0.0
    for j in range(horizon):
        constraints += [
            x[:, j + 1] == A_list[j] @ x[:, j] + B_list[j] @ u[:, j] + c_list[j],
            y[:, j] == C_list[j] @ x[:, j] + D_list[j] @ u[:, j] + d_list[j],
            u[:, j] >= bounds["u_min_norm"],
            u[:, j] <= bounds["u_max_norm"],
            y[:, j] + y_slack_low[:, j] >= bounds["y_min_norm"],
            y[:, j] - y_slack_high[:, j] <= bounds["y_max_norm"],
        ]
        du = u[:, j] - (u_prev if j == 0 else u[:, j - 1])
        objective += weights["q_y"] * cp.sum_squares(y[:, j] - refs[j].reshape(ny))
        objective += weights["r_u"] * cp.sum_squares(u[:, j] - u_refs[j].reshape(nu))
        objective += weights["s_du"] * cp.sum_squares(du)
        objective += weights["output_slack"] * (cp.sum(y_slack_low[:, j]) + cp.sum(y_slack_high[:, j]))
    problem = cp.Problem(cp.Minimize(objective), constraints)
    if warm_u is not None:
        u.value = warm_u.T
    start = time.perf_counter()
    problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, **solver_opts)
    elapsed = time.perf_counter() - start
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or u.value is None or y.value is None:
        raise RuntimeError(f"MPC QP failed with status {problem.status}")
    return np.asarray(u.value.T), np.asarray(y.value.T), float(problem.value), problem.status, elapsed


def make_constant_signal(length: int, value: float) -> np.ndarray:
    return np.full((length, 1), value, dtype=float)


def warmup_history(args, sampling_time: float, norm: Normalizer, controller_seed: int):
    plant = CascadedTanksPlant(
        Ts=sampling_time,
        x=np.asarray(args.initial_state, dtype=float),
        rk4_substeps=args.rk4_substeps,
        process_noise_std=args.process_noise_std,
        measurement_noise_std=args.measurement_noise_std,
        rng=np.random.default_rng(controller_seed),
    )
    u_hist, y_hist = [], []
    warm_u = float(np.clip(args.warmup_input, args.u_min, args.u_max))
    for _ in range(args.history_length):
        y = plant.step(warm_u)
        u_hist.append([warm_u])
        y_hist.append([y])
    return plant, norm.u_to_norm(np.asarray(u_hist)), norm.y_to_norm(np.asarray(y_hist))


def run_closed_loop(args: argparse.Namespace):
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    model, linear, norm, sampling_time, ckpt = train_or_load_model(args, dtype, device)
    bounds = {
        "u_min_norm": float(norm.u_to_norm(np.array([args.u_min]))[0]),
        "u_max_norm": float(norm.u_to_norm(np.array([args.u_max]))[0]),
        "y_min_norm": float(norm.y_to_norm(np.array([args.y_min]))[0]),
        "y_max_norm": float(norm.y_to_norm(np.array([args.y_max]))[0]),
    }
    weights = {"q_y": args.q_y, "r_u": args.r_u, "s_du": args.s_du, "output_slack": args.output_slack_weight}
    solver_opts = {"eps_abs": args.osqp_eps_abs, "eps_rel": args.osqp_eps_rel, "max_iter": args.osqp_max_iter}
    total_ref_len = args.simulation_steps + args.prediction_horizon + 1
    refs_phys = make_constant_signal(total_ref_len, args.reference_value)
    refs_norm = norm.y_to_norm(refs_phys)
    u_refs_norm = norm.u_to_norm(make_constant_signal(total_ref_len, args.u_ref))
    rows, summary = [], []

    for controller, seed_offset in [("linear", 0), ("resdynet_ltv", 1000)]:
        plant, u_hist, y_hist = warmup_history(args, sampling_time, norm, args.seed + seed_offset)
        u_prev = norm.u_to_norm(np.array([args.warmup_input]))
        warm_u = np.repeat(u_prev.reshape(1, -1), args.prediction_horizon, axis=0)
        costs, solve_times, sq_errors = [], [], []
        u_viol, y_viol = 0, 0
        for k in range(args.simulation_steps):
            refs = refs_norm[k : k + args.prediction_horizon]
            u_refs = u_refs_norm[k : k + args.prediction_horizon]
            if controller == "linear":
                x0_mpc = encode_linear_state(linear, y_hist, u_hist)
                terms = identified_linear_model(linear, args.prediction_horizon)
            else:
                x_hat = encode_state(model, y_hist, u_hist, dtype, device)
                x0_mpc = x_hat
                x_bar, _ = nominal_rollout(model, x_hat, warm_u, dtype, device)
                terms = linearize_resdynet(model, x_bar, warm_u, dtype, device)
            u_seq, y_pred, cost, status, solve_time = solve_ltv_mpc(
                x0_mpc, u_prev, refs, u_refs, terms, weights, bounds, solver_opts, warm_u
            )
            u_norm = u_seq[0]
            u_phys = float(norm.u_from_norm(u_norm.reshape(1))[0])
            y_phys = plant.step(u_phys)
            y_norm = norm.y_to_norm(np.array([[y_phys]])).reshape(1)
            err = y_phys - args.reference_value
            sq_errors.append(err * err)
            costs.append(cost)
            solve_times.append(solve_time)
            u_viol += int(u_phys < args.u_min - 1e-8 or u_phys > args.u_max + 1e-8)
            y_viol += int(y_phys < args.y_min - 1e-8 or y_phys > args.y_max + 1e-8)
            rows.append(
                {
                    "controller": controller,
                    "k": k,
                    "t": k * sampling_time,
                    "reference": args.reference_value,
                    "y": y_phys,
                    "u": u_phys,
                    "x1_simulator_only": plant.x[0],
                    "x2_simulator_only": plant.x[1],
                    "predicted_y0": float(norm.y_from_norm(y_pred[0]).reshape(-1)[0]),
                    "mpc_cost": cost,
                    "solve_time_sec": solve_time,
                    "qp_status": status,
                }
            )
            y_hist = np.vstack([y_hist[1:], y_norm.reshape(1, -1)])
            u_hist = np.vstack([u_hist[1:], u_norm.reshape(1, -1)])
            u_prev = u_norm
            warm_u = np.vstack([u_seq[1:], u_seq[-1:]])
        summary.append(
            {
                "controller": controller,
                "tracking_rmse": math.sqrt(sum(sq_errors) / len(sq_errors)),
                "accumulated_mpc_cost": sum(costs),
                "input_constraint_violations": u_viol,
                "output_constraint_violations": y_viol,
                "state_projection_count": plant.state_projection_count,
                "avg_qp_solve_time_sec": float(np.mean(solve_times)),
                "max_qp_solve_time_sec": float(np.max(solve_times)),
                "checkpoint_epoch": ckpt.get("epoch", ""),
                "checkpoint_val_nrmse": ckpt.get("val_nrmse", ""),
            }
        )
    return rows, summary, sampling_time


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output_dir: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"linear": "tab:gray", "resdynet_ltv": "tab:blue"}
    labels = {"linear": "Linear MPC", "resdynet_ltv": "ResDyNet LTV-MPC"}
    controllers = ["linear", "resdynet_ltv"]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    base = [row for row in rows if row["controller"] == "linear"]
    ax.plot([row["t"] for row in base], [row["reference"] for row in base], "k--", linewidth=1.5, label="reference")
    for ctrl in controllers:
        sub = [row for row in rows if row["controller"] == ctrl]
        ax.plot([row["t"] for row in sub], [row["y"] for row in sub], color=colors[ctrl], linewidth=1.6, label=labels[ctrl])
    ax.axhline(args.y_min, color="0.2", linestyle=":", linewidth=1.0)
    ax.axhline(args.y_max, color="0.2", linestyle=":", linewidth=1.0, label="output bounds")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("output")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.savefig(output_dir / "reference_vs_output.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    for ctrl in controllers:
        sub = [row for row in rows if row["controller"] == ctrl]
        ax.step([row["t"] for row in sub], [row["u"] for row in sub], where="post", color=colors[ctrl], linewidth=1.5, label=labels[ctrl])
    ax.axhline(args.u_min, color="k", linestyle="--", linewidth=1.0, label="input bounds")
    ax.axhline(args.u_max, color="k", linestyle="--", linewidth=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("input")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.savefig(output_dir / "control_inputs.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 3.8), constrained_layout=True)
    for ctrl in controllers:
        sub = [row for row in rows if row["controller"] == ctrl]
        ax.plot([row["t"] for row in sub], [row["y"] - row["reference"] for row in sub], color=colors[ctrl], linewidth=1.3, label=labels[ctrl])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("tracking error")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.savefig(output_dir / "tracking_error.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop MPC for Cascaded Tanks with ResDyNet successive linearization.")
    for key, value in DEFAULT_CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            parser.add_argument(flag, default=value)
        elif isinstance(value, list):
            parser.add_argument(flag, type=float, nargs="+", default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.latent_dim < args.linear_order:
        raise ValueError("--latent-dim must be >= --linear-order.")
    output_dir = Path(args.output_dir)
    rows, summary, sampling_time = run_closed_loop(args)
    write_csv(output_dir / "closed_loop_trajectories.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({**vars(args), "sampling_time": sampling_time, "tank_parameters": TANK_PARAMETERS}, indent=2)
    )
    save_plots(output_dir, rows, args)
    print(f"saved Cascaded Tanks MPC results to {output_dir}")
    for row in summary:
        print(
            f"{row['controller']}: RMSE={row['tracking_rmse']:.6g}, "
            f"cost={row['accumulated_mpc_cost']:.6g}, "
            f"u viol={row['input_constraint_violations']}, y viol={row['output_constraint_violations']}, "
            f"avg_qp={row['avg_qp_solve_time_sec']:.4g}s, max_qp={row['max_qp_solve_time_sec']:.4g}s"
        )


if __name__ == "__main__":
    main()
