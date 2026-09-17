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
from torch import Tensor

from main import ResDyNet
from run_rlc_latent_experiment import load_checkpoint


# ---------------------------------------------------------------------------
# Clear MPC configuration section.
# Values are intentionally conservative for a short, reproducible experiment.
# They can be overridden from the command line.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "system": "rlc",
    "checkpoint": "E5/checkpoints/seed_0_best.pt",
    "output_dir": "results/mpc",
    "simulation_steps": 180,
    "prediction_horizon": 25,
    "q_y": 20.0,
    "r_u": 0.05,
    "s_du": 0.5,
    "u_min": -2.0,
    "u_max": 2.0,
    "reference_value": 0.6,
    "initial_state": [0.0, 0.0],
    "osqp_eps_abs": 1e-5,
    "osqp_eps_rel": 1e-5,
    "osqp_max_iter": 10000,
    "dtype": "float32",
    "device": "cpu",
}


@dataclass(frozen=True)
class AffineNormalizer:
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
class RLCPlant:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Ts: float
    x: np.ndarray

    @classmethod
    def from_repository_parameters(cls, x0: list[float]) -> "RLCPlant":
        # Same continuous-time model and sample time documented in
        # generate_dataset_rlc.m.
        R, L, Ccap, Ts = 1.0, 0.5, 0.2, 0.02
        Ac = torch.tensor([[-R / L, -1.0 / L], [1.0 / Ccap, 0.0]], dtype=torch.float64)
        Bc = torch.tensor([[1.0 / L], [0.0]], dtype=torch.float64)
        aug = torch.zeros(3, 3, dtype=torch.float64)
        aug[:2, :2] = Ac
        aug[:2, 2:] = Bc
        exp_aug = torch.matrix_exp(aug * Ts)
        A = exp_aug[:2, :2].numpy()
        B = exp_aug[:2, 2:].numpy()
        return cls(
            A=A,
            B=B,
            C=np.array([[0.0, 1.0]], dtype=float),
            D=np.array([[0.0]], dtype=float),
            Ts=Ts,
            x=np.asarray(x0, dtype=float).reshape(2, 1),
        )

    def measure(self, u: np.ndarray) -> np.ndarray:
        return self.C @ self.x + self.D @ u.reshape(1, 1)

    def step(self, u: np.ndarray) -> np.ndarray:
        y = self.measure(u)
        self.x = self.A @ self.x + self.B @ u.reshape(1, 1)
        return y


def require_cvxpy():
    try:
        import cvxpy as cp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CVXPY is required for this experiment. Install it with "
            "`python3 -m pip install cvxpy osqp` in this environment."
        ) from exc
    return cp


def as_np(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


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


def nominal_rollout(
    model: ResDyNet,
    x0: np.ndarray,
    u_seq: np.ndarray,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    xs = [x0.reshape(-1)]
    ys = []
    x = xs[0]
    for u in u_seq:
        ys.append(model_g(model, x, u, dtype, device))
        x = model_f(model, x, u, dtype, device)
        xs.append(x)
    return np.asarray(xs), np.asarray(ys)


def linearize_resdynet(
    model: ResDyNet,
    x_bar: np.ndarray,
    u_bar: np.ndarray,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
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


def linear_branch_model(model: ResDyNet, horizon: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    Sf = as_np(model.evolver.S.weight)
    Sg = as_np(model.decoder.S.weight)
    nx = model.nx
    A = Sf[:, :nx]
    B = Sf[:, nx:]
    C = Sg[:, :nx]
    D = Sg[:, nx:]
    return (
        [A] * horizon,
        [B] * horizon,
        [C] * horizon,
        [D] * horizon,
        [np.zeros(nx)] * horizon,
        [np.zeros(model.ny)] * horizon,
    )


def solve_ltv_mpc(
    x0: np.ndarray,
    u_prev: np.ndarray,
    refs: np.ndarray,
    u_refs: np.ndarray,
    model_terms,
    weights: dict[str, float],
    bounds: dict[str, float],
    solver_opts: dict[str, float | int],
    warm_u: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float, str, float]:
    cp = require_cvxpy()
    A_list, B_list, C_list, D_list, c_list, d_list = model_terms
    horizon = len(A_list)
    nx = x0.size
    nu = B_list[0].shape[1]
    ny = C_list[0].shape[0]
    x = cp.Variable((nx, horizon + 1))
    u = cp.Variable((nu, horizon))
    y = cp.Variable((ny, horizon))

    constraints = [x[:, 0] == x0]
    objective = 0.0
    q, r, s = weights["q_y"], weights["r_u"], weights["s_du"]
    for j in range(horizon):
        constraints += [
            x[:, j + 1] == A_list[j] @ x[:, j] + B_list[j] @ u[:, j] + c_list[j],
            y[:, j] == C_list[j] @ x[:, j] + D_list[j] @ u[:, j] + d_list[j],
            u[:, j] >= bounds["u_min_norm"],
            u[:, j] <= bounds["u_max_norm"],
        ]
        du = u[:, j] - (u_prev if j == 0 else u[:, j - 1])
        objective += q * cp.sum_squares(y[:, j] - refs[j].reshape(ny))
        objective += r * cp.sum_squares(u[:, j] - u_refs[j].reshape(nu))
        objective += s * cp.sum_squares(du)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    if warm_u is not None:
        u.value = warm_u.T
    start = time.perf_counter()
    problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, **solver_opts)
    elapsed = time.perf_counter() - start
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or u.value is None:
        raise RuntimeError(f"MPC QP failed with status {problem.status}")
    return np.asarray(u.value.T), np.asarray(y.value.T), float(problem.value), problem.status, elapsed


def make_reference(steps: int, horizon: int, value: float) -> np.ndarray:
    return np.full((steps + horizon + 1, 1), value, dtype=float)


def run_closed_loop(args: argparse.Namespace) -> dict[str, object]:
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    cfg, model, normalizer_torch, _linear, ckpt = load_checkpoint(Path(args.checkpoint), dtype=dtype, device=device)
    model.eval()
    norm = AffineNormalizer(
        u_mean=as_np(normalizer_torch.u_mean).reshape(1),
        u_std=as_np(normalizer_torch.u_std).reshape(1),
        y_mean=as_np(normalizer_torch.y_mean).reshape(1),
        y_std=as_np(normalizer_torch.y_std).reshape(1),
    )

    plant0 = RLCPlant.from_repository_parameters(args.initial_state)
    y0 = plant0.measure(np.array([0.0])).reshape(1)
    y_hist0 = np.repeat(norm.y_to_norm(y0)[None, :], model.na, axis=0)
    u_hist0 = np.repeat(norm.u_to_norm(np.array([0.0]))[None, :], model.nb, axis=0)
    ref_phys = make_reference(args.simulation_steps, args.prediction_horizon, args.reference_value)
    ref_norm = norm.y_to_norm(ref_phys)
    u_ref_norm = np.repeat(norm.u_to_norm(np.array([[0.0]])), args.simulation_steps + args.prediction_horizon + 1, axis=0)
    bounds = {
        "u_min_norm": float(norm.u_to_norm(np.array([args.u_min]))[0]),
        "u_max_norm": float(norm.u_to_norm(np.array([args.u_max]))[0]),
    }
    weights = {"q_y": args.q_y, "r_u": args.r_u, "s_du": args.s_du}
    solver_opts = {"eps_abs": args.osqp_eps_abs, "eps_rel": args.osqp_eps_rel, "max_iter": args.osqp_max_iter}

    all_rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for controller in ("linear", "resdynet_ltv"):
        plant = RLCPlant.from_repository_parameters(args.initial_state)
        y_hist = y_hist0.copy()
        u_hist = u_hist0.copy()
        u_prev = norm.u_to_norm(np.array([0.0]))
        warm_u = np.repeat(u_prev[None, :], args.prediction_horizon, axis=0)
        costs, solve_times, sq_err = [], [], []
        violations = 0

        for k in range(args.simulation_steps):
            x_hat = encode_state(model, y_hist, u_hist, dtype, device)
            refs = ref_norm[k : k + args.prediction_horizon]
            u_refs = u_ref_norm[k : k + args.prediction_horizon]
            if controller == "linear":
                terms = linear_branch_model(model, args.prediction_horizon)
            else:
                x_bar, _ = nominal_rollout(model, x_hat, warm_u, dtype, device)
                terms = linearize_resdynet(model, x_bar, warm_u, dtype, device)
            u_seq, y_pred, cost, status, solve_time = solve_ltv_mpc(
                x_hat, u_prev, refs, u_refs, terms, weights, bounds, solver_opts, warm_u
            )
            u_norm = u_seq[0]
            u_phys = norm.u_from_norm(u_norm).reshape(1)
            y_phys = plant.step(u_phys).reshape(1)
            y_norm = norm.y_to_norm(y_phys).reshape(1)
            err = y_phys - ref_phys[k]
            sq_err.append(float(err @ err))
            costs.append(cost)
            solve_times.append(solve_time)
            violations += int(u_phys[0] < args.u_min - 1e-8 or u_phys[0] > args.u_max + 1e-8)
            all_rows.append(
                {
                    "controller": controller,
                    "k": k,
                    "t": k * plant.Ts,
                    "reference": ref_phys[k, 0],
                    "y": y_phys[0],
                    "u": u_phys[0],
                    "predicted_y0": norm.y_from_norm(y_pred[0]).reshape(-1)[0],
                    "stage_qp_cost": cost,
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
                "tracking_rmse": math.sqrt(sum(sq_err) / len(sq_err)),
                "accumulated_mpc_cost": sum(costs),
                "input_constraint_violations": violations,
                "avg_qp_solve_time_sec": float(np.mean(solve_times)),
                "max_qp_solve_time_sec": float(np.max(solve_times)),
                "checkpoint_epoch": ckpt.get("epoch", ""),
                "checkpoint_val_output_nrmse": ckpt.get("val_output_nrmse", ""),
            }
        )
    return {"rows": all_rows, "summary": summary, "config": vars(args)}


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
    controllers = ["linear", "resdynet_ltv"]
    colors = {"linear": "tab:gray", "resdynet_ltv": "tab:blue"}

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ref_rows = [row for row in rows if row["controller"] == controllers[0]]
    ax.plot([row["t"] for row in ref_rows], [row["reference"] for row in ref_rows], "k--", label="reference", linewidth=1.6)
    for ctrl in controllers:
        sub = [row for row in rows if row["controller"] == ctrl]
        ax.plot([row["t"] for row in sub], [row["y"] for row in sub], label=ctrl, linewidth=1.5, color=colors[ctrl])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("output")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.savefig(output_dir / "reference_vs_output.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.3), constrained_layout=True)
    for ctrl in controllers:
        sub = [row for row in rows if row["controller"] == ctrl]
        ax.step([row["t"] for row in sub], [row["u"] for row in sub], where="post", label=ctrl, linewidth=1.4, color=colors[ctrl])
    ax.axhline(args.u_min, color="k", linestyle="--", linewidth=1.0, label="input bounds")
    ax.axhline(args.u_max, color="k", linestyle="--", linewidth=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("input")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.savefig(output_dir / "control_inputs.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    for ctrl in controllers:
        sub = [row for row in rows if row["controller"] == ctrl]
        ax.plot([row["t"] for row in sub], [row["y"] - row["reference"] for row in sub], label=ctrl, linewidth=1.3, color=colors[ctrl])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("tracking error")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.savefig(output_dir / "tracking_error.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    for ctrl in controllers:
        sub = [row for row in rows if row["controller"] == ctrl]
        ax.plot([row["t"] for row in sub], [1000.0 * row["solve_time_sec"] for row in sub], label=ctrl, linewidth=1.1, color=colors[ctrl])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("QP solve time [ms]")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.savefig(output_dir / "qp_solve_time.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop MPC with ResDyNet successive linearization.")
    for key, value in DEFAULT_CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action="store_true", default=value)
        elif isinstance(value, list):
            parser.add_argument(flag, type=float, nargs="+", default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.system != "rlc":
        raise RuntimeError(
            "Cascaded Tanks closed-loop simulation is not run because this repository and the "
            "installed nonlinear_benchmarks bundle provide measured I/O data plus a qualitative "
            "ODE form, but not the true simulator parameters needed to apply new MPC inputs. "
            "Use --system rlc for the closed-loop experiment backed by repository equations and checkpoints."
        )
    output_dir = Path(args.output_dir)
    result = run_closed_loop(args)
    write_csv(output_dir / "closed_loop_trajectories.csv", result["rows"])
    write_csv(output_dir / "summary.csv", result["summary"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(result["config"], indent=2))
    save_plots(output_dir, result["rows"], args)
    print(f"saved MPC results to {output_dir}")
    for row in result["summary"]:
        print(
            f"{row['controller']}: RMSE={row['tracking_rmse']:.6g}, "
            f"cost={row['accumulated_mpc_cost']:.6g}, "
            f"avg_qp={row['avg_qp_solve_time_sec']:.4g}s, "
            f"max_qp={row['max_qp_solve_time_sec']:.4g}s"
        )


if __name__ == "__main__":
    main()
