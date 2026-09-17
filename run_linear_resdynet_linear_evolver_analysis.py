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
from torch.utils.data import DataLoader

from main import ResDyNet, RolloutWindowDataset, weighted_multistep_loss
from run_linear_resdynet_random_analysis import (
    LinearSystem,
    Trajectory,
    complex_list,
    load_or_generate_data,
    match_eigenvalues,
    split_train_val,
)


class PureLinearEvolver(nn.Module):
    """State transition x+ = S [x; u], with no nonlinear correction branch."""

    def __init__(self, nx: int, nu: int) -> None:
        super().__init__()
        self.S = nn.Linear(nx + nu, nx, bias=False)

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        return self.S(torch.cat((x, u), dim=-1))


def make_model(args: argparse.Namespace, ny: int, nu: int, seed: int, dtype: torch.dtype, device: torch.device) -> ResDyNet:
    torch.manual_seed(seed)
    hidden = (args.hidden_width,) * args.hidden_layers
    return ResDyNet(
        ny=ny,
        nu=nu,
        nx=args.latent_dim,
        na=args.history_length,
        nb=args.history_length,
        encoder_hidden_dims=hidden,
        decoder_hidden_dims=hidden,
        stream_dim=args.stream_dim or 2 * (args.latent_dim + nu) + 1,
        num_blocks=args.num_blocks,
        evolver_hidden_dims=(args.evolver_hidden_width,),
        activation=nn.Tanh,
        zero_nonlinear_outputs=False,
        evolver=PureLinearEvolver(args.latent_dim, nu),
    ).to(device=device, dtype=dtype)


@torch.no_grad()
def eval_metrics(model: ResDyNet, dataset: RolloutWindowDataset, batch_size: int) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    param = next(model.parameters())
    model.eval()
    sq = 0.0
    count = 0
    targets = []
    for y_past, u_past, u_future, y_future in loader:
        y_past = y_past.to(device=param.device, dtype=param.dtype)
        u_past = u_past.to(device=param.device, dtype=param.dtype)
        u_future = u_future.to(device=param.device, dtype=param.dtype)
        y_future = y_future.to(device=param.device, dtype=param.dtype)
        pred = model(y_past, u_past, u_future)
        sq += torch.sum((pred - y_future) ** 2).item()
        count += y_future.numel()
        targets.append(y_future.cpu().reshape(-1, y_future.shape[-1]))
    target = torch.cat(targets, dim=0)
    rmse = math.sqrt(sq / count)
    nrmse = rmse / torch.std(target).clamp_min(torch.finfo(target.dtype).eps).item()
    return {"rmse": rmse, "nrmse": nrmse}


def extract_linear_branches(model: ResDyNet) -> dict[str, Tensor]:
    Sf = model.evolver.S.weight.detach().cpu()
    Sg = model.decoder.S.weight.detach().cpu()
    return {
        "A_f": Sf[:, : model.nx].clone(),
        "B_f": Sf[:, model.nx :].clone(),
        "C_f": Sg[:, : model.nx].clone(),
        "D_f": Sg[:, model.nx :].clone(),
    }


@torch.no_grad()
def encoded_latents(model: ResDyNet, traj: Trajectory, history: int, batch_size: int, device, dtype) -> tuple[Tensor, Tensor]:
    idx_all = torch.arange(history, traj.u.shape[0])
    latents = []
    indices = []
    for start in range(0, idx_all.numel(), batch_size):
        idx = idx_all[start : start + batch_size]
        y_past = torch.stack([traj.y[i - history : i] for i in idx], dim=0)
        u_past = torch.stack([traj.u[i - history : i] for i in idx], dim=0)
        z = model.encode_regressor(y_past.to(device=device, dtype=dtype), u_past.to(device=device, dtype=dtype))
        latents.append(z.cpu())
        indices.append(idx)
    return torch.cat(latents, dim=0), torch.cat(indices, dim=0)


def fit_affine_T_c(x_phys: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
    design = torch.cat((x_phys, torch.ones(x_phys.shape[0], 1, dtype=x_phys.dtype)), dim=1)
    coeff = torch.linalg.lstsq(design, z).solution
    T = coeff[:-1, :].T.contiguous()
    c = coeff[-1, :].contiguous()
    return T, c


def affine_fit_error(x_phys: Tensor, z: Tensor, T: Tensor, c: Tensor) -> float:
    z_fit = x_phys @ T.T + c
    return (
        torch.sqrt(torch.mean((z - z_fit) ** 2))
        / torch.std(z).clamp_min(torch.finfo(z.dtype).eps)
    ).item()


def similarity_analysis(
    model: ResDyNet,
    calibration: Trajectory,
    test: Trajectory,
    system: LinearSystem,
    branches: dict[str, Tensor],
    args: argparse.Namespace,
    device,
    dtype,
) -> dict[str, object]:
    z_cal, idx_cal = encoded_latents(model, calibration, args.history_length, args.eval_batch_size, device, dtype)
    T, c = fit_affine_T_c(calibration.x[idx_cal], z_cal)
    cal_err = affine_fit_error(calibration.x[idx_cal], z_cal, T, c)
    z_test, idx_test = encoded_latents(model, test, args.history_length, args.eval_batch_size, device, dtype)
    test_err = affine_fit_error(test.x[idx_test], z_test, T, c)
    cond_T = torch.linalg.cond(T).item()
    out: dict[str, object] = {
        "T": T,
        "c": c,
        "T_cond": cond_T,
        "affine_calibration_nrmse": cal_err,
        "affine_test_nrmse": test_err,
        "similarity_A_error": float("nan"),
        "similarity_B_error": float("nan"),
    }
    if torch.linalg.matrix_rank(T).item() == T.shape[0]:
        T_inv = torch.linalg.inv(T)
        A_true_latent = T @ system.A @ T_inv
        B_true_latent = T @ system.B
        out["A_true_latent"] = A_true_latent
        out["B_true_latent"] = B_true_latent
        out["eig_A_true_latent"] = torch.linalg.eigvals(A_true_latent)
        out["similarity_A_error"] = torch.linalg.norm(branches["A_f"] - A_true_latent, ord="fro").item()
        out["similarity_B_error"] = torch.linalg.norm(branches["B_f"] - B_true_latent, ord="fro").item()
    return out


def tensor_json(v):
    if isinstance(v, Tensor):
        if torch.is_complex(v):
            return complex_list(v)
        return v.detach().cpu().tolist()
    if isinstance(v, complex):
        return {"real": v.real, "imag": v.imag, "abs": abs(v)}
    if isinstance(v, dict):
        return {k: tensor_json(val) for k, val in v.items()}
    if isinstance(v, list):
        return [tensor_json(val) for val in v]
    return v


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    t = torch.tensor(values, dtype=torch.float64)
    return torch.mean(t).item(), torch.std(t, unbiased=True).item() if len(values) > 1 else 0.0


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = [
        "test_nrmse",
        "test_rmse",
        "matched_eig_mean_abs_error",
        "spectral_radius_A_f",
        "fro_Af_minus_Atrue_coordinate_dependent",
        "T_cond",
        "affine_calibration_nrmse",
        "affine_test_nrmse",
        "similarity_A_error",
        "similarity_B_error",
    ]
    out = []
    for key in keys:
        vals = [float(r[key]) for r in rows]
        mean, std = mean_std(vals)
        out.append({"metric": key, "mean": mean, "std": std})
    return out


def format_eigs(eigs: Tensor) -> str:
    return "; ".join(f"{v.real.item():+.6g}{v.imag.item():+.6g}j" for v in eigs)


def train_seed(args, seed, datasets, trajectories, system, output_dir, dtype, device):
    train_ds, val_ds, test_ds = datasets
    train_traj, _val_traj, test_traj = trajectories
    model = make_model(args, ny=1, nu=1, seed=seed, dtype=dtype, device=device)
    if any(hasattr(model.evolver, attr) for attr in ("W_i", "W_o", "blocks")):
        raise RuntimeError("PureLinearEvolver unexpectedly contains nonlinear evolver modules.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.shuffle_seed_base + seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best = {"val_nrmse": float("inf"), "epoch": 0, "state": None}
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs + 1):
        train_m = eval_metrics(model, train_ds, args.eval_batch_size)
        val_m = eval_metrics(model, val_ds, args.eval_batch_size)
        row = {
            "seed": seed,
            "epoch": epoch,
            "train_rmse": train_m["rmse"],
            "train_nrmse": train_m["nrmse"],
            "val_rmse": val_m["rmse"],
            "val_nrmse": val_m["nrmse"],
            "wall_time_sec": time.perf_counter() - start,
        }
        history.append(row)
        print(
            f"Seed {seed:02d} | Epoch {epoch:03d} "
            f"train_NRMSE={train_m['nrmse']:.6g} val_NRMSE={val_m['nrmse']:.6g}",
            flush=True,
        )
        if val_m["nrmse"] < best["val_nrmse"]:
            best = {
                "val_nrmse": val_m["nrmse"],
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        if epoch == args.epochs:
            break
        model.train()
        for y_past, u_past, u_future, y_future in loader:
            y_past = y_past.to(device=device, dtype=dtype)
            u_past = u_past.to(device=device, dtype=dtype)
            u_future = u_future.to(device=device, dtype=dtype)
            y_future = y_future.to(device=device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_multistep_loss(model(y_past, u_past, u_future), y_future)
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

    model.load_state_dict(best["state"])
    model.eval()
    train_m = eval_metrics(model, train_ds, args.eval_batch_size)
    val_m = eval_metrics(model, val_ds, args.eval_batch_size)
    test_m = eval_metrics(model, test_ds, args.eval_batch_size)
    branches = extract_linear_branches(model)
    eig_f = torch.linalg.eigvals(branches["A_f"])
    eig_true = torch.linalg.eigvals(system.A)
    eig_matches, eig_mean_error = match_eigenvalues(eig_f, eig_true)
    sim = similarity_analysis(model, train_traj, test_traj, system, branches, args, device, dtype)
    summary = {
        "seed": seed,
        "best_epoch": best["epoch"],
        "train_rmse": train_m["rmse"],
        "train_nrmse": train_m["nrmse"],
        "val_rmse": val_m["rmse"],
        "val_nrmse": val_m["nrmse"],
        "test_rmse": test_m["rmse"],
        "test_nrmse": test_m["nrmse"],
        "eig_A_f": format_eigs(eig_f),
        "eig_A_true": format_eigs(eig_true),
        "matched_eig_mean_abs_error": eig_mean_error,
        "spectral_radius_A_f": torch.max(torch.abs(eig_f)).item(),
        "A_f_stable": bool(torch.all(torch.abs(eig_f) < 1.0).item()),
        "fro_Af_minus_Atrue_coordinate_dependent": torch.linalg.norm(branches["A_f"] - system.A, ord="fro").item(),
        "T_cond": float(sim["T_cond"]),
        "affine_calibration_nrmse": float(sim["affine_calibration_nrmse"]),
        "affine_test_nrmse": float(sim["affine_test_nrmse"]),
        "similarity_A_error": float(sim["similarity_A_error"]),
        "similarity_B_error": float(sim["similarity_B_error"]),
    }
    payload = {
        "seed": seed,
        "summary": summary,
        "A_f": branches["A_f"],
        "B_f": branches["B_f"],
        "C_f": branches["C_f"],
        "D_f": branches["D_f"],
        "A_true": system.A,
        "B_true": system.B,
        "C_true": system.C,
        "D_true": system.D,
        "eig_A_f": eig_f,
        "eig_A_true": eig_true,
        "eig_matches": eig_matches,
        "similarity": sim,
    }
    torch.save(payload, output_dir / f"matrices_seed_{seed}.pt")
    (output_dir / f"matrices_seed_{seed}.json").write_text(json.dumps(tensor_json(payload), indent=2))
    torch.save({"model_state": model.state_dict(), **payload}, output_dir / "checkpoints" / f"seed_{seed}_best.pt")
    print(f"Seed {seed:02d} eig(A_f): {summary['eig_A_f']}", flush=True)
    print(f"Seed {seed:02d} A_f:\n{branches['A_f']}", flush=True)
    print(f"Seed {seed:02d} B_f:\n{branches['B_f']}", flush=True)
    return history, summary, eig_f


def save_eigen_plot(path: Path, learned: list[tuple[int, Tensor]], eig_true: Tensor) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = torch.linspace(0, 2 * math.pi, 500)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot(torch.cos(theta), torch.sin(theta), "k--", linewidth=1, label="unit circle")
    ax.scatter(eig_true.real, eig_true.imag, marker="o", s=80, color="black", label="true poles")
    for seed, eigs in learned:
        ax.scatter(eigs.real, eigs.imag, marker="x", s=45, label=f"seed {seed}")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Order-3 linear dataset with ResDyNet encoder/decoder and pure-linear evolver.")
    p.add_argument("--dataset", default="linear_order3_dataset.mat")
    p.add_argument("--output-dir", default="results/linear_resdynet_pure_linear_evolver")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--history-length", type=int, default=20)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--latent-dim", type=int, default=3)
    p.add_argument("--hidden-width", type=int, default=64)
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--stream-dim", type=int, default=None)
    p.add_argument("--evolver-hidden-width", type=int, default=64)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--shuffle-seed-base", type=int, default=50000)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.latent_dim != 3:
        raise ValueError("This experiment requires --latent-dim 3.")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    train_full, test_traj, system, source = load_or_generate_data(args, dtype)
    train_traj, val_traj = split_train_val(train_full, args.val_fraction)
    train_ds = RolloutWindowDataset(train_traj.u, train_traj.y, args.history_length, args.history_length, args.horizon)
    val_ds = RolloutWindowDataset(val_traj.u, val_traj.y, args.history_length, args.history_length, args.horizon)
    test_ds = RolloutWindowDataset(test_traj.u, test_traj.y, args.history_length, args.history_length, args.horizon)
    config = {
        **vars(args),
        "data_source": source,
        "A_true": system.A.tolist(),
        "B_true": system.B.tolist(),
        "C_true": system.C.tolist(),
        "D_true": system.D.tolist(),
        "train_samples": train_traj.u.shape[0],
        "val_samples": val_traj.u.shape[0],
        "test_samples": test_traj.u.shape[0],
        "evolver": "PureLinearEvolver: x_next = S_f [x; u], no nonlinear branch",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"data source: {source}", flush=True)
    print("training uses only u,y; true x is used only after training for affine/similarity analysis", flush=True)

    all_history = []
    summaries = []
    learned_eigs = []
    for seed in args.seeds:
        history, summary, eigs = train_seed(
            args,
            seed,
            (train_ds, val_ds, test_ds),
            (train_traj, val_traj, test_traj),
            system,
            output_dir,
            dtype,
            device,
        )
        all_history.extend(history)
        summaries.append(summary)
        learned_eigs.append((seed, eigs))

    write_csv(output_dir / "epoch_history.csv", all_history)
    write_csv(output_dir / "seed_summary.csv", summaries)
    summary_rows = aggregate(summaries)
    write_csv(output_dir / "summary_mean_std.csv", summary_rows)
    save_eigen_plot(output_dir / "eigenvalues_complex_plane.png", learned_eigs, torch.linalg.eigvals(system.A))
    print(f"saved results to {output_dir}", flush=True)
    for row in summary_rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
