from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from main import (
    LinearStateSpace,
    ResDyNet,
    RolloutWindowDataset,
    apply_linear_informed_initialization,
    fit_n4sid_state_space,
    reconstructability_matrix,
    weighted_multistep_loss,
)


@dataclass
class Trajectory:
    name: str
    kind: str
    t: Tensor
    u: Tensor
    y: Tensor
    x: Tensor


@dataclass
class Normalizer:
    u_mean: Tensor
    u_std: Tensor
    y_mean: Tensor
    y_std: Tensor

    def uy(self, trajectory: Trajectory) -> tuple[Tensor, Tensor]:
        return (trajectory.u - self.u_mean) / self.u_std, (trajectory.y - self.y_mean) / self.y_std


def matlab_char(dataset) -> str:
    codes = dataset[()]
    return "".join(chr(int(c)) for c in codes.reshape(-1) if int(c) != 0)


def read_array(group, key: str, dtype: torch.dtype) -> Tensor:
    array = torch.as_tensor(group[key][()], dtype=dtype)
    if array.ndim == 2:
        array = array.T
    if array.ndim == 1:
        array = array.unsqueeze(-1)
    return array.contiguous()


def classify_latent_test(name: str) -> str:
    stress_tokens = ("constant", "free_response", "large_amplitude")
    return "stress_ood" if any(token in name for token in stress_tokens) else "nominal"


def read_trajectory(group, name: str, kind: str, dtype: torch.dtype) -> Trajectory:
    return Trajectory(
        name=name,
        kind=kind,
        t=read_array(group, "t", dtype),
        u=read_array(group, "u", dtype),
        y=read_array(group, "y", dtype),
        x=read_array(group, "x", dtype),
    )


def load_rlc_dataset(path: Path, dtype: torch.dtype) -> dict[str, object]:
    with h5py.File(path, "r") as f:
        train = read_trajectory(f["train"], "train", "train", dtype)
        val = read_trajectory(f["val"], "val", "val", dtype)
        test = read_trajectory(f["test"], "test", "nominal", dtype)
        affine = read_trajectory(f["affine_calibration"], "affine_calibration", "calibration", dtype)
        latent_tests = []
        refs = f["latent_test"][()]
        for ref in refs.reshape(-1):
            group = f[ref]
            name = matlab_char(group["name"])
            latent_tests.append(read_trajectory(group, name, classify_latent_test(name), dtype))
    return {
        "train": train,
        "val": val,
        "test": test,
        "affine_calibration": affine,
        "latent_test": latent_tests,
    }


def make_normalizer(train: Trajectory) -> Normalizer:
    eps = torch.finfo(train.u.dtype).eps
    return Normalizer(
        u_mean=train.u.mean(dim=0, keepdim=True),
        u_std=train.u.std(dim=0, keepdim=True).clamp_min(eps),
        y_mean=train.y.mean(dim=0, keepdim=True),
        y_std=train.y.std(dim=0, keepdim=True).clamp_min(eps),
    )


def scale_linear_realization_for_encoder(linear: LinearStateSpace, n: int, eps: float = 1e-12) -> LinearStateSpace:
    R = reconstructability_matrix(linear, n)
    row_scale = torch.max(torch.abs(R), dim=1).values.clamp_min(eps)
    P = torch.diag(1.0 / row_scale)
    P_inv = torch.diag(row_scale)
    return LinearStateSpace(A=P @ linear.A @ P_inv, B=P @ linear.B, C=linear.C @ P_inv, D=linear.D)


def nrmse_output(model: ResDyNet, dataset: RolloutWindowDataset, batch_size: int) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    param = next(model.parameters())
    was_training = model.training
    model.eval()
    sq = 0.0
    targets = []
    with torch.no_grad():
        for y_past, u_past, u_future, y_future in loader:
            y_past = y_past.to(param.device, param.dtype)
            u_past = u_past.to(param.device, param.dtype)
            u_future = u_future.to(param.device, param.dtype)
            y_future = y_future.to(param.device, param.dtype)
            pred = model(y_past, u_past, u_future)
            sq += torch.sum((pred - y_future) ** 2).item()
            targets.append(y_future.cpu().reshape(-1, y_future.shape[-1]))
    target = torch.cat(targets, dim=0)
    if was_training:
        model.train()
    return math.sqrt(sq / target.numel()) / max(torch.std(target).item(), torch.finfo(target.dtype).eps)


def build_model(args, ny: int, nu: int, dtype: torch.dtype, device: torch.device) -> ResDyNet:
    torch.manual_seed(args.seed)
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


def train_model(args, data: dict[str, object], output_dir: Path, dtype: torch.dtype, device: torch.device):
    train: Trajectory = data["train"]  # type: ignore[assignment]
    val: Trajectory = data["val"]  # type: ignore[assignment]
    normalizer = make_normalizer(train)
    u_train, y_train = normalizer.uy(train)
    u_val, y_val = normalizer.uy(val)

    assert train.x.numel() > 0 and val.x.numel() > 0
    print("sanity: physical x loaded but not passed to training, validation, N4SID, or model selection")

    n4sid_u = u_train[: args.max_n4sid_samples] if args.max_n4sid_samples else u_train
    n4sid_y = y_train[: args.max_n4sid_samples] if args.max_n4sid_samples else y_train
    linear = fit_n4sid_state_space(
        n4sid_u,
        n4sid_y,
        order=args.linear_order,
        num_block_rows=args.n4sid_block_rows,
        zero_direct_feedthrough=True,
    )
    linear = scale_linear_realization_for_encoder(linear, args.history_length)

    model = build_model(args, ny=y_train.shape[1], nu=u_train.shape[1], dtype=dtype, device=device)
    apply_linear_informed_initialization(model, linear, n=args.history_length)

    train_dataset = RolloutWindowDataset(u_train, y_train, args.history_length, args.history_length, args.horizon)
    val_dataset = RolloutWindowDataset(u_val, y_val, args.history_length, args.history_length, args.horizon)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 12345)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"seed_{args.seed}_best.pt"
    last_path = ckpt_dir / f"seed_{args.seed}_last.pt"

    start = time.perf_counter()
    for epoch in range(args.epochs + 1):
        val_score = nrmse_output(model, val_dataset, args.eval_batch_size)
        history.append({"epoch": epoch, "val_output_nrmse": val_score, "wall_time_sec": time.perf_counter() - start})
        if val_score < best_val:
            best_val = val_score
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_path, args, model, normalizer, linear, best_epoch, best_val)
        else:
            bad_epochs += 1

        if epoch == args.epochs or (args.patience and bad_epochs >= args.patience):
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
            if args.clip_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

    save_checkpoint(last_path, args, model, normalizer, linear, epoch, val_score)
    write_csv(output_dir / f"training_seed_{args.seed}.csv", history)
    print(f"training complete: seed={args.seed}, best val NRMSE={best_val:.6g} at epoch {best_epoch}")
    return best_path


def save_checkpoint(
    path: Path,
    args,
    model: ResDyNet,
    normalizer: Normalizer,
    linear: LinearStateSpace,
    epoch: int,
    val_nrmse: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
            "normalizer": {
                "u_mean": normalizer.u_mean.cpu(),
                "u_std": normalizer.u_std.cpu(),
                "y_mean": normalizer.y_mean.cpu(),
                "y_std": normalizer.y_std.cpu(),
            },
            "linear": {k: getattr(linear, k).cpu() for k in ("A", "B", "C", "D")},
            "epoch": epoch,
            "val_output_nrmse": val_nrmse,
        },
        path,
    )


def load_checkpoint(path: Path, dtype: torch.dtype, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    cfg = argparse.Namespace(**ckpt["config"])
    normalizer = Normalizer(**{k: v.to(dtype=dtype) for k, v in ckpt["normalizer"].items()})
    linear = LinearStateSpace(**{k: v.to(dtype=dtype) for k, v in ckpt["linear"].items()})
    model = build_model(cfg, ny=1, nu=1, dtype=dtype, device=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return cfg, model, normalizer, linear, ckpt


def make_regressor_batches(u: Tensor, y: Tensor, n: int, batch_size: int):
    indices = torch.arange(n, u.shape[0])
    for start in range(0, indices.numel(), batch_size):
        idx = indices[start : start + batch_size]
        y_past = torch.stack([y[i - n : i] for i in idx], dim=0)
        u_past = torch.stack([u[i - n : i] for i in idx], dim=0)
        yield idx, y_past, u_past


@torch.no_grad()
def encoded_latents(model: ResDyNet, u: Tensor, y: Tensor, n: int, batch_size: int, device, dtype) -> tuple[Tensor, Tensor]:
    latents = []
    aligned_indices = []
    for idx, y_past, u_past in make_regressor_batches(u, y, n, batch_size):
        z = model.encode_regressor(y_past.to(device=device, dtype=dtype), u_past.to(device=device, dtype=dtype))
        latents.append(z.cpu())
        aligned_indices.append(idx)
    return torch.cat(latents, dim=0), torch.cat(aligned_indices, dim=0)


@torch.no_grad()
def propagated_latents_and_outputs(
    model: ResDyNet,
    u: Tensor,
    y: Tensor,
    n: int,
    device,
    dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    y0 = y[:n].unsqueeze(0).to(device=device, dtype=dtype)
    u0 = u[:n].unsqueeze(0).to(device=device, dtype=dtype)
    encoder_calls = 0
    x = model.encode_regressor(y0, u0)
    encoder_calls += 1
    latents = []
    outputs = []
    for k in range(n, u.shape[0]):
        latents.append(x.squeeze(0).cpu())
        uk = u[k : k + 1].to(device=device, dtype=dtype)
        outputs.append(model.decoder(torch.cat((x, uk), dim=-1)).squeeze(0).cpu())
        if k < u.shape[0] - 1:
            x = model.evolver(x, uk)
    assert encoder_calls == 1, "propagated mode must call the encoder exactly once"
    return torch.stack(latents, dim=0), torch.stack(outputs, dim=0), torch.arange(n, u.shape[0])


def fit_affine_map(x_phys: Tensor, x_hat: Tensor) -> tuple[Tensor, Tensor]:
    assert x_phys.shape[0] == x_hat.shape[0]
    design = torch.cat((x_phys, torch.ones(x_phys.shape[0], 1, dtype=x_phys.dtype)), dim=1)
    coeff = torch.linalg.lstsq(design, x_hat).solution
    P = coeff[:-1, :].T.contiguous()
    c = coeff[-1, :].contiguous()
    return P, c


def apply_affine(P: Tensor, c: Tensor, x_phys: Tensor) -> Tensor:
    return x_phys @ P.T + c


def affine_metrics(x_hat: Tensor, x_aff: Tensor) -> dict[str, float]:
    numerator = torch.linalg.norm(x_hat - x_aff, ord="fro")
    denominator = torch.linalg.norm(x_hat, ord="fro")
    e_aff = (numerator / denominator.clamp_min(torch.finfo(x_hat.dtype).eps)).item()
    rmse = torch.sqrt(torch.mean((x_hat - x_aff) ** 2)).item()
    sse = torch.sum((x_hat - x_aff) ** 2)
    centered = x_hat - x_hat.mean(dim=0, keepdim=True)
    r2 = (1.0 - sse / torch.sum(centered**2).clamp_min(torch.finfo(x_hat.dtype).eps)).item()
    direct = torch.linalg.norm(x_hat - x_aff, ord="fro") / torch.linalg.norm(x_hat, ord="fro").clamp_min(
        torch.finfo(x_hat.dtype).eps
    )
    assert abs(direct.item() - e_aff) < 1e-12, "E_aff formula sanity check failed"
    return {"E_aff": e_aff, "affine_rmse": rmse, "affine_r2": r2}


def output_nrmse_single(y_hat: Tensor, y: Tensor) -> float:
    return (torch.sqrt(torch.mean((y_hat - y) ** 2)) / torch.std(y).clamp_min(torch.finfo(y.dtype).eps)).item()


def analyze_checkpoint(args, checkpoint_path: Path, output_dir: Path, dtype: torch.dtype, device: torch.device) -> None:
    cfg, model, normalizer, _linear, ckpt = load_checkpoint(checkpoint_path, dtype=dtype, device=device)
    data = load_rlc_dataset(Path(args.dataset), dtype=dtype)
    n = cfg.history_length

    affine_traj: Trajectory = data["affine_calibration"]  # type: ignore[assignment]
    u_aff, y_aff = normalizer.uy(affine_traj)
    x_enc_cal, idx_enc = encoded_latents(model, u_aff, y_aff, n, args.analysis_batch_size, device, dtype)
    x_roll_cal, y_roll_cal, idx_roll = propagated_latents_and_outputs(model, u_aff, y_aff, n, device, dtype)
    assert torch.equal(idx_enc, idx_roll), "encoded and propagated alignment mismatch on affine_calibration"
    x_phys_cal = affine_traj.x[idx_enc]
    assert x_phys_cal.shape[0] == x_enc_cal.shape[0], "temporal alignment sanity check failed"

    P_enc, c_enc = fit_affine_map(x_phys_cal, x_enc_cal)
    P_roll, c_roll = fit_affine_map(x_phys_cal, x_roll_cal)
    print("sanity: affine maps fitted once using affine_calibration only")
    rank_enc = torch.linalg.matrix_rank(P_enc).item()
    rank_roll = torch.linalg.matrix_rank(P_roll).item()
    relation_name = "affine change of coordinates" if cfg.latent_dim == 2 else "affine embedding"
    print(
        f"affine map ranks: enc={rank_enc}, roll={rank_roll}; "
        f"interpretation for n_x={cfg.latent_dim}: {relation_name}"
    )

    rows = []
    figure_dir = output_dir / "figures" / f"seed_{cfg.seed}"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_affine_fit(
        figure_dir / "calibration_affine_fit.png",
        "affine_calibration",
        x_enc_cal,
        apply_affine(P_enc, c_enc, x_phys_cal),
        x_roll_cal,
        apply_affine(P_roll, c_roll, x_phys_cal),
    )

    latent_tests: list[Trajectory] = data["latent_test"]  # type: ignore[assignment]
    representative_done = False
    for traj in latent_tests:
        u_norm, y_norm = normalizer.uy(traj)
        x_enc, idx_enc = encoded_latents(model, u_norm, y_norm, n, args.analysis_batch_size, device, dtype)
        x_roll, y_roll, idx_roll = propagated_latents_and_outputs(model, u_norm, y_norm, n, device, dtype)
        assert torch.equal(idx_enc, idx_roll), f"encoded/roll alignment mismatch on {traj.name}"
        x_phys = traj.x[idx_enc]
        y_aligned = y_norm[idx_enc]

        x_aff_enc = apply_affine(P_enc, c_enc, x_phys)
        x_aff_roll = apply_affine(P_roll, c_roll, x_phys)
        enc_metrics = affine_metrics(x_enc, x_aff_enc)
        roll_metrics = affine_metrics(x_roll, x_aff_roll)
        row = {
            "seed": cfg.seed,
            "checkpoint_epoch": ckpt["epoch"],
            "trajectory": traj.name,
            "kind": traj.kind,
            "E_aff_enc": enc_metrics["E_aff"],
            "E_aff_roll": roll_metrics["E_aff"],
            "affine_rmse_enc": enc_metrics["affine_rmse"],
            "affine_rmse_roll": roll_metrics["affine_rmse"],
            "affine_r2_enc": enc_metrics["affine_r2"],
            "affine_r2_roll": roll_metrics["affine_r2"],
            "propagated_output_nrmse": output_nrmse_single(y_roll, y_aligned),
        }
        rows.append(row)
        if not representative_done and traj.kind == "nominal":
            plot_affine_fit(
                figure_dir / "representative_unseen_affine_fit.png",
                traj.name,
                x_enc,
                x_aff_enc,
                x_roll,
                x_aff_roll,
            )
            representative_done = True

    write_csv(output_dir / f"latent_metrics_seed_{cfg.seed}.csv", rows)
    torch.save(
        {
            "P_enc": P_enc,
            "c_enc": c_enc,
            "P_roll": P_roll,
            "c_roll": c_roll,
            "calibration": "affine_calibration",
            "seed": cfg.seed,
        },
        output_dir / f"affine_maps_seed_{cfg.seed}.pt",
    )
    plot_eaff_scatter(output_dir / "figures" / f"seed_{cfg.seed}" / "E_aff_enc_vs_roll.png", rows)
    write_terminal_summary(rows)
    print("sanity: latent_test trajectories evaluated with frozen affine maps; no refitting performed")


def plot_affine_fit(path: Path, title: str, x_enc: Tensor, x_aff_enc: Tensor, x_roll: Tensor, x_aff_roll: Tensor) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(1000, x_enc.shape[0])
    dims = x_enc.shape[1]
    fig, axes = plt.subplots(dims, 2, figsize=(12, 3.0 * dims), squeeze=False, constrained_layout=True)
    for d in range(dims):
        axes[d, 0].plot(x_enc[:n, d], label="x_hat_enc", linewidth=1.0)
        axes[d, 0].plot(x_aff_enc[:n, d], label="P_enc x + c_enc", linewidth=1.0, alpha=0.85)
        axes[d, 0].set_title(f"encoded latent dim {d}")
        axes[d, 0].grid(True, alpha=0.3)
        axes[d, 1].plot(x_roll[:n, d], label="x_hat_roll", linewidth=1.0)
        axes[d, 1].plot(x_aff_roll[:n, d], label="P_roll x + c_roll", linewidth=1.0, alpha=0.85)
        axes[d, 1].set_title(f"propagated latent dim {d}")
        axes[d, 1].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 1].legend()
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_eaff_scatter(path: Path, rows: list[dict[str, object]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    colors = {"nominal": "tab:blue", "stress_ood": "tab:orange"}
    for kind in sorted({str(row["kind"]) for row in rows}):
        subset = [row for row in rows if row["kind"] == kind]
        ax.scatter(
            [float(row["E_aff_enc"]) for row in subset],
            [float(row["E_aff_roll"]) for row in subset],
            label=kind,
            color=colors.get(kind),
        )
        for row in subset:
            ax.annotate(str(row["trajectory"]), (float(row["E_aff_enc"]), float(row["E_aff_roll"])), fontsize=7)
    ax.set_xlabel("E_aff_enc")
    ax.set_ylabel("E_aff_roll")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_terminal_summary(rows: list[dict[str, object]]) -> None:
    nominal = [row for row in rows if row["kind"] == "nominal"]
    print("nominal latent_test summary (mean ± std):")
    for key in ["E_aff_enc", "E_aff_roll", "affine_rmse_enc", "affine_rmse_roll", "affine_r2_enc", "affine_r2_roll", "propagated_output_nrmse"]:
        values = torch.tensor([float(row[key]) for row in nominal], dtype=torch.float64)
        std = torch.std(values, unbiased=True).item() if len(values) > 1 else 0.0
        print(f"  {key}: {torch.mean(values).item():.6g} ± {std:.6g}")


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_metrics(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.glob("latent_metrics_seed_*.csv")):
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    if rows:
        write_csv(output_dir / "latent_metrics_all.csv", rows)


def parse_args():
    parser = argparse.ArgumentParser(description="E5 RLC latent affine-state experiment.")
    parser.add_argument("--dataset", default="rlc_resdynet_E5_dataset.mat")
    parser.add_argument("--output-dir", default="E5")
    parser.add_argument("--mode", choices=("train", "analyze", "train-analyze"), default="train-analyze")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--analysis-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--history-length", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--linear-order", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--stream-dim", type=int, default=None)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--evolver-hidden-width", type=int, default=64)
    parser.add_argument("--n4sid-block-rows", type=int, default=10)
    parser.add_argument("--max-n4sid-samples", type=int, default=10000)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    if args.mode == "analyze":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in analyze mode.")
        analyze_checkpoint(args, Path(args.checkpoint), output_dir, dtype, device)
        collect_metrics(output_dir)
        return

    seeds = args.seeds if args.seeds is not None else [args.seed]
    data = load_rlc_dataset(Path(args.dataset), dtype=dtype)
    for seed in seeds:
        args.seed = seed
        checkpoint_path = train_model(args, data, output_dir, dtype, device)
        if args.mode == "train-analyze":
            analyze_checkpoint(args, checkpoint_path, output_dir, dtype, device)
    collect_metrics(output_dir)


if __name__ == "__main__":
    main()
