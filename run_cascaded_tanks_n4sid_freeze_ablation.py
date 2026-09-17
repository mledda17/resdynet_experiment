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
    weighted_multistep_loss,
)
from run_cascaded_tanks_ablation import (
    dataset_to_tensors,
    scale_linear_realization_for_encoder,
    standardize_from_train,
    tensors_from_benchmark,
)


VARIANTS = ("Random", "N4SID-Joint", "N4SID-Frozen")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_batch_schedule(num_samples: int, batch_size: int, epochs: int, seed: int) -> list[list[Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    schedule = []
    for _ in range(epochs):
        perm = torch.randperm(num_samples, generator=generator)
        schedule.append([perm[start : start + batch_size] for start in range(0, num_samples, batch_size)])
    return schedule


def load_data(args: argparse.Namespace, dtype: torch.dtype):
    import nonlinear_benchmarks

    train_val, test = nonlinear_benchmarks.Cascaded_Tanks(atleast_2d=True)
    train_val_u, train_val_y = tensors_from_benchmark(train_val, dtype=dtype)
    test_u, test_y = tensors_from_benchmark(test, dtype=dtype)
    split_idx = int(train_val_u.shape[0] * (1.0 - args.val_fraction))
    train_u_raw, train_y_raw = train_val_u[:split_idx], train_val_y[:split_idx]
    val_u_raw, val_y_raw = train_val_u[split_idx:], train_val_y[split_idx:]
    _, normalized = standardize_from_train(train_u_raw, train_y_raw, (val_u_raw, val_y_raw), (test_u, test_y))
    (train_u, train_y), (val_u, val_y), (test_u, test_y) = normalized

    train_ds = RolloutWindowDataset(train_u, train_y, args.history_length, args.history_length, args.horizon)
    val_ds = RolloutWindowDataset(val_u, val_y, args.history_length, args.history_length, args.horizon)
    test_ds = RolloutWindowDataset(test_u, test_y, args.history_length, args.history_length, args.horizon)
    tensors = {
        "train": dataset_to_tensors(train_ds),
        "val": dataset_to_tensors(val_ds),
        "test": dataset_to_tensors(test_ds),
    }
    return tensors, train_u, train_y, train_y.shape[1], train_u.shape[1]


def make_model(
    args: argparse.Namespace,
    ny: int,
    nu: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    linear: LinearStateSpace | None,
) -> ResDyNet:
    torch.manual_seed(seed)
    hidden = (args.hidden_width,) * args.hidden_layers
    stream_dim = args.stream_dim or 2 * (args.latent_dim + nu) + 1
    model = ResDyNet(
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
        zero_nonlinear_outputs=True,
    ).to(device=device, dtype=dtype)
    if linear is not None:
        apply_linear_informed_initialization(model, linear, n=args.history_length)
    return model


def freeze_linear_backbone(model: ResDyNet) -> None:
    for module in (model.encoder.S, model.evolver.S, model.decoder.S):
        for param in module.parameters():
            param.requires_grad_(False)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def linear_requires_grad_signature(model: ResDyNet) -> dict[str, bool]:
    return {
        "encoder.S": next(model.encoder.S.parameters()).requires_grad,
        "evolver.S": next(model.evolver.S.parameters()).requires_grad,
        "decoder.S": next(model.decoder.S.parameters()).requires_grad,
    }


@torch.no_grad()
def max_prediction_difference(
    a: nn.Module,
    b: nn.Module,
    tensors: tuple[Tensor, Tensor, Tensor, Tensor],
    batch_size: int,
) -> float:
    y_past, u_past, u_future, _ = tensors
    pa = next(a.parameters())
    pb = next(b.parameters())
    max_diff = 0.0
    was_a, was_b = a.training, b.training
    a.eval()
    b.eval()
    for start in range(0, y_past.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        pred_a = a(
            y_past[sl].to(device=pa.device, dtype=pa.dtype),
            u_past[sl].to(device=pa.device, dtype=pa.dtype),
            u_future[sl].to(device=pa.device, dtype=pa.dtype),
        )
        pred_b = b(
            y_past[sl].to(device=pb.device, dtype=pb.dtype),
            u_past[sl].to(device=pb.device, dtype=pb.dtype),
            u_future[sl].to(device=pb.device, dtype=pb.dtype),
        )
        max_diff = max(max_diff, torch.max(torch.abs(pred_a - pred_b)).item())
    if was_a:
        a.train()
    if was_b:
        b.train()
    return max_diff


@torch.no_grad()
def eval_metrics(model: nn.Module, tensors: tuple[Tensor, Tensor, Tensor, Tensor], batch_size: int) -> dict[str, float]:
    y_past, u_past, u_future, y_future = tensors
    param = next(model.parameters())
    was_training = model.training
    model.eval()
    total_loss = 0.0
    sq = 0.0
    count = 0
    targets = []
    for start in range(0, y_future.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        yp = y_past[sl].to(device=param.device, dtype=param.dtype)
        up = u_past[sl].to(device=param.device, dtype=param.dtype)
        uf = u_future[sl].to(device=param.device, dtype=param.dtype)
        yf = y_future[sl].to(device=param.device, dtype=param.dtype)
        pred = model(yp, up, uf)
        loss = weighted_multistep_loss(pred, yf)
        total_loss += loss.item() * yf.shape[0]
        sq += torch.sum((pred - yf) ** 2).item()
        count += yf.shape[0]
        targets.append(yf.cpu().reshape(-1, yf.shape[-1]))
    target = torch.cat(targets, dim=0)
    rmse = math.sqrt(sq / target.numel())
    nrmse = rmse / torch.std(target).clamp_min(torch.finfo(target.dtype).eps).item()
    if was_training:
        model.train()
    return {"loss": total_loss / count, "rmse": rmse, "nrmse": nrmse}


def train_epoch(
    model: nn.Module,
    tensors: tuple[Tensor, Tensor, Tensor, Tensor],
    batches: list[Tensor],
    optimizer: torch.optim.Optimizer,
    clip_grad_norm: float | None,
) -> float:
    y_past, u_past, u_future, y_future = tensors
    param = next(model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = 0.0
    count = 0
    model.train()
    for idx in batches:
        yp = y_past[idx].to(device=param.device, dtype=param.dtype)
        up = u_past[idx].to(device=param.device, dtype=param.dtype)
        uf = u_future[idx].to(device=param.device, dtype=param.dtype)
        yf = y_future[idx].to(device=param.device, dtype=param.dtype)
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_multistep_loss(model(yp, up, uf), yf)
        loss.backward()
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(trainable, clip_grad_norm)
        optimizer.step()
        total += loss.item() * yf.shape[0]
        count += yf.shape[0]
    return total / count


def extract_backbone(model: ResDyNet, n_linear: int) -> dict[str, Tensor]:
    Sf = model.evolver.S.weight.detach().cpu()
    Sg = model.decoder.S.weight.detach().cpu()
    nx = model.nx
    return {
        "A": Sf[:n_linear, :n_linear].clone(),
        "B": Sf[:n_linear, nx:].clone(),
        "C": Sg[:, :n_linear].clone(),
        "D": Sg[:, nx:].clone(),
    }


def backbone_distances(seed: int, final: dict[str, Tensor], linear: LinearStateSpace, eps: float) -> dict[str, object]:
    reference = {
        "A": linear.A.detach().cpu(),
        "B": linear.B.detach().cpu(),
        "C": linear.C.detach().cpu(),
        "D": linear.D.detach().cpu(),
    }
    row: dict[str, object] = {"seed": seed, "model": "N4SID-Joint"}
    for key in ("A", "B", "C", "D"):
        abs_dist = torch.linalg.norm(final[key] - reference[key]).item()
        denom = torch.linalg.norm(reference[key]).item()
        if key == "D":
            denom = max(denom, eps)
        rel = abs_dist / max(denom, eps)
        row[f"abs_{key}"] = abs_dist
        row[f"d_{key}"] = rel
    return row


def clone_state_for_csv(t: Tensor) -> list[list[float]]:
    return [[float(v) for v in row] for row in t.tolist()]


def save_backbone_json(path: Path, seed: int, initial: dict[str, Tensor], final: dict[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "initial": {k: clone_state_for_csv(v) for k, v in initial.items()},
        "final": {k: clone_state_for_csv(v) for k, v in final.items()},
    }
    path.write_text(json.dumps(payload, indent=2))


def make_models(args, ny, nu, seed, dtype, device, linear) -> dict[str, ResDyNet]:
    models = {
        "Random": make_model(args, ny, nu, seed, dtype, device, linear=None),
        "N4SID-Joint": make_model(args, ny, nu, seed, dtype, device, linear=linear),
        "N4SID-Frozen": make_model(args, ny, nu, seed, dtype, device, linear=linear),
    }
    freeze_linear_backbone(models["N4SID-Frozen"])
    return models


def run_seed(args, seed, tensors, ny, nu, dtype, device, linear, output_dir):
    models = make_models(args, ny, nu, seed, dtype, device, linear)
    epoch0_diff = max_prediction_difference(
        models["N4SID-Joint"], models["N4SID-Frozen"], tensors["val"], args.eval_batch_size
    )
    joint_sig = linear_requires_grad_signature(models["N4SID-Joint"])
    frozen_sig = linear_requires_grad_signature(models["N4SID-Frozen"])
    if any(not v for v in joint_sig.values()):
        raise RuntimeError(f"N4SID-Joint linear branches are not all trainable: {joint_sig}")
    if any(v for v in frozen_sig.values()):
        raise RuntimeError(f"N4SID-Frozen linear branches are not all frozen: {frozen_sig}")
    if epoch0_diff > args.epoch0_tolerance:
        raise RuntimeError(
            f"N4SID-Joint and N4SID-Frozen differ at epoch 0: max abs diff={epoch0_diff:.3e}"
        )

    initial_backbone = extract_backbone(models["N4SID-Joint"], linear.order)
    optimizers = {
        name: torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
        for name, model in models.items()
    }
    schedule = make_batch_schedule(tensors["train"][0].shape[0], args.batch_size, args.epochs, args.shuffle_seed_base + seed)
    train_losses = {name: eval_metrics(model, tensors["train"], args.eval_batch_size)["loss"] for name, model in models.items()}
    best = {
        name: {
            "val_nrmse": float("inf"),
            "epoch": 0,
            "path": output_dir / "checkpoints" / f"{name.replace('-', '_')}_seed_{seed}_best.pt",
        }
        for name in models
    }
    histories = []
    start = time.perf_counter()
    for epoch in range(args.epochs + 1):
        epoch_rows = {}
        for name, model in models.items():
            val = eval_metrics(model, tensors["val"], args.eval_batch_size)
            row = {
                "seed": seed,
                "model": name,
                "epoch": epoch,
                "train_loss": train_losses[name],
                "val_loss": val["loss"],
                "val_rmse": val["rmse"],
                "val_nrmse": val["nrmse"],
                "wall_time_sec": time.perf_counter() - start,
            }
            histories.append(row)
            epoch_rows[name] = row
            if val["nrmse"] < best[name]["val_nrmse"]:
                best[name]["val_nrmse"] = val["nrmse"]
                best[name]["epoch"] = epoch
                best[name]["path"].parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "seed": seed,
                        "model": name,
                        "epoch": epoch,
                        "val_nrmse": val["nrmse"],
                    },
                    best[name]["path"],
                )

        print(f"Seed {seed:02d} | Epoch {epoch:03d}", flush=True)
        for name in VARIANTS:
            row = epoch_rows[name]
            print(
                f"{name:<13}: train={row['train_loss']:.6g} "
                f"val={row['val_loss']:.6g} val_RMSE={row['val_rmse']:.6g} "
                f"val_NRMSE={row['val_nrmse']:.6g}",
                flush=True,
            )

        if epoch == args.epochs:
            break
        for name, model in models.items():
            train_losses[name] = train_epoch(
                model, tensors["train"], schedule[epoch], optimizers[name], args.clip_grad_norm
            )

    seed_summaries = []
    for name, model in models.items():
        ckpt = torch.load(best[name]["path"], map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        test = eval_metrics(model, tensors["test"], args.eval_batch_size)
        final_val = [r for r in histories if r["model"] == name and r["epoch"] == args.epochs][0]["val_nrmse"]
        seed_summaries.append(
            {
                "seed": seed,
                "model": name,
                "best_val_nrmse": best[name]["val_nrmse"],
                "best_epoch": best[name]["epoch"],
                "final_val_nrmse": final_val,
                "test_nrmse": test["nrmse"],
                "training_time_sec": histories[-1]["wall_time_sec"],
                "avg_training_time_per_epoch_sec": histories[-1]["wall_time_sec"] / max(args.epochs, 1),
            }
        )

    final_backbone = extract_backbone(models["N4SID-Joint"], linear.order)
    save_backbone_json(output_dir / "backbone_matrices" / f"N4SID_Joint_seed_{seed}.json", seed, initial_backbone, final_backbone)
    torch.save(
        {"seed": seed, "initial": initial_backbone, "final": final_backbone},
        output_dir / "backbone_matrices" / f"N4SID_Joint_seed_{seed}.pt",
    )
    distance_row = backbone_distances(seed, final_backbone, linear, torch.finfo(dtype).eps)
    verification = {
        "seed": seed,
        "joint_frozen_epoch0_max_abs_prediction_diff": epoch0_diff,
        "joint_linear_requires_grad": json.dumps(joint_sig),
        "frozen_linear_requires_grad": json.dumps(frozen_sig),
    }
    return histories, seed_summaries, distance_row, verification


def mean_std(values: list[float]) -> tuple[float, float]:
    t = torch.tensor(values, dtype=torch.float64)
    return torch.mean(t).item(), torch.std(t, unbiased=True).item() if len(values) > 1 else 0.0


def aggregate_summary(seed_rows: list[dict[str, object]], param_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    params_by_model = {row["model"]: row for row in param_rows}
    out = []
    for model in VARIANTS:
        rows = [r for r in seed_rows if r["model"] == model]
        agg = {
            "model": model,
            "num_seeds": len(rows),
            "total_parameters": params_by_model[model]["total_parameters"],
            "trainable_parameters": params_by_model[model]["trainable_parameters"],
        }
        for key in (
            "best_val_nrmse",
            "best_epoch",
            "final_val_nrmse",
            "test_nrmse",
            "training_time_sec",
            "avg_training_time_per_epoch_sec",
        ):
            mean, std = mean_std([float(r[key]) for r in rows])
            agg[f"{key}_mean"] = mean
            agg[f"{key}_std"] = std
        out.append(agg)
    return out


def save_plots(output_dir: Path, histories: list[dict[str, object]], distance_rows: list[dict[str, object]], n4sid_ref: float) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Random": "tab:gray", "N4SID-Joint": "tab:blue", "N4SID-Frozen": "tab:orange"}
    specs = [
        ("val_nrmse", "validation NRMSE", "validation_nrmse_vs_epoch_mean_std.png"),
        ("val_loss", "validation loss", "validation_loss_vs_epoch_mean_std.png"),
        ("train_loss", "training loss", "training_loss_vs_epoch_mean_std.png"),
    ]
    for key, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
        for model in VARIANTS:
            rows = [r for r in histories if r["model"] == model]
            epochs = sorted({int(r["epoch"]) for r in rows})
            means, stds = [], []
            for epoch in epochs:
                vals = [float(r[key]) for r in rows if int(r["epoch"]) == epoch]
                mean, std = mean_std(vals)
                means.append(mean)
                stds.append(std)
            x = torch.tensor(epochs, dtype=torch.float64)
            m = torch.tensor(means)
            s = torch.tensor(stds)
            ax.plot(x, m, label=model, color=colors[model])
            ax.fill_between(x, m - s, m + s, color=colors[model], alpha=0.18)
        if key == "val_nrmse":
            ax.axhline(n4sid_ref, color="black", linestyle="--", linewidth=1.1, label="standalone N4SID")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)

    if distance_rows:
        labels = ("d_A", "d_B", "d_C", "d_D")
        means = [mean_std([float(r[label]) for r in distance_rows])[0] for label in labels]
        stds = [mean_std([float(r[label]) for r in distance_rows])[1] for label in labels]
        fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        ax.bar(labels, means, yerr=stds, capsize=4, color="tab:blue", alpha=0.85)
        ax.set_ylabel("relative Frobenius distance")
        ax.grid(True, axis="y", alpha=0.3)
        fig.savefig(output_dir / "n4sid_joint_backbone_relative_distances.png", dpi=220)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3-way Cascaded Tanks ablation: random vs N4SID joint vs N4SID frozen.")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--linear-order", type=int, default=6)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--history-length", type=int, default=50)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--stream-dim", type=int, default=None)
    p.add_argument("--hidden-width", type=int, default=64)
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--evolver-hidden-width", type=int, default=64)
    p.add_argument("--n4sid-block-rows", type=int, default=10)
    p.add_argument("--no-linear-state-scaling", action="store_true")
    p.add_argument("--shuffle-seed-base", type=int, default=30000)
    p.add_argument("--epoch0-tolerance", type=float, default=1e-10)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="results/cascaded_tanks_n4sid_freeze_ablation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.latent_dim < args.linear_order:
        raise ValueError("--latent-dim must be >= --linear-order.")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors, train_u, train_y, ny, nu = load_data(args, dtype)
    linear = fit_n4sid_state_space(
        train_u,
        train_y,
        order=args.linear_order,
        num_block_rows=args.n4sid_block_rows,
        zero_direct_feedthrough=True,
    )
    if not args.no_linear_state_scaling:
        linear = scale_linear_realization_for_encoder(linear, args.history_length)

    probe_models = make_models(args, ny, nu, args.seeds[0], dtype, device, linear)
    n4sid_ref = eval_metrics(probe_models["N4SID-Joint"], tensors["val"], args.eval_batch_size)["nrmse"]
    param_rows = [
        {
            "model": name,
            "total_parameters": count_total(model),
            "trainable_parameters": count_trainable(model),
        }
        for name, model in probe_models.items()
    ]
    del probe_models

    config = {
        **vars(args),
        "ny": ny,
        "nu": nu,
        "standalone_n4sid_validation_nrmse": n4sid_ref,
        "parameter_counts": param_rows,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(
        {"A": linear.A.detach().cpu(), "B": linear.B.detach().cpu(), "C": linear.C.detach().cpu(), "D": linear.D.detach().cpu()},
        output_dir / "n4sid_reference_matrices.pt",
    )
    write_csv(output_dir / "parameter_counts.csv", param_rows)
    print(f"standalone N4SID validation NRMSE={n4sid_ref:.6g}", flush=True)
    for row in param_rows:
        print(row, flush=True)

    all_histories = []
    seed_summaries = []
    distance_rows = []
    verification_rows = []
    for seed in args.seeds:
        histories, summaries, distance_row, verification = run_seed(
            args, seed, tensors, ny, nu, dtype, device, linear, output_dir
        )
        all_histories.extend(histories)
        seed_summaries.extend(summaries)
        distance_rows.append(distance_row)
        verification_rows.append(verification)

    aggregate = aggregate_summary(seed_summaries, param_rows)
    write_csv(output_dir / "epoch_history.csv", all_histories)
    write_csv(output_dir / "seed_summary.csv", seed_summaries)
    write_csv(output_dir / "summary_mean_std.csv", aggregate)
    write_csv(output_dir / "n4sid_joint_backbone_distances.csv", distance_rows)
    write_csv(output_dir / "epoch0_verification.csv", verification_rows)
    save_plots(output_dir, all_histories, distance_rows, n4sid_ref)
    print(f"saved results to {output_dir}", flush=True)
    for row in aggregate:
        print(row, flush=True)


if __name__ == "__main__":
    main()
