from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from main import ResDyNet, RolloutWindowDataset, count_parameters, weighted_multistep_loss
from run_cascaded_tanks_ablation import dataset_to_tensors, standardize_from_train, tensors_from_benchmark
from run_cascaded_tanks_skip_ablation import (
    PlainResDyNetCounterpart,
    make_batch_schedule,
    mean_std,
    synchronize_shared_initialization,
)


MODEL_ORDER = ("ResDyNet", "Plain")
MODEL_PREFIX = {"ResDyNet": "resdynet", "Plain": "plain"}
MODEL_COLORS = {"ResDyNet": "#2f5597", "Plain": "#c55a11"}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_normalized_splits(
    source: str, path: Path, val_fraction: float, dtype: torch.dtype
) -> tuple[dict[str, tuple[Tensor, Tensor]], int, int]:
    if source == "benchmark":
        import nonlinear_benchmarks

        train_val, test = nonlinear_benchmarks.Cascaded_Tanks(atleast_2d=True)
        train_val_u, train_val_y = tensors_from_benchmark(train_val, dtype)
        test_u, test_y = tensors_from_benchmark(test, dtype)
        split_index = int(train_val_u.shape[0] * (1.0 - val_fraction))
        raw = {
            "train": (train_val_u[:split_index], train_val_y[:split_index]),
            "val": (train_val_u[split_index:], train_val_y[split_index:]),
            "test": (test_u, test_y),
        }
    else:
        if not path.exists():
            raise FileNotFoundError(f"Cascaded Tanks dataset not found: {path}")
        with np.load(path, allow_pickle=True) as data:
            required = [f"{split}_{signal}" for split in ("train", "val", "test") for signal in ("u", "y")]
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError(f"Dataset is missing keys: {missing}")

            def tensor(key: str) -> Tensor:
                value = torch.as_tensor(data[key], dtype=dtype)
                return value.unsqueeze(-1) if value.ndim == 1 else value

            raw = {
                split: (tensor(f"{split}_u"), tensor(f"{split}_y"))
                for split in ("train", "val", "test")
            }

    _, normalized = standardize_from_train(raw["train"][0], raw["train"][1], raw["val"], raw["test"])
    splits = dict(zip(("train", "val", "test"), normalized, strict=True))
    return splits, splits["train"][1].shape[1], splits["train"][0].shape[1]


def make_tensors(
    splits: dict[str, tuple[Tensor, Tensor]], history_length: int, horizon: int
) -> dict[str, tuple[Tensor, Tensor, Tensor, Tensor]]:
    return {
        split: dataset_to_tensors(RolloutWindowDataset(u, y, history_length, history_length, horizon))
        for split, (u, y) in splits.items()
    }


def make_models(args: argparse.Namespace, ny: int, nu: int, seed: int, dtype: torch.dtype, device: torch.device):
    hidden = (args.hidden_width,) * args.hidden_layers
    stream_dim = args.stream_dim or 2 * (args.latent_dim + nu) + 1
    torch.manual_seed(seed)
    full = ResDyNet(
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
    torch.manual_seed(seed)
    plain = PlainResDyNetCounterpart(
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
    ).to(device=device, dtype=dtype)
    max_difference, copied_parameters = synchronize_shared_initialization(full, plain)
    if max_difference != 0.0:
        raise RuntimeError(f"Paired initialization failed for seed {seed}: max difference {max_difference}")
    return {"ResDyNet": full, "Plain": plain}, max_difference, copied_parameters


def verify_architectures(models: dict[str, nn.Module], copied_parameters: int) -> dict[str, object]:
    full = models["ResDyNet"]
    plain = models["Plain"]
    checks = {
        "same_encoder_ff": type(full.encoder.ff) is type(plain.encoder),
        "same_decoder_ff": type(full.decoder.ff) is type(plain.decoder),
        "same_num_evolver_blocks": len(full.evolver.blocks) == len(plain.evolver.blocks),
        "same_evolver_projections": (
            tuple(full.evolver.W_i.weight.shape) == tuple(plain.evolver.W_i.weight.shape)
            and tuple(full.evolver.W_o.weight.shape) == tuple(plain.evolver.W_o.weight.shape)
        ),
        "full_has_linear_paths": all(
            hasattr(module, "S") for module in (full.encoder, full.evolver, full.decoder)
        ),
        "plain_has_no_linear_paths": not any(
            hasattr(module, "S") for module in (plain.encoder, plain.evolver, plain.decoder)
        ),
        "full_blocks_are_residual": all(block.__class__.__name__ == "ResidualStreamBlock" for block in full.evolver.blocks),
        "plain_blocks_are_nonresidual": all(block.__class__.__name__ == "PlainStreamBlock" for block in plain.evolver.blocks),
        "all_plain_parameters_are_paired": copied_parameters == sum(p.numel() for p in plain.parameters()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Architecture isolation check failed: {checks}")
    return checks


@torch.no_grad()
def evaluate(model: nn.Module, tensors: tuple[Tensor, Tensor, Tensor, Tensor], batch_size: int) -> dict[str, float]:
    y_past, u_past, u_future, y_future = tensors
    parameter = next(model.parameters())
    was_training = model.training
    model.eval()
    loss_total = 0.0
    squared_total = 0.0
    windows = 0
    values = 0
    for start in range(0, y_future.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        yp = y_past[sl].to(device=parameter.device, dtype=parameter.dtype)
        up = u_past[sl].to(device=parameter.device, dtype=parameter.dtype)
        uf = u_future[sl].to(device=parameter.device, dtype=parameter.dtype)
        yf = y_future[sl].to(device=parameter.device, dtype=parameter.dtype)
        pred = model(yp, up, uf)
        loss_total += weighted_multistep_loss(pred, yf).item() * yf.shape[0]
        squared_total += (pred - yf).square().sum().item()
        windows += yf.shape[0]
        values += yf.numel()
    std = y_future.reshape(-1).std().clamp_min(torch.finfo(y_future.dtype).eps).item()
    rmse = math.sqrt(squared_total / values)
    if was_training:
        model.train()
    return {"loss": loss_total / windows, "rmse": rmse, "nrmse": rmse / std}


@torch.no_grad()
def evaluate_by_prediction_step(
    model: nn.Module, tensors: tuple[Tensor, Tensor, Tensor, Tensor], batch_size: int
) -> list[dict[str, float]]:
    y_past, u_past, u_future, y_future = tensors
    parameter = next(model.parameters())
    was_training = model.training
    model.eval()
    horizon = y_future.shape[1]
    squared = torch.zeros(horizon, dtype=torch.float64)
    counts = torch.zeros(horizon, dtype=torch.float64)
    for start in range(0, y_future.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        pred = model(
            y_past[sl].to(device=parameter.device, dtype=parameter.dtype),
            u_past[sl].to(device=parameter.device, dtype=parameter.dtype),
            u_future[sl].to(device=parameter.device, dtype=parameter.dtype),
        ).cpu().double()
        target = y_future[sl].double()
        squared += (pred - target).square().sum(dim=(0, 2))
        counts += target.shape[0] * target.shape[2]
    rows = []
    rmse = torch.sqrt(squared / counts)
    for step in range(horizon):
        target_std = y_future[:, step, :].reshape(-1).double().std().clamp_min(torch.finfo(torch.float64).eps)
        rows.append(
            {
                "prediction_step": step + 1,
                "rmse": rmse[step].item(),
                "nrmse": (rmse[step] / target_std).item(),
                "nrmse_percent": (100.0 * rmse[step] / target_std).item(),
            }
        )
    if was_training:
        model.train()
    return rows


def train_epoch(
    model: nn.Module,
    tensors: tuple[Tensor, Tensor, Tensor, Tensor],
    batches: list[Tensor],
    optimizer: torch.optim.Optimizer,
    clip_grad_norm: float | None,
) -> float:
    y_past, u_past, u_future, y_future = tensors
    parameter = next(model.parameters())
    model.train()
    total = 0.0
    count = 0
    for indices in batches:
        yp = y_past[indices].to(device=parameter.device, dtype=parameter.dtype)
        up = u_past[indices].to(device=parameter.device, dtype=parameter.dtype)
        uf = u_future[indices].to(device=parameter.device, dtype=parameter.dtype)
        yf = y_future[indices].to(device=parameter.device, dtype=parameter.dtype)
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_multistep_loss(model(yp, up, uf), yf)
        loss.backward()
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        total += loss.item() * yf.shape[0]
        count += yf.shape[0]
    return total / count


def train_pair(
    args: argparse.Namespace,
    training_horizon: int,
    seed: int,
    tensors: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]],
    long_test_tensors: tuple[Tensor, Tensor, Tensor, Tensor],
    ny: int,
    nu: int,
    dtype: torch.dtype,
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    models, max_init_difference, copied_parameters = make_models(args, ny, nu, seed, dtype, device)
    checks = verify_architectures(models, copied_parameters)
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=args.learning_rate) for name, model in models.items()
    }
    schedule = make_batch_schedule(
        tensors["train"][0].shape[0], args.batch_size, args.epochs, args.shuffle_seed_base + seed
    )
    train_losses = {
        name: evaluate(model, tensors["train"], args.eval_batch_size)["loss"] for name, model in models.items()
    }
    checkpoint_dir = output_dir / "checkpoints" / f"H{training_horizon}"
    best = {
        name: {
            "nrmse": float("inf"),
            "epoch": 0,
            "path": checkpoint_dir / f"{name}_seed_{seed}_best.pt",
        }
        for name in MODEL_ORDER
    }
    histories: list[dict[str, object]] = []
    start_time = time.perf_counter()
    for epoch in range(args.epochs + 1):
        epoch_rows = {}
        for name in MODEL_ORDER:
            val = evaluate(models[name], tensors["val"], args.eval_batch_size)
            row = {
                "training_horizon": training_horizon,
                "seed": seed,
                "model": name,
                "epoch": epoch,
                "train_loss": train_losses[name],
                "val_loss": val["loss"],
                "val_rmse": val["rmse"],
                "val_nrmse": val["nrmse"],
                "val_nrmse_percent": 100.0 * val["nrmse"],
                "wall_time_sec": time.perf_counter() - start_time,
            }
            histories.append(row)
            epoch_rows[name] = row
            if val["nrmse"] < best[name]["nrmse"]:
                best[name]["nrmse"] = val["nrmse"]
                best[name]["epoch"] = epoch
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": models[name].state_dict(),
                        "model": name,
                        "seed": seed,
                        "training_horizon": training_horizon,
                        "epoch": epoch,
                        "val_nrmse": val["nrmse"],
                    },
                    best[name]["path"],
                )
        print(f"H={training_horizon:03d} | Seed {seed:02d} | Epoch {epoch:03d}", flush=True)
        for name in MODEL_ORDER:
            row = epoch_rows[name]
            print(
                f"{name:<9}: train={row['train_loss']:.7g} val={row['val_loss']:.7g} "
                f"val_NRMSE={row['val_nrmse_percent']:.5f}%",
                flush=True,
            )
        if epoch == args.epochs:
            break
        for name in MODEL_ORDER:
            train_losses[name] = train_epoch(
                models[name], tensors["train"], schedule[epoch], optimizers[name], args.clip_grad_norm
            )

    summaries = []
    rollout_rows = []
    for name in MODEL_ORDER:
        checkpoint = torch.load(best[name]["path"], map_location=device, weights_only=False)
        models[name].load_state_dict(checkpoint["model_state"])
        test = evaluate(models[name], tensors["test"], args.eval_batch_size)
        summaries.append(
            {
                "training_horizon": training_horizon,
                "seed": seed,
                "model": name,
                "best_val_nrmse_percent": 100.0 * float(best[name]["nrmse"]),
                "best_epoch": int(best[name]["epoch"]),
                "test_nrmse_percent": 100.0 * test["nrmse"],
                "total_parameters": sum(parameter.numel() for parameter in models[name].parameters()),
                "trainable_parameters": count_parameters(models[name]),
                "checkpoint": str(best[name]["path"]),
                "max_shared_initial_parameter_difference": max_init_difference,
                "copied_shared_initial_parameters": copied_parameters,
            }
        )
        for row in evaluate_by_prediction_step(models[name], long_test_tensors, args.eval_batch_size):
            rollout_rows.append(
                {"training_horizon": training_horizon, "seed": seed, "model": name, **row}
            )
    verification = {
        "training_horizon": training_horizon,
        "seed": seed,
        "max_shared_initial_parameter_difference": max_init_difference,
        "copied_shared_initial_parameters": copied_parameters,
        **checks,
    }
    return histories, summaries, rollout_rows, verification


def aggregate_runs(rows: list[dict[str, object]], horizons: list[int]) -> list[dict[str, object]]:
    aggregate = []
    for horizon in horizons:
        for model in MODEL_ORDER:
            selected = [r for r in rows if int(r["training_horizon"]) == horizon and r["model"] == model]
            out: dict[str, object] = {
                "training_horizon": horizon,
                "model": model,
                "num_seeds": len(selected),
                "total_parameters": selected[0]["total_parameters"],
            }
            for key in ("best_val_nrmse_percent", "best_epoch", "test_nrmse_percent"):
                avg, std = mean_std([float(row[key]) for row in selected])
                out[f"{key}_mean"] = avg
                out[f"{key}_std"] = std
            aggregate.append(out)
    return aggregate


def horizon_effect_rows(seed_rows: list[dict[str, object]], horizons: list[int], seeds: list[int]):
    per_seed = []
    aggregate = []
    for horizon in horizons:
        for seed in seeds:
            by_model = {
                row["model"]: row
                for row in seed_rows
                if int(row["training_horizon"]) == horizon and int(row["seed"]) == seed
            }
            per_seed.append(
                {
                    "training_horizon": horizon,
                    "seed": seed,
                    "resdynet_best_val_nrmse_percent": by_model["ResDyNet"]["best_val_nrmse_percent"],
                    "plain_best_val_nrmse_percent": by_model["Plain"]["best_val_nrmse_percent"],
                    "delta_best_val_nrmse_percent": float(by_model["Plain"]["best_val_nrmse_percent"])
                    - float(by_model["ResDyNet"]["best_val_nrmse_percent"]),
                    "resdynet_test_nrmse_percent": by_model["ResDyNet"]["test_nrmse_percent"],
                    "plain_test_nrmse_percent": by_model["Plain"]["test_nrmse_percent"],
                    "delta_test_nrmse_percent": float(by_model["Plain"]["test_nrmse_percent"])
                    - float(by_model["ResDyNet"]["test_nrmse_percent"]),
                }
            )
        horizon_rows = [row for row in per_seed if int(row["training_horizon"]) == horizon]
        out: dict[str, object] = {"training_horizon": horizon}
        for key in (
            "resdynet_best_val_nrmse_percent",
            "plain_best_val_nrmse_percent",
            "delta_best_val_nrmse_percent",
            "resdynet_test_nrmse_percent",
            "plain_test_nrmse_percent",
            "delta_test_nrmse_percent",
        ):
            avg, std = mean_std([float(row[key]) for row in horizon_rows])
            out[f"{key}_mean"] = avg
            out[f"{key}_std"] = std
        aggregate.append(out)
    return per_seed, aggregate


def aggregate_rollouts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregate = []
    keys = sorted({(int(r["training_horizon"]), str(r["model"]), int(r["prediction_step"])) for r in rows})
    for horizon, model, step in keys:
        selected = [
            r for r in rows
            if int(r["training_horizon"]) == horizon and r["model"] == model and int(r["prediction_step"]) == step
        ]
        avg, std = mean_std([float(r["nrmse_percent"]) for r in selected])
        aggregate.append(
            {
                "training_horizon": horizon,
                "model": model,
                "prediction_step": step,
                "nrmse_percent_mean": avg,
                "nrmse_percent_std": std,
            }
        )
    return aggregate


def rollout_wide_rows(rows: list[dict[str, object]], horizons: list[int], seeds: list[int], eval_horizon: int):
    wide = []
    for step in range(1, eval_horizon + 1):
        out: dict[str, object] = {"prediction_step": step}
        for horizon in horizons:
            for model in MODEL_ORDER:
                prefix = f"{MODEL_PREFIX[model]}_H{horizon}"
                values = []
                for seed in seeds:
                    match = next(
                        row for row in rows
                        if int(row["training_horizon"]) == horizon
                        and row["model"] == model
                        and int(row["seed"]) == seed
                        and int(row["prediction_step"]) == step
                    )
                    value = float(match["nrmse_percent"])
                    out[f"{prefix}_seed{seed}"] = value
                    values.append(value)
                avg, std = mean_std(values)
                out[f"{prefix}_mean"] = avg
                out[f"{prefix}_std"] = std
        wide.append(out)
    return wide


def save_plots(output_dir: Path, effect_rows: list[dict[str, object]], rollout_rows: list[dict[str, object]]):
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, ylabel, filename in (
        ("best_val_nrmse_percent", "best validation NRMSE [%]", "horizon_vs_best_validation_nrmse.png"),
        ("test_nrmse_percent", "test NRMSE [%]", "horizon_vs_test_nrmse.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        x = [int(row["training_horizon"]) for row in effect_rows]
        for model in MODEL_ORDER:
            prefix = MODEL_PREFIX[model]
            mean = [float(row[f"{prefix}_{metric}_mean"]) for row in effect_rows]
            std = [float(row[f"{prefix}_{metric}_std"]) for row in effect_rows]
            ax.errorbar(x, mean, yerr=std, marker="o", capsize=3, label=model, color=MODEL_COLORS[model])
        ax.set_xlabel("training horizon H")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)

    horizons = sorted({int(row["training_horizon"]) for row in rollout_rows})
    fig, axes = plt.subplots(len(horizons), 1, figsize=(8.2, 2.7 * len(horizons)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, training_horizon in zip(axes, horizons, strict=True):
        for model in MODEL_ORDER:
            selected = [
                row for row in rollout_rows
                if int(row["training_horizon"]) == training_horizon and row["model"] == model
            ]
            x = np.asarray([int(row["prediction_step"]) for row in selected])
            mean = np.asarray([float(row["nrmse_percent_mean"]) for row in selected])
            std = np.asarray([float(row["nrmse_percent_std"]) for row in selected])
            ax.plot(x, mean, label=model, color=MODEL_COLORS[model])
            ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=MODEL_COLORS[model])
        ax.set_ylabel(f"H={training_horizon}\nNRMSE [%]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("recursive prediction step")
    axes[0].legend(ncol=2)
    fig.savefig(output_dir / "common_100_step_rollout_nrmse.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cascaded Tanks training-horizon ablation: ResDyNet vs Plain MLP.")
    parser.add_argument("--dataset-source", choices=("benchmark", "npz"), default="benchmark")
    parser.add_argument("--dataset-path", default="results/end_to_end_cascaded_tanks/dataset.npz")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default="results/cascaded_tanks_horizon_ablation")
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    parser.add_argument("--evaluation-horizon", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--stream-dim", type=int, default=None)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--evolver-hidden-width", type=int, default=64)
    parser.add_argument("--shuffle-seed-base", type=int, default=20000)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.horizons)) != len(args.horizons) or any(horizon < 1 for horizon in args.horizons):
        raise ValueError("Training horizons must be distinct positive integers.")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits, ny, nu = load_normalized_splits(
        args.dataset_source, Path(args.dataset_path), args.val_fraction, dtype
    )
    long_test_tensors = make_tensors(
        {"test": splits["test"]}, args.history_length, args.evaluation_horizon
    )["test"]

    all_histories: list[dict[str, object]] = []
    all_summaries: list[dict[str, object]] = []
    all_rollouts: list[dict[str, object]] = []
    verifications: list[dict[str, object]] = []
    for training_horizon in args.horizons:
        tensors = make_tensors(splits, args.history_length, training_horizon)
        for seed in args.seeds:
            histories, summaries, rollouts, verification = train_pair(
                args,
                training_horizon,
                seed,
                tensors,
                long_test_tensors,
                ny,
                nu,
                dtype,
                device,
                output_dir,
            )
            all_histories.extend(histories)
            all_summaries.extend(summaries)
            all_rollouts.extend(rollouts)
            verifications.append(verification)
            write_csv(output_dir / "epoch_history.csv", all_histories)
            write_csv(output_dir / "run_summary.csv", all_summaries)
            write_csv(output_dir / "common_rollout_per_seed.csv", all_rollouts)
            write_csv(output_dir / "architecture_verification.csv", verifications)
        del tensors

    aggregate = aggregate_runs(all_summaries, args.horizons)
    effect_per_seed, effect_aggregate = horizon_effect_rows(all_summaries, args.horizons, args.seeds)
    rollout_aggregate = aggregate_rollouts(all_rollouts)
    rollout_wide = rollout_wide_rows(all_rollouts, args.horizons, args.seeds, args.evaluation_horizon)
    write_csv(output_dir / "summary_mean_std.csv", aggregate)
    write_csv(output_dir / "horizon_effect_per_seed.csv", effect_per_seed)
    write_csv(output_dir / "horizon_effect_mean_std.csv", effect_aggregate)
    write_csv(output_dir / "common_rollout_mean_std.csv", rollout_aggregate)
    write_csv(output_dir / "common_rollout_tikz_wide.csv", rollout_wide)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    save_plots(output_dir, effect_aggregate, rollout_aggregate)
    print(f"Saved horizon ablation to {output_dir}", flush=True)
    for row in aggregate:
        print(row, flush=True)


if __name__ == "__main__":
    main()
