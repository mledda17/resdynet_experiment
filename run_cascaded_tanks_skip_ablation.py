from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from main import FeedForwardBranch, ResDyNet, count_parameters, weighted_multistep_loss
from run_cascaded_tanks_ablation import dataset_to_tensors, standardize_from_train, tensors_from_benchmark


class PlainStreamBlock(nn.Module):
    def __init__(self, stream_dim: int, hidden_dims: tuple[int, ...], activation: type[nn.Module] = nn.Tanh) -> None:
        super().__init__()
        self.ff = FeedForwardBranch(stream_dim, stream_dim, hidden_dims, activation=activation)

    def forward(self, h: Tensor) -> Tensor:
        return self.ff(h)


class PlainEvolver(nn.Module):
    def __init__(
        self,
        nx: int,
        nu: int,
        stream_dim: int,
        num_blocks: int,
        block_hidden_dims: tuple[int, ...],
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.W_i = nn.Linear(nx + nu, stream_dim, bias=False)
        self.blocks = nn.ModuleList(
            [PlainStreamBlock(stream_dim, block_hidden_dims, activation=activation) for _ in range(num_blocks)]
        )
        self.W_o = nn.Linear(stream_dim, nx, bias=False)

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        h = self.W_i(torch.cat((x, u), dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.W_o(h)


class PlainResDyNetCounterpart(nn.Module):
    def __init__(
        self,
        ny: int,
        nu: int,
        nx: int,
        na: int,
        nb: int,
        encoder_hidden_dims: tuple[int, ...],
        decoder_hidden_dims: tuple[int, ...],
        stream_dim: int,
        num_blocks: int,
        evolver_hidden_dims: tuple[int, ...],
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.ny, self.nu, self.nx, self.na, self.nb = ny, nu, nx, na, nb
        self.encoder = FeedForwardBranch(na * ny + nb * nu, nx, encoder_hidden_dims, activation=activation)
        self.evolver = PlainEvolver(nx, nu, stream_dim, num_blocks, evolver_hidden_dims, activation=activation)
        self.decoder = FeedForwardBranch(nx + nu, ny, decoder_hidden_dims, activation=activation)

    def encode_regressor(self, y_past: Tensor, u_past: Tensor) -> Tensor:
        batch_shape = y_past.shape[:-2]
        return self.encoder(
            torch.cat(
                (
                    y_past.reshape(*batch_shape, self.na * self.ny),
                    u_past.reshape(*batch_shape, self.nb * self.nu),
                ),
                dim=-1,
            )
        )

    def rollout(self, y_past: Tensor, u_past: Tensor, u_future: Tensor) -> Tensor:
        x = self.encode_regressor(y_past, u_past)
        outputs = []
        for h in range(u_future.shape[-2]):
            outputs.append(self.decoder(torch.cat((x, u_future[..., h, :]), dim=-1)))
            if h < u_future.shape[-2] - 1:
                x = self.evolver(x, u_future[..., h, :])
        return torch.stack(outputs, dim=-2)

    def forward(self, y_past: Tensor, u_past: Tensor, u_future: Tensor) -> Tensor:
        return self.rollout(y_past, u_past, u_future)


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
    out = []
    for _ in range(epochs):
        perm = torch.randperm(num_samples, generator=generator)
        out.append([perm[start : start + batch_size] for start in range(0, num_samples, batch_size)])
    return out


def load_cascaded_tanks_tensors(args, dtype: torch.dtype):
    from main import RolloutWindowDataset

    if args.dataset_source == "benchmark":
        import nonlinear_benchmarks

        train_val, test = nonlinear_benchmarks.Cascaded_Tanks(atleast_2d=True)
        train_val_u, train_val_y = tensors_from_benchmark(train_val, dtype=dtype)
        test_u, test_y = tensors_from_benchmark(test, dtype=dtype)
        split_idx = int(train_val_u.shape[0] * (1.0 - args.val_fraction))
        train_u_raw, train_y_raw = train_val_u[:split_idx], train_val_y[:split_idx]
        val_u_raw, val_y_raw = train_val_u[split_idx:], train_val_y[split_idx:]
    else:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Local Cascaded Tanks dataset not found: {dataset_path}")
        with np.load(dataset_path, allow_pickle=True) as data:
            required = [f"{split}_{signal}" for split in ("train", "val", "test") for signal in ("u", "y")]
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError(f"Dataset {dataset_path} is missing keys: {missing}")

            def tensor(key: str) -> Tensor:
                value = torch.as_tensor(data[key], dtype=dtype)
                return value.unsqueeze(-1) if value.ndim == 1 else value

            train_u_raw, train_y_raw = tensor("train_u"), tensor("train_y")
            val_u_raw, val_y_raw = tensor("val_u"), tensor("val_y")
            test_u, test_y = tensor("test_u"), tensor("test_y")
    _, normalized = standardize_from_train(train_u_raw, train_y_raw, (val_u_raw, val_y_raw), (test_u, test_y))
    (train_u, train_y), (val_u, val_y), (test_u, test_y) = normalized
    train_ds = RolloutWindowDataset(train_u, train_y, args.history_length, args.history_length, args.horizon)
    val_ds = RolloutWindowDataset(val_u, val_y, args.history_length, args.history_length, args.horizon)
    test_ds = RolloutWindowDataset(test_u, test_y, args.history_length, args.history_length, args.horizon)
    return dataset_to_tensors(train_ds), dataset_to_tensors(val_ds), dataset_to_tensors(test_ds), train_y.shape[1], train_u.shape[1]


def make_full_model(args, ny: int, nu: int, seed: int, dtype: torch.dtype, device: torch.device) -> ResDyNet:
    torch.manual_seed(seed)
    hidden = (args.full_hidden_width,) * args.hidden_layers
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
        evolver_hidden_dims=(args.full_evolver_hidden_width,),
        activation=nn.Tanh,
        zero_nonlinear_outputs=False,
    ).to(device=device, dtype=dtype)


def make_plain_model(args, ny: int, nu: int, seed: int, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(seed)
    hidden = (args.plain_hidden_width,) * args.hidden_layers
    stream_dim = args.stream_dim or 2 * (args.latent_dim + nu) + 1
    return PlainResDyNetCounterpart(
        ny=ny,
        nu=nu,
        nx=args.latent_dim,
        na=args.history_length,
        nb=args.history_length,
        encoder_hidden_dims=hidden,
        decoder_hidden_dims=hidden,
        stream_dim=stream_dim,
        num_blocks=args.num_blocks,
        evolver_hidden_dims=(args.plain_evolver_hidden_width,),
        activation=nn.Tanh,
    ).to(device=device, dtype=dtype)


@torch.no_grad()
def synchronize_shared_initialization(full: ResDyNet, plain: PlainResDyNetCounterpart) -> tuple[float, int]:
    """Copy every like-for-like nonlinear parameter into the plain counterpart."""

    pairs = [
        (full.encoder.ff, plain.encoder),
        (full.evolver.W_i, plain.evolver.W_i),
        (full.evolver.W_o, plain.evolver.W_o),
        (full.decoder.ff, plain.decoder),
    ]
    pairs.extend((full_block.ff, plain_block.ff) for full_block, plain_block in zip(full.evolver.blocks, plain.evolver.blocks, strict=True))
    copied_pairs = []
    copied_parameters = 0
    for full_module, plain_module in pairs:
        full_state = full_module.state_dict()
        plain_state = plain_module.state_dict()
        if full_state.keys() != plain_state.keys() or any(
            full_state[key].shape != plain_state[key].shape for key in full_state
        ):
            continue
        plain_module.load_state_dict(full_state)
        copied_pairs.append((full_module, plain_module))
        copied_parameters += sum(parameter.numel() for parameter in plain_module.parameters())
    max_difference = max(
        (
            torch.max(torch.abs(left.detach() - right.detach())).item()
            for full_module, plain_module in copied_pairs
            for left, right in zip(full_module.parameters(), plain_module.parameters(), strict=True)
        ),
        default=0.0,
    )
    return max_difference, copied_parameters


@torch.no_grad()
def eval_metrics(model: nn.Module, tensors: tuple[Tensor, Tensor, Tensor, Tensor], batch_size: int) -> dict[str, float]:
    y_past, u_past, u_future, y_future = tensors
    param = next(model.parameters())
    was_training = model.training
    model.eval()
    sq = 0.0
    total_loss = 0.0
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


def train_epoch(model: nn.Module, tensors: tuple[Tensor, Tensor, Tensor, Tensor], batches: list[Tensor], optimizer, clip_grad_norm: float | None) -> float:
    y_past, u_past, u_future, y_future = tensors
    param = next(model.parameters())
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
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        total += loss.item() * yf.shape[0]
        count += yf.shape[0]
    return total / count


def run_seed(args, seed: int, tensors, ny: int, nu: int, dtype: torch.dtype, device: torch.device, output_dir: Path):
    models = {
        "ResDyNet": make_full_model(args, ny, nu, seed, dtype, device),
        "Plain": make_plain_model(args, ny, nu, seed, dtype, device),
    }
    max_shared_init_difference, copied_initial_parameters = synchronize_shared_initialization(
        models["ResDyNet"], models["Plain"]
    )
    if max_shared_init_difference != 0.0:
        raise RuntimeError(f"Paired initialization failed for seed {seed}: {max_shared_init_difference}")
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=args.learning_rate) for name, model in models.items()}
    schedule = make_batch_schedule(tensors["train"][0].shape[0], args.batch_size, args.epochs, seed=args.shuffle_seed_base + seed)
    train_losses = {name: eval_metrics(model, tensors["train"], args.eval_batch_size)["loss"] for name, model in models.items()}
    best = {name: {"val_nrmse": float("inf"), "epoch": 0, "path": output_dir / "checkpoints" / f"{name}_seed_{seed}_best.pt"} for name in models}
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
                "val_nrmse_percent": 100.0 * val["nrmse"],
                "wall_time_sec": time.perf_counter() - start,
            }
            histories.append(row)
            epoch_rows[name] = row
            if val["nrmse"] < best[name]["val_nrmse"]:
                best[name]["val_nrmse"] = val["nrmse"]
                best[name]["epoch"] = epoch
                best[name]["path"].parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state": model.state_dict(), "seed": seed, "model": name, "epoch": epoch, "val_nrmse": val["nrmse"]}, best[name]["path"])
        print(f"Seed {seed} | Epoch {epoch:03d}", flush=True)
        for name in ("ResDyNet", "Plain"):
            r = epoch_rows[name]
            print(
                f"{name:<9}: train_loss={r['train_loss']:.6g}, "
                f"val_loss={r['val_loss']:.6g}, val_RMSE={r['val_rmse']:.6g}, val_NRMSE={r['val_nrmse']:.6g}",
                flush=True,
            )
        if epoch == args.epochs:
            break
        for name, model in models.items():
            train_losses[name] = train_epoch(model, tensors["train"], schedule[epoch], optimizers[name], args.clip_grad_norm)
    summaries = []
    for name, model in models.items():
        ckpt = torch.load(best[name]["path"], map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        test = eval_metrics(model, tensors["test"], args.eval_batch_size)
        final_val = [r for r in histories if r["model"] == name and r["epoch"] == args.epochs][0]["val_nrmse"]
        summaries.append(
            {
                "seed": seed,
                "model": name,
                "best_val_nrmse": best[name]["val_nrmse"],
                "best_val_nrmse_percent": 100.0 * best[name]["val_nrmse"],
                "best_epoch": best[name]["epoch"],
                "final_val_nrmse": final_val,
                "final_val_nrmse_percent": 100.0 * final_val,
                "test_nrmse": test["nrmse"],
                "test_nrmse_percent": 100.0 * test["nrmse"],
                "max_shared_initial_parameter_difference": max_shared_init_difference,
                "copied_shared_initial_parameters": copied_initial_parameters,
                "avg_training_time_per_epoch_sec": histories[-1]["wall_time_sec"] / max(args.epochs, 1),
            }
        )
    return histories, summaries


def mean_std(values: list[float]) -> tuple[float, float]:
    t = torch.tensor(values, dtype=torch.float64)
    return torch.mean(t).item(), torch.std(t, unbiased=True).item() if len(values) > 1 else 0.0


def aggregate_summary(
    seed_rows: list[dict[str, object]],
    trainable_param_counts: dict[str, int],
    total_param_counts: dict[str, int],
) -> list[dict[str, object]]:
    out = []
    for model in ("ResDyNet", "Plain"):
        rows = [r for r in seed_rows if r["model"] == model]
        agg = {
            "model": model,
            "num_seeds": len(rows),
            "total_parameters": total_param_counts[model],
            "trainable_parameters": trainable_param_counts[model],
        }
        for key in (
            "best_val_nrmse",
            "best_val_nrmse_percent",
            "best_epoch",
            "final_val_nrmse",
            "final_val_nrmse_percent",
            "test_nrmse",
            "test_nrmse_percent",
            "avg_training_time_per_epoch_sec",
        ):
            mean, std = mean_std([float(r[key]) for r in rows])
            agg[f"{key}_mean"] = mean
            agg[f"{key}_std"] = std
        out.append(agg)
    return out


def save_validation_nrmse_percent_csvs(
    output_dir: Path,
    histories: list[dict[str, object]],
    seeds: list[int],
    epochs: int,
) -> None:
    wide_rows = []
    mean_std_rows = []
    for epoch in range(epochs + 1):
        wide: dict[str, object] = {"epoch": epoch}
        aggregate: dict[str, object] = {"epoch": epoch}
        for model, prefix in (("ResDyNet", "resdynet"), ("Plain", "plain")):
            values = []
            for seed in seeds:
                match = next(
                    row
                    for row in histories
                    if row["model"] == model and int(row["seed"]) == seed and int(row["epoch"]) == epoch
                )
                value = float(match["val_nrmse_percent"])
                wide[f"{prefix}_seed{seed}"] = value
                values.append(value)
            mean, std = mean_std(values)
            aggregate[f"{prefix}_mean"] = mean
            aggregate[f"{prefix}_std"] = std
        wide_rows.append(wide)
        mean_std_rows.append(aggregate)
    write_csv(output_dir / "validation_nrmse_percent_curves.csv", wide_rows)
    write_csv(output_dir / "validation_nrmse_percent_mean_std.csv", mean_std_rows)


def save_plots(output_dir: Path, histories: list[dict[str, object]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        ("val_loss", "validation loss", "validation_loss_vs_epoch_mean_std.png"),
        ("val_nrmse", "validation NRMSE", "validation_nrmse_vs_epoch_mean_std.png"),
        ("train_loss", "training loss", "training_loss_vs_epoch_mean_std.png"),
    ]
    colors = {"ResDyNet": "tab:blue", "Plain": "tab:orange"}
    for key, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
        for model in ("ResDyNet", "Plain"):
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
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cascaded Tanks skip/residual ablation: full ResDyNet vs plain counterpart.")
    p.add_argument("--comparison", choices=("same_width", "parameter_matched", "both"), default="both")
    p.add_argument("--dataset-source", choices=("benchmark", "npz"), default="benchmark")
    p.add_argument("--dataset-path", default="results/end_to_end_cascaded_tanks/dataset.npz")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--history-length", type=int, default=50)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--stream-dim", type=int, default=None)
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--full-hidden-width", type=int, default=64)
    p.add_argument("--full-evolver-hidden-width", type=int, default=64)
    p.add_argument("--plain-hidden-width", type=int, default=64)
    p.add_argument("--plain-evolver-hidden-width", type=int, default=64)
    p.add_argument("--matched-plain-hidden-width", type=int, default=38)
    p.add_argument("--matched-plain-evolver-hidden-width", type=int, default=124)
    p.add_argument("--shuffle-seed-base", type=int, default=20000)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="results/cascaded_tanks_skip_ablation")
    return p.parse_args()


def run_comparison(
    args: argparse.Namespace,
    tensors,
    ny: int,
    nu: int,
    dtype: torch.dtype,
    device: torch.device,
    output_dir: Path,
    comparison_label: str,
    plain_hidden_width: int,
    plain_evolver_hidden_width: int,
) -> list[dict[str, object]]:
    local_args = copy.copy(args)
    local_args.plain_hidden_width = plain_hidden_width
    local_args.plain_evolver_hidden_width = plain_evolver_hidden_width
    output_dir.mkdir(parents=True, exist_ok=True)
    full_probe = make_full_model(local_args, ny, nu, local_args.seeds[0], dtype, device)
    plain_probe = make_plain_model(local_args, ny, nu, local_args.seeds[0], dtype, device)
    trainable_param_counts = {"ResDyNet": count_parameters(full_probe), "Plain": count_parameters(plain_probe)}
    total_param_counts = {
        "ResDyNet": sum(parameter.numel() for parameter in full_probe.parameters()),
        "Plain": sum(parameter.numel() for parameter in plain_probe.parameters()),
    }
    rel_diff = (
        abs(trainable_param_counts["Plain"] - trainable_param_counts["ResDyNet"])
        / trainable_param_counts["ResDyNet"]
    )
    print(
        f"{comparison_label} parameter counts: {total_param_counts}, "
        f"relative difference={rel_diff:.6%}",
        flush=True,
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                **vars(local_args),
                "comparison_label": comparison_label,
                "total_parameter_counts": total_param_counts,
                "trainable_parameter_counts": trainable_param_counts,
                "relative_parameter_difference": rel_diff,
            },
            indent=2,
        )
    )
    del full_probe, plain_probe

    all_histories = []
    seed_summaries = []
    for seed in local_args.seeds:
        histories, summaries = run_seed(local_args, seed, tensors, ny, nu, dtype, device, output_dir)
        for summary in summaries:
            model_name = str(summary["model"])
            summary["total_parameters"] = total_param_counts[model_name]
            summary["trainable_parameters"] = trainable_param_counts[model_name]
        all_histories.extend(histories)
        seed_summaries.extend(summaries)
        write_csv(output_dir / "epoch_history.csv", all_histories)
        write_csv(output_dir / "seed_summary.csv", seed_summaries)

    aggregate = aggregate_summary(seed_summaries, trainable_param_counts, total_param_counts)
    for row in all_histories:
        row["comparison"] = comparison_label
    for row in seed_summaries:
        row["comparison"] = comparison_label
    for row in aggregate:
        row["comparison"] = comparison_label
    write_csv(output_dir / "epoch_history.csv", all_histories)
    write_csv(output_dir / "seed_summary.csv", seed_summaries)
    write_csv(output_dir / "summary_mean_std.csv", aggregate)
    save_validation_nrmse_percent_csvs(output_dir, all_histories, local_args.seeds, local_args.epochs)
    write_csv(
        output_dir / "parameter_counts.csv",
        [
            {
                "model": "ResDyNet",
                "total_parameters": total_param_counts["ResDyNet"],
                "trainable_parameters": trainable_param_counts["ResDyNet"],
                "relative_difference": 0.0,
            },
            {
                "model": "Plain",
                "total_parameters": total_param_counts["Plain"],
                "trainable_parameters": trainable_param_counts["Plain"],
                "relative_difference": rel_diff,
            },
        ],
    )
    save_plots(output_dir, all_histories)
    print(f"saved results to {output_dir}", flush=True)
    for row in aggregate:
        print(row, flush=True)
    return aggregate


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    train_t, val_t, test_t, ny, nu = load_cascaded_tanks_tensors(args, dtype)
    tensors = {"train": train_t, "val": val_t, "test": test_t}

    comparisons = []
    if args.comparison in ("same_width", "both"):
        comparisons.append(("A_same_width", 64, 64))
    if args.comparison in ("parameter_matched", "both"):
        comparisons.append(
            (
                "B_parameter_matched",
                args.matched_plain_hidden_width,
                args.matched_plain_evolver_hidden_width,
            )
        )

    combined = []
    for label, plain_hidden_width, plain_evolver_hidden_width in comparisons:
        out_dir = root_output_dir / label if len(comparisons) > 1 else root_output_dir
        combined.extend(
            run_comparison(
                args,
                tensors,
                ny,
                nu,
                dtype,
                device,
                out_dir,
                label,
                plain_hidden_width,
                plain_evolver_hidden_width,
            )
        )
    write_csv(root_output_dir / "combined_summary_mean_std.csv", combined)


if __name__ == "__main__":
    main()
