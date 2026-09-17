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
    ResDyNet,
    RolloutWindowDataset,
    apply_linear_informed_initialization,
    fit_n4sid_state_space,
    weighted_multistep_loss,
)
from run_cascaded_tanks_ablation import (
    dataset_to_tensors,
    make_batch_schedule,
    scale_linear_realization_for_encoder,
)
from test_duffing_n4sid import load_duffing_dataset, make_normalizer


PROFILE_ORDER = ("uniform", "short_term", "long_term")
PROFILE_LABELS = {
    "uniform": "Uniform",
    "short_term": "Short-term emphasis",
    "long_term": "Long-term emphasis",
}
PROFILE_COLORS = {
    "uniform": "#2f5597",
    "short_term": "#c55a11",
    "long_term": "#548235",
}


def gamma_profiles(horizon: int, dtype: torch.dtype) -> dict[str, Tensor]:
    h = torch.arange(horizon, dtype=dtype)
    raw = {
        "uniform": torch.ones(horizon, dtype=dtype),
        "short_term": torch.pow(torch.tensor(0.8, dtype=dtype), h),
        "long_term": torch.pow(torch.tensor(0.8, dtype=dtype), horizon - 1 - h),
    }
    return {name: weights / weights.sum() for name, weights in raw.items()}


def make_model(args, ny: int, nu: int, seed: int, dtype: torch.dtype, device: torch.device) -> ResDyNet:
    torch.manual_seed(seed)
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_data(args, dtype: torch.dtype):
    splits, cfg = load_duffing_dataset(Path(args.dataset), dtype)
    normalizer = make_normalizer(splits["train"])
    normalized = {name: normalizer.uy(split) for name, split in splits.items()}
    tensors = {}
    for name, (u, y) in normalized.items():
        dataset = RolloutWindowDataset(
            u,
            y,
            args.history_length,
            args.history_length,
            args.horizon,
        )
        tensors[name] = dataset_to_tensors(dataset)
    if args.max_train_windows is not None:
        tensors["train"] = tuple(t[: args.max_train_windows] for t in tensors["train"])
    return normalized, tensors, cfg


@torch.no_grad()
def evaluate(
    model: ResDyNet,
    tensors: tuple[Tensor, Tensor, Tensor, Tensor],
    batch_size: int,
    gamma: Tensor | None,
    include_horizons: bool = False,
) -> dict[str, object]:
    y_past, u_past, u_future, y_future = tensors
    param = next(model.parameters())
    was_training = model.training
    model.eval()
    weighted_total = 0.0
    squared_total = 0.0
    count_windows = 0
    count_values = 0
    horizon_squared = torch.zeros(y_future.shape[1], dtype=torch.float64)
    horizon_count = torch.zeros(y_future.shape[1], dtype=torch.float64)
    horizon_targets: list[list[Tensor]] = [[] for _ in range(y_future.shape[1])]
    for start in range(0, y_future.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        yp = y_past[sl].to(device=param.device, dtype=param.dtype)
        up = u_past[sl].to(device=param.device, dtype=param.dtype)
        uf = u_future[sl].to(device=param.device, dtype=param.dtype)
        yf = y_future[sl].to(device=param.device, dtype=param.dtype)
        pred = model(yp, up, uf)
        err2 = (pred - yf).pow(2)
        weighted_total += weighted_multistep_loss(pred, yf, gamma).item() * yf.shape[0]
        squared_total += err2.sum().item()
        count_windows += yf.shape[0]
        count_values += yf.numel()
        if include_horizons:
            horizon_squared += err2.sum(dim=(0, 2)).cpu().double()
            horizon_count += yf.shape[0] * yf.shape[2]
            for h in range(yf.shape[1]):
                horizon_targets[h].append(yf[:, h, :].detach().cpu().reshape(-1).double())
    target_std = torch.std(y_future.reshape(-1).double()).clamp_min(torch.finfo(torch.float64).eps)
    result: dict[str, object] = {
        "loss": weighted_total / count_windows,
        "rmse": math.sqrt(squared_total / count_values),
        "nrmse": math.sqrt(squared_total / count_values) / target_std.item(),
    }
    if include_horizons:
        horizon_nrmse = []
        horizon_rmse = torch.sqrt(horizon_squared / horizon_count)
        for h in range(y_future.shape[1]):
            target_h = torch.cat(horizon_targets[h])
            std_h = torch.std(target_h).clamp_min(torch.finfo(torch.float64).eps)
            horizon_nrmse.append((horizon_rmse[h] / std_h).item())
        result["horizon_rmse"] = horizon_rmse.tolist()
        result["horizon_nrmse"] = horizon_nrmse
    if was_training:
        model.train()
    return result


def horizon_regions(values: list[float]) -> dict[str, float]:
    chunks = torch.tensor_split(torch.tensor(values, dtype=torch.float64), 3)
    return {
        "early_horizon_nrmse": chunks[0].mean().item(),
        "middle_horizon_nrmse": chunks[1].mean().item(),
        "late_horizon_nrmse": chunks[2].mean().item(),
        "full_horizon_mean_nrmse": torch.tensor(values, dtype=torch.float64).mean().item(),
    }


def initialize_model(args, ny, nu, seed, linear, dtype, device) -> ResDyNet:
    model = make_model(args, ny, nu, seed, dtype, device)
    if args.initialization == "n4sid":
        apply_linear_informed_initialization(model, linear, n=args.history_length)
    return model


def max_state_difference(left: dict[str, Tensor], right: dict[str, Tensor]) -> float:
    return max(torch.max(torch.abs(left[key] - right[key])).item() for key in left)


def verify_only_gamma_differs(args, linear, tensors, gammas, dtype, device) -> dict[str, object]:
    ny = tensors["train"][3].shape[-1]
    nu = tensors["train"][2].shape[-1]
    seed = args.seeds[0]
    states = []
    initial_predictions = []
    yp, up, uf, _ = tensors["val"]
    n = min(args.verify_windows, yp.shape[0])
    for _profile in PROFILE_ORDER:
        model = initialize_model(args, ny, nu, seed, linear, dtype, device)
        states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
        with torch.no_grad():
            initial_predictions.append(
                model(
                    yp[:n].to(device=device, dtype=dtype),
                    up[:n].to(device=device, dtype=dtype),
                    uf[:n].to(device=device, dtype=dtype),
                ).cpu()
            )
    schedule_a = make_batch_schedule(
        tensors["train"][0].shape[0], args.batch_size, args.epochs, args.shuffle_seed_base + seed
    )
    schedule_b = make_batch_schedule(
        tensors["train"][0].shape[0], args.batch_size, args.epochs, args.shuffle_seed_base + seed
    )
    schedule_equal = all(torch.equal(a, b) for ea, eb in zip(schedule_a, schedule_b) for a, b in zip(ea, eb))
    return {
        "profiles": PROFILE_ORDER,
        "gamma_sums": {name: gammas[name].sum().item() for name in PROFILE_ORDER},
        "max_initial_parameter_difference": max(
            max_state_difference(states[0], states[i]) for i in range(1, len(states))
        ),
        "max_initial_prediction_difference": max(
            torch.max(torch.abs(initial_predictions[0] - initial_predictions[i])).item()
            for i in range(1, len(initial_predictions))
        ),
        "minibatch_schedules_identical": schedule_equal,
        "differing_configuration_fields": ["gamma_h"],
    }


def train_one(args, profile, seed, gamma, linear, tensors, dtype, device, output_dir):
    y_train, u_train, uf_train, yf_train = tensors["train"]
    model = initialize_model(
        args,
        yf_train.shape[-1],
        uf_train.shape[-1],
        seed,
        linear,
        dtype,
        device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    schedule = make_batch_schedule(
        y_train.shape[0], args.batch_size, args.epochs, args.shuffle_seed_base + seed
    )
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    best_path = output_dir / "checkpoints" / f"{profile}_seed_{seed}_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    last_train_loss = float("nan")

    for epoch in range(args.epochs + 1):
        val = evaluate(model, tensors["val"], args.eval_batch_size, gamma)
        elapsed = time.perf_counter() - start_time
        history.append(
            {
                "profile": profile,
                "seed": seed,
                "epoch": epoch,
                "train_loss": last_train_loss,
                "validation_loss": val["loss"],
                "validation_rmse": val["rmse"],
                "validation_nrmse": val["nrmse"],
                "wall_time_sec": elapsed,
            }
        )
        print(
            f"Seed {seed:02d} | Epoch {epoch:03d} | {PROFILE_LABELS[profile]:19s}: "
            f"train={last_train_loss:.7g} val={float(val['loss']):.7g} "
            f"val_NRMSE={float(val['nrmse']):.7g}",
            flush=True,
        )
        if float(val["nrmse"]) < best_val:
            best_val = float(val["nrmse"])
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "profile": profile,
                    "gamma": gamma.cpu(),
                    "seed": seed,
                    "epoch": epoch,
                    "validation_nrmse": best_val,
                },
                best_path,
            )
        else:
            bad_epochs += 1
        if epoch == args.epochs or (args.patience > 0 and bad_epochs >= args.patience):
            break

        model.train()
        total = 0.0
        count = 0
        param = next(model.parameters())
        for idx in schedule[epoch]:
            yp = y_train[idx].to(device=param.device, dtype=param.dtype)
            up = u_train[idx].to(device=param.device, dtype=param.dtype)
            uf = uf_train[idx].to(device=param.device, dtype=param.dtype)
            yf = yf_train[idx].to(device=param.device, dtype=param.dtype)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_multistep_loss(model(yp, up, uf), yf, gamma)
            loss.backward()
            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            total += loss.item() * yf.shape[0]
            count += yf.shape[0]
        last_train_loss = total / count

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate(model, tensors["test"], args.eval_batch_size, gamma=None, include_horizons=True)
    horizon_nrmse = list(test["horizon_nrmse"])
    summary = {
        "profile": profile,
        "seed": seed,
        "best_validation_nrmse": best_val,
        "best_epoch": best_epoch,
        "test_rmse": test["rmse"],
        "test_nrmse": test["nrmse"],
        **horizon_regions(horizon_nrmse),
        "training_time_sec": time.perf_counter() - start_time,
        "checkpoint": str(best_path),
    }
    horizon_rows = [
        {
            "profile": profile,
            "seed": seed,
            "horizon_step": h + 1,
            "test_rmse": test["horizon_rmse"][h],
            "test_nrmse": horizon_nrmse[h],
        }
        for h in range(args.horizon)
    ]
    return history, summary, horizon_rows


def mean_std(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return tensor.mean().item(), tensor.std(unbiased=True).item() if len(values) > 1 else 0.0


def aggregate_seed_results(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = (
        "best_validation_nrmse",
        "best_epoch",
        "test_rmse",
        "test_nrmse",
        "early_horizon_nrmse",
        "middle_horizon_nrmse",
        "late_horizon_nrmse",
        "full_horizon_mean_nrmse",
        "training_time_sec",
    )
    output = []
    for profile in PROFILE_ORDER:
        selected = [row for row in seed_rows if row["profile"] == profile]
        row: dict[str, object] = {"profile": profile, "num_seeds": len(selected)}
        for key in keys:
            mean, std = mean_std([float(item[key]) for item in selected])
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        output.append(row)
    return output


def aggregate_horizons(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for profile in PROFILE_ORDER:
        for step in sorted({int(row["horizon_step"]) for row in rows}):
            values = [
                float(row["test_nrmse"])
                for row in rows
                if row["profile"] == profile and int(row["horizon_step"]) == step
            ]
            mean, std = mean_std(values)
            output.append(
                {
                    "profile": profile,
                    "horizon_step": step,
                    "test_nrmse_mean": mean,
                    "test_nrmse_std": std,
                    "num_seeds": len(values),
                }
            )
    return output


def save_plots(output_dir: Path, gammas: dict[str, Tensor], horizon_aggregate):
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    for profile in PROFILE_ORDER:
        rows = [row for row in horizon_aggregate if row["profile"] == profile]
        x = torch.tensor([int(row["horizon_step"]) for row in rows])
        mean = torch.tensor([float(row["test_nrmse_mean"]) for row in rows])
        std = torch.tensor([float(row["test_nrmse_std"]) for row in rows])
        ax.plot(x, mean, lw=1.8, color=PROFILE_COLORS[profile], label=PROFILE_LABELS[profile])
        ax.fill_between(x, mean - std, mean + std, color=PROFILE_COLORS[profile], alpha=0.18)
    ax.set(xlabel="Prediction step h", ylabel="Test NRMSE(h)", xlim=(1, len(gammas["uniform"])))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "test_nrmse_by_horizon_mean_std.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    steps = torch.arange(1, len(gammas["uniform"]) + 1)
    for profile in PROFILE_ORDER:
        ax.plot(steps, gammas[profile], lw=1.8, color=PROFILE_COLORS[profile], label=PROFILE_LABELS[profile])
    ax.set(xlabel="Prediction step h", ylabel="Normalized weight $\\gamma_h$", xlim=(1, len(steps)))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "normalized_gamma_profiles.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired gamma_h loss-weight ablation on the Duffing dataset.")
    parser.add_argument("--dataset", default="duffing_resdynet_dataset.mat")
    parser.add_argument("--output-dir", default="results/duffing_gamma_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--initialization", choices=("n4sid", "random"), default="random")
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--linear-order", type=int, default=6)
    parser.add_argument("--history-length", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--stream-dim", type=int, default=None)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--evolver-hidden-width", type=int, default=64)
    parser.add_argument("--n4sid-block-rows", type=int, default=10)
    parser.add_argument("--max-n4sid-samples", type=int, default=10000)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--verify-windows", type=int, default=128)
    parser.add_argument("--shuffle-seed-base", type=int, default=12345)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.initialization == "n4sid" and args.linear_order > args.latent_dim:
        raise ValueError("linear_order must not exceed latent_dim for N4SID initialization.")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized, tensors, dataset_cfg = prepare_data(args, dtype)
    train_u, train_y = normalized["train"]
    n4sid_u = train_u[: args.max_n4sid_samples] if args.max_n4sid_samples else train_u
    n4sid_y = train_y[: args.max_n4sid_samples] if args.max_n4sid_samples else train_y
    linear = None
    if args.initialization == "n4sid":
        linear = fit_n4sid_state_space(
            n4sid_u,
            n4sid_y,
            order=args.linear_order,
            num_block_rows=args.n4sid_block_rows,
            zero_direct_feedthrough=True,
        )
        linear = scale_linear_realization_for_encoder(linear, args.history_length)

    gammas = gamma_profiles(args.horizon, dtype)
    verification = verify_only_gamma_differs(args, linear, tensors, gammas, dtype, device)
    if (
        verification["max_initial_parameter_difference"] != 0.0
        or verification["max_initial_prediction_difference"] != 0.0
        or not verification["minibatch_schedules_identical"]
    ):
        raise RuntimeError(f"Pairing verification failed: {verification}")
    (output_dir / "pairing_verification.json").write_text(json.dumps(verification, indent=2))
    config = vars(args) | {
        "dataset_config": dataset_cfg,
        "gamma_profiles": {name: gamma.tolist() for name, gamma in gammas.items()},
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"pairing verification: {verification}", flush=True)

    histories = []
    seed_rows = []
    horizon_rows = []
    for seed in args.seeds:
        for profile in PROFILE_ORDER:
            history, summary, per_horizon = train_one(
                args, profile, seed, gammas[profile], linear, tensors, dtype, device, output_dir
            )
            histories.extend(history)
            seed_rows.append(summary)
            horizon_rows.extend(per_horizon)
            write_csv(output_dir / "epoch_history.csv", histories)
            write_csv(output_dir / "seed_summary.csv", seed_rows)
            write_csv(output_dir / "test_nrmse_by_horizon_per_seed.csv", horizon_rows)

    aggregate = aggregate_seed_results(seed_rows)
    horizon_aggregate = aggregate_horizons(horizon_rows)
    write_csv(output_dir / "summary_mean_std.csv", aggregate)
    write_csv(output_dir / "test_nrmse_by_horizon_mean_std.csv", horizon_aggregate)
    write_csv(
        output_dir / "gamma_profiles.csv",
        [
            {"horizon_step": h + 1, **{name: gammas[name][h].item() for name in PROFILE_ORDER}}
            for h in range(args.horizon)
        ],
    )
    save_plots(output_dir, gammas, horizon_aggregate)
    print(f"saved results to {output_dir}")
    for row in aggregate:
        print(row)


if __name__ == "__main__":
    main()
