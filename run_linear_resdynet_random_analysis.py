from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from main import ResDyNet, RolloutWindowDataset, weighted_multistep_loss


@dataclass(frozen=True)
class Trajectory:
    u: Tensor
    y: Tensor
    x: Tensor


@dataclass(frozen=True)
class LinearSystem:
    A: Tensor
    B: Tensor
    C: Tensor
    D: Tensor


def true_system(dtype: torch.dtype) -> LinearSystem:
    return LinearSystem(
        A=torch.tensor([[0.7, 1.0, 0.0], [0.0, 0.5, 1.0], [0.0, 0.0, 0.3]], dtype=dtype),
        B=torch.tensor([[1.0], [1.0], [1.0]], dtype=dtype),
        C=torch.tensor([[1.0, 1.0, 1.0]], dtype=dtype),
        D=torch.tensor([[0.0]], dtype=dtype),
    )


def read_mat_array(obj, key: str, dtype: torch.dtype) -> Tensor:
    arr = torch.as_tensor(obj[key][()], dtype=dtype)
    if arr.ndim == 2:
        # h5py gives MATLAB arrays in transposed memory order; scipy-loaded arrays
        # generally arrive as intended. This branch is used only by the h5 loader.
        arr = arr.T
    if arr.ndim == 1:
        arr = arr.unsqueeze(-1)
    return arr.contiguous()


def load_mat_dataset(path: Path, dtype: torch.dtype) -> tuple[Trajectory, Trajectory, LinearSystem | None]:
    try:
        import scipy.io as sio

        data = sio.loadmat(path)
        train = Trajectory(
            u=torch.as_tensor(data["u_train"], dtype=dtype).reshape(-1, 1),
            y=torch.as_tensor(data.get("y_train_noisy", data["y_train"]), dtype=dtype).reshape(-1, 1),
            x=torch.as_tensor(data["x_train"], dtype=dtype),
        )
        test = Trajectory(
            u=torch.as_tensor(data["u_test"], dtype=dtype).reshape(-1, 1),
            y=torch.as_tensor(data.get("y_test_noisy", data["y_test"]), dtype=dtype).reshape(-1, 1),
            x=torch.as_tensor(data["x_test"], dtype=dtype),
        )
        system = None
        if all(k in data for k in ("A_true", "B_true", "C_true", "D_true")):
            system = LinearSystem(
                A=torch.as_tensor(data["A_true"], dtype=dtype),
                B=torch.as_tensor(data["B_true"], dtype=dtype).reshape(-1, 1),
                C=torch.as_tensor(data["C_true"], dtype=dtype).reshape(1, -1),
                D=torch.as_tensor(data["D_true"], dtype=dtype).reshape(1, 1),
            )
        return train, test, system
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as f:
            root = f["dataset"] if "dataset" in f else f
            if "train" in root and "test" in root:
                train_group = root["train"]
                test_group = root["test"]
                train = Trajectory(
                    u=read_mat_array(train_group, "u", dtype),
                    y=read_mat_array(train_group, "y", dtype),
                    x=read_mat_array(train_group, "x", dtype),
                )
                test = Trajectory(
                    u=read_mat_array(test_group, "u", dtype),
                    y=read_mat_array(test_group, "y", dtype),
                    x=read_mat_array(test_group, "x", dtype),
                )
            else:
                train = Trajectory(
                    u=read_mat_array(root, "u_train", dtype),
                    y=read_mat_array(root, "y_train_noisy" if "y_train_noisy" in root else "y_train", dtype),
                    x=read_mat_array(root, "x_train", dtype),
                )
                test = Trajectory(
                    u=read_mat_array(root, "u_test", dtype),
                    y=read_mat_array(root, "y_test_noisy" if "y_test_noisy" in root else "y_test", dtype),
                    x=read_mat_array(root, "x_test", dtype),
                )
            system = None
            if all(k in root for k in ("A_true", "B_true", "C_true", "D_true")):
                system = LinearSystem(
                    A=read_mat_array(root, "A_true", dtype),
                    B=read_mat_array(root, "B_true", dtype),
                    C=read_mat_array(root, "C_true", dtype),
                    D=read_mat_array(root, "D_true", dtype),
                )
            return train, test, system


def generate_rich_input(n: int, u_max: float, seed: int, dtype: torch.dtype) -> Tensor:
    # This reproduces the structure of experiment.m. It is not MATLAB's exact RNG stream.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    u = torch.zeros(n, 1, dtype=dtype)
    edges = torch.round(torch.linspace(0, n, 4)).to(torch.long)

    start, end = int(edges[0]), int(edges[1])
    k = start
    while k < end:
        hold = int(torch.randint(3, 31, (1,), generator=generator).item())
        value = -u_max + 2.0 * u_max * torch.rand(1, generator=generator, dtype=dtype).item()
        k_end = min(k + hold, end)
        u[k:k_end, 0] = value
        k = k_end

    start, end = int(edges[1]), int(edges[2])
    t = torch.arange(end - start, dtype=dtype).unsqueeze(1)
    frequencies = torch.tensor([0.005, 0.011, 0.023, 0.047, 0.091, 0.150], dtype=dtype).unsqueeze(0)
    phases = 2.0 * math.pi * torch.rand(frequencies.shape[1], generator=generator, dtype=dtype).unsqueeze(0)
    sig = torch.sin(2.0 * math.pi * t @ frequencies + phases).sum(dim=1, keepdim=True)
    sig = sig / torch.max(torch.abs(sig)).clamp_min(torch.finfo(dtype).eps)
    u[start:end] = 0.9 * u_max * sig

    start, end = int(edges[2]), int(edges[3])
    u[start:end, 0] = u_max * (2.0 * torch.rand(end - start, generator=generator, dtype=dtype) - 1.0)
    return u


def simulate(system: LinearSystem, u: Tensor) -> Trajectory:
    x = torch.zeros(system.A.shape[0], dtype=u.dtype)
    xs = []
    ys = []
    for k in range(u.shape[0]):
        xs.append(x)
        uk = u[k]
        ys.append(system.C @ x + system.D @ uk)
        x = system.A @ x + system.B @ uk
    return Trajectory(u=u, y=torch.stack(ys, dim=0), x=torch.stack(xs, dim=0))


def load_or_generate_data(args: argparse.Namespace, dtype: torch.dtype) -> tuple[Trajectory, Trajectory, LinearSystem, str]:
    system = true_system(dtype)
    if args.dataset:
        path = Path(args.dataset)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        train, test, file_system = load_mat_dataset(path, dtype)
        if file_system is not None:
            system = file_system
        return train, test, system, f"loaded:{path}"
    if not args.generate_from_experiment_spec:
        raise FileNotFoundError(
            "No saved linear-system dataset was found/provided. Pass --dataset <file.mat> "
            "or explicitly use --generate-from-experiment-spec for a Python regeneration."
        )
    train = simulate(system, generate_rich_input(args.n_train, args.u_max, seed=1, dtype=dtype))
    test = simulate(system, generate_rich_input(args.n_test, args.u_max, seed=100, dtype=dtype))
    return train, test, system, "generated_from_experiment_spec_not_matlab_rng"


def split_train_val(train: Trajectory, val_fraction: float) -> tuple[Trajectory, Trajectory]:
    split = int(train.u.shape[0] * (1.0 - val_fraction))
    return (
        Trajectory(train.u[:split], train.y[:split], train.x[:split]),
        Trajectory(train.u[split:], train.y[split:], train.x[split:]),
    )


def make_model(args: argparse.Namespace, ny: int, nu: int, seed: int, dtype: torch.dtype, device: torch.device) -> ResDyNet:
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


@torch.no_grad()
def eval_metrics(model: ResDyNet, dataset: RolloutWindowDataset, batch_size: int) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    param = next(model.parameters())
    model.eval()
    total_loss = 0.0
    sq = 0.0
    count = 0
    targets = []
    for y_past, u_past, u_future, y_future in loader:
        y_past = y_past.to(device=param.device, dtype=param.dtype)
        u_past = u_past.to(device=param.device, dtype=param.dtype)
        u_future = u_future.to(device=param.device, dtype=param.dtype)
        y_future = y_future.to(device=param.device, dtype=param.dtype)
        pred = model(y_past, u_past, u_future)
        loss = weighted_multistep_loss(pred, y_future)
        total_loss += loss.item() * y_future.shape[0]
        sq += torch.sum((pred - y_future) ** 2).item()
        count += y_future.shape[0]
        targets.append(y_future.cpu().reshape(-1, y_future.shape[-1]))
    target = torch.cat(targets, dim=0)
    rmse = math.sqrt(sq / target.numel())
    nrmse = rmse / torch.std(target).clamp_min(torch.finfo(target.dtype).eps).item()
    return {"loss": total_loss / count, "rmse": rmse, "nrmse": nrmse}


def extract_branches(model: ResDyNet) -> dict[str, Tensor]:
    Sf = model.evolver.S.weight.detach().cpu()
    Sg = model.decoder.S.weight.detach().cpu()
    return {
        "A_f": Sf[:, : model.nx].clone(),
        "B_f": Sf[:, model.nx :].clone(),
        "C_f": Sg[:, : model.nx].clone(),
        "D_f": Sg[:, model.nx :].clone(),
    }


def complex_list(vals: Tensor) -> list[dict[str, float]]:
    return [{"real": float(v.real), "imag": float(v.imag), "abs": float(torch.abs(v))} for v in vals]


def match_eigenvalues(eig_f: Tensor, eig_true: Tensor) -> tuple[list[dict[str, object]], float]:
    remaining = list(range(eig_true.numel()))
    rows = []
    errors = []
    for i, val in enumerate(eig_f):
        dists = [torch.abs(val - eig_true[j]).item() for j in remaining]
        best_pos = min(range(len(remaining)), key=lambda p: dists[p])
        j = remaining.pop(best_pos)
        err = torch.abs(val - eig_true[j]).item()
        errors.append(err)
        rows.append(
            {
                "learned_index": i,
                "true_index": j,
                "lambda_f": complex(val),
                "lambda_true": complex(eig_true[j]),
                "abs_error": err,
            }
        )
    return rows, float(sum(errors) / len(errors))


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


def fit_T(x_phys: Tensor, x_hat: Tensor) -> Tensor:
    # x_hat ~= x_phys @ T.T
    return torch.linalg.lstsq(x_phys, x_hat).solution.T.contiguous()


def similarity_analysis(model: ResDyNet, calibration: Trajectory, test: Trajectory, system: LinearSystem, args, device, dtype) -> dict[str, float]:
    xhat_cal, idx_cal = encoded_latents(model, calibration, args.history_length, args.eval_batch_size, device, dtype)
    T = fit_T(calibration.x[idx_cal], xhat_cal)
    xhat_test, idx_test = encoded_latents(model, test, args.history_length, args.eval_batch_size, device, dtype)
    xhat_test_fit = test.x[idx_test] @ T.T
    latent_fit_nrmse = (
        torch.sqrt(torch.mean((xhat_test - xhat_test_fit) ** 2))
        / torch.std(xhat_test).clamp_min(torch.finfo(xhat_test.dtype).eps)
    ).item()
    T_inv = torch.linalg.inv(T)
    branches = extract_branches(model)
    A_sim = T @ system.A @ T_inv
    B_sim = T @ system.B
    C_sim = system.C @ T_inv
    return {
        "T_cond": torch.linalg.cond(T).item(),
        "latent_test_fit_nrmse": latent_fit_nrmse,
        "similarity_A_error": torch.linalg.norm(branches["A_f"] - A_sim, ord="fro").item(),
        "similarity_B_error": torch.linalg.norm(branches["B_f"] - B_sim, ord="fro").item(),
        "similarity_C_error": torch.linalg.norm(branches["C_f"] - C_sim, ord="fro").item(),
    }


def train_seed(args, seed, datasets, trajectories, system, output_dir, dtype, device):
    train_ds, val_ds, test_ds = datasets
    train_traj, val_traj, test_traj = trajectories
    model = make_model(args, ny=1, nu=1, seed=seed, dtype=dtype, device=device)
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
            f"train_RMSE={train_m['rmse']:.6g} train_NRMSE={train_m['nrmse']:.6g} "
            f"val_RMSE={val_m['rmse']:.6g} val_NRMSE={val_m['nrmse']:.6g}",
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
    branches = extract_branches(model)
    eig_f = torch.linalg.eigvals(branches["A_f"])
    eig_true = torch.linalg.eigvals(system.A)
    matches, mean_eig_err = match_eigenvalues(eig_f, eig_true)
    sim = similarity_analysis(model, train_traj, test_traj, system, args, device, dtype)
    summary = {
        "seed": seed,
        "best_epoch": best["epoch"],
        "train_rmse": train_m["rmse"],
        "train_nrmse": train_m["nrmse"],
        "val_rmse": val_m["rmse"],
        "val_nrmse": val_m["nrmse"],
        "test_rmse": test_m["rmse"],
        "test_nrmse": test_m["nrmse"],
        "eig_mean_abs_error": mean_eig_err,
        "spectral_radius_A_f": torch.max(torch.abs(eig_f)).item(),
        "spectral_radius_A_true": torch.max(torch.abs(eig_true)).item(),
        "A_f_stable": bool(torch.all(torch.abs(eig_f) < 1.0).item()),
        "fro_Af_minus_Atrue_coordinate_dependent": torch.linalg.norm(branches["A_f"] - system.A, ord="fro").item(),
        **sim,
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
        "eig_matches": matches,
    }
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **payload}, output_dir / "checkpoints" / f"seed_{seed}_best.pt")
    torch.save(payload, output_dir / f"matrices_seed_{seed}.pt")
    (output_dir / f"matrices_seed_{seed}.json").write_text(to_json(payload))
    return history, summary


def to_json(obj) -> str:
    def convert(v):
        if isinstance(v, Tensor):
            if torch.is_complex(v):
                return complex_list(v)
            return v.detach().cpu().tolist()
        if isinstance(v, complex):
            return {"real": v.real, "imag": v.imag, "abs": abs(v)}
        if isinstance(v, dict):
            return {k: convert(val) for k, val in v.items()}
        if isinstance(v, list):
            return [convert(val) for val in v]
        return v

    return json.dumps(convert(obj), indent=2)


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
        "eig_mean_abs_error",
        "similarity_A_error",
        "similarity_B_error",
        "similarity_C_error",
        "T_cond",
        "latent_test_fit_nrmse",
    ]
    out = []
    for key in keys:
        mean, std = mean_std([float(r[key]) for r in rows])
        out.append({"metric": key, "mean": mean, "std": std})
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random-initialized ResDyNet on the order-3 linear experiment.m system.")
    p.add_argument("--dataset", default=None, help="Saved MATLAB dataset containing u_train,y_train,x_train,u_test,y_test,x_test.")
    p.add_argument("--generate-from-experiment-spec", action="store_true")
    p.add_argument("--output-dir", default="results/linear_resdynet_random_analysis")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
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
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=3000)
    p.add_argument("--u-max", type=float, default=1.0)
    p.add_argument("--shuffle-seed-base", type=int, default=40000)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.latent_dim != 3:
        raise ValueError("This experiment is defined for --latent-dim 3.")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"data source: {source}", flush=True)
    print("training uses only u,y; true x is used after training for similarity analysis", flush=True)

    all_history = []
    summaries = []
    for seed in args.seeds:
        history, summary = train_seed(
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
        print(f"Seed {seed:02d} learned eig(A_f): {torch.load(output_dir / f'matrices_seed_{seed}.pt')['eig_A_f']}", flush=True)

    write_csv(output_dir / "epoch_history.csv", all_history)
    write_csv(output_dir / "seed_summary.csv", summaries)
    write_csv(output_dir / "summary_mean_std.csv", aggregate(summaries))
    print(f"saved results to {output_dir}", flush=True)
    for row in aggregate(summaries):
        print(row, flush=True)


if __name__ == "__main__":
    main()
