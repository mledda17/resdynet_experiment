from __future__ import annotations

import argparse

import torch

from main import (
    ResDyNet,
    apply_linear_informed_initialization,
    fit_n4sid_state_space,
    official_benchmark_split,
)


def standardize(train_u: torch.Tensor, train_y: torch.Tensor):
    u_mean = train_u.mean(dim=0, keepdim=True)
    u_std = train_u.std(dim=0, keepdim=True).clamp_min(torch.finfo(train_u.dtype).eps)
    y_mean = train_y.mean(dim=0, keepdim=True)
    y_std = train_y.std(dim=0, keepdim=True).clamp_min(torch.finfo(train_y.dtype).eps)
    return (train_u - u_mean) / u_std, (train_y - y_mean) / y_std


def random_windows(
    u: torch.Tensor,
    y: torch.Tensor,
    n: int,
    horizon: int,
    num_windows: int,
    seed: int,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    first_i = n
    last_i = u.shape[0] - horizon
    if last_i < first_i:
        raise ValueError("Trajectory is too short for the requested windows.")
    indices = torch.randint(first_i, last_i + 1, (num_windows,), generator=generator)

    y_past = torch.stack([y[i - n : i] for i in indices], dim=0)
    u_past = torch.stack([u[i - n : i] for i in indices], dim=0)
    u_future = torch.stack([u[i : i + horizon] for i in indices], dim=0)
    return y_past, u_past, u_future


@torch.no_grad()
def resdynet_rollout_with_states(model: ResDyNet, y_past: torch.Tensor, u_past: torch.Tensor, u_future: torch.Tensor):
    x = model.encode_regressor(y_past, u_past)
    states = []
    outputs = []
    for h in range(u_future.shape[1]):
        states.append(x)
        outputs.append(model.decoder(torch.cat((x, u_future[:, h, :]), dim=-1)))
        if h < u_future.shape[1] - 1:
            x = model.evolver(x, u_future[:, h, :])
    return torch.stack(states, dim=1), torch.stack(outputs, dim=1)


@torch.no_grad()
def linear_rollout(model: ResDyNet, y_past: torch.Tensor, u_past: torch.Tensor, u_future: torch.Tensor, n_l: int):
    regressor = torch.cat((y_past.reshape(y_past.shape[0], -1), u_past.reshape(u_past.shape[0], -1)), dim=-1)
    x_l = model.encoder.S(regressor)[:, :n_l]
    A = model.evolver.S.weight[:n_l, :n_l]
    B = model.evolver.S.weight[:n_l, model.nx :]
    C = model.decoder.S.weight[:, :n_l]
    D = model.decoder.S.weight[:, model.nx :]

    states = []
    outputs = []
    for h in range(u_future.shape[1]):
        states.append(x_l)
        outputs.append(x_l @ C.T + u_future[:, h, :] @ D.T)
        if h < u_future.shape[1] - 1:
            x_l = x_l @ A.T + u_future[:, h, :] @ B.T
    return torch.stack(states, dim=1), torch.stack(outputs, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exact ResDyNet linear predictor reproduction at initialization.")
    parser.add_argument("--benchmark", default="WienerHammerBenchMark")
    parser.add_argument("--linear-order", type=int, default=6)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--num-windows", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--num-block-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    train_val_splits, _ = official_benchmark_split(args.benchmark, val_fraction=args.val_fraction, dtype=dtype)
    if len(train_val_splits) != 1:
        raise ValueError("This script expects one continuous train_val trajectory.")
    split = train_val_splits[0]
    u_train_raw = split.u_train[: args.max_train_samples] if args.max_train_samples else split.u_train
    y_train_raw = split.y_train[: args.max_train_samples] if args.max_train_samples else split.y_train
    u_train, y_train = standardize(u_train_raw, y_train_raw)

    linear = fit_n4sid_state_space(
        u_train,
        y_train,
        order=args.linear_order,
        num_block_rows=args.num_block_rows,
        zero_direct_feedthrough=True,
    )

    ny = y_train.shape[1]
    nu = u_train.shape[1]
    n = args.linear_order
    nx = args.latent_dim
    if nx < args.linear_order:
        raise ValueError("--latent-dim must be >= --linear-order.")

    torch.manual_seed(args.seed)
    model = ResDyNet(
        ny=ny,
        nu=nu,
        nx=nx,
        na=n,
        nb=n,
        encoder_hidden_dims=(64, 64),
        decoder_hidden_dims=(64, 64),
        stream_dim=2 * (nx + nu) + 1,
        num_blocks=4,
        evolver_hidden_dims=(64,),
        zero_nonlinear_outputs=False,
    ).to(dtype=dtype)
    apply_linear_informed_initialization(model, linear, n=n)
    model.eval()

    y_past, u_past, u_future = random_windows(
        u_train,
        y_train,
        n=n,
        horizon=args.horizon,
        num_windows=args.num_windows,
        seed=args.seed,
    )

    x_res, y_res = resdynet_rollout_with_states(model, y_past, u_past, u_future)
    x_lin, y_lin = linear_rollout(model, y_past, u_past, u_future, n_l=args.linear_order)

    y_error = torch.max(torch.abs(y_res - y_lin)).item()
    active_state_error = torch.max(torch.abs(x_res[:, :, : args.linear_order] - x_lin)).item()
    inactive_state_max = torch.max(torch.abs(x_res[:, :, args.linear_order :])).item() if nx > args.linear_order else 0.0
    nonlinear_evolver_max = torch.max(
        torch.abs(
            model.evolver.W_o(
                torch.zeros(
                    args.num_windows,
                    model.evolver.stream_dim,
                    dtype=dtype,
                )
            )
        )
    ).item()

    print(f"benchmark: {args.benchmark}")
    print(f"dtype: {args.dtype}")
    print(f"linear order n_L: {args.linear_order}")
    print(f"latent dimension n_x: {nx}")
    print(f"horizon H: {args.horizon}")
    print(f"random rollout windows: {args.num_windows}")
    print(f"max |y_resdynet - y_linear|_inf: {y_error:.12e}")
    print(f"max |x_resdynet[1:n_L] - x_linear|_inf: {active_state_error:.12e}")
    print(f"max |x_resdynet[n_L+1:n_x]|_inf: {inactive_state_max:.12e}")
    print(f"sanity W_o(0) max abs: {nonlinear_evolver_max:.12e}")


if __name__ == "__main__":
    main()
