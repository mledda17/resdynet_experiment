from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset


def _as_tuple(widths: int | Sequence[int]) -> Tuple[int, ...]:
    if isinstance(widths, int):
        return (widths,)
    return tuple(widths)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class FeedForwardBranch(nn.Module):
    """L-layer MLP branch whose last layer is linear."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: int | Sequence[int],
        activation: type[nn.Module] = nn.Tanh,
        zero_output: bool = False,
    ) -> None:
        super().__init__()
        hidden = _as_tuple(hidden_dims)
        dims = (input_dim, *hidden, output_dim)
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            linear = nn.Linear(dims[idx], dims[idx + 1])
            layers.append(linear)
            if idx < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        self.output_layer = next(m for m in reversed(self.net) if isinstance(m, nn.Linear))
        if zero_output:
            self.zero_output_layer()

    def zero_output_layer(self) -> None:
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ProjectedResidualMLP(nn.Module):
    """phi(h) = phi_FF(h) + S h."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: int | Sequence[int],
        activation: type[nn.Module] = nn.Tanh,
        zero_output: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ff = FeedForwardBranch(
            input_dim,
            output_dim,
            hidden_dims,
            activation=activation,
            zero_output=zero_output,
        )
        self.S = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.ff(x) + self.S(x)


class ResidualStreamBlock(nn.Module):
    """h^b = h^{b-1} + phi_FF^{f,b}(h^{b-1})."""

    def __init__(
        self,
        stream_dim: int,
        hidden_dims: int | Sequence[int],
        activation: type[nn.Module] = nn.Tanh,
        zero_output: bool = False,
    ) -> None:
        super().__init__()
        self.ff = FeedForwardBranch(
            stream_dim,
            stream_dim,
            hidden_dims,
            activation=activation,
            zero_output=zero_output,
        )

    def forward(self, h: Tensor) -> Tensor:
        return h + self.ff(h)


class ResDyNetEvolver(nn.Module):
    """Equation (evolver_decomposition) from the provided paper excerpt."""

    def __init__(
        self,
        nx: int,
        nu: int,
        stream_dim: int,
        num_blocks: int,
        block_hidden_dims: int | Sequence[int],
        activation: type[nn.Module] = nn.Tanh,
        zero_output: bool = False,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks B must be positive.")
        if stream_dim < nx + nu:
            raise ValueError("stream_dim n_h must satisfy n_h >= n_x + n_u.")
        self.nx = nx
        self.nu = nu
        self.input_dim = nx + nu
        self.stream_dim = stream_dim
        self.S = nn.Linear(self.input_dim, nx, bias=False)
        self.W_i = nn.Linear(self.input_dim, stream_dim, bias=False)
        self.W_o = nn.Linear(stream_dim, nx, bias=False)
        self.blocks = nn.ModuleList(
            [
                ResidualStreamBlock(
                    stream_dim,
                    block_hidden_dims,
                    activation=activation,
                    zero_output=zero_output,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        h = torch.cat((x, u), dim=-1)
        h0 = self.W_i(h)
        hb = h0
        for block in self.blocks:
            hb = block(hb)
        nonlinear = self.W_o(hb - h0)
        return self.S(h) + nonlinear


class PlainMLPEvolver(nn.Module):
    """Feedforward evolver baseline phi_FF([x,u]) with no residual decomposition."""

    def __init__(
        self,
        nx: int,
        nu: int,
        hidden_dims: Sequence[int],
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nu = nu
        self.input_dim = nx + nu
        self.ff = FeedForwardBranch(self.input_dim, nx, hidden_dims, activation=activation)

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        return self.ff(torch.cat((x, u), dim=-1))


class LinearPlusMLPCorrectionEvolver(nn.Module):
    """Evolver baseline f(h) = S_f h + psi_MLP(h)."""

    def __init__(
        self,
        nx: int,
        nu: int,
        hidden_dims: Sequence[int],
        activation: type[nn.Module] = nn.Tanh,
        zero_output: bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nu = nu
        self.input_dim = nx + nu
        self.S = nn.Linear(self.input_dim, nx, bias=False)
        self.correction = FeedForwardBranch(
            self.input_dim,
            nx,
            hidden_dims,
            activation=activation,
            zero_output=zero_output,
        )

    def zero_output_layer(self) -> None:
        self.correction.zero_output_layer()

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        h = torch.cat((x, u), dim=-1)
        return self.S(h) + self.correction(h)


class ResDyNet(nn.Module):
    def __init__(
        self,
        ny: int,
        nu: int,
        nx: int,
        na: int,
        nb: int,
        encoder_hidden_dims: int | Sequence[int],
        decoder_hidden_dims: int | Sequence[int],
        stream_dim: int,
        num_blocks: int,
        evolver_hidden_dims: int | Sequence[int],
        activation: type[nn.Module] = nn.Tanh,
        zero_nonlinear_outputs: bool = False,
        evolver: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.ny = ny
        self.nu = nu
        self.nx = nx
        self.na = na
        self.nb = nb
        self.regressor_dim = na * ny + nb * nu

        self.encoder = ProjectedResidualMLP(
            self.regressor_dim,
            nx,
            encoder_hidden_dims,
            activation=activation,
            zero_output=zero_nonlinear_outputs,
        )
        self.evolver = evolver or ResDyNetEvolver(
            nx,
            nu,
            stream_dim,
            num_blocks,
            evolver_hidden_dims,
            activation=activation,
            zero_output=zero_nonlinear_outputs,
        )
        self.decoder = ProjectedResidualMLP(
            nx + nu,
            ny,
            decoder_hidden_dims,
            activation=activation,
            zero_output=zero_nonlinear_outputs,
        )

    def encode_regressor(self, y_past: Tensor, u_past: Tensor) -> Tensor:
        batch_shape = y_past.shape[:-2]
        r_y = y_past.reshape(*batch_shape, self.na * self.ny)
        r_u = u_past.reshape(*batch_shape, self.nb * self.nu)
        return self.encoder(torch.cat((r_y, r_u), dim=-1))

    def rollout(self, y_past: Tensor, u_past: Tensor, u_future: Tensor) -> Tensor:
        """Roll out H predictions; the encoder is called exactly once."""

        x = self.encode_regressor(y_past, u_past)
        predictions: list[Tensor] = []
        horizon = u_future.shape[-2]
        for h in range(horizon):
            predictions.append(self.decoder(torch.cat((x, u_future[..., h, :]), dim=-1)))
            if h < horizon - 1:
                x = self.evolver(x, u_future[..., h, :])
        return torch.stack(predictions, dim=-2)

    def forward(self, y_past: Tensor, u_past: Tensor, u_future: Tensor) -> Tensor:
        return self.rollout(y_past, u_past, u_future)


@dataclass(frozen=True)
class LinearStateSpace:
    A: Tensor
    B: Tensor
    C: Tensor
    D: Tensor

    @property
    def order(self) -> int:
        return self.A.shape[0]


def observability_matrix(A: Tensor, C: Tensor, n: int) -> Tensor:
    blocks = []
    Apow = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    for _ in range(n):
        blocks.append(C @ Apow)
        Apow = Apow @ A
    return torch.cat(blocks, dim=0)


def toeplitz_matrix(A: Tensor, B: Tensor, C: Tensor, D: Tensor, n: int) -> Tensor:
    ny, nu = D.shape
    rows = []
    for row in range(n):
        cols = []
        for col in range(n):
            if col > row:
                cols.append(torch.zeros(ny, nu, dtype=A.dtype, device=A.device))
            elif col == row:
                cols.append(D)
            else:
                cols.append(C @ torch.linalg.matrix_power(A, row - col - 1) @ B)
        rows.append(torch.cat(cols, dim=1))
    return torch.cat(rows, dim=0)


def controllability_window_matrix(A: Tensor, B: Tensor, n: int) -> Tensor:
    return torch.cat([torch.linalg.matrix_power(A, p) @ B for p in range(n - 1, -1, -1)], dim=1)


def reconstructability_matrix(linear: LinearStateSpace, n: int) -> Tensor:
    """Build R_L exactly as in equation (linear_state_estimator)."""

    A, B, C, D = linear.A, linear.B, linear.C, linear.D
    O = observability_matrix(A, C, n)
    O_dagger = torch.linalg.solve(O.T @ O, O.T)
    A_n = torch.linalg.matrix_power(A, n)
    T = toeplitz_matrix(A, B, C, D, n)
    C_window = controllability_window_matrix(A, B, n)
    return torch.cat((A_n @ O_dagger, C_window - A_n @ O_dagger @ T), dim=1)


@torch.no_grad()
def apply_linear_informed_initialization(model: ResDyNet, linear: LinearStateSpace, n: int) -> None:
    """Initialize S_e, S_f, S_g and zero all nonlinear output layers."""

    if model.na != n or model.nb != n:
        raise ValueError("The linear-informed initialization requires n_a = n_b = n.")
    nL = linear.order
    if nL > model.nx:
        raise ValueError("The linear model order n_L must satisfy n_L <= n_x.")
    if linear.B.shape[1] != model.nu or linear.C.shape[0] != model.ny:
        raise ValueError("Linear model dimensions do not match ResDyNet dimensions.")
    if not hasattr(model.evolver, "S"):
        raise TypeError("Linear-informed initialization requires an evolver with an S linear branch.")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    A = linear.A.to(device=device, dtype=dtype)
    B = linear.B.to(device=device, dtype=dtype)
    C = linear.C.to(device=device, dtype=dtype)
    D = linear.D.to(device=device, dtype=dtype)
    linear = LinearStateSpace(A, B, C, D)

    R_L = reconstructability_matrix(linear, n)
    Se = torch.zeros(model.nx, model.regressor_dim, dtype=dtype, device=device)
    Se[:nL, :] = R_L
    model.encoder.S.weight.copy_(Se)

    Sf = torch.zeros(model.nx, model.nx + model.nu, dtype=dtype, device=device)
    Sf[:nL, :nL] = A
    Sf[:nL, model.nx :] = B
    model.evolver.S.weight.copy_(Sf)

    Sg = torch.zeros(model.ny, model.nx + model.nu, dtype=dtype, device=device)
    Sg[:, :nL] = C
    Sg[:, model.nx :] = D
    model.decoder.S.weight.copy_(Sg)

    model.encoder.ff.zero_output_layer()
    model.decoder.ff.zero_output_layer()
    if isinstance(model.evolver, ResDyNetEvolver):
        for block in model.evolver.blocks:
            block.ff.zero_output_layer()
    elif hasattr(model.evolver, "zero_output_layer"):
        model.evolver.zero_output_layer()


def weighted_multistep_loss(pred: Tensor, target: Tensor, gamma: Optional[Tensor] = None) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have identical shapes, got {pred.shape} and {target.shape}.")
    err = (target - pred).pow(2).sum(dim=-1)
    if gamma is None:
        return err.mean()
    weights = gamma.to(device=pred.device, dtype=pred.dtype)
    weights = weights * (weights.numel() / weights.sum().clamp_min(torch.finfo(pred.dtype).eps))
    while weights.ndim < err.ndim:
        weights = weights.unsqueeze(0)
    return (err * weights).mean()


class RolloutWindowDataset(Dataset[Tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Windows for admissible i in {max(na, nb), ..., N-H}."""

    def __init__(self, u: Tensor, y: Tensor, na: int, nb: int, horizon: int) -> None:
        if u.ndim != 2 or y.ndim != 2:
            raise ValueError("u and y must be two-dimensional tensors with shape [time, dim].")
        if u.shape[0] != y.shape[0]:
            raise ValueError("u and y must have the same number of time samples.")
        if horizon < 1:
            raise ValueError("horizon H must be positive.")
        self.u = u
        self.y = y
        self.na = na
        self.nb = nb
        self.horizon = horizon
        self.start = max(na, nb)
        self.stop = u.shape[0] - horizon
        if self.stop < self.start:
            raise ValueError("Need at least H + max(na, nb) + 1 samples.")

    def __len__(self) -> int:
        return self.stop - self.start + 1

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        i = self.start + idx
        y_past = self.y[i - self.na : i]
        u_past = self.u[i - self.nb : i]
        u_future = self.u[i : i + self.horizon]
        y_future = self.y[i : i + self.horizon]
        return y_past, u_past, u_future, y_future


@dataclass(frozen=True)
class DatasetSplit:
    u_train: Tensor
    y_train: Tensor
    u_val: Tensor
    y_val: Tensor
    u_test: Tensor
    y_test: Tensor


@dataclass(frozen=True)
class TrainValidationSplit:
    u_train: Tensor
    y_train: Tensor
    u_val: Tensor
    y_val: Tensor


def split_trajectory(
    u: Tensor,
    y: Tensor,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> DatasetSplit:
    """Chronologically split one input-output trajectory into train/val/test."""

    if u.ndim != 2 or y.ndim != 2:
        raise ValueError("u and y must be two-dimensional tensors with shape [time, dim].")
    if u.shape[0] != y.shape[0]:
        raise ValueError("u and y must have the same number of time samples.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1.")

    n_samples = u.shape[0]
    train_end = int(n_samples * train_fraction)
    val_end = train_end + int(n_samples * val_fraction)
    if train_end < 1 or val_end >= n_samples:
        raise ValueError("Fractions produce an empty train or test split.")
    if val_fraction > 0.0 and val_end <= train_end:
        raise ValueError("Fractions produce an empty validation split.")

    return DatasetSplit(
        u_train=u[:train_end],
        y_train=y[:train_end],
        u_val=u[train_end:val_end],
        y_val=y[train_end:val_end],
        u_test=u[val_end:],
        y_test=y[val_end:],
    )


def split_train_validation(
    u: Tensor,
    y: Tensor,
    train_fraction: float = 0.85,
) -> TrainValidationSplit:
    """Chronologically split a train_val trajectory into train and validation."""

    if u.ndim != 2 or y.ndim != 2:
        raise ValueError("u and y must be two-dimensional tensors with shape [time, dim].")
    if u.shape[0] != y.shape[0]:
        raise ValueError("u and y must have the same number of time samples.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    train_end = int(u.shape[0] * train_fraction)
    if train_end < 1 or train_end >= u.shape[0]:
        raise ValueError("train_fraction produces an empty train or validation split.")
    return TrainValidationSplit(
        u_train=u[:train_end],
        y_train=y[:train_end],
        u_val=u[train_end:],
        y_val=y[train_end:],
    )


@dataclass(frozen=True)
class RolloutDatasets:
    train: Dataset[Tuple[Tensor, Tensor, Tensor, Tensor]]
    val: Dataset[Tuple[Tensor, Tensor, Tensor, Tensor]]
    test: Dataset[Tuple[Tensor, Tensor, Tensor, Tensor]]


def make_rollout_datasets(
    u: Tensor,
    y: Tensor,
    na: int,
    nb: int,
    horizon: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> RolloutDatasets:
    split = split_trajectory(u, y, train_fraction=train_fraction, val_fraction=val_fraction)
    return RolloutDatasets(
        train=RolloutWindowDataset(split.u_train, split.y_train, na, nb, horizon),
        val=RolloutWindowDataset(split.u_val, split.y_val, na, nb, horizon),
        test=RolloutWindowDataset(split.u_test, split.y_test, na, nb, horizon),
    )


def _to_2d_tensor(array: object, dtype: torch.dtype = torch.float32) -> Tensor:
    tensor = torch.as_tensor(array, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a one- or two-dimensional trajectory, got shape {tuple(tensor.shape)}.")
    return tensor


def _as_dataset_list(data: object) -> list[object]:
    if isinstance(data, (list, tuple)):
        return list(data)
    return [data]


def _unpack_input_output_data(data: object, dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
    if hasattr(data, "u") and hasattr(data, "y"):
        u, y = data.u, data.y
    else:
        u, y = data
    return _to_2d_tensor(u, dtype=dtype), _to_2d_tensor(y, dtype=dtype)


def official_benchmark_split(
    benchmark_name: str,
    val_fraction: float = 0.15,
    dtype: torch.dtype = torch.float32,
    **benchmark_kwargs: object,
) -> tuple[list[TrainValidationSplit], list[tuple[Tensor, Tensor]]]:
    """Load a nonlinear_benchmarks dataset using its official train/test split.

    The official train_val portion is split chronologically into train and
    validation. The official test portion is returned untouched.
    """

    import nonlinear_benchmarks

    benchmark = getattr(nonlinear_benchmarks, benchmark_name)
    train_val_raw, test_raw = benchmark(atleast_2d=True, **benchmark_kwargs)

    train_val_splits = []
    for dataset in _as_dataset_list(train_val_raw):
        u, y = _unpack_input_output_data(dataset, dtype=dtype)
        train_val_splits.append(split_train_validation(u, y, train_fraction=1.0 - val_fraction))

    test_trajectories = [_unpack_input_output_data(dataset, dtype=dtype) for dataset in _as_dataset_list(test_raw)]
    return train_val_splits, test_trajectories


def make_official_benchmark_rollout_datasets(
    benchmark_name: str,
    na: int,
    nb: int,
    horizon: int,
    val_fraction: float = 0.15,
    dtype: torch.dtype = torch.float32,
    **benchmark_kwargs: object,
) -> tuple[RolloutDatasets, list[int | None]]:
    """Create rollout datasets from a nonlinear_benchmarks benchmark.

    Returns the rollout datasets and the official test state-initialization
    window lengths, when provided by the benchmark objects.
    """

    import nonlinear_benchmarks

    benchmark = getattr(nonlinear_benchmarks, benchmark_name)
    train_val_raw, test_raw = benchmark(atleast_2d=True, **benchmark_kwargs)

    train_datasets = []
    val_datasets = []
    for dataset in _as_dataset_list(train_val_raw):
        u, y = _unpack_input_output_data(dataset, dtype=dtype)
        split = split_train_validation(u, y, train_fraction=1.0 - val_fraction)
        train_datasets.append(RolloutWindowDataset(split.u_train, split.y_train, na, nb, horizon))
        val_datasets.append(RolloutWindowDataset(split.u_val, split.y_val, na, nb, horizon))

    test_datasets = []
    init_lengths: list[int | None] = []
    for dataset in _as_dataset_list(test_raw):
        u, y = _unpack_input_output_data(dataset, dtype=dtype)
        test_datasets.append(RolloutWindowDataset(u, y, na, nb, horizon))
        init_lengths.append(getattr(dataset, "state_initialization_window_length", None))

    rollout_datasets = RolloutDatasets(
        train=ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0],
        val=ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0],
        test=ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0],
    )
    return rollout_datasets, init_lengths


def make_resdynet(
    ny: int,
    nu: int,
    nx: int,
    na: int,
    nb: int,
    encoder_hidden_dims: int | Sequence[int],
    decoder_hidden_dims: int | Sequence[int],
    stream_dim: int,
    num_blocks: int,
    evolver_hidden_dims: int | Sequence[int],
    linear: Optional[LinearStateSpace] = None,
    activation: type[nn.Module] = nn.Tanh,
) -> ResDyNet:
    model = ResDyNet(
        ny,
        nu,
        nx,
        na,
        nb,
        encoder_hidden_dims,
        decoder_hidden_dims,
        stream_dim,
        num_blocks,
        evolver_hidden_dims,
        activation=activation,
        zero_nonlinear_outputs=linear is not None,
    )
    if linear is not None:
        apply_linear_informed_initialization(model, linear, na)
    return model


def make_random_resdynet_baseline(
    ny: int,
    nu: int,
    nx: int,
    na: int,
    nb: int,
    encoder_hidden_dims: int | Sequence[int],
    decoder_hidden_dims: int | Sequence[int],
    stream_dim: int,
    num_blocks: int,
    evolver_hidden_dims: int | Sequence[int],
    activation: type[nn.Module] = nn.Tanh,
) -> ResDyNet:
    return ResDyNet(
        ny,
        nu,
        nx,
        na,
        nb,
        encoder_hidden_dims,
        decoder_hidden_dims,
        stream_dim,
        num_blocks,
        evolver_hidden_dims,
        activation=activation,
        zero_nonlinear_outputs=False,
    )


def _plain_evolver_param_count(nx: int, nu: int, hidden_dims: Sequence[int]) -> int:
    dims = (nx + nu, *hidden_dims, nx)
    return sum(dims[i + 1] * (dims[i] + 1) for i in range(len(dims) - 1))


def matched_plain_hidden_dims(
    target_evolver_params: int,
    nx: int,
    nu: int,
    depth: int,
    max_width: int = 4096,
) -> Tuple[int, ...]:
    """Choose equal-width hidden layers with closest parameter count."""

    if depth < 1:
        return ()
    best_width = 1
    best_gap = float("inf")
    for width in range(1, max_width + 1):
        dims = (width,) * depth
        gap = abs(_plain_evolver_param_count(nx, nu, dims) - target_evolver_params)
        if gap < best_gap:
            best_width = width
            best_gap = gap
    return (best_width,) * depth


def make_plain_mlp_evolver_baseline(
    ny: int,
    nu: int,
    nx: int,
    na: int,
    nb: int,
    encoder_hidden_dims: int | Sequence[int],
    decoder_hidden_dims: int | Sequence[int],
    stream_dim: int,
    num_blocks: int,
    evolver_hidden_dims: int | Sequence[int],
    plain_depth: Optional[int] = None,
    activation: type[nn.Module] = nn.Tanh,
) -> ResDyNet:
    reference = ResDyNetEvolver(nx, nu, stream_dim, num_blocks, evolver_hidden_dims, activation=activation)
    target = count_parameters(reference)
    block_depth = len(_as_tuple(evolver_hidden_dims)) + 1
    depth = plain_depth if plain_depth is not None else max(1, num_blocks * block_depth)
    hidden_dims = matched_plain_hidden_dims(target, nx, nu, depth)
    return ResDyNet(
        ny,
        nu,
        nx,
        na,
        nb,
        encoder_hidden_dims,
        decoder_hidden_dims,
        stream_dim,
        num_blocks,
        evolver_hidden_dims,
        activation=activation,
        evolver=PlainMLPEvolver(nx, nu, hidden_dims, activation=activation),
    )


def train_model(
    model: ResDyNet,
    u: Tensor,
    y: Tensor,
    horizon: int,
    epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    gamma: Optional[Tensor] = None,
    shuffle: bool = True,
    clip_grad_norm: Optional[float] = None,
) -> list[float]:
    dataset = RolloutWindowDataset(u, y, model.na, model.nb, horizon)
    return train_on_dataset(
        model,
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        gamma=gamma,
        shuffle=shuffle,
        clip_grad_norm=clip_grad_norm,
    )


def train_on_dataset(
    model: ResDyNet,
    dataset: Dataset[Tuple[Tensor, Tensor, Tensor, Tensor]],
    epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    gamma: Optional[Tensor] = None,
    shuffle: bool = True,
    clip_grad_norm: Optional[float] = None,
) -> list[float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        count = 0
        for y_past, u_past, u_future, y_future in loader:
            param = next(model.parameters())
            y_past = y_past.to(device=param.device, dtype=param.dtype)
            u_past = u_past.to(device=param.device, dtype=param.dtype)
            u_future = u_future.to(device=param.device, dtype=param.dtype)
            y_future = y_future.to(device=param.device, dtype=param.dtype)
            optimizer.zero_grad(set_to_none=True)
            pred = model(y_past, u_past, u_future)
            loss = weighted_multistep_loss(pred, y_future, gamma)
            loss.backward()
            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            batch = y_future.shape[0]
            total += loss.detach().item() * batch
            count += batch
        history.append(total / count)
    return history


@torch.no_grad()
def evaluate_model(
    model: ResDyNet,
    dataset: Dataset[Tuple[Tensor, Tensor, Tensor, Tensor]],
    batch_size: int,
    gamma: Optional[Tensor] = None,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for y_past, u_past, u_future, y_future in loader:
        param = next(model.parameters())
        y_past = y_past.to(device=param.device, dtype=param.dtype)
        u_past = u_past.to(device=param.device, dtype=param.dtype)
        u_future = u_future.to(device=param.device, dtype=param.dtype)
        y_future = y_future.to(device=param.device, dtype=param.dtype)
        pred = model(y_past, u_past, u_future)
        loss = weighted_multistep_loss(pred, y_future, gamma)
        batch = y_future.shape[0]
        total += loss.item() * batch
        count += batch
    if was_training:
        model.train()
    return total / count


def least_squares_linear_state_space(
    x: Tensor,
    u: Tensor,
    y: Tensor,
) -> LinearStateSpace:
    """Convenience helper when state samples are available."""

    if x.shape[0] != u.shape[0] or x.shape[0] != y.shape[0]:
        raise ValueError("x, u, and y must share the same time dimension.")
    xu_state = torch.cat((x[:-1], u[:-1]), dim=1)
    AB = torch.linalg.lstsq(xu_state, x[1:]).solution.T
    xu_output = torch.cat((x, u), dim=1)
    CD = torch.linalg.lstsq(xu_output, y).solution.T
    nx = x.shape[1]
    return LinearStateSpace(A=AB[:, :nx], B=AB[:, nx:], C=CD[:, :nx], D=CD[:, nx:])


def _torch_to_numpy(tensor: Tensor):
    return tensor.detach().cpu().numpy()


def fit_n4sid_state_space(
    u: Tensor,
    y: Tensor,
    order: int,
    num_block_rows: Optional[int] = None,
    zero_direct_feedthrough: bool = True,
) -> LinearStateSpace:
    """Estimate A, B, C, D using N4SID/PO-MOESP subspace identification.

    Inputs are assumed to be preprocessed as desired by the experiment pipeline
    e.g. zero mean and unit standard deviation, as in the referenced paper.
    """

    if u.ndim != 2 or y.ndim != 2:
        raise ValueError("u and y must be two-dimensional tensors with shape [time, dim].")
    if u.shape[0] != y.shape[0]:
        raise ValueError("u and y must have the same number of time samples.")
    if order < 1:
        raise ValueError("order must be positive.")
    if num_block_rows is None:
        num_block_rows = max(order + 1, 2 * order)
    if num_block_rows <= order:
        raise ValueError("num_block_rows should be larger than the requested system order.")

    import os

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

    import pandas as pd
    from nfoursid.nfoursid import NFourSID

    u_columns = [f"u{i}" for i in range(u.shape[1])]
    y_columns = [f"y{i}" for i in range(y.shape[1])]
    data = {
        **{name: _torch_to_numpy(u[:, idx]) for idx, name in enumerate(u_columns)},
        **{name: _torch_to_numpy(y[:, idx]) for idx, name in enumerate(y_columns)},
    }
    dataframe = pd.DataFrame(data)

    n4sid = NFourSID(
        dataframe,
        output_columns=y_columns,
        input_columns=u_columns,
        num_block_rows=num_block_rows,
    )
    n4sid.subspace_identification()
    state_space, _ = n4sid.system_identification(rank=order)

    dtype = y.dtype
    device = y.device
    A = torch.as_tensor(state_space.a, dtype=dtype, device=device)
    B = torch.as_tensor(state_space.b, dtype=dtype, device=device)
    C = torch.as_tensor(state_space.c, dtype=dtype, device=device)
    D = torch.zeros(y.shape[1], u.shape[1], dtype=dtype, device=device)
    if not zero_direct_feedthrough:
        D = torch.as_tensor(state_space.d, dtype=dtype, device=device)
    return LinearStateSpace(A=A, B=B, C=C, D=D)


def fit_n4sid_state_space_from_trajectories(
    trajectories: Sequence[tuple[Tensor, Tensor]],
    order: int,
    num_block_rows: Optional[int] = None,
    zero_direct_feedthrough: bool = True,
) -> LinearStateSpace:
    """Estimate N4SID model from one or more trajectories.

    For multiple trajectories, rows are concatenated for the subspace-ID call.
    Prefer passing one continuous training trajectory when available.
    """

    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    if len(trajectories) == 1:
        u, y = trajectories[0]
        return fit_n4sid_state_space(
            u,
            y,
            order=order,
            num_block_rows=num_block_rows,
            zero_direct_feedthrough=zero_direct_feedthrough,
        )

    u0, y0 = trajectories[0]
    nu, ny = u0.shape[1], y0.shape[1]
    for u, y in trajectories:
        if u.ndim != 2 or y.ndim != 2 or u.shape[1] != nu or y.shape[1] != ny:
            raise ValueError("All trajectories must be two-dimensional and share input/output dimensions.")
    return fit_n4sid_state_space(
        torch.cat([u for u, _ in trajectories], dim=0),
        torch.cat([y for _, y in trajectories], dim=0),
        order=order,
        num_block_rows=num_block_rows,
        zero_direct_feedthrough=zero_direct_feedthrough,
    )


def estimate_markov_parameters_bla(
    u: Tensor,
    y: Tensor,
    fir_horizon: int,
    ridge: float = 0.0,
) -> Tensor:
    """Least-squares BLA/FIR estimate of Markov parameters G_0, ..., G_M.

    The fitted linear convolution model is
    y_k = G_0 u_k + G_1 u_{k-1} + ... + G_M u_{k-M} + residual_k.
    Returned shape is [M + 1, n_y, n_u].
    """

    if u.ndim != 2 or y.ndim != 2:
        raise ValueError("u and y must be two-dimensional tensors with shape [time, dim].")
    if u.shape[0] != y.shape[0]:
        raise ValueError("u and y must have the same number of time samples.")
    if fir_horizon < 1:
        raise ValueError("fir_horizon must be at least 1.")
    if u.shape[0] <= fir_horizon:
        raise ValueError("Need more samples than fir_horizon.")

    rows = []
    targets = []
    for k in range(fir_horizon, u.shape[0]):
        rows.append(torch.cat([u[k - lag] for lag in range(fir_horizon + 1)], dim=0))
        targets.append(y[k])
    phi = torch.stack(rows, dim=0)
    target = torch.stack(targets, dim=0)

    if ridge > 0:
        gram = phi.T @ phi
        reg = ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        theta = torch.linalg.solve(gram + reg, phi.T @ target)
    else:
        theta = torch.linalg.lstsq(phi, target).solution

    nu = u.shape[1]
    ny = y.shape[1]
    return theta.T.reshape(ny, fir_horizon + 1, nu).permute(1, 0, 2).contiguous()


def estimate_markov_parameters_bla_from_trajectories(
    trajectories: Iterable[tuple[Tensor, Tensor]],
    fir_horizon: int,
    ridge: float = 0.0,
) -> Tensor:
    """BLA/FIR Markov estimate from multiple independent trajectories."""

    if fir_horizon < 1:
        raise ValueError("fir_horizon must be at least 1.")

    rows = []
    targets = []
    nu = None
    ny = None
    for u, y in trajectories:
        if u.ndim != 2 or y.ndim != 2:
            raise ValueError("Each u and y must have shape [time, dim].")
        if u.shape[0] != y.shape[0]:
            raise ValueError("Each u and y pair must share the same time dimension.")
        if u.shape[0] <= fir_horizon:
            continue
        nu = u.shape[1] if nu is None else nu
        ny = y.shape[1] if ny is None else ny
        if u.shape[1] != nu or y.shape[1] != ny:
            raise ValueError("All trajectories must share input and output dimensions.")
        for k in range(fir_horizon, u.shape[0]):
            rows.append(torch.cat([u[k - lag] for lag in range(fir_horizon + 1)], dim=0))
            targets.append(y[k])

    if not rows or nu is None or ny is None:
        raise ValueError("No trajectory is long enough for the requested fir_horizon.")

    phi = torch.stack(rows, dim=0)
    target = torch.stack(targets, dim=0)
    if ridge > 0:
        gram = phi.T @ phi
        reg = ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        theta = torch.linalg.solve(gram + reg, phi.T @ target)
    else:
        theta = torch.linalg.lstsq(phi, target).solution
    return theta.T.reshape(ny, fir_horizon + 1, nu).permute(1, 0, 2).contiguous()


def realize_markov_parameters_ho_kalman(
    markov: Tensor,
    order: int,
    hankel_rows: Optional[int] = None,
    hankel_cols: Optional[int] = None,
) -> LinearStateSpace:
    """Realize Markov parameters with a truncated Ho-Kalman/ERA construction."""

    if markov.ndim != 3:
        raise ValueError("markov must have shape [num_parameters, n_y, n_u].")
    if order < 1:
        raise ValueError("order must be positive.")

    num_markov, ny, nu = markov.shape
    max_dynamic_lags = num_markov - 1
    hankel_rows = hankel_rows or max(1, (order + ny - 1) // ny)
    hankel_cols = hankel_cols or max(1, (order + nu - 1) // nu)
    if hankel_rows * ny < order or hankel_cols * nu < order:
        raise ValueError("Hankel dimensions must be large enough for the requested order.")
    if hankel_rows + hankel_cols > max_dynamic_lags:
        raise ValueError("Need fir_horizon >= hankel_rows + hankel_cols for Ho-Kalman realization.")

    def block_hankel(offset: int) -> Tensor:
        block_rows = []
        for i in range(hankel_rows):
            block_rows.append(torch.cat([markov[offset + i + j] for j in range(hankel_cols)], dim=1))
        return torch.cat(block_rows, dim=0)

    H0 = block_hankel(1)
    H1 = block_hankel(2)
    U, singular_values, Vh = torch.linalg.svd(H0, full_matrices=False)
    U_r = U[:, :order]
    s_r = singular_values[:order]
    V_r = Vh[:order, :].T
    sqrt_s = torch.diag(torch.sqrt(s_r))
    inv_sqrt_s = torch.diag(torch.rsqrt(s_r))

    A = inv_sqrt_s @ U_r.T @ H1 @ V_r @ inv_sqrt_s
    B = sqrt_s @ Vh[:order, :nu]
    C = U_r[:ny, :] @ sqrt_s
    D = markov[0]
    return LinearStateSpace(A=A, B=B, C=C, D=D)


def fit_bla_state_space(
    u: Tensor,
    y: Tensor,
    order: int,
    fir_horizon: int,
    hankel_rows: Optional[int] = None,
    hankel_cols: Optional[int] = None,
    ridge: float = 0.0,
) -> LinearStateSpace:
    """Estimate a zero-offset BLA state-space model from input-output data.

    This first fits the least-squares FIR BLA Markov parameters, then realizes
    them as an order-n_L state-space model for ResDyNet initialization.
    """

    markov = estimate_markov_parameters_bla(u, y, fir_horizon=fir_horizon, ridge=ridge)
    return realize_markov_parameters_ho_kalman(
        markov,
        order=order,
        hankel_rows=hankel_rows,
        hankel_cols=hankel_cols,
    )


def fit_bla_state_space_from_trajectories(
    trajectories: Iterable[tuple[Tensor, Tensor]],
    order: int,
    fir_horizon: int,
    hankel_rows: Optional[int] = None,
    hankel_cols: Optional[int] = None,
    ridge: float = 0.0,
) -> LinearStateSpace:
    """Estimate BLA state-space model from multiple independent trajectories."""

    markov = estimate_markov_parameters_bla_from_trajectories(
        trajectories,
        fir_horizon=fir_horizon,
        ridge=ridge,
    )
    return realize_markov_parameters_ho_kalman(
        markov,
        order=order,
        hankel_rows=hankel_rows,
        hankel_cols=hankel_cols,
    )


if __name__ == "__main__":
    print("ResDyNet PyTorch components are ready to import.")
