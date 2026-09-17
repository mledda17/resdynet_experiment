from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class CascadedTanksConfig:
    k1: float = 0.5
    k2: float = 0.4
    k3: float = 0.2
    k4: float = 0.3
    sampling_time: float = 1.0
    rk4_substeps: int = 20
    process_noise_std: float = 0.0
    measurement_noise_std: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class Trajectory:
    t: np.ndarray
    u: np.ndarray
    y: np.ndarray
    x: np.ndarray


class CascadedTanks:
    """Continuous-time Cascaded Tanks simulator.

    The physical state is intentionally kept inside this class. Identification
    and MPC code should consume only the sampled input/output trajectories.
    """

    def __init__(self, config: CascadedTanksConfig | None = None) -> None:
        self.config = config or CascadedTanksConfig()
        if self.config.sampling_time <= 0:
            raise ValueError("sampling_time must be positive.")
        if self.config.rk4_substeps < 1:
            raise ValueError("rk4_substeps must be at least 1.")
        self.rng = np.random.default_rng(self.config.seed)
        self.x = np.zeros(2, dtype=float)
        self.state_projection_count = 0

    def reset(self, x0: np.ndarray | list[float] | tuple[float, float]) -> None:
        x = np.asarray(x0, dtype=float).reshape(2)
        if np.any(x < 0.0):
            raise ValueError(f"Initial tank levels must be nonnegative, got {x}.")
        self.x = x.copy()
        self.state_projection_count = 0

    def dynamics(self, x: np.ndarray, u: float, w: np.ndarray | None = None) -> np.ndarray:
        if np.any(x < 0.0):
            if np.any(x < -1e-7):
                self.state_projection_count += 1
        w_vec = np.zeros(2, dtype=float) if w is None else np.asarray(w, dtype=float).reshape(2)
        x_safe = np.maximum(x, 0.0)
        cfg = self.config
        return np.array(
            [
                -cfg.k1 * np.sqrt(x_safe[0]) + cfg.k4 * u + w_vec[0],
                cfg.k2 * np.sqrt(x_safe[0]) - cfg.k3 * np.sqrt(x_safe[1]) + w_vec[1],
            ],
            dtype=float,
        )

    def output(self, noisy: bool = True) -> float:
        e = self.rng.normal(0.0, self.config.measurement_noise_std) if noisy else 0.0
        return float(self.x[1] + e)

    def step(self, u: float) -> float:
        y = self.output(noisy=True)
        cfg = self.config
        w = self.rng.normal(0.0, cfg.process_noise_std, size=2)
        dt = cfg.sampling_time / cfg.rk4_substeps
        for _ in range(cfg.rk4_substeps):
            x0 = self.x
            k1 = self.dynamics(x0, u, w)
            k2 = self.dynamics(x0 + 0.5 * dt * k1, u, w)
            k3 = self.dynamics(x0 + 0.5 * dt * k2, u, w)
            k4 = self.dynamics(x0 + dt * k3, u, w)
            self.x = x0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if np.any(self.x < 0.0):
                if np.any(self.x < -1e-7):
                    self.state_projection_count += 1
                self.x = np.maximum(self.x, 0.0)
        return y

    def simulate(self, u: np.ndarray, x0: np.ndarray | list[float] | tuple[float, float]) -> Trajectory:
        u_arr = np.asarray(u, dtype=float).reshape(-1)
        self.reset(x0)
        x = np.zeros((u_arr.size, 2), dtype=float)
        y = np.zeros((u_arr.size, 1), dtype=float)
        for k, uk in enumerate(u_arr):
            x[k] = self.x
            y[k, 0] = self.step(float(uk))
        t = np.arange(u_arr.size, dtype=float).reshape(-1, 1) * self.config.sampling_time
        return Trajectory(t=t, u=u_arr.reshape(-1, 1), y=y, x=x)


def steady_state_for_input(u: float, config: CascadedTanksConfig | None = None) -> np.ndarray:
    cfg = config or CascadedTanksConfig()
    x1 = (cfg.k4 * u / cfg.k1) ** 2
    x2 = ((cfg.k2 / cfg.k3) * np.sqrt(x1)) ** 2
    return np.array([x1, x2], dtype=float)


def generate_piecewise_constant_inputs(
    n_samples: int,
    hold_length: int,
    mean: float,
    std: float,
    u_min: float,
    u_max: float,
    seed: int,
) -> np.ndarray:
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    if hold_length < 1:
        raise ValueError("hold_length must be positive.")
    if u_min < 0:
        raise ValueError("u_min must be nonnegative for this physical plant.")
    rng = np.random.default_rng(seed)
    values = rng.normal(mean, std, size=(n_samples + hold_length - 1) // hold_length)
    values = np.clip(values, u_min, u_max)
    return np.repeat(values, hold_length)[:n_samples].reshape(-1, 1)


def shifted_sine_sweep(
    n_samples: int,
    sampling_time: float,
    offset: float,
    amplitude: float,
    f_start: float,
    f_end: float,
) -> np.ndarray:
    t = np.arange(n_samples, dtype=float) * sampling_time
    duration = max(t[-1], sampling_time) if n_samples > 1 else sampling_time
    chirp_rate = (f_end - f_start) / duration
    phase = 2.0 * np.pi * (f_start * t + 0.5 * chirp_rate * t**2)
    return (offset + amplitude * np.sin(phase)).reshape(-1, 1)


def sinusoidal_reference(
    n_samples: int,
    sampling_time: float,
    offset: float,
    amplitude: float,
    frequency: float,
) -> np.ndarray:
    t = np.arange(n_samples, dtype=float) * sampling_time
    return (offset + amplitude * np.sin(2.0 * np.pi * frequency * t)).reshape(-1, 1)


def save_dataset_npz(path: Path, config: CascadedTanksConfig, splits: dict[str, Trajectory], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"config": np.array([asdict(config)], dtype=object), "metadata": np.array([metadata], dtype=object)}
    for name, traj in splits.items():
        payload[f"{name}_t"] = traj.t
        payload[f"{name}_u"] = traj.u
        payload[f"{name}_y"] = traj.y
        payload[f"{name}_x"] = traj.x
    np.savez_compressed(path, **payload)
