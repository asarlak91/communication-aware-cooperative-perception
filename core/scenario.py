from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.stats import truncnorm


@dataclass
class Scenario:
    """Vehicle states over a finite cooperative-perception horizon."""

    positions: np.ndarray  # shape: (N, H)
    velocities: np.ndarray  # shape: (N,)
    ego_positions: np.ndarray  # shape: (H,)
    ego_velocity: float
    times: np.ndarray  # shape: (H,)

    @property
    def N(self) -> int:
        return int(self.positions.shape[0])

    @property
    def H(self) -> int:
        return int(self.positions.shape[1])

    @property
    def distances(self) -> np.ndarray:
        return np.abs(self.positions - self.ego_positions[None, :])


def generate_scenario(
    N: int = 10,
    road_interval: Tuple[float, float] = (20.0, 300.0),
    v_mu: float = 18.0,
    v_sigma: float = 5.0,
    v_min: float = 5.0,
    v_max: float = 30.0,
    ego_position: float = 0.0,
    ego_velocity: float = 20.0,
    horizon_s: float = 10.0,
    dt: float = 1.0,
    seed: int = 42,
) -> Scenario:
    """Generate a paper-style 1-D highway scenario.

    Conditioned on N points inside a bounded road segment, a homogeneous
    Poisson point process has uniformly distributed point locations. We use
    that conditional construction so N can be fixed exactly, matching the
    simulation setting used for most paper experiments.
    """

    if N < 1:
        raise ValueError("N must be positive")
    if road_interval[1] <= road_interval[0]:
        raise ValueError("road_interval must have positive length")
    if dt <= 0 or horizon_s < 0:
        raise ValueError("dt must be positive and horizon_s nonnegative")

    rng = np.random.default_rng(seed)
    initial_positions = np.sort(rng.uniform(road_interval[0], road_interval[1], N))

    a = (v_min - v_mu) / v_sigma
    b = (v_max - v_mu) / v_sigma
    velocities = truncnorm.rvs(
        a,
        b,
        loc=v_mu,
        scale=v_sigma,
        size=N,
        random_state=rng,
    )

    times = np.arange(0.0, horizon_s + 1e-12, dt)
    positions = initial_positions[:, None] + velocities[:, None] * times[None, :]
    ego_positions = ego_position + ego_velocity * times

    return Scenario(
        positions=positions,
        velocities=np.asarray(velocities, dtype=float),
        ego_positions=np.asarray(ego_positions, dtype=float),
        ego_velocity=float(ego_velocity),
        times=np.asarray(times, dtype=float),
    )
