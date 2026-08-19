from __future__ import annotations

import numpy as np


def apply_packet_mask(
    image: np.ndarray,
    alpha: float,
    grid: int = 4,
    rng: np.random.Generator | None = None,
    fill_value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop each image block independently with probability 1-alpha."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if grid <= 0:
        raise ValueError("grid must be positive")
    if image is None or image.ndim not in (2, 3):
        raise ValueError("image must be a valid numpy array")
    if rng is None:
        rng = np.random.default_rng()

    corrupted = image.copy()
    h, w = corrupted.shape[:2]
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    dropped = rng.random((grid, grid)) < (1.0 - alpha)

    for row in range(grid):
        for col in range(grid):
            if dropped[row, col]:
                corrupted[ys[row]:ys[row + 1], xs[col]:xs[col + 1], ...] = fill_value

    return corrupted, dropped


def expected_dropped_blocks(alpha: float, grid: int = 4) -> float:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return grid * grid * (1.0 - alpha)
