from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np

from core.perception_metrics import (
    PerceptionFeatures,
    dinkelbach_quadratic_matrix,
    direct_objective,
    fractional_components,
    selection_vector,
)


def _validate_M(N: int, M: int) -> None:
    if M < 1 or M > N:
        raise ValueError(f"M must be in [1, N]; got M={M}, N={N}")


def exact_select(features: PerceptionFeatures, M: int) -> np.ndarray:
    """Exact combinatorial minimizer of f1+f2+f3 for small N."""

    N = len(features.location_cost)
    _validate_M(N, M)
    best_indices = None
    best_cost = np.inf

    for combo in combinations(range(N), M):
        s = selection_vector(list(combo), N)
        cost = direct_objective(s, features)
        if cost < best_cost:
            best_cost = cost
            best_indices = np.asarray(combo, dtype=int)

    assert best_indices is not None
    return best_indices


def _exact_dinkelbach_subproblem(features: PerceptionFeatures, M: int, eta: float) -> np.ndarray:
    """Exact oracle for min_s G(s)-eta D(s), sum s=M, s binary."""

    N = len(features.location_cost)
    best_indices = None
    best_value = np.inf
    for combo in combinations(range(N), M):
        s = selection_vector(list(combo), N)
        G, D = fractional_components(s, features)
        value = G - eta * D
        if value < best_value:
            best_value = value
            best_indices = np.asarray(combo, dtype=int)
    assert best_indices is not None
    return best_indices


def _sdp_dinkelbach_subproblem(features: PerceptionFeatures, M: int, eta: float) -> np.ndarray:
    """SDP relaxation of the binary QCQP Dinkelbach subproblem.

    The paper motivates a Lagrangian/SDP relaxation for the nonconvex QCQP.
    This implementation uses an equivalent standard lifted SDP relaxation:
      X approximately equals s s^T,
      diag(X)=s,
      [1 s^T; s X] is positive semidefinite.

    The relaxed solution is rounded to M helpers, then improved by one-swap
    local search using the original transformed objective.
    """

    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError("cvxpy is required for method='sdp'") from exc

    N = len(features.location_cost)
    Q = dinkelbach_quadratic_matrix(features)
    r = features.visual_range

    s_var = cp.Variable(N)
    X = cp.Variable((N, N), symmetric=True)
    block = cp.bmat(
        [
            [np.ones((1, 1)), cp.reshape(s_var, (1, N), order="C")],
            [cp.reshape(s_var, (N, 1), order="C"), X],
        ]
    )

    objective = cp.Minimize(cp.trace(Q @ X) + 1.0 - eta * (r @ s_var))
    constraints = [
        cp.diag(X) == s_var,
        s_var >= 0.0,
        s_var <= 1.0,
        cp.sum(s_var) == M,
        block >> 0,
    ]

    problem = cp.Problem(objective, constraints)
    installed = set(cp.installed_solvers())
    candidates = [name for name in ("CLARABEL", "SCS") if name in installed]
    if not candidates:
        raise RuntimeError("No SDP-capable cvxpy solver found; install SCS or CLARABEL")

    last_error = None
    for solver in candidates:
        try:
            kwargs = {"verbose": False}
            if solver == "SCS":
                kwargs.update({"eps": 1e-5, "max_iters": 20000})
            problem.solve(solver=solver, **kwargs)
            if s_var.value is not None:
                break
        except Exception as exc:  # pragma: no cover - solver dependent
            last_error = exc

    if s_var.value is None:
        raise RuntimeError(f"SDP solve failed: {last_error}")

    scores = np.asarray(s_var.value).reshape(-1)
    selected = np.argsort(scores)[-M:]

    def transformed_cost(indices: Iterable[int]) -> float:
        s = selection_vector(list(indices), N)
        G, D = fractional_components(s, features)
        return float(G - eta * D)

    # One-swap local improvement after rounding.
    selected_set = set(int(i) for i in selected)
    improved = True
    while improved:
        improved = False
        current = transformed_cost(selected_set)
        outside = [i for i in range(N) if i not in selected_set]
        for out_idx in list(selected_set):
            for in_idx in outside:
                candidate = set(selected_set)
                candidate.remove(out_idx)
                candidate.add(in_idx)
                value = transformed_cost(candidate)
                if value + 1e-12 < current:
                    selected_set = candidate
                    improved = True
                    current = value
                    break
            if improved:
                break

    return np.asarray(sorted(selected_set), dtype=int)


def dinkelbach_select(
    features: PerceptionFeatures,
    M: int,
    subproblem: str = "exact",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> tuple[np.ndarray, dict]:
    """Dinkelbach helper selection with exact or SDP-relaxed QCQP oracle."""

    N = len(features.location_cost)
    _validate_M(N, M)

    # Start from the best proximity-based feasible set.
    initial = np.argsort(features.location_cost)[:M]
    s = selection_vector(initial, N)
    G, D = fractional_components(s, features)
    if D <= 0:
        raise ValueError("Visual-range denominator must be positive")
    eta = G / D

    history = []
    selected = initial
    for k in range(max_iter):
        if subproblem == "exact":
            selected = _exact_dinkelbach_subproblem(features, M, eta)
        elif subproblem == "sdp":
            selected = _sdp_dinkelbach_subproblem(features, M, eta)
        else:
            raise ValueError("subproblem must be 'exact' or 'sdp'")

        s = selection_vector(selected, N)
        G, D = fractional_components(s, features)
        residual = G - eta * D
        objective = G / D
        history.append(
            {
                "iteration": k,
                "eta": float(eta),
                "G": float(G),
                "D": float(D),
                "residual": float(residual),
                "objective": float(objective),
                "selected": selected.tolist(),
            }
        )

        if abs(residual) <= tol:
            break
        eta = G / D

    info = {
        "method": f"dinkelbach_{subproblem}",
        "iterations": len(history),
        "history": history,
        "objective": float(direct_objective(selection_vector(selected, N), features)),
    }
    return np.asarray(selected, dtype=int), info


def proximity_select(features: PerceptionFeatures, M: int) -> np.ndarray:
    return np.argsort(features.location_cost)[:M].astype(int)


def velocity_select(velocities: np.ndarray, M: int) -> np.ndarray:
    return np.argsort(np.asarray(velocities, dtype=float))[:M].astype(int)


def random_select(N: int, M: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(N, size=M, replace=False)).astype(int)


def greedy_select(features: PerceptionFeatures, M: int) -> np.ndarray:
    N = len(features.location_cost)
    selected: list[int] = []
    remaining = set(range(N))
    while len(selected) < M:
        best_idx = None
        best_cost = np.inf
        for i in remaining:
            candidate = selected + [i]
            s = selection_vector(candidate, N)
            cost = direct_objective(s, features)
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        assert best_idx is not None
        selected.append(best_idx)
        remaining.remove(best_idx)
    return np.asarray(sorted(selected), dtype=int)
