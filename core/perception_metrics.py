from dataclasses import dataclass

import numpy as np


@dataclass
class PerceptionFeatures:
    """Per-candidate quantities used by the helper-selection objective."""

    location_cost: np.ndarray  # a_i = sum_t distance_i(t)
    visual_range: np.ndarray  # r_i = sum_t R_i(t)
    motion_blur_cost: np.ndarray  # c_i = sum_t blur_i(t)
    visual_range_matrix: np.ndarray  # shape: (N, H)
    motion_blur_matrix: np.ndarray  # shape: (N, H)


def effective_visual_range(
    distances: np.ndarray,
    visibility: np.ndarray | float = 1.0,
    theta_rad: np.ndarray | float = 0.0,
    sensor_max: float = np.inf,
) -> np.ndarray:
    """Effective visual range R_it.

    R_it = min(R_sensor, distance_it * visibility_it * cos(theta_it)).
    Negative off-axis contributions are clipped to zero.
    """

    distances = np.asarray(distances, dtype=float)
    visibility = np.asarray(visibility, dtype=float)
    theta_rad = np.asarray(theta_rad, dtype=float)
    raw = distances * visibility * np.cos(theta_rad)
    raw = np.maximum(raw, 0.0)
    return np.minimum(float(sensor_max), raw)


def motion_blur_straight(
    velocities: np.ndarray,
    horizon_len: int,
    exposure_s: float = 1e-3,
    focal_length_m: float = 5e-3,
    object_depth_m: float = 50.0,
    pixel_size_m: float = 5e-6,
) -> np.ndarray:
    """Straight-motion form of the paper's motion-blur metric.

    blur_it = v_i * exposure * focal_length / (depth * pixel_size).
    """

    velocities = np.asarray(velocities, dtype=float)
    if object_depth_m <= 0 or pixel_size_m <= 0:
        raise ValueError("object_depth_m and pixel_size_m must be positive")
    per_vehicle = (
        velocities
        * float(exposure_s)
        * float(focal_length_m)
        / (float(object_depth_m) * float(pixel_size_m))
    )
    return np.repeat(per_vehicle[:, None], horizon_len, axis=1)


def build_perception_features(
    positions: np.ndarray,
    ego_positions: np.ndarray,
    velocities: np.ndarray,
    visibility: np.ndarray | float = 1.0,
    theta_rad: np.ndarray | float = 0.0,
    sensor_max: float = np.inf,
    exposure_s: float = 1e-3,
    focal_length_m: float = 5e-3,
    object_depth_m: float = 50.0,
    pixel_size_m: float = 5e-6,
) -> PerceptionFeatures:
    positions = np.asarray(positions, dtype=float)
    ego_positions = np.asarray(ego_positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)

    distances = np.abs(positions - ego_positions[None, :])
    ranges = effective_visual_range(distances, visibility, theta_rad, sensor_max)
    blur = motion_blur_straight(
        velocities,
        horizon_len=positions.shape[1],
        exposure_s=exposure_s,
        focal_length_m=focal_length_m,
        object_depth_m=object_depth_m,
        pixel_size_m=pixel_size_m,
    )

    return PerceptionFeatures(
        location_cost=distances.sum(axis=1),
        visual_range=ranges.sum(axis=1),
        motion_blur_cost=blur.sum(axis=1),
        visual_range_matrix=ranges,
        motion_blur_matrix=blur,
    )


def selection_vector(indices: np.ndarray | list[int], N: int) -> np.ndarray:
    s = np.zeros(N, dtype=float)
    s[np.asarray(indices, dtype=int)] = 1.0
    return s


def f1_location(s: np.ndarray, features: PerceptionFeatures) -> float:
    return float(np.dot(np.asarray(s, dtype=float), features.location_cost))


def f2_visual_range(s: np.ndarray, features: PerceptionFeatures, eps: float = 1e-12) -> float:
    total = float(np.dot(np.asarray(s, dtype=float), features.visual_range))
    return 1.0 / (total + eps)


def f3_motion_blur(s: np.ndarray, features: PerceptionFeatures) -> float:
    return float(np.dot(np.asarray(s, dtype=float), features.motion_blur_cost))


def direct_objective(s: np.ndarray, features: PerceptionFeatures) -> float:
    """Paper-style composite objective f1 + f2 + f3."""

    return f1_location(s, features) + f2_visual_range(s, features) + f3_motion_blur(s, features)


def fractional_components(s: np.ndarray, features: PerceptionFeatures) -> tuple[float, float]:
    """Exact fractional representation of f1 + f2 + f3.

    Let A(s)=f1(s), C(s)=f3(s), B(s)=sum_i s_i r_i. Then

        A + 1/B + C = ((A+C)B + 1) / B.

    This gives a mathematically consistent G(s)/D(s) form for Dinkelbach.
    """

    s = np.asarray(s, dtype=float)
    a_plus_c = features.location_cost + features.motion_blur_cost
    B = float(np.dot(s, features.visual_range))
    A_plus_C = float(np.dot(s, a_plus_c))
    G = A_plus_C * B + 1.0
    D = B
    return G, D


def dinkelbach_quadratic_matrix(features: PerceptionFeatures) -> np.ndarray:
    """Symmetric Q satisfying s^T Q s = (a+c)^T s * r^T s."""

    u = features.location_cost + features.motion_blur_cost
    r = features.visual_range
    return 0.5 * (np.outer(u, r) + np.outer(r, u))


def normalize_features_for_selection(
    features: PerceptionFeatures,
    eps: float = 1e-12,
) -> PerceptionFeatures:
    """
    Normalize the three helper-selection criteria to comparable scales.

    The published paper reports the component metrics in physical units,
    but its composite objective is on a normalized scale. Because the
    original MATLAB normalization is unavailable, this reimplementation
    uses per-scenario max normalization.

    Raw features should still be retained for reporting physical metrics.
    """

    d_scale = max(
        float(np.max(features.location_cost)),
        eps,
    )

    r_scale = max(
        float(np.max(features.visual_range)),
        eps,
    )

    b_scale = max(
        float(np.max(features.motion_blur_cost)),
        eps,
    )

    return PerceptionFeatures(
        location_cost=features.location_cost / d_scale,
        visual_range=features.visual_range / r_scale,
        motion_blur_cost=features.motion_blur_cost / b_scale,
        visual_range_matrix=features.visual_range_matrix / r_scale,
        motion_blur_matrix=features.motion_blur_matrix / b_scale,
    )
