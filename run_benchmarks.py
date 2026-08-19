from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.helper_selection import (
    dinkelbach_select,
    greedy_select,
    proximity_select,
    random_select,
    velocity_select,
)

from core.perception_metrics import (
    build_perception_features,
    direct_objective,
    selection_vector,
    normalize_features_for_selection,
)

from core.scenario import generate_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible helper-selection benchmarks")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--M-min", type=int, default=2)
    parser.add_argument("--M-max", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results/generated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    methods = ["Proposed", "Proximity", "Random", "Velocity", "Greedy"]

    for seed in range(args.seeds):
        scenario = generate_scenario(N=args.N, seed=seed)
        # features = build_perception_features(
        #     scenario.positions,
        #     scenario.ego_positions,
        #     scenario.velocities,
        #     sensor_max=np.inf,
        #     visibility=1.0,
        #     theta_rad=0.0,
        # )

        raw_features = build_perception_features(
        scenario.positions,
        scenario.ego_positions,
        scenario.velocities,
        sensor_max=np.inf,
        visibility=1.0,
        theta_rad=0.0,
        )
        features = normalize_features_for_selection(raw_features)

        for M in range(args.M_min, min(args.M_max, args.N) + 1):
            proposed, _ = dinkelbach_select(features, M=M, subproblem="exact")
            selections = {
                "Proposed": proposed,
                "Proximity": proximity_select(features, M),
                "Random": random_select(args.N, M, seed=10_000 + seed * 10 + M),
                "Velocity": velocity_select(scenario.velocities, M),
                "Greedy": greedy_select(features, M),
            }
            for method in methods:
                idx = selections[method]
                s = selection_vector(idx, args.N)
                rows.append(
                    {
                        "seed": seed,
                        "M": M,
                        "method": method,
                        "objective": direct_objective(s, features),
                        "mean_distance": float(
                            raw_features.location_cost[idx].mean() / scenario.H
                        ),
                        "mean_visual_range": float(
                            raw_features.visual_range[idx].mean() / scenario.H
                        ),
                        "mean_motion_blur": float(
                            raw_features.motion_blur_cost[idx].mean() / scenario.H
                        ),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out / "helper_selection_benchmark.csv", index=False)

    summary = (
        df.groupby(["M", "method"], as_index=False)
        .agg(objective_mean=("objective", "mean"), objective_std=("objective", "std"))
    )
    summary.to_csv(out / "helper_selection_summary.csv", index=False)

    # This generated plot is intentionally separate from the published-paper figure.
    plt.figure(figsize=(8, 5))
    for method in methods:
        sub = summary[summary["method"] == method]
        plt.errorbar(
            sub["M"],
            sub["objective_mean"],
            yerr=sub["objective_std"],
            marker="o",
            capsize=3,
            label=method,
        )
    plt.xlabel("Number of selected helpers")
    plt.ylabel("Composite selection objective")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "generated_objective_comparison.png", dpi=180)
    plt.close()

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
