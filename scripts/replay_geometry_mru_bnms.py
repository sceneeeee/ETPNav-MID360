#!/usr/bin/env python3
"""CPU-only MRU replay over saved Geometry V0-BNMS pre-NMS score maps."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vlnce_baselines.waypoint_provider.mid360_geometry import (
    GEOMETRY_RADIAL_UTILITY_MRU,
    apply_radial_utility,
    select_geometry_candidates,
)

DISTANCES = np.arange(1, 13, dtype=np.float64) * 0.25


def load(path):
    with open(str(path)) as handle:
        return json.load(handle)


def dump(path, value):
    with open(str(path), "w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def percentile(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        raise SystemExit("replay has missing or non-finite candidate distances")
    return {
        "count": int(len(values)), "mean_m": float(values.mean()),
        "median_m": float(np.median(values)), "p95_m": float(np.percentile(values, 95)),
        "percentage_gte_2_5m": float(np.mean(values >= 2.5) * 100.0),
        "percentage_eq_3_0m": float(np.mean(values == 3.0) * 100.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    telemetry = load(args.telemetry)
    candidates, counts, map_ties, selected_ties = [], [], [], []
    histogram = {"{:.2f}".format(value): 0 for value in DISTANCES}
    total_maps = all_zero_maps = 0

    for episode in telemetry["episodes"].values():
        for record in episode["waypoint_trace"]:
            if "geometry_pre_nms_score" not in record:
                continue
            raw = np.asarray(record["geometry_pre_nms_score"], dtype=np.float32)
            if raw.shape != (120, 12):
                raise SystemExit("saved pre-NMS map is not 120x12")
            score = apply_radial_utility(raw, GEOMETRY_RADIAL_UTILITY_MRU)
            if not np.all(np.isfinite(score)):
                raise SystemExit("MRU replay produced non-finite score")
            total_maps += 1
            if not np.any(score > 0.0):
                all_zero_maps += 1
            angle_indexes, radial_indexes = select_geometry_candidates(score, 5)
            chosen_scores = score[angle_indexes, radial_indexes]
            distances = DISTANCES[radial_indexes]
            counts.append(int(len(distances)))
            candidates.extend(distances.tolist())
            for distance in distances:
                histogram["{:.2f}".format(distance)] += 1
            max_score = float(score.max())
            map_ties.append(int(np.sum(score == max_score)) if max_score > 0.0 else 0)
            selected_ties.append(int(len(chosen_scores) - len(np.unique(chosen_scores))))

    if not total_maps:
        raise SystemExit("no saved Geometry pre-NMS evidence was available")
    report = {
        "kind": "CPU-only sanity replay; no evaluation, fitting, or parameter selection",
        "source_telemetry": str(args.telemetry),
        "mru_formula": "x * exp(1 - x), x = 2d / 3.0",
        "maps_replayed": total_maps,
        "all_zero_map_count": all_zero_maps,
        "candidate_distance_m": percentile(candidates),
        "candidate_count": {"total": int(sum(counts)), "per_map_mean": float(np.mean(counts)),
                            "per_map_median": float(np.median(counts)), "per_map_min": int(min(counts)),
                            "per_map_max": int(max(counts))},
        "radial_histogram": histogram,
        "tie_frequency": {
            "maps_with_tied_global_positive_maximum": int(sum(value > 1 for value in map_ties)),
            "percentage_maps_with_tied_global_positive_maximum": float(np.mean(np.asarray(map_ties) > 1) * 100.0),
            "mean_cells_at_global_positive_maximum": float(np.mean(map_ties)),
            "candidate_sets_with_exact_score_tie": int(sum(value > 0 for value in selected_ties)),
            "percentage_candidate_sets_with_exact_score_tie": float(np.mean(np.asarray(selected_ties) > 0) * 100.0),
        },
        "sanity_verdict": (
            "PASS" if candidates and all_zero_maps < total_maps
            else "FAIL: all replay maps are zero or candidate output is empty"
        ),
        "all_zero_map_interpretation": (
            "Some saved Geometry V0 maps were already fully blocked; MRU preserves "
            "their exact zero semantics.  This is not a global all-zero regression."
            if all_zero_maps else "No saved map was all zero."
        ),
    }
    if not candidates:
        raise SystemExit("MRU replay produced empty candidate output")
    dump(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
