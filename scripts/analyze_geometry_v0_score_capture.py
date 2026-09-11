#!/usr/bin/env python3
"""Offline analysis of opt-in Geometry V0 pre-NMS score captures.

This program never starts Habitat or navigation.  It reads the saved score
maps from detailed telemetry and applies only the existing NMS implementation
to counterfactual copies.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# A script is executed with ``scripts/`` rather than the repository root on
# sys.path.  Add the root explicitly so this offline analysis imports the
# project's *current* NMS implementation, rather than maintaining a second
# production-NMS copy here.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

FAILURES = ("1685", "1024", "1477", "1636", "1375", "1478")
CONTROLS = ("53", "821", "164", "1679")
DISTANCES = np.arange(1, 13, dtype=np.float64) * 0.25


def load(path):
    with open(path) as handle:
        return json.load(handle)


def json_default(value):
    """Convert scalar NumPy counters while keeping report serialization strict."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("unsupported JSON value: {}".format(type(value).__name__))


def score_map(record):
    score = np.asarray(record["geometry_pre_nms_score"], dtype=np.float32)
    if score.shape != (120, 12):
        raise ValueError("captured Geometry score is not 120x12")
    return score


def radial_summary(score):
    """One radial profile, treating score > 0 as a traversable cell."""
    positive = score > 0
    ray_max = np.max(score, axis=1, keepdims=True)
    # A completely blocked ray (max=0) has no valid maximum cell.
    ray_maximum = positive & (score == ray_max)
    rows = []
    for radial_index, distance in enumerate(DISTANCES):
        rows.append({
            "distance_m": float(distance),
            "mean_score_across_angles": float(np.mean(score[:, radial_index])),
            "maximum_score_across_angles": float(np.max(score[:, radial_index])),
            "positive_valid_angular_cells": int(np.sum(positive[:, radial_index])),
            "fraction_of_angles_where_radial_bin_is_ray_maximum": float(
                np.mean(ray_maximum[:, radial_index])
            ),
        })
    maxima = np.argwhere(score == np.max(score))
    # Stable angle, then radial order makes ties explicit and reproducible.
    flattened = [
        {"angle_bin": int(angle), "distance_m": float(DISTANCES[radial]),
         "score": float(score[angle, radial])}
        for angle, radial in sorted(maxima.tolist())
    ]
    ranked = sorted(
        ({"angle_bin": int(angle), "distance_m": float(DISTANCES[radial]),
          "score": float(score[angle, radial])}
         for angle, radial in np.ndindex(score.shape)),
        key=lambda item: (-item["score"], item["angle_bin"], item["distance_m"]),
    )
    return {
        "radial_bins": rows,
        "global_max_score": float(np.max(score)),
        "global_max_distance_bins_m": sorted({item["distance_m"] for item in flattened}),
        "global_max_cells": flattened,
        "top_20_pre_nms_bins": ranked[:20],
    }


def nms_candidates(score, radial_sigma, angular_sigma):
    """Exact project NMS semantics, parameterized only for offline analysis."""
    # NMS calls columns x and rows y.  Geometry's columns are radial bins and
    # rows are angle bins, so its sigma=(7, 5) means radial=7, angular=5.
    # Duplicate the wrapping/call contract so only NMS changes offline.
    import torch
    from vlnce_baselines.waypoint_pred.utils import nms
    tensor = torch.from_numpy(np.asarray(score, dtype=np.float32))
    wrapped = torch.cat((tensor[-1:, :], tensor, tensor[:1, :]), dim=0)
    selected = nms(wrapped[None, None], max_predictions=5,
                   sigma=(radial_sigma, angular_sigma)).squeeze(0).squeeze(0)[1:-1]
    return [
        {"angle_bin": int(angle), "distance_m": float(DISTANCES[radial]),
         "score": float(score[angle, radial])}
        for angle, radial in selected.nonzero().cpu().numpy().tolist()
    ]


def bounded_radial_nms_candidates(score, radial_sigma=7.0, angular_sigma=5.0):
    """Project NMS with only radial circularity disabled, for offline audit.

    The production caller continues to wrap angle rows before NMS.  This is a
    line-for-line replay of ``nms``' greedy loop, except its neighbourhood is
    created with ``circular_x=False``.  In the Geometry map, x/width is radial
    distance and y/height is the already angle-wrapped dimension.
    """
    import torch
    from vlnce_baselines.waypoint_pred.utils import neighborhoods

    tensor = torch.from_numpy(np.asarray(score, dtype=np.float32))
    wrapped = torch.cat((tensor[-1:, :], tensor, tensor[:1, :]), dim=0)
    pred = wrapped[None, None]
    shape = pred.shape
    output = torch.zeros_like(pred)
    flat_pred = pred.reshape((shape[0], -1))
    supp_pred = pred.clone()
    flat_output = output.reshape((shape[0], -1))
    for _ in range(5):
        flat_supp_pred = supp_pred.reshape((shape[0], -1))
        _, index = torch.max(flat_supp_pred, dim=1)
        batch = torch.arange(0, shape[0])
        flat_output[batch, index] = flat_pred[batch, index]
        y = index / shape[-1]
        x = index % shape[-1]
        center = torch.stack([x, y], dim=1).float()
        mask = neighborhoods(
            center, shape[-1], shape[-2], (radial_sigma, angular_sigma),
            circular_x=False,
        )
        supp_pred *= (1 - mask.unsqueeze(1))
    output[output < 0] = 0
    selected = output.squeeze(0).squeeze(0)[1:-1]
    return [
        {"angle_bin": int(angle), "distance_m": float(DISTANCES[radial]),
         "score": float(score[angle, radial])}
        for angle, radial in selected.nonzero().cpu().numpy().tolist()
    ]


def radial_circularity_unit_example():
    """Minimal CPU example: radial bin 11 suppresses radial bin 0 in prod."""
    score = np.zeros((120, 12), dtype=np.float32)
    score[40, 11] = 2.0  # 3.00 m, deliberately selected first.
    score[40, 0] = 1.0   # 0.25 m, same angle, only one radial step via wrap.
    return {
        "input_positive_cells": [
            {"angle_bin": 40, "distance_bin": 11, "distance_m": 3.0, "score": 2.0},
            {"angle_bin": 40, "distance_bin": 0, "distance_m": 0.25, "score": 1.0},
        ],
        "production_circular_radial_selected": nms_candidates(score, 7.0, 5.0),
        "bounded_radial_selected": bounded_radial_nms_candidates(score),
        "verdict": (
            "production selects only 3.00 m because radial bin 11 suppresses "
            "bin 0 through width-axis circularity; bounded radial NMS selects both"
        ),
    }


def distance_distribution(items):
    distances = [item["distance_m"] for item in items]
    if not distances:
        return {
            "candidate_count": 0, "mean_m": None, "median_m": None,
            "p95_m": None, "fraction_gte_2_5m": None,
            "fraction_eq_3_0m": None,
        }
    values = np.asarray(distances, dtype=np.float64)
    return {
        "candidate_count": int(len(values)),
        "mean_m": float(values.mean()),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95)),
        "fraction_gte_2_5m": float(np.mean(values >= 2.5)),
        "fraction_eq_3_0m": float(np.mean(values == 3.0)),
    }


def candidate_variant_summary(captures, candidate_key):
    candidates = [candidate for item in captures for candidate in item[candidate_key]]
    result = distance_distribution(candidates)
    result["mean_candidate_count_per_map"] = float(
        np.mean([len(item[candidate_key]) for item in captures])
    ) if captures else None
    return result


def aggregate(captures):
    if not captures:
        return {"capture_count": 0}
    maps = np.stack([item["score"] for item in captures])
    positive = maps > 0
    ray_max = maps.max(axis=2, keepdims=True)
    maxima = positive & (maps == ray_max)
    radial = []
    for index, distance in enumerate(DISTANCES):
        radial.append({
            "distance_m": float(distance),
            "mean_score_across_captures_and_angles": float(maps[:, :, index].mean()),
            "maximum_score": float(maps[:, :, index].max()),
            "positive_valid_angular_cells": int(positive[:, :, index].sum()),
            "fraction_of_angles_where_radial_bin_is_ray_maximum": float(maxima[:, :, index].mean()),
        })
    nonzero_maps = maps[np.any(positive, axis=(1, 2))]
    global_histogram = {"{:.2f}".format(distance): 0 for distance in DISTANCES}
    for score in nonzero_maps:
        # Ties count once in every radial bin attaining the global maximum.
        for radial in np.unique(np.argwhere(score == score.max())[:, 1]):
            global_histogram["{:.2f}".format(DISTANCES[radial])] += 1
    return {
        "captured_map_count": len(captures),
        "all_zero_map_count": int(len(captures) - len(nonzero_maps)),
        "global_max_radial_bin_histogram_nonzero_maps": global_histogram,
        "radial_bins": radial,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    telemetry = load(args.telemetry)
    captures, per_decision = [], []
    for episode_id, episode in telemetry["episodes"].items():
        for record in episode["waypoint_trace"]:
            if "geometry_pre_nms_score" not in record:
                continue
            score = score_map(record)
            analysis = radial_summary(score)
            analysis.update({
                "episode_id": episode_id, "step_index": record["step_index"],
                "post_nms_candidates": [
                    {"angle": item["angle"], "distance": item["distance"]}
                    for item in record["candidates"]
                ],
                "selected_waypoint": record.get("selected_waypoint"),
                "selected_candidate_index": record.get("selected_candidate_index"),
                "selected_graph_node_or_ghost_id": record.get("selected_graph_node_or_ghost_id"),
                "ghost_node_count": record.get("ghost_node_count"),
                "nms_counterfactual": {
                    "production_sigma_radial7_angular5": nms_candidates(score, 7.0, 5.0),
                    "reduced_radial_sigma1_angular5": nms_candidates(score, 1.0, 5.0),
                    "angular_only_radial0_angular5": nms_candidates(score, 0.0, 5.0),
                    "bounded_radial_sigma7_angular5": bounded_radial_nms_candidates(score),
                },
            })
            per_decision.append(analysis)
            captures.append({"episode_id": episode_id, "score": score,
                             "post_nms_candidates": analysis["post_nms_candidates"],
                             "selected_waypoint": analysis["selected_waypoint"],
                             "production": analysis["nms_counterfactual"]["production_sigma_radial7_angular5"],
                             "reduced_radial": analysis["nms_counterfactual"]["reduced_radial_sigma1_angular5"],
                             "radial_disabled": analysis["nms_counterfactual"]["angular_only_radial0_angular5"],
                             "bounded_radial": analysis["nms_counterfactual"]["bounded_radial_sigma7_angular5"]})
    by_group = {"original_only_failures": [], "both_success_controls": []}
    for capture in captures:
        if capture["episode_id"] in FAILURES:
            by_group["original_only_failures"].append(capture)
        if capture["episode_id"] in CONTROLS:
            by_group["both_success_controls"].append(capture)
    zero_score_audit = []
    for record in per_decision:
        if record["global_max_score"] == 0:
            zero_score_audit.append({
                "episode_id": record["episode_id"],
                "step_index": record["step_index"],
                "candidate_list_empty": not record["post_nms_candidates"],
                "selected_waypoint": record["selected_waypoint"],
                "selected_candidate_index": record["selected_candidate_index"],
                "selected_graph_node_or_ghost_id": record["selected_graph_node_or_ghost_id"],
                "ghost_node_count": record["ghost_node_count"],
            })
    output = {
        "definition": {
            "positive_valid": "score > 0", "ray_maximum": "positive cell whose score equals its ray maximum; all-zero rays contribute none",
            "score_map": "exact pre-NMS Geometry V0 score, no value transformation",
            "global_max_histogram": "counts each nonzero score map once per tied radial maximum; all-zero maps are excluded",
        },
        "groups": {
            "all_captures": aggregate(captures),
            **{name: aggregate(items) for name, items in by_group.items()},
        },
        "nms_counterfactual_all_captures": {
            "production_sigma_radial7_angular5": candidate_variant_summary(captures, "production"),
            "reduced_radial_sigma1_angular5": candidate_variant_summary(captures, "reduced_radial"),
            "angular_only_radial0_angular5": candidate_variant_summary(captures, "radial_disabled"),
            "bounded_radial_sigma7_angular5": candidate_variant_summary(captures, "bounded_radial"),
        },
        "radial_circularity_cpu_unit_example": radial_circularity_unit_example(),
        "zero_score_decision_audit": zero_score_audit,
        "latency_status": {
            "targeted_diagnostic": {
                "mean_ms": 87.616786,
                "p95_ms": 104.155916,
                "peak_allocated_vram_mib": 1017.471,
            },
            "historical_geometry_v0": {"mean_ms_approx": 85.71, "p95_ms_approx": 98.20},
            "previous_valid_renderer_formal20": {"mean_ms": 142.18, "p95_ms": 229.18},
            "interpretation": (
                "latency is not stable across runs; 142 ms is not yet the canonical "
                "deployment cost. No optimization was performed."
            ),
        },
    }
    Path(args.output).write_text(json.dumps(output, indent=2, default=json_default) + "\n")


if __name__ == "__main__":
    main()
