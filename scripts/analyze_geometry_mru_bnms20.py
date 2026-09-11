#!/usr/bin/env python3
"""Analyze the formal frozen-20 Geometry MRU run without altering artifacts."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

EPISODE_IDS = [
    "53", "1685", "821", "1024", "1477", "1636", "675", "164", "1375", "1679",
    "250", "2", "475", "1356", "252", "1439", "1016", "1285", "1478", "25",
]


def load(path):
    with open(str(path)) as handle:
        return json.load(handle)


def dump(path, value):
    with open(str(path), "w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def only(root, pattern):
    paths = list(Path(root).glob(pattern))
    if len(paths) != 1:
        raise SystemExit("expected one {} under {}, got {}".format(pattern, root, paths))
    return paths[0]


def read_run(root, label):
    root = Path(root)
    result = {
        "stats": load(only(root, "stats_ep_ckpt_*_val_unseen_r0_w1.json")),
        "aggregate": load(only(root, "stats_ckpt_*_val_unseen.json")),
        "telemetry": load(only(root, "telemetry_ep_ckpt_*_val_unseen_r0_w1.json")),
        "sensor": load(only(root, "sensor_validity_ep_ckpt_*_val_unseen_r0_w1.json")),
    }
    if list(result["stats"]) != EPISODE_IDS or list(result["telemetry"]["episodes"]) != EPISODE_IDS:
        raise SystemExit("{} has an unexpected frozen episode order".format(label))
    if set(result["sensor"]) != set(EPISODE_IDS) or not all(
        item["rgb_valid"] and item["depth_valid"] for item in result["sensor"].values()
    ):
        raise SystemExit("{} has invalid or incomplete sensor guard evidence".format(label))
    return result


def numeric_summary(values, thresholds=False):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        raise SystemExit("missing or non-finite telemetry values")
    result = {"count": int(len(values)), "mean": float(values.mean()),
              "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
              "minimum": float(values.min()), "maximum": float(values.max())}
    if thresholds:
        result.update({"percentage_lte_1_0m": float(np.mean(values <= 1.0) * 100.0),
                       "percentage_gte_2_5m": float(np.mean(values >= 2.5) * 100.0),
                       "percentage_eq_3_0m": float(np.mean(values == 3.0) * 100.0)})
    return result


def waypoint_distribution(run):
    candidates, selected, counts = [], [], []
    histogram = {"{:.2f}".format(index * 0.25): 0 for index in range(1, 13)}
    for episode in run["telemetry"]["episodes"].values():
        for decision in episode["waypoint_trace"]:
            entries = decision["candidates"]
            counts.append(len(entries))
            for item in entries:
                distance = item["distance"]
                candidates.append(distance)
                histogram["{:.2f}".format(distance)] += 1
            if decision.get("selected_waypoint") is not None:
                selected.append(decision["selected_waypoint"]["distance"])
    return {"candidate_count": {"total": int(sum(counts)), **numeric_summary(counts)},
            "candidate_distance_m": numeric_summary(candidates, thresholds=True),
            "selected_waypoint_distance_m": numeric_summary(selected, thresholds=True),
            "radial_histogram": histogram}


def metrics(run):
    aggregate, telemetry = run["aggregate"], run["telemetry"]["summary"]
    latency, memory = telemetry["inference_latency_ms"], telemetry["cuda_memory"]
    collisions = [decision.get("collision") for episode in run["telemetry"]["episodes"].values()
                  for decision in episode["waypoint_trace"] if "collision" in decision]
    return {
        "SR": aggregate["success"], "SPL": aggregate["spl"],
        "NE": aggregate["distance_to_goal"], "OS": aggregate["oracle_success"],
        "nDTW": aggregate["ndtw"], "SDTW": aggregate["sdtw"],
        "mean_steps": aggregate["steps_taken"], "mean_path_length_m": aggregate["path_length"],
        "collision_statistics": {"aggregate_collision_metric": aggregate["collisions"],
            "collision_events": int(sum(bool(item) for item in collisions)),
            "actions_with_collision_outcome": len(collisions)},
        "ghost_count": aggregate["ghost_cnt"],
        "latency_ms": {"mean": latency["mean"], "p50": latency["p50"],
                       "p95": latency["p95"], "maximum": latency["maximum"]},
        "peak_allocated_vram_mib": memory["peak_allocated_mib"],
    }


def delta(left, right):
    result = {}
    for key in left:
        if key not in right:
            continue
        if isinstance(left[key], (int, float)) and isinstance(right[key], (int, float)):
            result[key] = left[key] - right[key]
        elif isinstance(left[key], dict) and isinstance(right[key], dict):
            nested = delta(left[key], right[key])
            if nested:
                result[key] = nested
    return result


def candidates(decision):
    return sorted((round(item["angle"], 8), round(item["distance"], 8)) for item in decision["candidates"])


def selected(decision):
    value = decision.get("selected_waypoint")
    waypoint = None if value is None else tuple(sorted((key, round(item, 8) if isinstance(item, float) else item) for key, item in value.items()))
    return decision.get("selected_graph_node_or_ghost_id"), waypoint


def trace_position(decision):
    return tuple(round(value, 8) for value in decision["agent_position"]), round(decision["agent_heading"], 8)


def first_difference(old, new, projector):
    for index, (old_item, new_item) in enumerate(zip(old, new)):
        old_value, new_value = projector(old_item), projector(new_item)
        if old_value != new_value:
            return {"step_index": index, "geometry_bnms": old_value, "mru": new_value}
    if len(old) != len(new):
        index = min(len(old), len(new))
        return {"step_index": index,
                "geometry_bnms": "trace ended" if len(old) <= len(new) else "continues",
                "mru": "trace ended" if len(new) <= len(old) else "continues"}
    return None


def episode_divergence(old_episode, new_episode):
    old, new = old_episode["waypoint_trace"], new_episode["waypoint_trace"]
    return {
        "first_candidate_set_divergence": first_difference(old, new, candidates),
        "first_selected_node_or_waypoint_divergence": first_difference(old, new, selected),
        "trajectory_divergence_point": first_difference(old, new, trace_position),
        "STOP_difference": first_difference(old, new, lambda item: item["stop"]),
    }


def comparison(bnms, mru):
    categories = {"BNMS_fail_to_MRU_success": [], "BNMS_success_to_MRU_fail": [],
                  "both_success": [], "both_fail": []}
    changed, per_episode = [], []
    for episode_id in EPISODE_IDS:
        old_success = bool(bnms["stats"][episode_id]["success"])
        new_success = bool(mru["stats"][episode_id]["success"])
        category = ("BNMS_fail_to_MRU_success" if not old_success and new_success else
                    "BNMS_success_to_MRU_fail" if old_success and not new_success else
                    "both_success" if new_success else "both_fail")
        categories[category].append(episode_id)
        record = {"episode_id": episode_id, "geometry_bnms_success": old_success,
                  "mru_success": new_success, **episode_divergence(
                      bnms["telemetry"]["episodes"][episode_id],
                      mru["telemetry"]["episodes"][episode_id])}
        per_episode.append(record)
        if old_success != new_success:
            changed.append(record)
    return {"outcome_categories": categories, "changed_outcomes": changed,
            "per_episode_divergence": per_episode}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--mru-run", required=True)
    parser.add_argument("--bnms-run", required=True)
    parser.add_argument("--original-run", required=True)
    args = parser.parse_args()
    mru = read_run(args.mru_run, "Geometry MRU")
    bnms = read_run(args.bnms_run, "Geometry V0-BNMS")
    original = read_run(args.original_run, "Original ETP")
    mru_metrics, bnms_metrics, original_metrics = metrics(mru), metrics(bnms), metrics(original)
    root = Path(args.output)
    dump(root / "waypoint_distribution.json", {
        "geometry_mru_bnms": waypoint_distribution(mru),
        "geometry_v0_bnms": waypoint_distribution(bnms),
        "original_etp": waypoint_distribution(original),
    })
    dump(root / "comparison_vs_geometry_bnms.json", {
        "geometry_mru_bnms": mru_metrics, "geometry_v0_bnms": bnms_metrics,
        "mru_minus_geometry_bnms": delta(mru_metrics, bnms_metrics),
        "episode_comparison": comparison(bnms, mru),
    })
    dump(root / "comparison_vs_original.json", {
        "geometry_mru_bnms": mru_metrics, "original_etp": original_metrics,
        "mru_minus_original": delta(mru_metrics, original_metrics),
        "waypoint_distance_distribution_comparison": {
            "geometry_mru_bnms": waypoint_distribution(mru),
            "original_etp": waypoint_distribution(original),
        },
    })


if __name__ == "__main__":
    main()
