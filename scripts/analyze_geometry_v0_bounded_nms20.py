#!/usr/bin/env python3
"""Validate and compare the frozen Geometry V0-BNMS evaluation."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


EPISODE_IDS = [
    "53", "1685", "821", "1024", "1477", "1636", "675", "164", "1375", "1679",
    "250", "2", "475", "1356", "252", "1439", "1016", "1285", "1478", "25",
]
ORIGINAL_REFERENCE = {
    "SR": 0.800, "SPL": 0.6983661897804773, "NE": 2.9216403543949125, "OS": 0.800,
}


def load(path):
    with open(str(path)) as handle:
        return json.load(handle)


def dump(path, value):
    with open(str(path), "w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def only(root, pattern):
    matches = list(Path(root).glob(pattern))
    if len(matches) != 1:
        raise SystemExit("expected one {} under {}, got {}".format(pattern, root, matches))
    return matches[0]


def read_run(root, expected_name):
    root = Path(root)
    stats = load(only(root, "stats_ep_ckpt_*_val_unseen_r0_w1.json"))
    aggregate = load(only(root, "stats_ckpt_*_val_unseen.json"))
    telemetry = load(only(root, "telemetry_ep_ckpt_*_val_unseen_r0_w1.json"))
    sensor = load(only(root, "sensor_validity_ep_ckpt_*_val_unseen_r0_w1.json"))
    if list(stats) != EPISODE_IDS or list(telemetry["episodes"]) != EPISODE_IDS:
        raise SystemExit("{} does not retain frozen episode order".format(expected_name))
    if set(sensor) != set(EPISODE_IDS) or not all(
        item["rgb_valid"] and item["depth_valid"] for item in sensor.values()
    ):
        raise SystemExit("{} has incomplete/invalid sensor guard evidence".format(expected_name))
    return {"stats": stats, "aggregate": aggregate, "telemetry": telemetry, "sensor": sensor}


def percentiles(values, thresholds=False):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        raise SystemExit("missing or non-finite waypoint distances")
    result = {"count": int(len(values)), "mean": float(values.mean()),
              "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
    if thresholds:
        result.update({"percentage_gte_2_5m": float(np.mean(values >= 2.5) * 100),
                       "percentage_eq_3_0m": float(np.mean(values == 3.0) * 100)})
    return result


def distribution(run):
    candidates, selected, counts = [], [], []
    for episode in run["telemetry"]["episodes"].values():
        for decision in episode["waypoint_trace"]:
            items = decision["candidates"]
            counts.append(len(items))
            candidates.extend(item["distance"] for item in items)
            if decision.get("selected_waypoint") is not None:
                selected.append(decision["selected_waypoint"]["distance"])
    count_summary = percentiles(counts)
    count_summary["total_candidates"] = int(sum(counts))
    return {"candidate_count": count_summary,
            "candidate_distance_m": percentiles(candidates, thresholds=True),
            "selected_waypoint_distance_m": percentiles(selected)}


def summary(run):
    aggregate, telemetry = run["aggregate"], run["telemetry"]["summary"]
    latency, memory = telemetry["inference_latency_ms"], telemetry["cuda_memory"]
    outcomes = [decision.get("collision") for episode in run["telemetry"]["episodes"].values()
                for decision in episode["waypoint_trace"] if "collision" in decision]
    collision_events = int(sum(bool(value) for value in outcomes))
    return {
        "SR": aggregate["success"], "SPL": aggregate["spl"],
        "NE": aggregate["distance_to_goal"], "OS": aggregate["oracle_success"],
        "mean_steps": aggregate["steps_taken"], "mean_path_length_m": aggregate["path_length"],
        "nDTW": aggregate["ndtw"], "SDTW": aggregate["sdtw"],
        "collision_statistics": {"aggregate_collision_metric": aggregate["collisions"],
            "collision_events": collision_events, "actions_with_collision_outcome": len(outcomes),
            "telemetry_collision_event_rate": (float(collision_events) / len(outcomes) if outcomes else 0.0)},
        "ghost_count": aggregate["ghost_cnt"],
        "mean_latency_ms": latency["mean"], "p50_latency_ms": latency["p50"],
        "p95_latency_ms": latency["p95"], "max_latency_ms": latency["maximum"],
        "peak_allocated_vram_mib": memory["peak_allocated_mib"],
    }


def candidate_signature(decision):
    return sorted((round(x["angle"], 8), round(x["distance"], 8)) for x in decision["candidates"])


def decision_signature(decision):
    selected = decision.get("selected_waypoint")
    selected = None if selected is None else tuple(sorted(
        (key, round(value, 8) if isinstance(value, float) else value) for key, value in selected.items()))
    return decision["stop"], decision.get("selected_graph_node_or_ghost_id"), selected


def first_differences(legacy_episode, bounded_episode):
    legacy_trace, bounded_trace = legacy_episode["waypoint_trace"], bounded_episode["waypoint_trace"]
    candidate, decision = None, None
    for index, (old, new) in enumerate(zip(legacy_trace, bounded_trace)):
        if candidate is None and candidate_signature(old) != candidate_signature(new):
            candidate = {"step_index": index, "legacy": candidate_signature(old), "bounded": candidate_signature(new)}
        if decision is None and decision_signature(old) != decision_signature(new):
            decision = {"step_index": index, "legacy": decision_signature(old), "bounded": decision_signature(new)}
    if len(legacy_trace) != len(bounded_trace):
        ended = {"step_index": min(len(legacy_trace), len(bounded_trace)),
                 "legacy": "trace ended" if len(legacy_trace) <= len(bounded_trace) else "continues",
                 "bounded": "trace ended" if len(bounded_trace) <= len(legacy_trace) else "continues"}
        candidate = candidate or ended
        decision = decision or ended
    return candidate, decision


def compare(legacy, bounded):
    categories = {"legacy_fail_to_bounded_success": [], "legacy_success_to_bounded_fail": [],
                  "unchanged_success": [], "unchanged_fail": []}
    changed = []
    per_episode = []
    for episode_id in EPISODE_IDS:
        old, new = legacy["stats"][episode_id], bounded["stats"][episode_id]
        old_success, new_success = bool(old["success"]), bool(new["success"])
        key = ("legacy_fail_to_bounded_success" if not old_success and new_success else
               "legacy_success_to_bounded_fail" if old_success and not new_success else
               "unchanged_success" if new_success else "unchanged_fail")
        categories[key].append(episode_id)
        candidate, decision = first_differences(
            legacy["telemetry"]["episodes"][episode_id], bounded["telemetry"]["episodes"][episode_id])
        record = {"episode_id": episode_id, "legacy_success": old_success, "bounded_success": new_success,
                  "first_candidate_set_divergence": candidate,
                  "first_high_level_decision_divergence": decision}
        per_episode.append(record)
        if old_success != new_success:
            changed.append(record)
    return {"outcome_categories": categories, "changed_outcome_count": len(changed),
            "changed_outcomes_first_divergence": changed, "per_episode": per_episode}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--bounded-run", required=True)
    parser.add_argument("--legacy-run", required=True)
    parser.add_argument("--original-run", required=True)
    args = parser.parse_args()
    bounded = read_run(args.bounded_run, "Geometry V0-BNMS")
    legacy = read_run(args.legacy_run, "valid-renderer legacy Geometry V0")
    original = read_run(args.original_run, "restored Original-20")
    bounded_metrics, legacy_metrics, original_metrics = summary(bounded), summary(legacy), summary(original)
    if any(abs(original_metrics[key] - value) > 1e-12 for key, value in ORIGINAL_REFERENCE.items()):
        raise SystemExit("restored Original-20 metrics differ from frozen reference")
    delta = lambda left, right: {
        key: left[key] - right[key] for key in left if key in right
        and isinstance(left[key], (int, float)) and isinstance(right[key], (int, float))
    }
    root = Path(args.output)
    dump(root / "waypoint_distribution.json", {
        "Geometry V0-BNMS": distribution(bounded), "valid_renderer_legacy_geometry_v0": distribution(legacy),
    })
    dump(root / "comparison_vs_legacy_geometry.json", {
        "Geometry V0-BNMS": bounded_metrics, "valid_renderer_legacy_geometry_v0": legacy_metrics,
        "bounded_minus_legacy": delta(bounded_metrics, legacy_metrics),
        "episode_comparison": compare(legacy, bounded),
    })
    dump(root / "comparison_vs_original.json", {
        "Geometry V0-BNMS": bounded_metrics, "restored_original_20": original_metrics,
        "bounded_minus_original": delta(bounded_metrics, original_metrics),
    })


if __name__ == "__main__":
    main()
