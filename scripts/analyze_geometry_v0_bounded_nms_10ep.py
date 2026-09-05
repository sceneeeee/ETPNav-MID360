#!/usr/bin/env python3
"""Compare the bounded-radial Geometry V0 panel with its diagnostic baseline."""
from __future__ import print_function

import argparse
import json
import math
from pathlib import Path

import numpy as np


EPISODE_IDS = (
    "1685", "1024", "1477", "1636", "1375", "1478",
    "53", "821", "164", "1679",
)


def load(path):
    with open(str(path)) as handle:
        return json.load(handle)


def only(directory, pattern):
    matches = list(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise SystemExit(
            "expected exactly one {} under {}, found {}".format(
                pattern, directory, matches
            )
        )
    return matches[0]


def read_run(directory):
    directory = Path(directory)
    stats = load(only(directory, "stats_ep_ckpt_*_val_unseen_r0_w1.json"))
    aggregate = load(only(directory, "stats_ckpt_*_val_unseen.json"))
    telemetry = load(only(directory, "telemetry_ep_ckpt_*_val_unseen_r0_w1.json"))
    sensor = load(only(directory, "sensor_validity_ep_ckpt_*_val_unseen_r0_w1.json"))
    if set(stats) != set(EPISODE_IDS):
        raise SystemExit("unexpected episode panel: {}".format(sorted(stats)))
    if set(telemetry["episodes"]) != set(EPISODE_IDS):
        raise SystemExit("telemetry episode panel does not match stats")
    if set(sensor) != set(EPISODE_IDS) or not all(
        item["rgb_valid"] and item["depth_valid"] for item in sensor.values()
    ):
        raise SystemExit("sensor guard evidence is incomplete or invalid")
    return {"stats": stats, "aggregate": aggregate, "telemetry": telemetry}


def percentile(values, thresholds=False):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean_m": None, "median_m": None, "p95_m": None}
    result = {
        "count": int(len(values)), "mean_m": float(values.mean()),
        "median_m": float(np.median(values)), "p95_m": float(np.percentile(values, 95)),
    }
    if thresholds:
        result["percentage_gte_2_5m"] = float(np.mean(values >= 2.5) * 100.0)
        result["percentage_eq_3_0m"] = float(np.mean(values == 3.0) * 100.0)
    return result


def waypoint_distribution(telemetry):
    candidates = []
    selected = []
    for episode in telemetry["episodes"].values():
        for decision in episode["waypoint_trace"]:
            candidates.extend(item["distance"] for item in decision["candidates"])
            waypoint = decision.get("selected_waypoint")
            if waypoint is not None:
                selected.append(waypoint["distance"])
    if not all(math.isfinite(value) for value in candidates + selected):
        raise SystemExit("non-finite waypoint distance in telemetry")
    return {
        "candidate_distance": percentile(candidates, thresholds=True),
        "selected_waypoint_distance": percentile(selected),
    }


def aggregate_summary(aggregate):
    return {
        "SR": aggregate["success"], "SPL": aggregate["spl"],
        "NE": aggregate["distance_to_goal"], "OS": aggregate["oracle_success"],
        "mean_steps": aggregate["steps_taken"],
        "mean_path_length": aggregate["path_length"],
    }


def episode_summary(stats, telemetry):
    return [
        {
            "episode_id": episode_id, "success": stats[episode_id]["success"],
            "SPL": stats[episode_id]["spl"],
            "NE": stats[episode_id]["distance_to_goal"],
            "steps": stats[episode_id]["steps_taken"],
            "path_length": stats[episode_id]["path_length"],
            "collisions": int(sum(
                item.get("collision", False)
                for item in telemetry["episodes"][episode_id]["waypoint_trace"]
            )),
            "collision_rate": stats[episode_id]["collisions"],
            "ghost_count": stats[episode_id]["ghost_cnt"],
        }
        for episode_id in EPISODE_IDS
    ]


def candidate_set(decision):
    return sorted((round(item["angle"], 8), round(item["distance"], 8))
                  for item in decision["candidates"])


def selected_signature(decision):
    selected = decision.get("selected_waypoint")
    if selected is None:
        return None
    return tuple(sorted((key, round(value, 8) if isinstance(value, float) else value)
                        for key, value in selected.items()))


def trajectory_difference(baseline, bounded):
    left, right = baseline["trajectory"], bounded["trajectory"]
    for index, (a, b) in enumerate(zip(left, right)):
        if not np.allclose(a, b, rtol=0.0, atol=1e-5):
            return {"diverged": True, "first_position_index": index}
    return {"diverged": len(left) != len(right), "first_position_index": min(len(left), len(right)) if len(left) != len(right) else None}


def compare_episode(episode_id, baseline, bounded):
    baseline_trace = baseline["telemetry"]["episodes"][episode_id]["waypoint_trace"]
    bounded_trace = bounded["telemetry"]["episodes"][episode_id]["waypoint_trace"]
    first_candidate_change = None
    first_decision_change = None
    for index, (old, new) in enumerate(zip(baseline_trace, bounded_trace)):
        if first_candidate_change is None and candidate_set(old) != candidate_set(new):
            first_candidate_change = {
                "step_index": index, "baseline": candidate_set(old),
                "bounded": candidate_set(new),
            }
        old_decision = (old["stop"], old.get("selected_graph_node_or_ghost_id"), selected_signature(old))
        new_decision = (new["stop"], new.get("selected_graph_node_or_ghost_id"), selected_signature(new))
        if first_decision_change is None and old_decision != new_decision:
            first_decision_change = {
                "step_index": index, "baseline": old_decision, "bounded": new_decision,
            }
    if first_candidate_change is None and len(baseline_trace) != len(bounded_trace):
        first_candidate_change = {"step_index": min(len(baseline_trace), len(bounded_trace)), "baseline": "trace ended", "bounded": "trace ended"}
    if first_decision_change is None and len(baseline_trace) != len(bounded_trace):
        first_decision_change = {"step_index": min(len(baseline_trace), len(bounded_trace)), "baseline": "trace ended", "bounded": "trace ended"}
    baseline_stats, bounded_stats = baseline["stats"][episode_id], bounded["stats"][episode_id]
    return {
        "episode_id": episode_id,
        "outcome_changed": bool(baseline_stats["success"] != bounded_stats["success"]),
        "first_candidate_set_change": first_candidate_change,
        "first_differing_high_level_decision": first_decision_change,
        "trajectory": trajectory_difference(
            baseline["telemetry"]["episodes"][episode_id],
            bounded["telemetry"]["episodes"][episode_id],
        ),
        "stop_behavior": {
            "baseline_stop_steps": [item["step_index"] for item in baseline_trace if item["stop"]],
            "bounded_stop_steps": [item["step_index"] for item in bounded_trace if item["stop"]],
        },
    }


def classification(comparisons):
    outcome_changes = sum(item["outcome_changed"] for item in comparisons)
    trajectory_changes = sum(item["trajectory"]["diverged"] for item in comparisons)
    any_change = any(
        item["first_candidate_set_change"] is not None
        or item["first_differing_high_level_decision"] is not None
        or item["trajectory"]["diverged"]
        for item in comparisons
    )
    if outcome_changes >= 2 or trajectory_changes >= 2:
        label = "MATERIAL"
    elif any_change:
        label = "MINOR"
    else:
        label = "NEGLIGIBLE"
    return {
        "classification": label,
        "rule": "MATERIAL iff at least two outcomes or trajectories differ; MINOR iff any difference otherwise exists; NEGLIGIBLE iff none exists.",
        "outcome_change_count": int(outcome_changes),
        "trajectory_divergence_count": int(trajectory_changes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bounded-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bounded, baseline = read_run(args.bounded_run), read_run(args.baseline_run)
    comparisons = [compare_episode(episode_id, baseline, bounded) for episode_id in EPISODE_IDS]
    report = {
        "controlled_variable": "Geometry-specific NMS boundary semantics only: angle circular; radial bounded/noncircular.",
        "bounded_nms": {
            "aggregate": aggregate_summary(bounded["aggregate"]),
            "per_episode": episode_summary(bounded["stats"], bounded["telemetry"]),
            "waypoint_distribution": waypoint_distribution(bounded["telemetry"]),
        },
        "geometry_v0_diagnostic_baseline": {
            "aggregate": aggregate_summary(baseline["aggregate"]),
            "per_episode": episode_summary(baseline["stats"], baseline["telemetry"]),
            "waypoint_distribution": waypoint_distribution(baseline["telemetry"]),
        },
        "episode_comparison": comparisons,
        "decision_gate": classification(comparisons),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
