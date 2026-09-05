#!/usr/bin/env python3
"""Validate and summarize the frozen valid-renderer Geometry V0 evaluation."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


EPISODE_IDS = [
    "53", "1685", "821", "1024", "1477", "1636", "675", "164", "1375", "1679",
    "250", "2", "475", "1356", "252", "1439", "1016", "1285", "1478", "25",
]
ORIGINAL = {
    "SR": 0.800,
    "SPL": 0.6983661897804773,
    "NE": 2.9216403543949125,
    "OS": 0.800,
    "mean_latency_ms": 38.670588801230764,
    "p95_latency_ms": 45.910937749795266,
    "peak_allocated_mib": 1017.55419921875,
    "mean_steps": 71.55,
    "mean_path_length": 10.830972146987914,
}
HISTORICAL = {
    "SR": 0.500, "SPL": 0.321, "NE": 5.055, "OS": 0.500,
    "mean_latency_ms": 85.71, "p95_latency_ms": 98.20,
    "peak_allocated_mib": 1017.47, "mean_path_length": 22.19,
    "mean_steps": 145.8,
}


def load(path):
    with open(path) as handle:
        return json.load(handle)


def dump(path, value):
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def only(root, pattern):
    paths = list(root.glob(pattern))
    if len(paths) != 1:
        raise SystemExit("expected exactly one {} under {}, found {}".format(pattern, root, paths))
    return paths[0]


def percentile_summary(values, include_thresholds=False):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        raise SystemExit("waypoint telemetry contains no finite values")
    summary = {
        "count": int(len(values)), "mean": float(np.mean(values)),
        "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
        "minimum": float(np.min(values)), "maximum": float(np.max(values)),
    }
    if include_thresholds:
        summary.update({
            "percentage_lte_1_0m": float(np.mean(values <= 1.0) * 100.0),
            "percentage_gte_2_5m": float(np.mean(values >= 2.5) * 100.0),
            "percentage_eq_3_0m": float(np.mean(values == 3.0) * 100.0),
        })
    return summary


def waypoint_distribution(telemetry):
    candidate_distances, selected_distances, candidate_counts = [], [], []
    for episode in telemetry["episodes"].values():
        for decision in episode["waypoint_trace"]:
            candidates = decision["candidates"]
            candidate_counts.append(len(candidates))
            for candidate in candidates:
                angle, distance = candidate["angle"], candidate["distance"]
                if not math.isfinite(angle) or not math.isfinite(distance):
                    raise SystemExit("non-finite raw waypoint candidate")
                candidate_distances.append(distance)
            selected = decision.get("selected_waypoint")
            if selected is not None:
                distance = selected["distance"]
                if not math.isfinite(distance):
                    raise SystemExit("non-finite selected waypoint")
                selected_distances.append(distance)
    if not candidate_distances:
        raise SystemExit("no raw waypoint candidates recorded")
    if not selected_distances:
        raise SystemExit("no selected waypoints recorded")
    return {
        "candidate_distance_m": percentile_summary(candidate_distances, include_thresholds=True),
        "selected_waypoint_distance_m": percentile_summary(selected_distances),
        "candidate_count_per_decision": {
            "count": len(candidate_counts), "mean": float(np.mean(candidate_counts)),
            "median": float(np.median(candidate_counts)),
        },
    }


def metrics(stats, aggregate, telemetry):
    latency = telemetry["summary"]["inference_latency_ms"]
    memory = telemetry["summary"]["cuda_memory"]
    return {
        "SR": aggregate["success"], "SPL": aggregate["spl"],
        "NE": aggregate["distance_to_goal"], "OS": aggregate["oracle_success"],
        "mean_latency_ms": latency["mean"], "p95_latency_ms": latency["p95"],
        "max_latency_ms": latency["maximum"],
        "allocated_mib": memory["allocated_mib"], "reserved_mib": memory["reserved_mib"],
        "peak_allocated_mib": memory["peak_allocated_mib"],
        "peak_reserved_mib": memory["peak_reserved_mib"],
        "mean_steps": aggregate["steps_taken"], "mean_path_length": aggregate["path_length"],
    }


def historical_classification(current):
    # The record is approximate, so this deliberately uses broad, disclosed
    # reproducibility bands; it never changes the experiment configuration.
    reproduced = (abs(current["SR"] - HISTORICAL["SR"]) <= 0.01 and
                  abs(current["SPL"] - HISTORICAL["SPL"]) <= 0.01 and
                  abs(current["NE"] - HISTORICAL["NE"]) <= 0.10 and
                  abs(current["OS"] - HISTORICAL["OS"]) <= 0.01)
    close = (abs(current["SR"] - HISTORICAL["SR"]) <= 0.10 and
             abs(current["SPL"] - HISTORICAL["SPL"]) <= 0.10 and
             abs(current["NE"] - HISTORICAL["NE"]) <= 1.0 and
             abs(current["OS"] - HISTORICAL["OS"]) <= 0.10)
    return "reproduced" if reproduced else ("functionally_close" if close else "materially_different")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--original-telemetry", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    native = root / "geometry_v0_baseline20_valid_renderer"
    stats = load(only(native, "stats_ep_ckpt_*_val_unseen_r0_w1.json"))
    aggregate = load(only(native, "stats_ckpt_*_val_unseen.json"))
    telemetry = load(only(native, "telemetry_ep_ckpt_*_val_unseen_r0_w1.json"))
    sensor = load(only(native, "sensor_validity_ep_ckpt_*_val_unseen_r0_w1.json"))
    if list(stats) != EPISODE_IDS:
        raise SystemExit("frozen episode order mismatch: {}".format(list(stats)))
    if list(telemetry["episodes"]) != EPISODE_IDS:
        raise SystemExit("telemetry episode order mismatch")
    if set(sensor) != set(EPISODE_IDS) or not all(
        item["rgb_valid"] and item["depth_valid"] for item in sensor.values()
    ):
        raise SystemExit("sensor validity evidence is incomplete or invalid")
    config = yaml.safe_load(Path(args.config).read_text())
    if "maximum_candidate_distance_m" in config:
        raise SystemExit("V0.1 maximum_candidate_distance_m cap is forbidden")
    geometry = metrics(stats, aggregate, telemetry)
    original_telemetry = load(args.original_telemetry)
    distributions = {
        "geometry_v0": waypoint_distribution(telemetry),
        "original_valid_renderer": waypoint_distribution(original_telemetry),
    }
    original_stats = load("outputs/original_baseline20_valid_renderer/original_baseline20_valid_renderer/stats_ep_ckpt_59_val_unseen_r0_w1.json")
    categories = {"both_succeed": [], "original_only_succeeds": [], "geometry_only_succeeds": [], "both_fail": []}
    for episode_id in EPISODE_IDS:
        original_success = bool(original_stats[episode_id]["success"])
        geometry_success = bool(stats[episode_id]["success"])
        key = "both_succeed" if original_success and geometry_success else (
            "original_only_succeeds" if original_success else (
                "geometry_only_succeeds" if geometry_success else "both_fail"))
        categories[key].append(episode_id)
    deltas = {key: geometry[key] - ORIGINAL[key] for key in ORIGINAL}
    comparison = {
        "original_valid_renderer": ORIGINAL, "geometry_v0": geometry,
        "geometry_minus_original": deltas, "episode_outcomes": categories,
        "historical_geometry_v0_reference": HISTORICAL,
        "historical_comparison": {
            "classification": historical_classification(geometry),
            "geometry_minus_historical": {key: geometry[key] - HISTORICAL[key] for key in HISTORICAL},
        },
    }
    dump(root / "waypoint_distribution.json", distributions)
    dump(root / "comparison_vs_original.json", comparison)
    dump(root / "episodes.json", [
        {"episode_id": episode_id, "success": stats[episode_id]["success"],
         "SPL": stats[episode_id]["spl"], "NE": stats[episode_id]["distance_to_goal"],
         "oracle_success": stats[episode_id]["oracle_success"], "steps": stats[episode_id]["steps_taken"],
         "path_length": stats[episode_id]["path_length"], "collisions": stats[episode_id]["collisions"],
         "ghost_count": stats[episode_id]["ghost_cnt"], "nDTW": stats[episode_id]["ndtw"], "SDTW": stats[episode_id]["sdtw"]}
        for episode_id in EPISODE_IDS
    ])


if __name__ == "__main__":
    main()
