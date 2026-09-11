#!/usr/bin/env python3
"""Create stable, compact evidence files for the frozen Original-20 run."""
import argparse
import json
from pathlib import Path


EPISODE_IDS = [
    "53", "1685", "821", "1024", "1477", "1636", "675", "164", "1375", "1679",
    "250", "2", "475", "1356", "252", "1439", "1016", "1285", "1478", "25",
]
HISTORICAL = {
    "SR": 0.800000,
    "SPL": 0.6983661897804773,
    "NE": 2.9216403543949125,
    "OS": 0.800000,
    "mean_trajectory_length": 10.830972146987914,
    "mean_steps": 71.55,
    "waypoint_latency_mean_ms": 39.08902862649593,
    "peak_vram_mib": 1017.55419921875,
}


def load(path):
    with open(path) as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    stats = load(next(root.glob("stats_ep_ckpt_*_val_unseen_r0_w1.json")))
    aggregate = load(next(root.glob("stats_ckpt_*_val_unseen.json")))
    telemetry = load(next(root.glob("telemetry_ep_ckpt_*_val_unseen_r0_w1.json")))
    sensor = load(next(root.glob("sensor_validity_ep_ckpt_*_val_unseen_r0_w1.json")))
    actual_ids = list(stats)
    if actual_ids != EPISODE_IDS:
        raise SystemExit("frozen episode order mismatch: {}".format(actual_ids))
    if set(sensor) != set(EPISODE_IDS) or not all(
        item["rgb_valid"] and item["depth_valid"] for item in sensor.values()
    ):
        raise SystemExit("sensor validity evidence is incomplete or invalid")

    episodes = []
    for episode_id in EPISODE_IDS:
        metrics = stats[episode_id]
        trace = telemetry["episodes"].get(episode_id, {})
        episodes.append({
            "episode_id": episode_id,
            "scene": trace.get("scene_id", sensor[episode_id]["scene_id"]),
            "success": metrics["success"], "SPL": metrics["spl"],
            "NE": metrics["distance_to_goal"], "distance_to_goal": metrics["distance_to_goal"],
            "oracle_success": metrics["oracle_success"], "steps": metrics["steps_taken"],
            "trajectory_length": metrics["path_length"], "path_length": metrics["path_length"],
            "collisions": metrics["collisions"], "ghost_count": metrics["ghost_cnt"],
            "nDTW": metrics["ndtw"], "SDTW": metrics["sdtw"],
        })
    current = {
        "SR": aggregate["success"], "SPL": aggregate["spl"],
        "NE": aggregate["distance_to_goal"], "OS": aggregate["oracle_success"],
        "mean_trajectory_length": aggregate["path_length"], "mean_steps": aggregate["steps_taken"],
        "waypoint_latency_mean_ms": telemetry["summary"]["inference_latency_ms"].get("mean"),
        "waypoint_latency_p95_ms": telemetry["summary"]["inference_latency_ms"].get("p95"),
        "peak_vram_mib": telemetry["summary"]["cuda_memory"]["peak_allocated_mib"],
    }
    differences = {key: current[key] - value for key, value in HISTORICAL.items() if current.get(key) is not None}
    historical_stats = load(Path("data/logs/eval_results/local_r2r_20ep/stats_ep_ckpt_59_val_unseen_r0_w1.json"))
    outcome_match = {episode_id: stats[episode_id]["success"] == historical_stats[episode_id]["success"] for episode_id in EPISODE_IDS}
    exact_metrics = all(abs(differences[key]) <= 1e-6 for key in ("SR", "SPL", "NE", "OS"))
    close_metrics = all(abs(differences[key]) <= 1e-3 for key in ("SR", "SPL", "NE", "OS"))
    if exact_metrics and all(outcome_match.values()):
        classification = "A. RESTORED"
    elif close_metrics:
        classification = "B. FUNCTIONALLY RESTORED, NOT BIT-EXACT"
    else:
        classification = "C. NOT RESTORED"
    summary = {
        "classification": classification,
        "episode_identities_match_frozen_record": actual_ids == EPISODE_IDS,
        "sensor_validity_complete": True,
        "current": current, "historical": HISTORICAL,
        "absolute_difference": {key: abs(value) for key, value in differences.items()},
        "per_episode_success_match_historical": outcome_match,
    }
    for name, value in (("episodes.json", episodes), ("telemetry.json", telemetry),
                        ("sensor_validity.json", sensor), ("summary.json", summary)):
        with open(root / name, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
