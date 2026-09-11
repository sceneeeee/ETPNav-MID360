#!/usr/bin/env python3
"""Assemble bounded per-process Original ETP forensic evidence."""
from __future__ import print_function

import argparse
import glob
import json
import os
import sys


# Keep direct ``python scripts/...`` invocations consistent with the other
# repository utilities, including the forensic runner's final collection step.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vlnce_baselines.diagnostic_serialization import json_safe


def read(path):
    with open(path) as handle:
        return json.load(handle)


def write(path, data):
    with open(path, "w") as handle:
        json.dump(json_safe(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def telemetry(path):
    files = glob.glob(os.path.join(path, "**", "telemetry_*.json"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(path, "**", "forensic_trace.json"), recursive=True)
    return read(files[0]) if len(files) == 1 else None


def stats(path):
    files = glob.glob(os.path.join(path, "**", "stats_ep_*.json"), recursive=True)
    return read(files[0]) if len(files) == 1 else None


def candidates_match(raw, report):
    if not report or not report.get("episodes"):
        return None
    episode = next(iter(report["episodes"].values()))
    trace = episode.get("waypoint_trace", [])
    if not trace:
        return None
    serialized = trace[0].get("candidates")
    expected = [{"angle": a, "distance": d, "panorama_image_index": i}
                for a, d, i in zip(raw["cand_angles"], raw["cand_distances"], raw["cand_img_idxes"])]
    return serialized == expected


def trajectory_signature(report):
    if not report or not report.get("episodes"):
        return None
    episode = next(iter(report["episodes"].values()))
    return {"waypoint_trace": [{key: step.get(key) for key in
                                 ("agent_position", "agent_heading", "candidates",
                                  "selected_waypoint", "selected_graph_node_or_ghost_id", "stop")}
                                for step in episode.get("waypoint_trace", [])],
            "trajectory": episode.get("trajectory"), "metrics": episode.get("metrics")}


def candidate_signature(raw):
    return list(zip(raw["cand_angles"], raw["cand_distances"], raw["cand_img_idxes"]))


def observation_summary(audit):
    views = audit.get("views", [])
    rgb = [view.get("rgb", {}) for view in views]
    depth = [view.get("depth", {}) for view in views]
    return {
        "view_count": len(views),
        "rgb_nonzero_count": sum(item.get("max", 0) > 0 for item in rgb),
        "rgb_distinct_hashes": len(set(item.get("hash") for item in rgb)),
        "depth_valid_count": sum(item.get("valid_fraction", 0) > 0 for item in depth),
        "depth_distinct_hashes": len(set(item.get("hash") for item in depth)),
        "valid": bool(views) and all(item.get("max", 0) > 0 for item in rgb)
        and all(item.get("valid_fraction", 0) > 0 for item in depth),
    }


def run_record(root, label):
    raw = read(os.path.join(root, "raw_%s.json" % label))
    report = telemetry(os.path.join(root, label))
    metrics = stats(os.path.join(root, label))
    episode_metrics = next(iter(metrics.values())) if metrics else {}
    trace = trajectory_signature(report)
    return {"raw": raw, "telemetry": report, "metrics_by_episode": metrics,
            "episode53": {"success": episode_metrics.get("success"),
                          "spl": episode_metrics.get("spl"),
                          "ne": episode_metrics.get("distance_to_goal"),
                          "os": episode_metrics.get("oracle_success"),
                          "steps": len(trace["waypoint_trace"]) if trace else None,
                          "trajectory_length": episode_metrics.get("path_length")},
            "raw_serialized_exact_match": candidates_match(raw, report)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    records = []
    for number in range(1, 6):
        label = "run%d" % number
        record = run_record(root, label)
        write(os.path.join(root, "episode53_run%d.json" % number), record)
        records.append(record)
    write(os.path.join(root, "raw_waypoint_step0.json"), records[0]["raw"])
    write(os.path.join(root, "observation_audit.json"), records[0]["raw"]["observation_audit"])
    off = run_record(root, "telemetry_off")
    on = run_record(root, "telemetry_on")
    comparison = {"telemetry_off": off, "telemetry_on": on,
                  "raw_candidates_identical": candidate_signature(off["raw"]) == candidate_signature(on["raw"]),
                  "trajectory_equivalent": trajectory_signature(off["telemetry"]) == trajectory_signature(on["telemetry"]),
                  "comparison_fields": ["candidate sets", "selected waypoint", "graph action", "pose trajectory", "STOP", "final metrics"]}
    write(os.path.join(root, "telemetry_compare.json"), comparison)
    signatures = [json.dumps(json_safe({"raw": r["raw"], "trajectory": trajectory_signature(r["telemetry"])}), sort_keys=True) for r in records]
    first = records[0]
    observation = observation_summary(first["raw"].get("observation_audit", {}))
    summary = {"episode": "53", "runs": 5,
               "classification": "DETERMINISTIC" if len(set(signatures)) == 1 else "NONDETERMINISTIC",
               "final_metrics_identical": all(r["metrics_by_episode"] == first["metrics_by_episode"] for r in records),
               "raw_step0_candidates_identical": all(candidate_signature(r["raw"]) == candidate_signature(first["raw"]) for r in records),
               "observation_audit": observation,
               "raw_serialized_exact_match": [r["raw_serialized_exact_match"] for r in records],
               "telemetry_non_interference": comparison["raw_candidates_identical"]}
    write(os.path.join(root, "summary.json"), summary)
    # Always create the requested placeholder; the historical command fills it.
    write(os.path.join(root, "historical_compare.json"), {"status": "not run"})


if __name__ == "__main__":
    main()
