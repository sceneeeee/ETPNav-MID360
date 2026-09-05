#!/usr/bin/env python3
"""Write provenance for the frozen MRU run, and reject control drift."""
import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

EXPECTED = {
    "dataset": "7db0a39bef374cf5ce986473f8b35c9721f962c6c1a9af069257aa710df16365",
    "checkpoint": "4e70d2a1b4a6cfb32158901b7300cbc2be657641f16c9a17f35ca8570470d8b3",
    "auxiliary": "09d0f42cbd801e05b0fa0212b901d409f033f4d0bce2fce1fa8a1331b502159f",
}


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("output", "checkpoint", "auxiliary", "dataset", "config", "episode_ids"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args()
    hashes = {"dataset": digest(args.dataset), "checkpoint": digest(args.checkpoint),
              "auxiliary": digest(args.auxiliary)}
    if hashes != EXPECTED:
        raise SystemExit("provenance SHA256 mismatch: {}".format(hashes))
    values = yaml.safe_load(Path(args.config).read_text())
    baseline = yaml.safe_load(Path("configs/mid360_waypoint.yaml").read_text())
    if "maximum_candidate_distance_m" in values:
        raise SystemExit("distance cap is forbidden")
    if values.get("radial_utility") != "mru" or {
        key: value for key, value in values.items() if key != "radial_utility"
    } != baseline:
        raise SystemExit("MRU config changes something other than radial utility")
    git = lambda *items: subprocess.check_output(["git", *items], universal_newlines=True).strip()
    value = {
        "label": "Geometry Mid-Range Radial Utility + Bounded Radial NMS",
        "short_label": "Geometry MRU-BNMS", "split": "val_unseen",
        "ordered_episode_ids": ast.literal_eval(args.episode_ids),
        "dataset_path": args.dataset, "dataset_sha256": hashes["dataset"],
        "checkpoint_path": args.checkpoint, "checkpoint_sha256": hashes["checkpoint"],
        "auxiliary_waypoint_weight_path": args.auxiliary,
        "auxiliary_waypoint_weight_sha256": hashes["auxiliary"],
        "provenance_matches_restored_formal_runs": True, "provider": "mid360_geometry",
        "controlled_change": "Pre-NMS MRU radial utility only; bounded Geometry NMS is unchanged.",
        "radial_utility": {"name": "MRU", "formula": "x * exp(1 - x), x = 2*d/3.0",
            "D_max_m": 3.0, "analytic_peak_m": 1.5, "hard_cutoff": None,
            "learned_parameters": None, "fitted_parameters": None},
        "geometry_parameters": {"angular_bins": 120, "radial_bins": 12,
            "radial_spacing_m": 0.25, "maximum_range_m": 3.0, "sigma": [7, 5],
            "maximum_candidates": 5, "config": values},
        "renderer_bootstrap": "scripts/renderer_bootstrap.bash:etp_apply_known_good_renderer_environment",
        "sensor_guard": "require_valid_sensor_observations at every evaluation reset",
        "branch": git("branch", "--show-current"), "HEAD": git("rev-parse", "HEAD"),
        "git_status_at_manifest_creation": git("status", "--short"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "bash scripts/run_geometry_mru_bnms20_valid_renderer.bash",
    }
    with open(str(Path(args.output) / "run_manifest.json"), "w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
