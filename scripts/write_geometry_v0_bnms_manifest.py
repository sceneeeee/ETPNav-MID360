#!/usr/bin/env python3
"""Write the provenance manifest for the formal Geometry V0-BNMS run."""
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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--auxiliary", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episode-ids", required=True)
    args = parser.parse_args()
    hashes = {"dataset": sha256(args.dataset), "checkpoint": sha256(args.checkpoint),
              "auxiliary": sha256(args.auxiliary)}
    if hashes != EXPECTED:
        raise SystemExit("provenance SHA256 mismatch: {}".format(hashes))
    values = yaml.safe_load(Path(args.config).read_text())
    if "maximum_candidate_distance_m" in values:
        raise SystemExit("Geometry V0.1 distance cap is forbidden")
    git = lambda *items: subprocess.check_output(
        ["git", *items], universal_newlines=True
    ).strip()
    manifest = {
        "label": "Geometry V0 + Bounded Radial NMS", "short_label": "Geometry V0-BNMS",
        "split": "val_unseen", "ordered_episode_ids": ast.literal_eval(args.episode_ids),
        "dataset_path": args.dataset, "dataset_sha256": hashes["dataset"],
        "checkpoint_path": args.checkpoint, "checkpoint_sha256": hashes["checkpoint"],
        "auxiliary_waypoint_weight_path": args.auxiliary,
        "auxiliary_waypoint_weight_sha256": hashes["auxiliary"],
        "provenance_matches_restored_formal_runs": True,
        "provider": "mid360_geometry",
        "controlled_change": "Geometry-specific NMS boundary semantics only: angular circular; radial bounded/noncircular.",
        "geometry_parameters": {"score": "S(a,k)=d/3", "angular_bins": 120,
            "radial_bins": 12, "sigma": [7, 5], "maximum_candidates": 5,
            "radial_distance_cap": None, "config": values},
        "renderer_bootstrap": "scripts/renderer_bootstrap.bash:etp_apply_known_good_renderer_environment",
        "sensor_guard": "require_valid_sensor_observations at every evaluation reset",
        "branch": git("branch", "--show-current"), "HEAD": git("rev-parse", "HEAD"),
        "git_status_at_manifest_creation": git("status", "--short"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "bash scripts/run_geometry_v0_bounded_nms20.bash",
    }
    with open(str(Path(args.output) / "run_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
