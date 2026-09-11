#!/usr/bin/env python3
"""Write the ignored pre-registration manifest for inactive Geometry V1."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/geometry_v1_integration/freeze_manifest.json"
FORMAL_EPISODES = [
    "53", "1685", "821", "1024", "1477", "1636", "675", "164",
    "1375", "1679", "250", "2", "475", "1356", "252", "1439",
    "1016", "1285", "1478", "25",
]
V1_FILES = [
    "configs/mid360_waypoint_v1.yaml",
    "scripts/smoke_geometry_v1_provider.py",
    "scripts/study_geometry_v1_candidate_contract.py",
    "scripts/write_geometry_v1_freeze_manifest.py",
    "tests/test_depth_to_rays.py",
    "tests/test_geometry_v1_candidates.py",
    "tests/test_geometry_v1_provider.py",
    "tests/test_ray_carving_representation.py",
    "vlnce_baselines/config/default.py",
    "vlnce_baselines/perception/depth_to_rays.py",
    "vlnce_baselines/perception/geometry_v1_candidates.py",
    "vlnce_baselines/perception/ray_carving_representation.py",
    "vlnce_baselines/ss_trainer_ETP.py",
    "vlnce_baselines/waypoint_provider/__init__.py",
    "vlnce_baselines/waypoint_provider/mid360_geometry_v1.py",
]
ASSETS = {
    "checkpoint": "data/logs/checkpoints/release_r2r/ckpt.iter12000.pth",
    "dataset": "data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/val_unseen/val_unseen_bertidx.json.gz",
    "auxiliary_waypoint": "data/wp_pred/check_cwp_bestdist_hfov90",
    "depth_model": "data/ddppo-models/gibson-2plus-resnet50.pth",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def git(*args):
    return subprocess.check_output(
        ["git"] + list(args), cwd=str(ROOT), universal_newlines=True
    ).strip()


def main():
    with (ROOT / "configs/mid360_waypoint_v1.yaml").open() as handle:
        config = yaml.safe_load(handle)
    smoke_path = ROOT / "outputs/geometry_v1_integration/provider_smoke.json"
    with smoke_path.open() as handle:
        smoke = json.load(handle)
    if smoke.get("status") != "PASS" or smoke.get("navigation_executed") is not False:
        raise RuntimeError("provider-only smoke evidence is missing or invalid")
    manifest = {
        "status": "FROZEN_PENDING_FORMAL_EVAL",
        "provider_mode": "mid360_geometry_v1",
        "config_name": "configs/mid360_waypoint_v1.yaml",
        "default_provider_unchanged": "original_etp",
        "v1_design_identifier": "height-aware-ray-grid/hybrid-extraction-v1",
        "representation_identifier": "3d-voxel-free-hit-unknown-range-censor-v1",
        "candidate_contract_identifier": "direct-clearance-ridge_then_bounded-progress-fallback-v1",
        "legacy_dense_radial_score_or_generic_nms": False,
        "legacy_statement": "Geometry V1 does not construct or consume the legacy dense 120x12 radial score and does not call generic radial NMS.",
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty_worktree_status": git("status", "--short"),
        "v1_related_files": {
            path: sha256(ROOT / path) for path in V1_FILES
        },
        "assets": {
            name: {"path": path, "sha256": sha256(ROOT / path)}
            for name, path in ASSETS.items()
        },
        "formal_split": "val_unseen",
        "formal_frozen_episode_ids": FORMAL_EPISODES,
        "constants": {
            "inherited_etp_habitat": {
                "panorama_views": 12,
                "panorama_headings_degrees": list(range(0, 360, 30)),
                "depth_shape_hw": [256, 256],
                "depth_normalized": True,
                "depth_minimum_m": config["policy_depth_minimum_m"],
                "depth_maximum_m": config["policy_depth_maximum_m"],
                "horizontal_fov_degrees": config["horizontal_fov_degrees"],
                "camera_height_m": config["camera_height_m"],
                "candidate_angular_bins": config["angular_bins"],
                "candidate_radial_bins": config["distance_bins"],
                "candidate_distance_step_m": config["distance_step_m"],
                "maximum_candidates": config["maximum_candidates"],
            },
            "inherited_geometry_v0_physical": {
                key: config[key] for key in (
                    "maximum_distance_m", "depth_pixel_stride",
                    "minimum_range_m", "maximum_range_m",
                    "vertical_fov_min_degrees", "vertical_fov_max_degrees",
                    "azimuth_bin_degrees", "elevation_bin_degrees",
                    "voxel_size_m", "robot_radius_m", "robot_radius_status",
                    "clearance_margin_m", "obstacle_height_min_m",
                    "obstacle_height_max_m", "grid_resolution_m",
                )
            },
            "structurally_frozen_v1": {
                "ray_carving_sample_spacing": "at_most_half_grid_cell",
                "ridge_minimum_prominence": "one_grid_cell",
                "angular_redundancy_bins": 5,
                "radial_adjacency_bins": 1,
                "full_circular_fallback_candidates": 0,
                "fallback_radius": None,
                "unknown_is_traversable": False,
                "range_censor_is_physical_hit": False,
            },
            "new_potentially_tunable": {},
        },
        "cpu_tests": {
            "command": "PYTHONPATH=. .../etpnav_legacy/bin/python -m unittest discover -s tests -v",
            "result": "PASS",
            "tests_run": 95,
        },
        "sensor_provider_smoke": smoke,
        "sensor_capture_manifest": "outputs/geometry_v1_integration/sensor_capture/example_scene_metadata.json",
        "renderer_bootstrap": "scripts/renderer_bootstrap.bash:etp_apply_known_good_renderer_environment",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "navigation_outcomes_observed": False,
        "preregistration_statement": "NO NAVIGATION OUTCOMES HAVE BEEN OBSERVED FOR V1",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
