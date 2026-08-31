#!/usr/bin/env python3
"""Compare learned and geometric proposals on fixed public-scene poses.

This is an engineering smoke test only. It does not compute R2R-CE metrics.
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import OrderedDict
from types import SimpleNamespace

import cv2
import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (REPO_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(REPO_ROOT)

from etp_baseline_smoke import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_DDPPO,
    DEFAULT_WAYPOINT,
    check_config,
    load_policy,
    panorama_observations,
    synchronize,
)
from vlnce_baselines.waypoint_provider.mid360_geometry import (  # noqa: E402
    MID360GeometryConfig,
    MID360GeometryWaypointProvider,
    default_mid360_config_path,
)
from vlnce_baselines.waypoint_provider.types import (  # noqa: E402
    validate_waypoint_output,
)


def candidate_records(output):
    return [
        {
            "angle_radians_ccw": float(angle),
            "distance_m": float(distance),
            "panorama_image_index": int(image_index),
        }
        for angle, distance, image_index in zip(
            output["cand_angles"][0],
            output["cand_distances"][0],
            output["cand_img_idxes"][0],
        )
    ]


def topdown_canvas(points, extent_m=3.25, size=600):
    canvas = np.full((size, size, 3), 250, dtype=np.uint8)
    center = size // 2
    cv2.line(canvas, (center, 0), (center, size - 1), (220, 220, 220), 1)
    cv2.line(canvas, (0, center), (size - 1, center), (220, 220, 220), 1)
    if points.size:
        columns = np.rint(
            center + points[:, 0] / extent_m * center
        ).astype(np.int64)
        rows = np.rint(
            center + points[:, 2] / extent_m * center
        ).astype(np.int64)
        valid = (
            (rows >= 0) & (rows < size) & (columns >= 0) & (columns < size)
        )
        canvas[rows[valid], columns[valid]] = (110, 80, 20)
    cv2.circle(canvas, (center, center), 7, (0, 0, 0), -1)
    cv2.arrowedLine(
        canvas,
        (center, center),
        (center, center - 45),
        (0, 0, 0),
        3,
        tipLength=0.25,
    )
    return canvas


def draw_candidates(canvas, candidates, color, label, extent_m=3.25):
    image = canvas.copy()
    center = image.shape[0] // 2
    scale = center / extent_m
    for index, candidate in enumerate(candidates):
        angle = candidate["angle_radians_ccw"]
        distance = candidate["distance_m"]
        x = -math.sin(angle) * distance
        z = -math.cos(angle) * distance
        endpoint = (
            int(round(center + x * scale)),
            int(round(center + z * scale)),
        )
        cv2.line(image, (center, center), endpoint, color, 2)
        cv2.circle(image, endpoint, 7, color, -1)
        cv2.putText(
            image,
            str(index),
            (endpoint[0] + 5, endpoint[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return image


def occupancy_image(occupancy):
    image = np.where(occupancy, 0, 255).astype(np.uint8)
    return cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)


def heatmap_image(score):
    normalized = np.clip(score.T * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return cv2.resize(colored, (600, 240), interpolation=cv2.INTER_NEAREST)


def save_debug(debug, original, geometry, output_dir, detailed, extent_m):
    os.makedirs(output_dir, exist_ok=True)
    base = topdown_canvas(debug["filtered_pointcloud"], extent_m=extent_m)
    original_image = draw_candidates(
        base,
        candidate_records(original),
        (30, 30, 220),
        "Original learned",
        extent_m=extent_m,
    )
    geometry_image = draw_candidates(
        base,
        candidate_records(geometry),
        (40, 170, 40),
        "MID360-like geometry",
        extent_m=extent_m,
    )
    comparison = np.concatenate((original_image, geometry_image), axis=1)
    cv2.imwrite(os.path.join(output_dir, "candidate_comparison.png"), comparison)
    if detailed:
        traversal = debug["traversability"]
        cv2.imwrite(os.path.join(output_dir, "topdown_pointcloud.png"), base)
        cv2.imwrite(
            os.path.join(output_dir, "obstacle_occupancy.png"),
            occupancy_image(traversal.occupancy),
        )
        cv2.imwrite(
            os.path.join(output_dir, "inflated_occupancy.png"),
            occupancy_image(traversal.inflated_occupancy),
        )
        cv2.imwrite(
            os.path.join(output_dir, "geometry_heatmap_120x12.png"),
            heatmap_image(traversal.score),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/example_scene")
    parser.add_argument("--output-dir", default="outputs/example_scene/comparison")
    parser.add_argument("--pose-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--config", default=default_mid360_config_path())
    return parser.parse_args()


def validate_capture_metadata(metadata, geometry_config, pose_count):
    expected_headings = list(range(0, 360, 30))
    checks = (
        (metadata.get("test_kind") == "ENGINEERING_SMOKE_TEST_ONLY", "test kind"),
        (metadata.get("pose_count", 0) >= pose_count, "pose count"),
        (metadata.get("headings_degrees") == expected_headings, "heading order"),
        (metadata.get("rgb_shape") == [224, 224, 3], "RGB shape"),
        (metadata.get("depth_shape") == [256, 256, 1], "depth shape"),
        (
            abs(metadata.get("camera_height_m", -1) - geometry_config.camera_height_m)
            <= 1e-6,
            "camera height",
        ),
        (
            abs(
                metadata.get("horizontal_fov_degrees", -1)
                - geometry_config.horizontal_fov_degrees
            )
            <= 1e-6,
            "horizontal FoV",
        ),
        (
            abs(
                metadata.get("policy_depth", {}).get("minimum_m", -1)
                - geometry_config.policy_depth_minimum_m
            )
            <= 1e-6,
            "policy minimum depth",
        ),
        (
            abs(
                metadata.get("policy_depth", {}).get("maximum_m", -1)
                - geometry_config.policy_depth_maximum_m
            )
            <= 1e-6,
            "policy maximum depth",
        ),
    )
    failed = [name for passed, name in checks if not passed]
    if failed:
        raise RuntimeError(
            "capture/provider configuration mismatch: {}".format(failed)
        )


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("sample-scene provider comparison requires CUDA")
    paths = [
        os.path.join(args.input_dir, "fixed_12view_policy_observation.npz")
    ]
    paths.extend(
        sorted(
            glob.glob(
                os.path.join(
                    args.input_dir,
                    "poses",
                    "pose_*",
                    "fixed_12view_policy_observation.npz",
                )
            )
        )
    )
    paths = paths[: args.pose_count]
    if len(paths) != args.pose_count or not all(os.path.isfile(path) for path in paths):
        raise RuntimeError(
            "expected {} captured poses under {}, found {}".format(
                args.pose_count, args.input_dir, len(paths)
            )
        )

    geometry_config = MID360GeometryConfig.from_yaml(args.config)
    metadata_path = os.path.join(args.input_dir, "example_scene_metadata.json")
    with open(metadata_path, "r") as handle:
        capture_metadata = json.load(handle)
    validate_capture_metadata(capture_metadata, geometry_config, args.pose_count)

    policy_args = SimpleNamespace(
        config=DEFAULT_CONFIG,
        checkpoint=DEFAULT_CHECKPOINT,
        waypoint_checkpoint=DEFAULT_WAYPOINT,
        ddppo_checkpoint=DEFAULT_DDPPO,
        episodes=1,
    )
    initialization = OrderedDict()
    etp_config = check_config(policy_args, initialization)
    policy, waypoint = load_policy(
        etp_config, policy_args, device, initialization
    )
    geometry_provider = MID360GeometryWaypointProvider(
        rgb_encoder=policy.net.rgb_encoder,
        depth_encoder=policy.net.depth_encoder,
        rgb_pool=policy.net.space_pool_rgb,
        depth_pool=policy.net.space_pool_depth,
        pano_angle_features=policy.net.pano_angle_fts,
        pano_image_indexes=policy.net.pano_img_idxes,
        config=geometry_config,
    )
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    pose_results = []
    with torch.no_grad():
        for pose_index, path in enumerate(paths):
            observations = panorama_observations(device, path)
            synchronize(device)
            started = time.perf_counter()
            original = policy.net(
                mode="waypoint",
                waypoint_predictor=waypoint,
                observations=observations,
                in_train=False,
            )
            synchronize(device)
            original_ms = (time.perf_counter() - started) * 1000.0

            synchronize(device)
            started = time.perf_counter()
            geometry = policy.net(
                mode="waypoint",
                waypoint_provider=geometry_provider,
                observations=observations,
                in_train=False,
            )
            synchronize(device)
            geometry_ms = (time.perf_counter() - started) * 1000.0
            validate_waypoint_output(original)
            validate_waypoint_output(geometry)
            pano_rgb_difference = float(
                torch.max(torch.abs(original["pano_rgb"] - geometry["pano_rgb"])).item()
            )
            pano_depth_difference = float(
                torch.max(
                    torch.abs(original["pano_depth"] - geometry["pano_depth"])
                ).item()
            )
            debug = geometry_provider.last_debug[0]
            pose_output_dir = os.path.join(
                args.output_dir, "pose_{:03d}".format(pose_index)
            )
            save_debug(
                debug,
                original,
                geometry,
                pose_output_dir,
                detailed=pose_index == 0,
                extent_m=geometry_config.maximum_range_m,
            )
            pose_results.append(
                OrderedDict(
                    [
                        ("pose_index", pose_index),
                        ("observation", path),
                        ("original_candidates", candidate_records(original)),
                        ("geometry_candidates", candidate_records(geometry)),
                        ("original_latency_ms", original_ms),
                        ("geometry_latency_ms", geometry_ms),
                        (
                            "merged_point_count",
                            int(debug["merged_pointcloud"].shape[0]),
                        ),
                        (
                            "filtered_point_count",
                            int(debug["filtered_pointcloud"].shape[0]),
                        ),
                        (
                            "geometry_score_shape",
                            list(debug["traversability"].score.shape),
                        ),
                        ("pano_rgb_max_abs_difference", pano_rgb_difference),
                        (
                            "pano_depth_max_abs_difference",
                            pano_depth_difference,
                        ),
                    ]
                )
            )

    report = OrderedDict(
        [
            ("status", "PASS"),
            ("test_kind", "ENGINEERING_SMOKE_TEST_ONLY_NOT_R2R_CE_BENCHMARK"),
            ("seed", args.seed),
            ("pose_count", len(pose_results)),
            ("geometry_config", geometry_config.values),
            ("poses", pose_results),
            (
                "mean_original_latency_ms",
                float(np.mean([pose["original_latency_ms"] for pose in pose_results])),
            ),
            (
                "mean_geometry_latency_ms",
                float(np.mean([pose["geometry_latency_ms"] for pose in pose_results])),
            ),
            (
                "mean_original_candidate_count",
                float(
                    np.mean(
                        [len(pose["original_candidates"]) for pose in pose_results]
                    )
                ),
            ),
            (
                "mean_geometry_candidate_count",
                float(
                    np.mean(
                        [len(pose["geometry_candidates"]) for pose in pose_results]
                    )
                ),
            ),
            (
                "peak_cuda_memory_mib",
                torch.cuda.max_memory_allocated(device) / 2 ** 20,
            ),
        ]
    )
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "comparison_results.json")
    rendered = json.dumps(report, indent=2)
    with open(output_path, "w") as handle:
        handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
