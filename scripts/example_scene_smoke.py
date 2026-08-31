#!/usr/bin/env python3
"""Capture a deterministic 12-view ETP observation in a public Habitat scene.

This is an engineering smoke utility, not an R2R-CE episode or benchmark.
Generated images, tensors, and metadata belong under the ignored ``outputs/`` tree.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict

import cv2
import habitat_sim
import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCENE = os.path.join(
    os.path.dirname(REPO_ROOT),
    "datasets",
    "habitat",
    "scene_datasets",
    "habitat-test-scenes",
    "van-gogh-room.glb",
)
HEADINGS_DEGREES = tuple(range(0, 360, 30))
CAMERA_HEIGHT_M = 1.25
HORIZONTAL_FOV_DEGREES = 90.0
POLICY_MIN_DEPTH_M = 0.0
POLICY_MAX_DEPTH_M = 10.0


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sensor_uuid(kind, heading_degrees):
    if heading_degrees == 0:
        return kind
    return "{}_{}".format(kind, heading_degrees)


def make_sensor(kind, heading_degrees):
    spec = habitat_sim.SensorSpec()
    spec.uuid = sensor_uuid(kind, heading_degrees)
    spec.sensor_type = (
        habitat_sim.SensorType.COLOR
        if kind == "rgb"
        else habitat_sim.SensorType.DEPTH
    )
    resolution = 224 if kind == "rgb" else 256
    spec.resolution = [resolution, resolution]
    spec.position = [0.0, CAMERA_HEIGHT_M, 0.0]
    spec.orientation = [0.0, math.radians(heading_degrees), 0.0]
    spec.parameters["hfov"] = str(HORIZONTAL_FOV_DEGREES)
    return spec


def make_configuration(scene):
    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = scene
    simulator_config.gpu_device_id = 0

    agent_config = habitat_sim.agent.AgentConfiguration()
    sensors = [make_sensor("rgb", 0), make_sensor("depth", 0)]
    sensors.extend(make_sensor("rgb", heading) for heading in HEADINGS_DEGREES[1:])
    sensors.extend(
        make_sensor("depth", heading) for heading in HEADINGS_DEGREES[1:]
    )
    agent_config.sensor_specifications = sensors
    return habitat_sim.Configuration(simulator_config, [agent_config])


def ordered_observations(sensor_observations, normalize_depth=True):
    observations = OrderedDict()
    observations["rgb"] = sensor_observations["rgb"][..., :3]
    depth = sensor_observations["depth"][..., np.newaxis]
    if normalize_depth:
        depth = (depth - POLICY_MIN_DEPTH_M) / (
            POLICY_MAX_DEPTH_M - POLICY_MIN_DEPTH_M
        )
        depth = np.clip(depth, 0.0, 1.0)
    observations["depth"] = depth.astype(np.float32, copy=False)
    for heading in HEADINGS_DEGREES[1:]:
        rgb_key = sensor_uuid("rgb", heading)
        depth_key = sensor_uuid("depth", heading)
        observations[rgb_key] = sensor_observations[rgb_key][..., :3]
        depth = sensor_observations[depth_key][..., np.newaxis]
        if normalize_depth:
            depth = (depth - POLICY_MIN_DEPTH_M) / (
                POLICY_MAX_DEPTH_M - POLICY_MIN_DEPTH_M
            )
            depth = np.clip(depth, 0.0, 1.0)
        observations[depth_key] = depth.astype(np.float32, copy=False)
    return observations


def to_policy_tensors(observations, device):
    tensors = OrderedDict()
    for key, value in observations.items():
        tensor = torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0)
        tensors[key] = tensor.to(device)
    return tensors


def save_debug_frames(policy_observations, metric_observations, output_dir):
    rgb = policy_observations["rgb"]
    depth = metric_observations["depth"][..., 0]
    valid_depth = np.isfinite(depth) & (depth > 0)
    depth_debug = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid_depth):
        upper = float(np.percentile(depth[valid_depth], 99.0))
        depth_debug[valid_depth] = np.clip(
            depth[valid_depth] / max(upper, 1e-6) * 255.0, 0, 255
        ).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, "rgb_heading_000.png"), rgb[..., ::-1])
    cv2.imwrite(os.path.join(output_dir, "depth_heading_000.png"), depth_debug)

    tiles = []
    for heading in HEADINGS_DEGREES:
        rgb_key = sensor_uuid("rgb", heading)
        frame = policy_observations[rgb_key]
        frame = cv2.resize(frame, (168, 168), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame,
            "{:03d} deg".format(heading),
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(frame)
    rows = [np.concatenate(tiles[i : i + 4], axis=1) for i in range(0, 12, 4)]
    cv2.imwrite(os.path.join(output_dir, "rgb_12view_contact_sheet.png"), np.concatenate(rows, axis=0))


def save_observation_arrays(policy_observations, metric_observations, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    save_debug_frames(policy_observations, metric_observations, output_dir)
    np.savez_compressed(
        os.path.join(output_dir, "fixed_12view_policy_observation.npz"),
        **policy_observations
    )
    metric_depth = {
        sensor_uuid("range_m", heading): metric_observations[
            sensor_uuid("depth", heading)
        ]
        for heading in HEADINGS_DEGREES
    }
    np.savez_compressed(
        os.path.join(output_dir, "fixed_12view_depth_meters.npz"),
        **metric_depth
    )


def validate_sensor_observations(policy_observations, metric_observations):
    for heading in HEADINGS_DEGREES:
        rgb_key = sensor_uuid("rgb", heading)
        depth_key = sensor_uuid("depth", heading)
        rgb = policy_observations[rgb_key]
        metric_depth = metric_observations[depth_key][..., 0]
        policy_depth = policy_observations[depth_key][..., 0]
        if rgb.shape != (224, 224, 3) or np.ptp(rgb) == 0:
            raise RuntimeError(
                "invalid or constant RGB sensor output at heading {}".format(
                    heading
                )
            )
        valid_depth = np.isfinite(metric_depth) & (metric_depth > 0)
        if metric_depth.shape != (256, 256) or not np.any(valid_depth):
            raise RuntimeError(
                "depth sensor has no valid metric samples at heading {}".format(
                    heading
                )
            )
        if not np.all(np.isfinite(policy_depth)):
            raise RuntimeError(
                "policy depth contains non-finite samples at heading {}".format(
                    heading
                )
            )
        if np.min(policy_depth) < 0.0 or np.max(policy_depth) > 1.0:
            raise RuntimeError(
                "policy depth is outside [0, 1] at heading {}".format(heading)
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--output-dir", default="outputs/example_scene")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pose-count", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    scene = os.path.abspath(args.scene)
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isfile(scene):
        raise RuntimeError("example scene does not exist: {}".format(scene))
    if args.pose_count < 1:
        raise RuntimeError("pose-count must be positive")
    os.makedirs(output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    started = time.perf_counter()
    simulator = habitat_sim.Simulator(make_configuration(scene))
    init_ms = (time.perf_counter() - started) * 1000.0
    try:
        simulator.seed(args.seed)
        if hasattr(simulator.pathfinder, "seed"):
            simulator.pathfinder.seed(args.seed)
        if not simulator.pathfinder.is_loaded:
            raise RuntimeError("scene navmesh was not loaded")

        positions = []
        pose_records = []
        policy_observations = None
        metric_observations = None
        for pose_index in range(args.pose_count):
            position = simulator.pathfinder.get_random_navigable_point()
            if not simulator.pathfinder.is_navigable(position):
                raise RuntimeError("selected pose is not navigable")
            positions.append(position)
            state = habitat_sim.AgentState()
            state.position = position
            state.rotation = np.quaternion(1.0, 0.0, 0.0, 0.0)
            simulator.initialize_agent(0, state)

            sensor_observations = simulator.get_sensor_observations()
            current_policy = ordered_observations(
                sensor_observations, normalize_depth=True
            )
            current_metric = ordered_observations(
                sensor_observations, normalize_depth=False
            )
            validate_sensor_observations(current_policy, current_metric)
            pose_dir = (
                output_dir
                if pose_index == 0
                else os.path.join(
                    output_dir, "poses", "pose_{:03d}".format(pose_index)
                )
            )
            save_observation_arrays(current_policy, current_metric, pose_dir)
            pose_records.append(
                {
                    "pose_index": pose_index,
                    "position": [float(value) for value in position],
                    "observation_dir": os.path.relpath(pose_dir, output_dir),
                }
            )
            if pose_index == 0:
                policy_observations = current_policy
                metric_observations = current_metric

        reset_state = habitat_sim.AgentState()
        reset_state.position = positions[0]
        reset_state.rotation = np.quaternion(1.0, 0.0, 0.0, 0.0)
        simulator.initialize_agent(0, reset_state)

        action_results = []
        for action in ("move_forward", "turn_left", "turn_right"):
            action_started = time.perf_counter()
            action_observation = simulator.step(action)
            action_results.append(
                {
                    "action": action,
                    "elapsed_ms": (time.perf_counter() - action_started) * 1000.0,
                    "collided": bool(action_observation.get("collided", False)),
                    "rgb_shape": list(action_observation["rgb"].shape),
                    "depth_shape": list(action_observation["depth"].shape),
                }
            )

        depth_ranges = {}
        for heading in HEADINGS_DEGREES:
            key = sensor_uuid("depth", heading)
            depth = metric_observations[key][..., 0]
            valid = np.isfinite(depth) & (depth > 0)
            depth_ranges[key] = {
                "valid_pixels": int(np.count_nonzero(valid)),
                "minimum": float(np.min(depth[valid])) if np.any(valid) else None,
                "maximum": float(np.max(depth[valid])) if np.any(valid) else None,
            }

        metadata = {
            "status": "PASS",
            "test_kind": "ENGINEERING_SMOKE_TEST_ONLY",
            "seed": args.seed,
            "scene": scene,
            "scene_sha256": sha256_file(scene),
            "scene_format": ".glb",
            "navmesh_loaded": True,
            "navigable_area": float(simulator.pathfinder.navigable_area),
            "agent_position": [float(value) for value in positions[0]],
            "agent_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "camera_height_m": CAMERA_HEIGHT_M,
            "horizontal_fov_degrees": HORIZONTAL_FOV_DEGREES,
            "headings_degrees": list(HEADINGS_DEGREES),
            "pose_count": args.pose_count,
            "poses": pose_records,
            "heading_convention": "positive Habitat sensor yaw; original ETP input order",
            "rgb_shape": list(policy_observations["rgb"].shape),
            "depth_shape": list(policy_observations["depth"].shape),
            "policy_depth": {
                "normalized": True,
                "minimum_m": POLICY_MIN_DEPTH_M,
                "maximum_m": POLICY_MAX_DEPTH_M,
            },
            "sensor_validation": "all 12 RGB/depth views valid",
            "depth_ranges_m": depth_ranges,
            "simulator_init_ms": init_ms,
            "actions": action_results,
        }
        metadata_path = os.path.join(output_dir, "example_scene_metadata.json")
        with open(metadata_path, "w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=False)
        print(json.dumps(metadata, indent=2, sort_keys=False))
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
