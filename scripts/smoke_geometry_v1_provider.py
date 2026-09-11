#!/usr/bin/env python3
"""One-observation Geometry V1 provider smoke; never runs navigation."""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from types import SimpleNamespace

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
from vlnce_baselines.perception.ray_carving_representation import (  # noqa: E402
    EpistemicState,
)
from vlnce_baselines.ss_trainer_ETP import RLTrainer  # noqa: E402
from vlnce_baselines.waypoint_provider.mid360_geometry_v1 import (  # noqa: E402
    GEOMETRY_V1_DESIGN_ID,
    GEOMETRY_V1_MODE,
    MID360GeometryV1Config,
    MID360GeometryV1WaypointProvider,
    default_mid360_v1_config_path,
)
from vlnce_baselines.waypoint_provider.types import validate_waypoint_output  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observation",
        default="outputs/geometry_v1_integration/sensor_capture/"
                "fixed_12view_policy_observation.npz",
    )
    parser.add_argument(
        "--output", default="outputs/geometry_v1_integration/provider_smoke.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Geometry V1 provider smoke requires CUDA")
    device = torch.device("cuda:0")
    policy_args = SimpleNamespace(
        config=DEFAULT_CONFIG,
        checkpoint=DEFAULT_CHECKPOINT,
        waypoint_checkpoint=DEFAULT_WAYPOINT,
        ddppo_checkpoint=DEFAULT_DDPPO,
        episodes=1,
    )
    initialization = OrderedDict()
    etp_config = check_config(policy_args, initialization)
    policy, _ = load_policy(etp_config, policy_args, device, initialization)
    config = MID360GeometryV1Config.from_yaml(default_mid360_v1_config_path())
    provider = MID360GeometryV1WaypointProvider(
        rgb_encoder=policy.net.rgb_encoder,
        depth_encoder=policy.net.depth_encoder,
        rgb_pool=policy.net.space_pool_rgb,
        depth_pool=policy.net.space_pool_depth,
        pano_angle_features=policy.net.pano_angle_fts,
        pano_image_indexes=policy.net.pano_img_idxes,
        config=config,
    )
    observations = panorama_observations(device, args.observation)
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        output = policy.net(
            mode="waypoint", waypoint_provider=provider,
            observations=observations, in_train=False,
        )
    synchronize(device)
    total_ms = (time.perf_counter() - started) * 1000.0
    validate_waypoint_output(output)
    # Exercise the exact trainer adapter which marks candidate features for
    # GraphMap insertion.  This remains valid for a zero-candidate observation.
    adapter = object.__new__(RLTrainer)
    adapter.envs = SimpleNamespace(num_envs=1)
    vp_inputs = adapter._vp_feature_variable(output)
    with torch.no_grad():
        pano_embeds, _ = policy.net(mode="panorama", **vp_inputs)
    graphmap_candidate_embed_count = int(
        pano_embeds[0][vp_inputs["nav_types"][0] == 1].shape[0]
    )

    debug = provider.last_debug[0]
    grid = debug["grid"]
    state = grid.voxel_state()
    rays = debug["ray_result"].rays
    candidates = debug["candidates"]
    polar = debug["polar_evidence"]
    if not all(np.all(np.isfinite(ray.direction_xyz)) and
               np.isfinite(ray.range_m) for ray in rays):
        raise RuntimeError("V1 emitted non-finite rays")
    if not all(polar.safe[candidate.angle_index, candidate.distance_index]
               for candidate in candidates):
        raise RuntimeError("V1 emitted a safety-invalid candidate")
    feature_tensors = (
        output["pano_rgb"], output["pano_depth"],
        output["cand_rgb"][0], output["cand_depth"][0],
        output["cand_angle_fts"][0],
    )
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in feature_tensors):
        raise RuntimeError("V1 output contains non-finite features")
    if len(candidates) > config.maximum_candidates:
        raise RuntimeError("V1 candidate cap exceeded")

    result = OrderedDict([
        ("status", "PASS"),
        ("test_kind", "SENSOR_PROVIDER_ONLY_NO_NAVIGATION"),
        ("provider", GEOMETRY_V1_MODE),
        ("design_id", GEOMETRY_V1_DESIGN_ID),
        ("observation", args.observation),
        ("input_depth_shapes", [list(shape) for shape in debug["ray_result"].input_shapes]),
        ("sampled_ray_count", debug["ray_result"].sampled_count),
        ("retained_ray_count", len(rays)),
        ("hit_ray_count", debug["ray_result"].hit_count),
        ("range_censored_ray_count", debug["ray_result"].censored_count),
        ("invalid_ray_count", debug["ray_result"].invalid_count),
        ("free_voxel_count", int(grid.free_evidence.sum())),
        ("occupied_voxel_count", int(grid.occupied_evidence.sum())),
        ("range_censored_voxel_count", int(grid.range_censored.sum())),
        ("unknown_voxel_count", int(np.count_nonzero(state == EpistemicState.UNKNOWN))),
        ("candidate_count", len(candidates)),
        ("candidate_kinds", debug["candidate_kinds"]),
        ("candidate_angles", [float(value) for value in output["cand_angles"][0]]),
        ("candidate_distances_m", [float(value) for value in output["cand_distances"][0]]),
        ("candidate_image_indexes", [int(value) for value in output["cand_img_idxes"][0]]),
        ("candidate_rgb_shape", list(output["cand_rgb"][0].shape)),
        ("candidate_depth_shape", list(output["cand_depth"][0].shape)),
        ("candidate_location_shape", list(output["cand_angle_fts"][0].shape)),
        ("panorama_rgb_shape", list(output["pano_rgb"].shape)),
        ("panorama_depth_shape", list(output["pano_depth"].shape)),
        ("graphmap_candidate_embed_count", graphmap_candidate_embed_count),
        ("provider_total_ms", total_ms),
        ("stage_timings_ms", provider.last_timings[0]),
        ("legacy_dense_score_or_nms_used", False),
        ("navigation_executed", False),
    ])
    parent = os.path.dirname(args.output)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
