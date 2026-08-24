#!/usr/bin/env python3
"""Staged smoke tests for the original ETPNav R2R baseline.

The policy/planner checks use synthetic sensor tensors, so they do not need
licensed Matterport3D scenes.  The simulator check accepts any local Habitat
scene and exercises the real 0.1.7 renderer, reset, and one action.
"""

import argparse
import gc
import json
import os
import resource
import sys
import time
from collections import OrderedDict

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)


DEFAULT_CHECKPOINT = "data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
DEFAULT_WAYPOINT = "data/wp_pred/check_cwp_bestdist_hfov90"
DEFAULT_DDPPO = "data/ddppo-models/gibson-2plus-resnet50.pth"
DEFAULT_CONFIG = "run_r2r/iter_train.yaml"
PANORAMA_VIEW_COUNT = 12


def elapsed_ms(start):
    return (time.perf_counter() - start) * 1000.0


def materialized_file(path):
    if not os.path.isfile(path):
        raise RuntimeError("missing asset: {}".format(path))
    with open(path, "rb") as handle:
        prefix = handle.read(48)
    if prefix.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError("asset is still a Git LFS pointer: {}".format(path))
    return os.path.getsize(path)


def current_rss_mib():
    with open("/proc/self/status", "r") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return None


def check_assets(args, result):
    paths = OrderedDict(
        [
            ("checkpoint", args.checkpoint),
            ("waypoint", args.waypoint_checkpoint),
            ("ddppo", args.ddppo_checkpoint),
            ("connectivity", "data/connectivity_graphs.pkl"),
            (
                "r2r_val_unseen",
                "data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/"
                "val_unseen/val_unseen_bertidx.json.gz",
            ),
        ]
    )
    result["assets_bytes"] = {
        name: materialized_file(path) for name, path in paths.items()
    }


def check_config(args, result):
    import habitat
    import habitat_sim
    import habitat_baselines
    import habitat_extensions  # noqa: F401
    import vlnce_baselines  # noqa: F401
    from vlnce_baselines.config.default import get_config

    opts = [
        "SIMULATOR_GPU_IDS",
        "[0]",
        "TORCH_GPU_IDS",
        "[0]",
        "TORCH_GPU_ID",
        "0",
        "GPU_NUMBERS",
        "1",
        "NUM_ENVIRONMENTS",
        "1",
        "EVAL.EPISODE_COUNT",
        str(args.episodes),
        "MODEL.pretrained_path",
        "None",
        "VIDEO_OPTION",
        "[]",
    ]
    config = get_config(args.config, opts)
    result["versions"] = {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "habitat": getattr(habitat, "__version__", "0.1.7"),
        "habitat_sim": "0.1.7",
        "habitat_baselines_source": habitat_baselines.__file__,
    }
    result["config"] = {
        "environments": config.NUM_ENVIRONMENTS,
        "gpu_numbers": config.GPU_NUMBERS,
        "episodes": config.EVAL.EPISODE_COUNT,
        "split": config.EVAL.SPLIT,
        "pretrained_path": config.MODEL.pretrained_path,
        "dataset_path": config.TASK_CONFIG.DATASET.DATA_PATH,
        "scenes_dir": config.TASK_CONFIG.DATASET.SCENES_DIR,
    }
    return config


def simulator_smoke(scene, result):
    import habitat_sim

    materialized_file(scene)
    sim_config = habitat_sim.SimulatorConfiguration()
    sim_config.scene_id = scene
    sim_config.gpu_device_id = 0

    rgb = habitat_sim.SensorSpec()
    rgb.uuid = "color_sensor"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [64, 64]
    rgb.position = [0.0, 1.5, 0.0]

    depth = habitat_sim.SensorSpec()
    depth.uuid = "depth_sensor"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [64, 64]
    depth.position = [0.0, 1.5, 0.0]

    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [rgb, depth]
    config = habitat_sim.Configuration(sim_config, [agent_config])

    started = time.perf_counter()
    simulator = habitat_sim.Simulator(config)
    init_ms = elapsed_ms(started)
    try:
        reset_obs = simulator.reset()
        step_started = time.perf_counter()
        step_obs = simulator.step("move_forward")
        result["simulator"] = {
            "scene": scene,
            "init_ms": init_ms,
            "reset": sorted(reset_obs.keys()),
            "step_ms": elapsed_ms(step_started),
            "step": sorted(step_obs.keys()),
            "rgb_shape": list(step_obs["color_sensor"].shape),
            "depth_shape": list(step_obs["depth_sensor"].shape),
            "pathfinder_loaded": bool(simulator.pathfinder.is_loaded),
        }
    finally:
        simulator.close()


def load_policy(config, args, device, result):
    from gym import spaces
    from transformers import logging as transformers_logging
    from vlnce_baselines.models.Policy_ViewSelection_ETP import (
        PolicyViewSelectionETP,
    )
    from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM

    transformers_logging.set_verbosity_error()

    observation_space = spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
            "depth": spaces.Box(
                0.0, 1.0, shape=(256, 256, 1), dtype=np.float32
            ),
        }
    )
    action_space = spaces.Discrete(6)

    config.defrost()
    config.MODEL.pretrained_path = None
    config.MODEL.DEPTH_ENCODER.ddppo_checkpoint = args.ddppo_checkpoint
    config.MODEL.TORCH_GPU_ID = 0
    config.freeze()

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    policy = PolicyViewSelectionETP.from_config(
        config=config,
        observation_space=observation_space,
        action_space=action_space,
    )
    policy.to(device)
    result["policy_init_ms"] = elapsed_ms(started)

    checkpoint_started = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    first_key = next(iter(state_dict))
    wrapped = first_key.startswith("net.module.")
    if wrapped:
        device_id = device.index if device.index is not None else 0
        policy.net = torch.nn.DataParallel(
            policy.net, device_ids=[device_id], output_device=device_id
        )
    incompatible = policy.load_state_dict(state_dict, strict=False)
    if wrapped:
        policy.net = policy.net.module
    result["checkpoint"] = {
        "iteration": checkpoint.get("iteration"),
        "load_ms": elapsed_ms(checkpoint_started),
        "state_tensors": len(state_dict),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint does not exactly match policy: missing={}, unexpected={}".format(
                list(incompatible.missing_keys),
                list(incompatible.unexpected_keys),
            )
        )
    del checkpoint
    del state_dict
    gc.collect()

    waypoint = BinaryDistPredictor_TRM(device=device)
    waypoint_state = torch.load(args.waypoint_checkpoint, map_location="cpu")
    waypoint.load_state_dict(waypoint_state["predictor"]["state_dict"])
    waypoint.to(device)
    del waypoint_state

    policy.eval()
    waypoint.eval()
    result["policy_parameters"] = sum(p.numel() for p in policy.parameters())
    result["waypoint_parameters"] = sum(p.numel() for p in waypoint.parameters())
    result["rss_mib_after_policy"] = current_rss_mib()
    if device.type == "cuda":
        result["gpu_mib_after_policy"] = torch.cuda.memory_allocated(device) / 2 ** 20
        result["gpu_peak_mib_after_policy"] = (
            torch.cuda.max_memory_allocated(device) / 2 ** 20
        )
    return policy, waypoint


def panorama_observations(device):
    observations = OrderedDict()
    heading_step = 360 // PANORAMA_VIEW_COUNT
    headings = [
        "" if angle == 0 else "_{}".format(angle)
        for angle in range(0, 360, heading_step)
    ]
    for suffix in headings:
        observations["rgb{}".format(suffix)] = torch.zeros(
            (1, 224, 224, 3), dtype=torch.uint8, device=device
        )
        observations["depth{}".format(suffix)] = torch.zeros(
            (1, 256, 256, 1), dtype=torch.float32, device=device
        )
    return observations


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def waypoint_smoke(policy, waypoint, device, args, result):
    observations = panorama_observations(device)
    timings = []
    output = None
    with torch.no_grad():
        for _ in range(args.waypoint_runs):
            synchronize(device)
            started = time.perf_counter()
            output = policy.net(
                mode="waypoint",
                waypoint_predictor=waypoint,
                observations=observations,
                in_train=False,
            )
            synchronize(device)
            timings.append(elapsed_ms(started))
    result["waypoint"] = {
        "runs": len(timings),
        "latency_ms": timings,
        "mean_ms": sum(timings) / len(timings),
        "candidate_count": len(output["cand_angles"][0]),
        "candidate_angles": output["cand_angles"][0],
        "candidate_distances": output["cand_distances"][0],
    }
    return output


def planner_smoke(policy, waypoint_output, device, args, result):
    candidate_indexes = waypoint_output["cand_img_idxes"][0]
    panorama_view_count = waypoint_output["pano_rgb"].shape[1]
    candidate_mask = np.zeros(panorama_view_count, dtype=np.bool_)
    candidate_mask[candidate_indexes] = True

    rgb_features = torch.cat(
        [
            waypoint_output["cand_rgb"][0],
            waypoint_output["pano_rgb"][0][~candidate_mask],
        ],
        dim=0,
    ).unsqueeze(0)
    depth_features = torch.cat(
        [
            waypoint_output["cand_depth"][0],
            waypoint_output["pano_depth"][0][~candidate_mask],
        ],
        dim=0,
    ).unsqueeze(0)
    location_features = torch.cat(
        [
            waypoint_output["cand_angle_fts"][0],
            waypoint_output["pano_angle_fts"][~candidate_mask],
        ],
        dim=0,
    ).unsqueeze(0).to(device)
    candidate_count = len(candidate_indexes)
    nav_types = torch.tensor(
        [[1] * candidate_count + [0] * (panorama_view_count - candidate_count)],
        dtype=torch.long,
        device=device,
    )
    view_lens = torch.tensor([panorama_view_count], dtype=torch.long, device=device)
    text_ids = torch.tensor(
        [[101, 2023, 2003, 1037, 3231, 102]], dtype=torch.long, device=device
    )
    text_masks = torch.ones_like(text_ids, dtype=torch.bool)

    with torch.no_grad():
        text_embeddings = policy.net(
            mode="language", txt_ids=text_ids, txt_masks=text_masks
        )
        pano_embeddings, pano_masks = policy.net(
            mode="panorama",
            rgb_fts=rgb_features,
            dep_fts=depth_features,
            loc_fts=location_features,
            nav_types=nav_types,
            view_lens=view_lens,
        )
        current_embedding = torch.sum(
            pano_embeddings * pano_masks.unsqueeze(2), dim=1
        ) / torch.sum(pano_masks, dim=1, keepdim=True)
        ghost_embedding = pano_embeddings[:, 0]
        zeros = torch.zeros_like(current_embedding)
        graph_embeddings = torch.stack(
            [zeros[0], current_embedding[0], ghost_embedding[0]], dim=0
        ).unsqueeze(0)
        graph_steps = torch.tensor([[0, 1, 0]], dtype=torch.long, device=device)
        graph_positions = torch.zeros((1, 3, 7), device=device)
        graph_masks = torch.ones((1, 3), dtype=torch.bool, device=device)
        graph_visited = torch.tensor(
            [[False, True, False]], dtype=torch.bool, device=device
        )
        graph_distances = torch.zeros((1, 3, 3), device=device)

        timings = []
        planner_output = None
        for _ in range(args.planner_runs):
            synchronize(device)
            started = time.perf_counter()
            planner_output = policy.net(
                mode="navigation",
                txt_embeds=text_embeddings,
                txt_masks=text_masks,
                gmap_vp_ids=[[None, "v0", "g0"]],
                gmap_step_ids=graph_steps,
                gmap_img_fts=graph_embeddings,
                gmap_pos_fts=graph_positions,
                gmap_masks=graph_masks,
                gmap_visited_masks=graph_visited,
                gmap_pair_dists=graph_distances,
            )
            synchronize(device)
            timings.append(elapsed_ms(started))

    result["planner"] = {
        "runs": len(timings),
        "latency_ms": timings,
        "mean_ms": sum(timings) / len(timings),
        "global_logits_shape": list(planner_output["global_logits"].shape),
        "selected_index": int(
            planner_output["global_logits"].argmax(dim=1).item()
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--waypoint-checkpoint", default=DEFAULT_WAYPOINT)
    parser.add_argument("--ddppo-checkpoint", default=DEFAULT_DDPPO)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--scene")
    parser.add_argument("--policy", action="store_true")
    parser.add_argument("--waypoint", action="store_true")
    parser.add_argument("--planner", action="store_true")
    parser.add_argument("--waypoint-runs", type=int, default=1)
    parser.add_argument("--planner-runs", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.waypoint or args.planner:
        args.policy = True
    if args.planner:
        args.waypoint = True

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device + (":0" if args.device == "cuda" else ""))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    result = OrderedDict()
    result["status"] = "PASS"
    result["device"] = str(device)
    check_assets(args, result)
    config = check_config(args, result)
    if args.scene:
        simulator_smoke(args.scene, result)
    if args.policy:
        policy, waypoint = load_policy(config, args, device, result)
        waypoint_output = None
        if args.waypoint:
            waypoint_output = waypoint_smoke(
                policy, waypoint, device, args, result
            )
        if args.planner:
            planner_smoke(policy, waypoint_output, device, args, result)
        result["rss_mib_final"] = current_rss_mib()
        result["peak_rss_mib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss / 1024.0
        if device.type == "cuda":
            result["gpu_peak_mib_final"] = (
                torch.cuda.max_memory_allocated(device) / 2 ** 20
            )

    rendered = json.dumps(result, indent=2, sort_keys=False)
    print(rendered)
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
