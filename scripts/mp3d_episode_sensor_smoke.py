#!/usr/bin/env python3
"""Formal, policy-free sensor smoke at an R2R episode's exact reset pose."""

from __future__ import print_function

import argparse
import json
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", default="53")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="run_r2r/iter_train.yaml")
    args = parser.parse_args()

    from habitat_baselines.common.environments import get_env_class
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.common.env_utils import construct_envs
    from vlnce_baselines.common.r2r_runtime import (
        initialize_runtime_config,
        prepare_r2r_evaluation_environment,
    )
    from vlnce_baselines.evaluation_sensor_diagnostics import (
        invalid_sensor_reason,
        summarize_observations,
    )
    from vlnce_baselines.diagnostic_serialization import json_safe

    options = [
        "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]",
        "TORCH_GPU_ID", "0", "GPU_NUMBERS", "1", "NUM_ENVIRONMENTS", "1",
        "EVAL.SPLIT", "val_unseen", "EVAL.EPISODE_IDS", "['{}']".format(args.episode_id),
    ]
    config = get_config(args.config, options)
    # Match run.py's launcher-owned runtime setup, then the evaluator's exact
    # pre-environment sensor configuration without constructing a policy/trainer.
    initialize_runtime_config(config, local_rank=0)
    config = prepare_r2r_evaluation_environment(config)
    envs = construct_envs(
        config, get_env_class(config.ENV_NAME),
        episodes_allowed=[str(args.episode_id)], auto_reset_done=False,
    )
    try:
        observations = envs.reset()
        episode = envs.current_episodes()[0]
        pose = envs.call(["get_pos_ori"])[0]
        views = summarize_observations(observations[0], include_hash=True)
        rgb = [item for key, item in views.items() if key == "rgb" or key.startswith("rgb_")]
        depth = [item for key, item in views.items() if key == "depth" or key.startswith("depth_")]
        report = {
            "episode_id": str(episode.episode_id), "scene": episode.scene_id,
            "start_position": list(pose[0]), "start_rotation": list(pose[1]),
            "views": views,
            "rgb_nonzero_count": sum(item["max"] > 0 for item in rgb),
            "depth_valid_count": sum(item["valid_fraction"] > 0 for item in depth),
            "rgb_distinct_hashes": len(set(item["hash"] for item in rgb)),
            "depth_distinct_hashes": len(set(item["hash"] for item in depth)),
        }
        report["valid"] = invalid_sensor_reason(views) is None
        parent = os.path.dirname(os.path.abspath(args.output))
        if not os.path.exists(parent):
            os.makedirs(parent)
        with open(args.output, "w") as handle:
            json.dump(json_safe(report), handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps(json_safe(report), sort_keys=True))
        if not report["valid"]:
            raise RuntimeError(
                "Invalid Habitat sensor observations: RGB/Depth frames are zero. "
                "Check renderer/GL environment."
            )
    finally:
        envs.close()


if __name__ == "__main__":
    main()
