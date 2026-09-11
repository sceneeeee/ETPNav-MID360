#!/usr/bin/env python3
"""Regress the provider against the pre-refactor sample-scene golden output."""

import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict
from types import SimpleNamespace

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
)
def tensor_fingerprint(value):
    if isinstance(value, list):
        tensors = value
    else:
        tensors = [value]
    cpu_tensors = [tensor.detach().cpu().contiguous() for tensor in tensors]
    flattened = torch.cat([tensor.double().reshape(-1) for tensor in cpu_tensors])
    digest = hashlib.sha256()
    for tensor in cpu_tensors:
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return OrderedDict(
        [
            ("shapes", [list(tensor.shape) for tensor in tensors]),
            ("sha256", digest.hexdigest()),
            ("sum", float(flattened.sum().item())),
            ("l2", float(torch.norm(flattened).item())),
            ("minimum", float(flattened.min().item())),
            ("maximum", float(flattened.max().item())),
        ]
    )


def compare_fingerprints(actual, expected, tolerance):
    differences = OrderedDict()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if actual_value["shapes"] != expected_value["shapes"]:
            raise AssertionError(
                "shape mismatch for {}: {} != {}".format(
                    key, actual_value["shapes"], expected_value["shapes"]
                )
            )
        if actual_value["sha256"] != expected_value["sha256"]:
            raise AssertionError(
                "tensor bytes mismatch for {}: {} != {}".format(
                    key,
                    actual_value["sha256"],
                    expected_value["sha256"],
                )
            )
        numeric_differences = {
            field: abs(actual_value[field] - expected_value[field])
            for field in ("sum", "l2", "minimum", "maximum")
        }
        if max(numeric_differences.values()) > tolerance:
            raise AssertionError(
                "fingerprint mismatch for {}: {}".format(
                    key, numeric_differences
                )
            )
        differences[key] = numeric_differences
    return differences


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observation-npz",
        default="outputs/example_scene/fixed_12view_policy_observation.npz",
    )
    parser.add_argument("--output", default="outputs/example_scene/provider_equivalence.json")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--golden",
        default="tests/fixtures/example_scene_waypoint_golden.json",
    )
    parser.add_argument("--print-current", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("equivalence test requires the validated CUDA path")
    policy_args = SimpleNamespace(
        config=DEFAULT_CONFIG,
        checkpoint=DEFAULT_CHECKPOINT,
        waypoint_checkpoint=DEFAULT_WAYPOINT,
        ddppo_checkpoint=DEFAULT_DDPPO,
        episodes=1,
    )
    result = OrderedDict()
    config = check_config(policy_args, result)
    policy, waypoint = load_policy(config, policy_args, device, result)
    observations = panorama_observations(device, cli.observation_npz)

    with torch.no_grad():
        torch.manual_seed(cli.seed)
        provider = policy.net(
            mode="waypoint",
            waypoint_predictor=waypoint,
            observations=observations,
            in_train=False,
        )
    fingerprint_keys = (
        "cand_rgb",
        "cand_depth",
        "cand_angle_fts",
        "pano_rgb",
        "pano_depth",
        "pano_angle_fts",
    )
    fingerprints = OrderedDict(
        (key, tensor_fingerprint(provider[key])) for key in fingerprint_keys
    )
    report = OrderedDict(
        [
            ("status", "PASS"),
            ("seed", cli.seed),
            ("observation", cli.observation_npz),
            ("tolerance", cli.tolerance),
            ("candidate_count", len(provider["cand_angles"][0])),
            ("candidate_angles", provider["cand_angles"][0]),
            ("candidate_distances", provider["cand_distances"][0]),
            ("candidate_img_idxes", provider["cand_img_idxes"][0].tolist()),
            ("panorama_img_idxes", provider["pano_img_idxes"].tolist()),
            ("fingerprints", fingerprints),
        ]
    )
    if cli.print_current:
        print(json.dumps(report, indent=2))
        return
    with open(cli.golden, "r") as handle:
        golden = json.load(handle, object_pairs_hook=OrderedDict)
    for key in (
        "candidate_angles",
        "candidate_distances",
        "candidate_img_idxes",
        "panorama_img_idxes",
    ):
        if report[key] != golden[key]:
            raise AssertionError(
                "golden candidate mismatch for {}: {} != {}".format(
                    key, report[key], golden[key]
                )
            )
    differences = compare_fingerprints(
        fingerprints, golden["fingerprints"], cli.tolerance
    )
    report["golden"] = cli.golden
    report["maximum_fingerprint_differences"] = differences
    rendered = json.dumps(report, indent=2)
    print(rendered)
    output_dir = os.path.dirname(cli.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(cli.output, "w") as handle:
        handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
