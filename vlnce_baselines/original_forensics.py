"""Opt-in, small JSON evidence captured at the Original ETP provider seam."""

import hashlib
import json
import os

import torch

from vlnce_baselines.diagnostic_serialization import json_safe
from vlnce_baselines.evaluation_sensor_diagnostics import (
    compact_layer_comparison,
    summarize_observations,
)


def _scalar(value):
    return float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)


def _view_summary(value, include_depth_fields=False):
    cpu = value.detach().cpu().contiguous()
    values = cpu.float()
    result = {
        "shape": list(cpu.shape), "dtype": str(cpu.dtype),
        "min": _scalar(values.min()), "max": _scalar(values.max()),
        "mean": _scalar(values.mean()),
        "hash": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
    }
    if include_depth_fields:
        result.update({
            "median": _scalar(values.median()),
            "valid_fraction": _scalar((values > 0).float().mean()),
        })
    return result


def capture(path, observations, logits, output, distance_step_m=0.25):
    """Write only first-step scalars; never persist image/depth tensors."""
    if not path:
        return
    if os.path.exists(path):
        with open(path) as handle:
            existing = json.load(handle)
        # The provider can be called repeatedly during a rollout.  Preserve the
        # first provider output, while allowing the trainer to have written the
        # two earlier sensor layers first.
        if "cand_angles" in existing:
            return
    # These are the predictor's direct, pre-NMS scores (logits), not a
    # post-processing probability map.
    score_map = logits.reshape(logits.shape[0], 120, 12)[0].detach().cpu()
    flat = score_map.reshape(-1)
    top_count = min(20, flat.numel())
    scores, indices = torch.topk(flat, top_count)
    top_bins = []
    for score, index in zip(scores, indices):
        angle_bin = int(index.item() // 12)
        distance_bin = int(index.item() % 12)
        top_bins.append({"angle_bin": angle_bin, "distance_bin": distance_bin,
                         "distance_m": (distance_bin + 1) * distance_step_m,
                         "score": float(score.item())})
    distance_bins = []
    for distance_bin in range(12):
        column = score_map[:, distance_bin]
        distance_bins.append({"distance_bin": distance_bin,
                              "distance_m": (distance_bin + 1) * distance_step_m,
                              "max_score": float(column.max().item()),
                              "mean_score": float(column.mean().item())})
    views = []
    angle_count = 0
    for depth_key, depth in observations.items():
        if "depth" not in depth_key:
            continue
        rgb_key = depth_key.replace("depth", "rgb")
        packed_index = (12 - angle_count) % 12
        heading = 0 if depth_key == "depth" else int(depth_key.rsplit("_", 1)[1])
        views.append({"habitat_observation_key": depth_key,
                      "habitat_observation_index": angle_count,
                      "physical_heading_degrees": heading,
                      "etp_packed_index": packed_index,
                      "waypoint_predictor_index": packed_index,
                      "rgb": _view_summary(observations[rgb_key][0]),
                      "depth": _view_summary(depth[0], include_depth_fields=True)})
        angle_count += 1
    existing = {}
    if os.path.exists(path):
        with open(path) as handle:
            existing = json.load(handle)
    provider_views = summarize_observations(observations, batch_index=0, include_hash=True)
    raw = {"cand_angles": [float(x) for x in output["cand_angles"][0]],
           "cand_distances": [float(x) for x in output["cand_distances"][0]],
           "cand_img_idxes": [int(x) for x in output["cand_img_idxes"][0]],
           "heatmap": {"shape": [120, 12], "score_type": "raw predictor pre-NMS logits",
                       "top_20_bins": top_bins, "distance_bin_scores": distance_bins},
           "observation_audit": {"source": "batch passed directly to OriginalETPWaypointProvider.predict",
                                 "depth_unit": "as supplied by Habitat/policy batch; inspect numeric range below",
                                 "depth_normalization": "no additional normalization in OriginalETPWaypointProvider",
                                 "views": views},
           "sensor_layers": existing.get("sensor_layers", {}),
           "provider_input": provider_views}
    layers = raw["sensor_layers"]
    if "raw_habitat" in layers and "batched_pre_policy" in layers:
        raw["sensor_layer_comparison"] = compact_layer_comparison(
            layers["raw_habitat"], layers["batched_pre_policy"], provider_views
        )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(json_safe(raw), handle, indent=2, sort_keys=True)
        handle.write("\n")
