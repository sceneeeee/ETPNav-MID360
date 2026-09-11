"""Small, evaluation-only Habitat sensor checks and scalar diagnostics."""

from __future__ import print_function

import hashlib
import json
import os

import numpy as np
import torch

from vlnce_baselines.diagnostic_serialization import json_safe


def _as_cpu_array(value, batch_index=None):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if batch_index is not None:
        value = value[batch_index]
    return np.ascontiguousarray(value)


def _is_sensor_key(key, kind):
    return key == kind or key.startswith(kind + "_")


def summarize_observations(observations, batch_index=None, include_hash=False):
    """Return scalar-only RGB/depth statistics, never sensor arrays."""
    summary = {}
    for key in sorted(observations):
        if not (_is_sensor_key(key, "rgb") or _is_sensor_key(key, "depth")):
            continue
        array = _as_cpu_array(observations[key], batch_index)
        values = array.astype(np.float64, copy=False)
        record = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }
        if _is_sensor_key(key, "depth"):
            record["median"] = float(np.median(values))
            record["valid_fraction"] = float(np.mean(values > 0))
        if include_hash:
            record["hash"] = hashlib.sha256(array.tobytes()).hexdigest()
        summary[key] = record
    return summary


def invalid_sensor_reason(summary):
    """Return a reason for an all-black/all-invalid panorama, else None."""
    rgb = [record for key, record in summary.items() if _is_sensor_key(key, "rgb")]
    depth = [record for key, record in summary.items() if _is_sensor_key(key, "depth")]
    if rgb and all(record["min"] == 0 and record["max"] == 0 for record in rgb):
        return "RGB"
    if depth and all(record["valid_fraction"] == 0 for record in depth):
        return "Depth"
    return None


def require_valid_sensor_observations(observations, batch_index=None):
    summary = summarize_observations(observations, batch_index=batch_index)
    reason = invalid_sensor_reason(summary)
    if reason is not None:
        raise RuntimeError(
            "Invalid Habitat sensor observations: {0} frames are zero. "
            "Check renderer/GL environment.".format(reason)
        )
    return summary


def write_forensic_layers(path, raw, batched):
    """Persist the two pre-provider layers to the opt-in forensic JSON."""
    if not path:
        return
    payload = {}
    if os.path.exists(path):
        with open(path) as handle:
            payload = json.load(handle)
    payload["sensor_layers"] = {"raw_habitat": raw, "batched_pre_policy": batched}
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "w") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def compact_layer_comparison(raw, batched, provider):
    """Match per-sensor scalar max/mean values across the three boundaries."""
    result = []
    for key in sorted(set(raw) | set(batched) | set(provider)):
        if not (_is_sensor_key(key, "rgb") or _is_sensor_key(key, "depth")):
            continue
        row = {"sensor_key": key}
        for label, layer in (("raw_habitat", raw), ("batched", batched), ("provider", provider)):
            record = layer.get(key)
            row[label] = None if record is None else {
                "max": record["max"], "mean": record["mean"]
            }
        result.append(row)
    return result
