#!/usr/bin/env python3
"""CPU-only structural study for Geometry V1; never imports Habitat.

This deliberately evaluates *failure modes*, not navigation outcomes.  The
occupancy patterns use the production 120x12/0.25 m contract and the same
inflation/traversability code as Geometry V0.  It compares three minimal
geometry-only families:

* endpoint_clearance: rank free endpoints by Euclidean obstacle clearance;
* clearance_ridge: retain only strict radial clearance maxima;
* support_clearance: rank clearance times circular angular free support.

The script is an analysis artifact, not a provider implementation.
"""
import json
import math
from pathlib import Path

import numpy as np

from vlnce_baselines.perception.local_traversability import (
    TraversabilityConfig, compute_local_traversability,
)
from vlnce_baselines.waypoint_provider.mid360_geometry import (
    select_geometry_candidates,
)


CFG = TraversabilityConfig(120, 12, 0.25, 0.10, 0.02, 0.08, 1.60, 0.05)
OUT = Path("outputs/geometry_v1_design/synthetic_results.json")


def _points_from_predicate(predicate):
    """Dense vertical obstacle samples whose horizontal cells satisfy predicate."""
    xy = []
    axis = np.arange(-3.2, 3.201, 0.05)
    for x in axis:
        for z in axis:
            if predicate(x, z):
                xy.append((x, z))
    if not xy:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray([(x, 0.50, z) for x, z in xy], dtype=np.float32)


def scenes():
    # x is right, z is backward; forward is negative z.
    return {
        "open_space": _points_from_predicate(lambda x, z: False),
        "straight_corridor": _points_from_predicate(lambda x, z: abs(x) >= 0.75),
        "approaching_wall": _points_from_predicate(lambda x, z: z <= -1.8),
        "doorway": _points_from_predicate(
            lambda x, z: abs(z + 1.5) < 0.03 and abs(x) >= 0.35
        ),
        "t_junction": _points_from_predicate(
            lambda x, z: (abs(x) >= 0.60 and z > -1.5) or z <= -1.5
        ),
        "asymmetric_obstacle": _points_from_predicate(
            lambda x, z: x > 0.25 and -2.6 < z < -0.35
        ),
        "narrow_passage": _points_from_predicate(lambda x, z: abs(x) >= 0.27),
        "mixed_range": _points_from_predicate(
            lambda x, z: (z <= -0.9 and x < -0.2) or (z <= -2.6 and x >= -0.2)
        ),
    }


def clearance_map(occupancy, resolution):
    """Exact CPU EDT for this tiny grid; inf means no observed obstacle."""
    rows, cols = np.nonzero(occupancy)
    size = occupancy.shape[0]
    if not len(rows):
        return np.full(occupancy.shape, np.inf, dtype=np.float64)
    rr, cc = np.indices((size, size))
    squared = (rr[..., None] - rows) ** 2 + (cc[..., None] - cols) ** 2
    return np.sqrt(np.min(squared, axis=-1)) * resolution


def endpoint_samples(clearance, resolution):
    center = clearance.shape[0] // 2
    values = np.zeros((120, 12), dtype=np.float64)
    for a in range(120):
        angle = a / 120.0 * 2.0 * math.pi
        for r in range(12):
            d = (r + 1) * 0.25
            col = int(round(math.sin(angle) * d / resolution)) + center
            row = int(round(-math.cos(angle) * d / resolution)) + center
            values[a, r] = clearance[row, col]
    return values


def support_score(free, clearance):
    """Dimensionless product: normalized endpoint clearance × local support."""
    # A 5-bin radius is inherited from angular NMS, not tuned here.
    support = sum(np.roll(free, shift, axis=0) for shift in range(-5, 6)) / 11.0
    finite = np.where(np.isfinite(clearance), clearance, 1.0)
    # Per-radius angular normalization removes a metric coefficient.  In an
    # obstacle-free annulus every valid cell consequently ties, exposing the
    # required deterministic-tie policy rather than hiding it.
    denom = np.maximum(np.max(np.where(free, finite, 0.0), axis=0), 1e-12)
    return np.where(free, (finite / denom[None, :]) * support, 0.0)


def ridge_score(free, clearance):
    finite = np.where(np.isfinite(clearance), clearance, 1e9)
    result = np.zeros_like(finite)
    # Strict interior extrema only: neither a range-boundary nor a flat region
    # may claim to be a geometry-determined radial optimum.
    mask = (free[:, 1:-1] & (finite[:, 1:-1] > finite[:, :-2])
            & (finite[:, 1:-1] > finite[:, 2:]))
    result[:, 1:-1] = np.where(mask, finite[:, 1:-1], 0.0)
    return result


def summarize(score, free):
    a, r = select_geometry_candidates(score.astype(np.float32), max_candidates=5)
    selected = [{"angle_bin": int(x), "distance_m": round((int(y) + 1) * .25, 2)}
                for x, y in zip(a, r)]
    all_best = []
    for angle in (0, 15, 30, 45, 60, 75, 90, 105):
        valid = np.flatnonzero(free[angle] & np.isfinite(score[angle]) & (score[angle] > 0))
        if len(valid):
            best = valid[np.argmax(score[angle, valid])]
            all_best.append(round((int(best) + 1) * .25, 2))
    return {"selected": selected, "sampled_per_direction_best_m": all_best,
            "unique_sampled_radial_bins": len(set(all_best))}


def main():
    results = {}
    for name, points in scenes().items():
        traversal = compute_local_traversability(points, CFG)
        free = traversal.score > 0
        clearance = endpoint_samples(clearance_map(
            traversal.occupancy, traversal.grid_resolution_m), traversal.grid_resolution_m)
        # A raw inf score is converted to a common finite value only so the
        # legacy deterministic selector can expose its row-major tie behavior.
        endpoint = np.where(free, np.where(np.isfinite(clearance), clearance, 1e6), 0)
        results[name] = {
            "endpoint_clearance": summarize(endpoint, free),
            "clearance_ridge": summarize(ridge_score(free, clearance), free),
            "support_clearance": summarize(support_score(free, clearance), free),
        }
    # I2/I3 check on a non-symmetric scene.  A +90-degree map rotation is
    # exactly +30 of the 120 angular bins.  It is deliberately an occupancy
    # test rather than a navigation test.
    original = scenes()["asymmetric_obstacle"]
    rotated = original.copy()
    rotated[:, [0, 2]] = np.stack((-original[:, 2], original[:, 0]), axis=1)
    rotation_errors = {}
    for label, cloud in (("original", original), ("rotated", rotated)):
        traversal = compute_local_traversability(cloud, CFG)
        free = traversal.score > 0
        clearance = endpoint_samples(clearance_map(
            traversal.occupancy, traversal.grid_resolution_m), traversal.grid_resolution_m)
        endpoint = np.where(free, np.where(np.isfinite(clearance), clearance, 1e6), 0)
        scores = {
            "endpoint_clearance": endpoint,
            "clearance_ridge": ridge_score(free, clearance),
            "support_clearance": support_score(free, clearance),
        }
        if label == "original":
            reference = scores
        else:
            for name, score in scores.items():
                rotation_errors[name] = float(np.max(np.abs(
                    score - np.roll(reference[name], 30, axis=0))))
    results["rotation_check_90_degrees"] = rotation_errors
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
