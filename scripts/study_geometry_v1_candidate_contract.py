#!/usr/bin/env python3
"""CPU-only Geometry V1 candidate-contract fixture study.

The fixtures are height-complete synthetic ray fans.  They exercise the
inactive epistemic representation and compare three *candidate semantics*:
binary dense-map plus legacy bounded NMS, direct clearance ridges, and the
selected direct-plus-explicit-fallback hybrid.  They are not Habitat scenes
and must never be used as navigation metrics.
"""
import json
import math
from pathlib import Path

import numpy as np

from vlnce_baselines.perception.geometry_v1_candidates import (
    ETP_ANGLE_BINS,
    extract_geometry_v1_candidates,
    extract_geometry_v1_structural_candidates,
    polar_evidence_from_grid,
)
from vlnce_baselines.perception.ray_carving_representation import (
    RayCarvingGrid,
    RayObservation,
)
from vlnce_baselines.waypoint_provider.mid360_geometry import (
    select_geometry_candidates,
)


OUT = Path("outputs/geometry_v1_candidate_contract/synthetic_results.json")


def ray(y, angle, distance, kind):
    theta = angle / float(ETP_ANGLE_BINS) * 2.0 * math.pi
    return RayObservation((0.0, y, 0.0),
                          (math.sin(theta), 0.0, -math.cos(theta)),
                          distance, kind)


def fan(distance, kind):
    # 10 cm is deliberately finer than the 25 cm ETP bins: a censor endpoint
    # must not alias into the preceding candidate bin.
    grid = RayCarvingGrid(4.0, 0.0, 2.0, 0.10, 0.10, 1.50)
    grid.integrate_many(
        ray(y, angle, distance(angle), kind(angle))
        for y in np.arange(0.10, 1.51, 0.10)
        for angle in range(ETP_ANGLE_BINS)
    )
    return grid


def full_open():
    """An intentionally rotationally symmetric range-censored open scan."""
    grid = RayCarvingGrid(4.0, 0.0, 2.0, 0.10, 0.10, 1.50)
    grid.integrate_many(
        ray(y, angle, 3.0, "censored")
        for y in np.arange(0.10, 1.51, 0.10)
        for angle in range(ETP_ANGLE_BINS)
    )
    centers = grid.y_min_m + np.arange(grid.ny) * grid.resolution_m
    levels = ((centers >= grid.obstacle_height_min_m) &
              (centers <= grid.obstacle_height_max_m))
    grid.free_evidence[levels] = True
    return grid


def angular_delta(angle, center=0):
    return min((angle - center) % 120, (center - angle) % 120)


def corridor():
    def distance(angle):
        s = abs(math.sin(angle / 120.0 * 2.0 * math.pi))
        return min(3.0, 0.9 / s) if s > 1e-8 else 3.0
    return fan(distance, lambda angle: "hit" if distance(angle) < 3.0 else "censored")


def sector_scene(open_sectors, near_hit=0.75, wall_hit=1.5):
    """A raw-ray abstraction: named sectors are directly seen past a wall."""
    def is_open(angle):
        return any(angular_delta(angle, center) <= width
                   for center, width in open_sectors)
    def distance(angle):
        if is_open(angle):
            return 3.0
        if angular_delta(angle, 0) <= 30:
            return wall_hit
        return near_hit
    def kind(angle):
        return "censored" if is_open(angle) else "hit"
    return fan(distance, kind)


def narrow_passage():
    def distance(angle):
        s = abs(math.sin(angle / 120.0 * 2.0 * math.pi))
        return min(3.0, 0.30 / s) if s > 1e-8 else 3.0
    return fan(distance, lambda angle: "hit" if distance(angle) < 3.0 else "censored")


def large_room():
    # Physical returns at the sensor horizon, rather than a censor boundary.
    grid = fan(lambda _: 3.0, lambda _: "hit")
    centers = grid.y_min_m + np.arange(grid.ny) * grid.resolution_m
    levels = ((centers >= grid.obstacle_height_min_m) &
              (centers <= grid.obstacle_height_max_m))
    # The fixture intentionally certifies the entire interior.  Endpoint hits
    # remain occupied, so this is a uniform room rather than a censor fan.
    grid.free_evidence[levels] = True
    return grid


def mixed_near_far():
    def distance(angle):
        return 1.0 if 0 <= angle < 30 else (2.5 if angle < 60 else 0.75)
    return fan(distance, lambda _: "hit")


def partial_occlusion():
    def distance(angle):
        return 0.75 if angular_delta(angle, 0) <= 5 else 3.0
    return fan(distance, lambda angle: "hit" if angular_delta(angle, 0) <= 5 else "censored")


def candidate_rows(candidates):
    return [{"angle_bin": int(c.angle_index), "distance_m": c.distance_m,
             "kind": c.kind, "rank": float(c.rank)} for c in candidates]


def report_scene(grid):
    polar = polar_evidence_from_grid(grid)
    # This deliberately has no radial preference.  Any returned location is
    # therefore a legacy NMS/tie consequence, not claimed geometry salience.
    dense_score = polar.safe.astype(np.float32)
    a, r = select_geometry_candidates(dense_score, max_candidates=5)
    direct, _ = extract_geometry_v1_structural_candidates(grid)
    hybrid, _ = extract_geometry_v1_candidates(grid)
    return {
        "safe_polar_cells": int(polar.safe.sum()),
        "legacy_dense_binary_before_nms_positive_cells": int((dense_score > 0).sum()),
        "legacy_dense_binary_after_nms": [
            {"angle_bin": int(angle), "distance_m": (int(radius) + 1) * .25}
            for angle, radius in zip(a, r)
        ],
        "direct_structural": candidate_rows(direct),
        "hybrid": candidate_rows(hybrid),
    }


def nms_toys():
    def run(entries):
        score = np.zeros((120, 12), dtype=np.float32)
        for angle, radius, value in entries:
            score[angle, radius] = value
        a, r = select_geometry_candidates(score, max_candidates=5)
        return [{"angle_bin": int(x), "radial_bin": int(y),
                 "distance_m": (int(y) + 1) * .25}
                for x, y in zip(a, r)]
    return {
        "one_heading_all_radial_bins": run([(60, r, float(r + 1)) for r in range(12)]),
        "same_heading_two_end_modes": run([(60, 0, 9.0), (60, 6, 8.0), (60, 11, 10.0)]),
        "near_heading_different_radial_modes": run([
            (60, 11, 10.0), (64, 0, 9.0), (66, 6, 8.0), (70, 0, 7.0),
        ]),
    }


def main():
    scenes = {
        "open_space": full_open,
        "long_corridor": corridor,
        "front_wall": lambda: sector_scene([], near_hit=0.75, wall_hit=1.5),
        "doorway": lambda: sector_scene([(0, 4)]),
        "t_junction": lambda: sector_scene([(15, 4), (105, 4)]),
        "l_junction": lambda: sector_scene([(20, 4)]),
        "asymmetric_obstacle": lambda: sector_scene([(12, 7)], near_hit=0.5),
        "narrow_passage": narrow_passage,
        "large_open_room": large_room,
        "mixed_near_far_structure": mixed_near_far,
        "range_censored_open_direction": full_open,
        "partial_occlusion": partial_occlusion,
        "rotated_equivalent_geometry_90deg": lambda: sector_scene([(42, 7)], near_hit=0.5),
    }
    result = {
        "scope": "CPU synthetic raw-ray fixtures only; no Habitat or navigation outcome",
        "scenes": {name: report_scene(factory()) for name, factory in scenes.items()},
        "bounded_nms_toys": nms_toys(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
