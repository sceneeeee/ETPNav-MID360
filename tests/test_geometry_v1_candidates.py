"""CPU-only contract tests for the inactive Geometry V1 candidate prototype."""
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from vlnce_baselines.perception.geometry_v1_candidates import (
    ETP_ANGLE_BINS,
    GeometryV1Candidate,
    _select,
    extract_geometry_v1_candidates,
)
from vlnce_baselines.perception.ray_carving_representation import (
    EpistemicState,
    RayCarvingGrid,
    RayObservation,
)
from vlnce_baselines.waypoint_provider.original_etp import construct_waypoint_output


def _ray(y, angle_index, distance, kind):
    theta = angle_index / float(ETP_ANGLE_BINS) * 2.0 * math.pi
    return RayObservation(
        (0.0, y, 0.0), (math.sin(theta), 0.0, -math.cos(theta)), distance,
        kind,
    )


def _fan(range_for_angle, kind_for_angle=None):
    """A collision-height-complete 360-degree synthetic raw-ray scan."""
    grid = RayCarvingGrid(4.0, 0.0, 2.0, 0.25, 0.25, 1.50)
    rays = []
    for y in np.arange(0.25, 1.51, 0.25):
        for angle in range(ETP_ANGLE_BINS):
            kind = "censored" if kind_for_angle is None else kind_for_angle(angle)
            rays.append(_ray(y, angle, range_for_angle(angle), kind))
    grid.integrate_many(rays)
    return grid


def _fully_observed_censored_open():
    """Raw censor rays plus complete collision-height free evidence."""
    grid = RayCarvingGrid(4.0, 0.0, 2.0, 0.10, 0.10, 1.50)
    rays = [_ray(y, angle, 3.0, "censored")
            for y in np.arange(0.10, 1.51, 0.10)
            for angle in range(ETP_ANGLE_BINS)]
    grid.integrate_many(rays)
    centers = grid.y_min_m + np.arange(grid.ny) * grid.resolution_m
    levels = ((centers >= grid.obstacle_height_min_m) &
              (centers <= grid.obstacle_height_max_m))
    grid.free_evidence[levels] = True
    return grid


def _structural_ridge_grid():
    """Complete free evidence with two physical side obstacles around a ridge."""
    grid = _fully_observed_censored_open()
    centers = grid.y_min_m + np.arange(grid.ny) * grid.resolution_m
    levels = ((centers >= grid.obstacle_height_min_m) &
              (centers <= grid.obstacle_height_max_m))
    for point in ((0.30, 0.10, -1.00), (0.30, 0.10, -2.00)):
        _, row, col = grid._index(point)
        grid.occupied_evidence[levels, row, col] = True
    return grid


class GeometryV1CandidateTest(unittest.TestCase):
    def extract(self, grid):
        return extract_geometry_v1_candidates(grid)[0]

    def test_open_space_is_explicitly_underdetermined(self):
        candidates = self.extract(_fully_observed_censored_open())
        self.assertEqual(candidates, ())

    def test_range_censored_open_direction_is_not_a_structure(self):
        candidates = self.extract(_fully_observed_censored_open())
        self.assertFalse(any(c.kind == "clearance_ridge" for c in candidates))

    def test_corridor_uses_labeled_progress_fallback(self):
        def distance(angle):
            theta = angle / 120.0 * 2.0 * math.pi
            side = abs(math.sin(theta))
            return min(3.0, 0.9 / side) if side > 1e-8 else 3.0

        def kind(angle):
            return "hit" if distance(angle) < 3.0 else "censored"

        candidates = self.extract(_fan(distance, kind))
        self.assertGreaterEqual(len(candidates), 2)
        self.assertTrue(all(c.kind == "progress_fallback" for c in candidates))
        self.assertTrue(all(c.distance_m == 2.75 for c in candidates))

    def test_unknown_rejects_candidate_path(self):
        grid = _fully_observed_censored_open()
        # A missing collision-height cell in front makes this ray uncertified.
        _, row, col = grid._index((0.0, 0.10, -1.0))
        grid.free_evidence[:, row, col] = False
        candidates, polar = extract_geometry_v1_candidates(grid)
        self.assertFalse(polar.safe[0, 10])
        self.assertFalse(any(c.angle_index == 0 for c in candidates))

    def test_safety_invalid_rejects_candidate_path(self):
        grid = _fully_observed_censored_open()
        invalid = np.zeros((grid.nxy, grid.nxy), dtype=bool)
        _, row, col = grid._index((0.0, 0.10, -1.0))
        invalid[row, col] = True
        _, polar = extract_geometry_v1_candidates(grid, safety_invalid=invalid)
        self.assertFalse(polar.safe[0, 10])

    def test_structural_clearance_ridge_can_be_extracted(self):
        candidates = self.extract(_structural_ridge_grid())
        self.assertTrue(any(c.kind == "clearance_ridge" for c in candidates))

    def test_fallback_does_not_activate_when_a_ridge_exists(self):
        candidates = self.extract(_structural_ridge_grid())
        self.assertTrue(candidates)
        self.assertTrue(all(c.kind == "clearance_ridge" for c in candidates))

    def test_maximum_candidate_count(self):
        grid = _fully_observed_censored_open()
        candidates, _ = extract_geometry_v1_candidates(grid, maximum_candidates=3)
        self.assertLessEqual(len(candidates), 3)

    def test_deterministic_ordering(self):
        grid = _fully_observed_censored_open()
        first = self.extract(grid)
        second = self.extract(grid)
        self.assertEqual(first, second)

    def test_equal_rank_over_cap_orbit_is_not_coordinate_truncated(self):
        orbit = [GeometryV1Candidate(angle, 4, "clearance_ridge", 1.0)
                 for angle in (0, 20, 40, 60, 80, 100)]
        self.assertEqual(_select(orbit, 5), ())

    def test_radially_distinct_structures_are_not_radially_suppressed(self):
        candidates = _select([
            GeometryV1Candidate(0, 1, "clearance_ridge", 2.0),
            GeometryV1Candidate(0, 10, "clearance_ridge", 1.0),
        ], 5)
        self.assertEqual([candidate.distance_index for candidate in candidates], [1, 10])

    def test_quarter_turn_rotation_is_equivariant_for_asymmetric_scan(self):
        def distance(angle):
            return 2.25 if 5 <= angle <= 15 else 0.75

        a = self.extract(_fan(distance, lambda _: "hit"))
        b = self.extract(_fan(lambda angle: distance((angle - 30) % 120),
                              lambda _: "hit"))
        self.assertEqual(
            {(c.angle_index + 30) % 120 for c in a},
            {c.angle_index for c in b},
        )

    def test_collision_state_keeps_censor_distinct_from_hit(self):
        grid = _fully_observed_censored_open()
        _, row, col = grid._index((0.0, 0.25, -3.0))
        self.assertTrue(grid.range_censored[:, row, col].any())
        self.assertFalse(grid.occupied_evidence.any())

    def test_existing_feature_adapter_accepts_direct_candidate_indices(self):
        candidates = self.extract(_structural_ridge_grid())
        encoded = SimpleNamespace(
            batch_size=1,
            rgb_features=torch.zeros(1, 12, 4),
            depth_features=torch.zeros(1, 12, 3),
        )
        output = construct_waypoint_output(
            encoded,
            [torch.tensor([c.angle_index for c in candidates])],
            [torch.tensor([c.distance_index for c in candidates])],
            torch.zeros(12, 128),
            np.arange(12),
        )
        self.assertEqual(len(output["cand_angles"][0]), len(candidates))
        self.assertEqual(len(output["cand_distances"][0]), len(candidates))
        self.assertEqual(output["cand_rgb"][0].shape[0], len(candidates))
        self.assertEqual(output["cand_depth"][0].shape[0], len(candidates))


if __name__ == "__main__":
    unittest.main()
