import math
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

_MODULE_PATH = Path(__file__).parents[1] / "vlnce_baselines/perception/ray_carving_representation.py"
_SPEC = importlib.util.spec_from_file_location("ray_carving_representation_study", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
EpistemicState = _MODULE.EpistemicState
RayCarvingGrid = _MODULE.RayCarvingGrid
RayObservation = _MODULE.RayObservation


def ray(origin, direction, distance, kind="hit"):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    return RayObservation(tuple(origin), tuple(direction), distance, kind)


class RayCarvingRepresentationTest(unittest.TestCase):
    def grid(self):
        return RayCarvingGrid(4.0, 0.0, 2.0, 0.1, 0.1, 1.5)

    def idx(self, grid, point):
        return grid._index(point)

    def test_first_hit_occlusion_and_physical_hit(self):
        grid = self.grid()
        grid.integrate(ray((0, 1, 0), (1, 0, 0), 2.0))
        states = grid.voxel_state()
        self.assertEqual(states[self.idx(grid, (1, 1, 0))], EpistemicState.OBSERVED_FREE)
        self.assertEqual(states[self.idx(grid, (2, 1, 0))], EpistemicState.OCCUPIED)
        self.assertEqual(states[self.idx(grid, (3, 1, 0))], EpistemicState.UNKNOWN)

    def test_range_censoring_is_not_an_obstacle(self):
        grid = self.grid()
        grid.integrate(ray((0, 1, 0), (1, 0, 0), 3.0, "censored"))
        end = self.idx(grid, (3, 1, 0))
        self.assertTrue(grid.range_censored[end])
        self.assertFalse(grid.occupied_evidence[end])

    def test_beyond_usable_range_is_censored_prefix_not_a_hit(self):
        # The depth adapter must convert a farther valid surface into this
        # usable-range censored ray; it must never retain its remote endpoint.
        grid = self.grid()
        grid.integrate(ray((0, 1, 0), (1, 0, 0), 3.5, "censored"))
        self.assertFalse(grid.occupied_evidence.any())
        self.assertTrue(grid.range_censored[self.idx(grid, (3.5, 1, 0))])

    def test_invalid_is_no_evidence(self):
        grid = self.grid()
        grid.integrate(ray((0, 1, 0), (1, 0, 0), 2.0, "invalid"))
        self.assertFalse(grid.free_evidence.any())
        self.assertFalse(grid.occupied_evidence.any())

    def test_high_ray_does_not_certify_collision_column(self):
        grid = self.grid()
        grid.integrate(ray((0, 1.9, 0), (1, 0, 0), 3.0, "censored"))
        column = grid.collision_column_state()
        _, row, col = self.idx(grid, (1, 1, 0))
        self.assertEqual(column[row, col], EpistemicState.UNKNOWN)

    def test_overhang_is_3d_occupied_but_not_collision_occupied(self):
        grid = self.grid()
        grid.integrate(ray((0, 1.9, 0), (1, 0, 0), 2.0))
        _, row, col = self.idx(grid, (2, 1.9, 0))
        self.assertTrue(grid.occupied_evidence[:, row, col].any())
        self.assertNotEqual(grid.collision_column_state()[row, col], EpistemicState.OCCUPIED)

    def test_low_obstacle_survives_free_evidence_and_safety_is_separate(self):
        grid = self.grid()
        grid.integrate(ray((0, 1, 0), (1, 0, 0), 2.0))
        before = grid.voxel_state().copy()
        safety = grid.safety_invalid(0.2)
        _, row, col = self.idx(grid, (2, 1, 0))
        self.assertTrue(safety[row, col])
        np.testing.assert_array_equal(before, grid.voxel_state())

    def test_doorway_and_narrow_passage_remain_unknown_until_column_observed(self):
        # Two collision-height wall returns leave a geometric gap, but a gap is
        # not declared free merely because it lacks a return.
        grid = self.grid()
        grid.integrate_many([
            ray((0, 0.8, 0), (2, 0, -1), math.sqrt(5)),
            ray((0, 0.8, 0), (2, 0, 1), math.sqrt(5)),
        ])
        _, row, col = self.idx(grid, (2, 0.8, 0))
        self.assertEqual(grid.collision_column_state()[row, col], EpistemicState.UNKNOWN)

    def test_mixed_height_returns_preserve_both_evidence_kinds(self):
        grid = self.grid()
        grid.integrate(ray((0, 1.0, 0), (1, 0, 0), 2.0))
        grid.integrate(ray((3, 1.0, 0), (-1, 0, 0), 2.0, "censored"))
        index = self.idx(grid, (2, 1.0, 0))
        self.assertTrue(grid.occupied_evidence[index])
        self.assertTrue(grid.free_evidence[index])

    def test_multi_view_overlap_is_idempotent_boolean_evidence(self):
        one, two = self.grid(), self.grid()
        observation = ray((0, 1, 0), (1, 0, 0), 2.0)
        one.integrate(observation)
        two.integrate_many([observation, observation])
        np.testing.assert_array_equal(one.voxel_state(), two.voxel_state())

    def test_independent_ray_can_observe_behind_other_ray_hit(self):
        grid = self.grid()
        grid.integrate(ray((0, 1, 0), (1, 0, 0), 1.0))
        grid.integrate(ray((3, 1, 0), (-1, 0, 0), 3.0, "censored"))
        self.assertTrue(grid.free_evidence[self.idx(grid, (2, 1, 0))])

    def test_rotation_consistency(self):
        a, b = self.grid(), self.grid()
        a.integrate(ray((0, 1, 0), (1, 0, 0), 2.0))
        b.integrate(ray((0, 1, 0), (0, 0, 1), 2.0))
        _, ar, ac = self.idx(a, (2, 1, 0))
        _, br, bc = self.idx(b, (0, 1, 2))
        self.assertTrue(a.occupied_evidence[:, ar, ac].any())
        self.assertTrue(b.occupied_evidence[:, br, bc].any())

    def test_deterministic_fusion(self):
        rays = [ray((0, 1, 0), (1, 0, 0), 2.0), ray((0, 1, 0), (0, 0, 1), 2.0)]
        a, b = self.grid(), self.grid()
        a.integrate_many(rays); b.integrate_many(rays)
        np.testing.assert_array_equal(a.voxel_state(), b.voxel_state())

    def test_vectorized_many_matches_reference_single_ray_semantics(self):
        observations = [
            ray((0, 1.0, 0), (1, 0.1, 0), 2.13),
            ray((0, 1.2, 0), (0, -0.2, -1), 3.17, "censored"),
            ray((0.2, 0.7, 0), (-1, 0.3, 1), 1.41),
        ]
        vectorized, reference = self.grid(), self.grid()
        vectorized.integrate_many(observations)
        for observation in observations:
            reference._integrate_reference(observation)
        np.testing.assert_array_equal(
            vectorized.free_evidence, reference.free_evidence
        )
        np.testing.assert_array_equal(
            vectorized.occupied_evidence, reference.occupied_evidence
        )
        np.testing.assert_array_equal(
            vectorized.range_censored, reference.range_censored
        )
