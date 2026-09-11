import unittest

import numpy as np

from vlnce_baselines.perception.local_traversability import (
    TraversabilityConfig,
    compute_local_traversability,
)
from vlnce_baselines.waypoint_provider.mid360_geometry import (
    select_geometry_candidates,
)


class LocalTraversabilityTest(unittest.TestCase):
    def config(self):
        return TraversabilityConfig(
            angular_bins=120,
            distance_bins=12,
            distance_step_m=0.25,
            robot_radius_m=0.15,
            clearance_margin_m=0.05,
            obstacle_height_min_m=0.1,
            obstacle_height_max_m=1.5,
            grid_resolution_m=0.05,
        )

    def test_empty_cloud_marks_all_rays_free_with_distance_score(self):
        result = compute_local_traversability(
            np.empty((0, 3), dtype=np.float32), self.config()
        )

        self.assertEqual(result.score.shape, (120, 12))
        self.assertTrue(np.all(result.score[:, 1:] > result.score[:, :-1]))
        self.assertFalse(result.occupancy.any())
        self.assertFalse(result.inflated_occupancy.any())

    def test_forward_obstacle_blocks_far_forward_bins_but_not_side_ray(self):
        obstacle = np.array([[0.0, 0.5, -1.0]], dtype=np.float32)

        result = compute_local_traversability(obstacle, self.config())

        self.assertGreater(result.score[0, 0], 0.0)
        self.assertEqual(result.score[0, 4], 0.0)
        self.assertGreater(result.score[20, 4], 0.0)


    def test_outer_candidate_considers_obstacle_inside_footprint_margin(self):
        obstacle = np.array([[0.0, 0.5, -3.10]], dtype=np.float32)

        result = compute_local_traversability(obstacle, self.config())

        self.assertEqual(result.score[0, 11], 0.0)
        self.assertGreater(result.score[20, 11], 0.0)

    def test_candidate_selection_uses_120_by_12_contract(self):
        result = compute_local_traversability(
            np.empty((0, 3), dtype=np.float32), self.config()
        )

        angle_indexes, distance_indexes = select_geometry_candidates(
            result.score, max_candidates=5
        )

        self.assertLessEqual(len(angle_indexes), 5)
        self.assertEqual(len(angle_indexes), len(distance_indexes))
        self.assertTrue(np.all(distance_indexes == 11))


if __name__ == "__main__":
    unittest.main()
