import unittest

import numpy as np

from vlnce_baselines.perception.depth_to_pointcloud import (
    backproject_depth,
    depth_panorama_to_pointcloud,
)


class DepthToPointCloudTest(unittest.TestCase):
    def test_center_pixel_projects_forward_in_habitat_coordinates(self):
        depth = np.zeros((3, 3), dtype=np.float32)
        depth[1, 1] = 2.0

        points = backproject_depth(
            depth, horizontal_fov_degrees=90.0, camera_height_m=1.25
        )

        np.testing.assert_allclose(points, [[0.0, 1.25, -2.0]], atol=1e-6)

    def test_positive_ninety_degree_heading_rotates_forward_to_left(self):
        depth = np.ones((1, 1), dtype=np.float32)

        points = depth_panorama_to_pointcloud(
            [depth, depth],
            headings_degrees=[0.0, 90.0],
            horizontal_fov_degrees=90.0,
            camera_height_m=1.25,
        )

        np.testing.assert_allclose(
            points,
            [[0.0, 1.25, -1.0], [-1.0, 1.25, 0.0]],
            atol=1e-6,
        )

    def test_invalid_and_out_of_range_depth_are_discarded(self):
        depth = np.array([[0.0, np.nan, 2.0, 4.0]], dtype=np.float32)

        points = backproject_depth(
            depth,
            horizontal_fov_degrees=90.0,
            camera_height_m=0.0,
            minimum_depth_m=0.1,
            maximum_depth_m=3.0,
        )

        self.assertEqual(points.shape, (1, 3))
        self.assertAlmostEqual(points[0, 2], -2.0)


if __name__ == "__main__":
    unittest.main()
