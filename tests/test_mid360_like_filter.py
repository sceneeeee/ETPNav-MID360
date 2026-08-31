import unittest

import numpy as np

from vlnce_baselines.perception.mid360_like_filter import MID360LikeFilter


class MID360LikeFilterTest(unittest.TestCase):
    def test_range_vertical_fov_and_voxel_filters_are_deterministic(self):
        points = np.array(
            [
                [0.0, 0.0, -0.05],
                [0.0, 0.0, -1.01],
                [0.01, 0.01, -1.02],
                [0.0, 2.0, -1.0],
                [0.0, 0.0, -4.0],
            ],
            dtype=np.float32,
        )
        lidar_filter = MID360LikeFilter(
            minimum_range_m=0.1,
            maximum_range_m=3.0,
            vertical_fov_min_degrees=-45.0,
            vertical_fov_max_degrees=45.0,
            azimuth_bin_degrees=1.0,
            elevation_bin_degrees=1.0,
            voxel_size_m=0.1,
        )

        first = lidar_filter(points)
        second = lidar_filter(points[::-1])

        self.assertEqual(first.shape, (1, 3))
        np.testing.assert_allclose(first, second, atol=1e-6)
        self.assertAlmostEqual(float(first[0, 2]), -1.01, places=5)


if __name__ == "__main__":
    unittest.main()
