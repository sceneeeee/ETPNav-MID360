import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from vlnce_baselines.perception.depth_to_pointcloud import backproject_depth
from vlnce_baselines.perception.depth_to_rays import depth_panorama_to_rays
from vlnce_baselines.perception.mid360_like_filter import MID360LikeFilter
from vlnce_baselines.perception.ray_carving_representation import RayCarvingGrid


def config(**updates):
    values = dict(
        policy_depth_minimum_m=0.0,
        policy_depth_maximum_m=10.0,
        horizontal_fov_degrees=90.0,
        camera_height_m=1.25,
        depth_pixel_stride=1,
        minimum_range_m=0.10,
        maximum_range_m=3.5,
        vertical_fov_min_degrees=-60.0,
        vertical_fov_max_degrees=60.0,
        azimuth_bin_degrees=1.0,
        elevation_bin_degrees=1.0,
        voxel_size_m=0.05,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def panorama(size=3, fill=0.0):
    result = {}
    for heading in range(0, 360, 30):
        key = "depth" if heading == 0 else "depth_{}".format(heading)
        result[key] = torch.full((1, size, size, 1), fill, dtype=torch.float32)
    return result


class DepthToRaysTest(unittest.TestCase):
    def test_center_pixel_endpoint_matches_existing_v0_backprojection(self):
        observations = panorama()
        observations["depth"][0, 1, 1, 0] = 0.2

        result = depth_panorama_to_rays(observations, 0, config())

        self.assertEqual(len(result.rays), 1)
        ray = result.rays[0]
        endpoint = np.asarray(ray.origin_xyz) + np.asarray(ray.direction_xyz) * ray.range_m
        expected = backproject_depth(
            np.array([[0, 0, 0], [0, 2.0, 0], [0, 0, 0]], dtype=np.float32),
            90.0, 1.25,
        )[0]
        np.testing.assert_allclose(endpoint, expected, rtol=0.0, atol=1e-7)
        self.assertAlmostEqual(np.linalg.norm(ray.direction_xyz), 1.0, places=12)
        self.assertEqual((ray.view_index, ray.pixel_row, ray.pixel_column), (0, 1, 1))

    def test_off_axis_z_depth_is_converted_to_euclidean_range(self):
        observations = panorama()
        observations["depth"][0, 1, 2, 0] = 0.2

        ray = depth_panorama_to_rays(observations, 0, config()).rays[0]
        endpoint = np.asarray(ray.origin_xyz) + np.asarray(ray.direction_xyz) * ray.range_m
        expected = backproject_depth(
            np.array([[0, 0, 0], [0, 0, 2.0], [0, 0, 0]], dtype=np.float32),
            90.0, 1.25,
        )[0]
        np.testing.assert_allclose(endpoint, expected, rtol=0.0, atol=1e-7)
        self.assertGreater(ray.range_m, ray.measured_z_depth_m)

    def test_all_retained_pixel_endpoints_match_v0_at_tight_tolerance(self):
        observations = panorama(size=7)
        observations["depth"].fill_(0.1)

        result = depth_panorama_to_rays(observations, 0, config())

        self.assertEqual(len(result.rays), 49)
        metric = np.ones((7, 7), dtype=np.float32)
        expected = backproject_depth(metric, 90.0, 1.25)
        expected_by_pixel = {
            (row, column): expected[row * 7 + column]
            for row in range(7) for column in range(7)
        }
        for ray in result.rays:
            endpoint = (np.asarray(ray.origin_xyz) +
                        np.asarray(ray.direction_xyz) * ray.range_m)
            np.testing.assert_allclose(
                endpoint,
                expected_by_pixel[(ray.pixel_row, ray.pixel_column)],
                rtol=0.0,
                atol=2e-7,
            )

    def test_retained_hit_endpoints_match_current_mid360_filter(self):
        observations = panorama(size=7)
        observations["depth"].fill_(0.1)
        cfg = config()
        result = depth_panorama_to_rays(observations, 0, cfg)
        endpoints = np.asarray([
            np.asarray(ray.origin_xyz) + np.asarray(ray.direction_xyz) * ray.range_m
            for ray in result.rays
        ], dtype=np.float32)
        v0_points = backproject_depth(
            np.ones((7, 7), dtype=np.float32), 90.0, 1.25
        )
        retained = MID360LikeFilter(
            cfg.minimum_range_m, cfg.maximum_range_m,
            cfg.vertical_fov_min_degrees, cfg.vertical_fov_max_degrees,
            cfg.azimuth_bin_degrees, cfg.elevation_bin_degrees,
            cfg.voxel_size_m, cfg.camera_height_m,
        )(v0_points)
        endpoint_order = np.lexsort((endpoints[:, 2], endpoints[:, 1], endpoints[:, 0]))
        retained_order = np.lexsort((retained[:, 2], retained[:, 1], retained[:, 0]))
        np.testing.assert_allclose(
            endpoints[endpoint_order], retained[retained_order],
            rtol=0.0, atol=3e-7,
        )

    def test_positive_ninety_degree_view_rotates_forward_to_left(self):
        observations = panorama()
        observations["depth_90"][0, 1, 1, 0] = 0.2

        ray = depth_panorama_to_rays(observations, 0, config()).rays[0]
        endpoint = np.asarray(ray.origin_xyz) + np.asarray(ray.direction_xyz) * ray.range_m
        np.testing.assert_allclose(endpoint, [-2.0, 1.25, 0.0], atol=1e-7)
        self.assertEqual(ray.heading_degrees, 90.0)
        self.assertEqual(ray.view_index, 3)

    def test_surface_beyond_usable_range_becomes_censored_prefix(self):
        observations = panorama()
        observations["depth"][0, 1, 1, 0] = 0.4

        ray = depth_panorama_to_rays(observations, 0, config()).rays[0]
        self.assertEqual(ray.kind, "censored")
        self.assertEqual(ray.range_m, 3.5)
        self.assertAlmostEqual(ray.measured_z_depth_m, 4.0, places=6)

    def test_renderer_max_depth_is_censored_not_a_hit(self):
        observations = panorama()
        observations["depth"][0, 1, 1, 0] = 1.0

        ray = depth_panorama_to_rays(observations, 0, config()).rays[0]
        self.assertEqual(ray.kind, "censored")
        grid = RayCarvingGrid(4.0, 0.0, 2.0, 0.1, 0.1, 1.5)
        grid.integrate(ray)
        self.assertTrue(grid.range_censored.any())
        self.assertFalse(grid.occupied_evidence.any())

    def test_zero_nan_and_below_minimum_depth_are_invalid(self):
        observations = panorama()
        observations["depth"][0, 0, 0, 0] = float("nan")
        observations["depth"][0, 1, 1, 0] = 0.005

        result = depth_panorama_to_rays(observations, 0, config())
        self.assertEqual(result.rays, ())
        self.assertEqual(result.invalid_count, result.sampled_count)

    def test_outside_mid360_vertical_fov_does_not_carve(self):
        observations = panorama()
        observations["depth"][0, 0, 1, 0] = 0.2

        result = depth_panorama_to_rays(
            observations, 0,
            config(vertical_fov_min_degrees=-10.0,
                   vertical_fov_max_degrees=10.0),
        )
        self.assertEqual(result.rays, ())
        self.assertEqual(result.outside_vertical_fov_count, 1)

    def test_missing_panorama_view_is_rejected(self):
        observations = panorama()
        del observations["depth_330"]
        with self.assertRaisesRegex(ValueError, "depth_330"):
            depth_panorama_to_rays(observations, 0, config())

    def test_twelve_input_shapes_and_deterministic_output(self):
        observations = panorama(fill=0.2)
        first = depth_panorama_to_rays(observations, 0, config())
        second = depth_panorama_to_rays(observations, 0, config())
        self.assertEqual(first.rays, second.rays)
        self.assertEqual(first.input_shapes, ((3, 3),) * 12)
        self.assertEqual(first.hit_count, len(first.rays))


if __name__ == "__main__":
    unittest.main()
