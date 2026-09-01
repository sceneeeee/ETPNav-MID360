import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vlnce_baselines.perception.geometry_provider import (
    DepthReconstructedGeometryProvider,
    GeometryFrame,
    GeometryProvider,
)
from vlnce_baselines.perception.depth_to_pointcloud import (
    depth_panorama_to_pointcloud,
)
from vlnce_baselines.waypoint_provider.mid360_geometry import (
    MID360GeometryConfig,
    MID360GeometryWaypointProvider,
    default_mid360_config_path,
)


class _Config(object):
    policy_depth_minimum_m = 0.0
    policy_depth_maximum_m = 10.0
    horizontal_fov_degrees = 90.0
    camera_height_m = 1.25
    minimum_range_m = 0.0
    maximum_range_m = 10.0
    depth_pixel_stride = 1


class GeometryProviderTest(unittest.TestCase):
    @staticmethod
    def _observations():
        observations = {}
        for heading in range(0, 360, 30):
            suffix = "" if heading == 0 else "_{}".format(heading)
            depth = torch.zeros(1, 1, 1, 1)
            depth[0, 0, 0, 0] = 0.2
            observations["depth{}".format(suffix)] = depth
        return observations

    @staticmethod
    def _config():
        return MID360GeometryConfig.from_yaml(default_mid360_config_path())

    def _provider(self, geometry_provider=None):
        return MID360GeometryWaypointProvider(
            rgb_encoder=None,
            depth_encoder=None,
            rgb_pool=None,
            depth_pool=None,
            pano_angle_features=None,
            pano_image_indexes=None,
            config=self._config(),
            geometry_provider=geometry_provider,
        )

    def _predict_without_encoders(self, provider, observations):
        score = np.zeros((120, 12), dtype=np.float32)
        with patch(
            "vlnce_baselines.waypoint_provider.mid360_geometry.encode_panorama",
            return_value=SimpleNamespace(batch_size=1),
        ), patch(
            "vlnce_baselines.waypoint_provider.mid360_geometry."
            "compute_local_traversability",
            return_value=SimpleNamespace(score=score),
        ), patch(
            "vlnce_baselines.waypoint_provider.mid360_geometry."
            "select_geometry_candidates",
            return_value=(np.array([0]), np.array([0])),
        ), patch(
            "vlnce_baselines.waypoint_provider.mid360_geometry."
            "construct_waypoint_output",
            return_value={"unchanged": "output"},
        ):
            return provider.predict(observations)

    def test_depth_reconstruction_keeps_v0_depth_heading_contract(self):
        observations = self._observations()

        frame = DepthReconstructedGeometryProvider(_Config()).get_geometry(
            observations, 0
        )

        self.assertEqual(frame.source, "v0_depth_reconstruction")
        self.assertEqual(frame.headings_degrees, list(range(0, 360, 30)))
        self.assertEqual(frame.pointcloud.shape, (12, 3))
        np.testing.assert_allclose(frame.pointcloud[0], [0.0, 1.25, -2.0])
        expected = depth_panorama_to_pointcloud(
            [np.full((1, 1, 1), 2.0, dtype=np.float32)] * 12,
            headings_degrees=list(range(0, 360, 30)),
            horizontal_fov_degrees=90.0,
            camera_height_m=1.25,
            minimum_depth_m=0.0,
            maximum_depth_m=10.0,
            pixel_stride=1,
        )
        np.testing.assert_array_equal(frame.pointcloud, expected)

    def test_default_provider_uses_exact_v0_depth_reconstruction(self):
        provider = self._provider()
        observations = self._observations()

        output = self._predict_without_encoders(provider, observations)

        config = provider.config
        expected = depth_panorama_to_pointcloud(
            [np.full((1, 1, 1), 2.0, dtype=np.float32)] * 12,
            headings_degrees=list(range(0, 360, 30)),
            horizontal_fov_degrees=config.horizontal_fov_degrees,
            camera_height_m=config.camera_height_m,
            minimum_depth_m=config.minimum_range_m,
            maximum_depth_m=config.maximum_range_m,
            pixel_stride=config.depth_pixel_stride,
        )
        self.assertEqual(output, {"unchanged": "output"})
        self.assertEqual(
            provider.last_debug[0]["geometry_source"], "v0_depth_reconstruction"
        )
        np.testing.assert_array_equal(
            provider.last_debug[0]["merged_pointcloud"], expected
        )

    def test_injected_pointcloud_reaches_frozen_geometry_scorer(self):
        pointcloud = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        class NativeGeometryProvider(GeometryProvider):
            def get_geometry(self, observations, batch_index):
                return GeometryFrame(pointcloud, "v1_native_pointcloud")

        provider = self._provider(NativeGeometryProvider())
        output = self._predict_without_encoders(provider, self._observations())

        self.assertEqual(output, {"unchanged": "output"})
        self.assertEqual(
            provider.last_debug[0]["geometry_source"], "v1_native_pointcloud"
        )
        np.testing.assert_array_equal(
            provider.last_debug[0]["merged_pointcloud"], pointcloud
        )

    def test_geometry_frame_rejects_invalid_pointclouds(self):
        with self.assertRaisesRegex(ValueError, "Nx3"):
            GeometryFrame(np.zeros((3, 2)), "test")
        with self.assertRaisesRegex(ValueError, "finite"):
            GeometryFrame(np.array([[0.0, np.nan, 1.0]]), "test")

    def test_waypoint_provider_rejects_non_provider_geometry_source(self):
        with self.assertRaisesRegex(TypeError, "GeometryProvider"):
            self._provider(geometry_provider=object())


if __name__ == "__main__":
    unittest.main()
