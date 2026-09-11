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
from vlnce_baselines.waypoint_pred.utils import nms
from vlnce_baselines.perception.depth_to_pointcloud import (
    depth_panorama_to_pointcloud,
)
from vlnce_baselines.waypoint_provider.mid360_geometry import (
    MID360GeometryConfig,
    MID360GeometryWaypointProvider,
    apply_radial_utility,
    default_mid360_config_path,
    select_geometry_candidates,
)
from vlnce_baselines.waypoint_provider.original_etp import (
    OriginalETPWaypointProvider,
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
    def test_mru_has_the_frozen_twelve_bin_score_vector(self):
        traversability = np.tile(
            np.arange(1, 13, dtype=np.float32) / 12.0, (120, 1)
        )

        score = apply_radial_utility(traversability, "mru")

        expected = np.array([
            0.383495982, 0.649244680, 0.824360635, 0.930408283,
            0.984467011, 1.000000000, 0.987562012, 0.955375081,
            0.909795990, 0.855695198, 0.796763382, 0.735758882,
        ], dtype=np.float32)
        np.testing.assert_allclose(score[0], expected, rtol=0.0, atol=5e-7)
        print("MRU 12-bin score vector:", score[0].tolist())

    def test_mru_keeps_blocked_cells_exactly_zero(self):
        traversability = np.tile(
            np.arange(1, 13, dtype=np.float32) / 12.0, (120, 1)
        )
        traversability[7, [0, 5, 11]] = 0.0

        score = apply_radial_utility(traversability, "mru")

        np.testing.assert_array_equal(score[7, [0, 5, 11]], 0.0)
        self.assertGreater(score[7, 1], 0.0)

    def test_mru_is_finite_unimodal_and_has_no_hard_cutoff(self):
        traversability = np.tile(
            np.arange(1, 13, dtype=np.float32) / 12.0, (120, 1)
        )

        score = apply_radial_utility(traversability, "mru")[0]

        self.assertTrue(np.all(np.isfinite(score)))
        self.assertTrue(np.all(np.diff(score[:6]) > 0.0))
        self.assertEqual(int(np.argmax(score)), 5)  # 1.50 m, zero-based bin 5.
        self.assertTrue(np.all(np.diff(score[5:]) < 0.0))
        self.assertGreater(score[-1], 0.0)  # valid 3.00 m bin is not cut off.

    def test_mru_disabled_preserves_v0_bnms_score_and_candidates(self):
        baseline = np.zeros((120, 12), dtype=np.float32)
        baseline[20, :12] = np.arange(1, 13, dtype=np.float32) / 12.0
        baseline[80, :9] = np.arange(1, 10, dtype=np.float32) / 12.0

        disabled = apply_radial_utility(baseline, "v0_linear")

        np.testing.assert_array_equal(disabled, baseline)
        self.assertEqual(
            self._candidate_indexes(disabled),
            self._candidate_indexes(baseline),
        )

    def test_mru_does_not_couple_generic_original_etp_to_geometry(self):
        # Original ETP remains its independent learned-proposal implementation;
        # MRU is only applied by the Geometry provider before bounded NMS.
        self.assertFalse(hasattr(OriginalETPWaypointProvider, "radial_utility"))
        self.assertFalse(hasattr(OriginalETPWaypointProvider, "apply_radial_utility"))

    @staticmethod
    def _candidate_indexes(score, max_candidates=5):
        angles, distances = select_geometry_candidates(score, max_candidates)
        return set(zip(angles.tolist(), distances.tolist()))

    def test_geometry_nms_wraps_angular_bin_zero_to_119(self):
        score = np.zeros((120, 12), dtype=np.float32)
        score[0, 6] = 10.0
        score[119, 6] = 9.0

        candidates = self._candidate_indexes(score, max_candidates=2)

        self.assertEqual(candidates, {(0, 6)})

    def test_geometry_nms_does_not_wrap_radial_bin_zero_to_11(self):
        score = np.zeros((120, 12), dtype=np.float32)
        score[60, 11] = 10.0
        score[60, 0] = 9.0

        candidates = self._candidate_indexes(score, max_candidates=2)

        self.assertEqual(candidates, {(60, 0), (60, 11)})

    def test_radial_bin_11_no_longer_suppresses_bin_zero_at_same_angle(self):
        score = np.zeros((120, 12), dtype=np.float32)
        score[60, 11] = 10.0
        score[60, 0] = 9.0
        score[60, 6] = 8.0

        candidates = self._candidate_indexes(score, max_candidates=2)

        self.assertIn((60, 11), candidates)
        self.assertIn((60, 0), candidates)
        self.assertNotIn((60, 6), candidates)

    def test_bounded_nms_keeps_non_boundary_geometry_candidates_unchanged(self):
        score = np.zeros((120, 12), dtype=np.float32)
        score[30, 6] = 10.0
        score[50, 7] = 9.0
        score[80, 10] = 8.0

        legacy_score = torch.from_numpy(score)
        legacy_wrapped = torch.cat(
            (legacy_score[-1:, :], legacy_score, legacy_score[:1, :]), dim=0
        )
        legacy = nms(
            legacy_wrapped[None, None], max_predictions=5, sigma=(7.0, 5.0)
        ).squeeze(0).squeeze(0)[1:-1, :].nonzero().cpu().numpy()
        angle_indexes, distance_indexes = select_geometry_candidates(score)

        np.testing.assert_array_equal(score, legacy_score.numpy())
        np.testing.assert_array_equal(
            np.stack((angle_indexes, distance_indexes), axis=1), legacy
        )
        self.assertEqual(
            self._candidate_indexes(score),
            {(30, 6), (50, 7), (80, 10)},
        )

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
            provider.last_debug[0]["pre_nms_score"],
            np.zeros((120, 12), dtype=np.float32),
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
