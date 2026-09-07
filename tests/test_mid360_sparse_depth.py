import unittest
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from vlnce_baselines.perception.depth_to_pointcloud import backproject_depth
from vlnce_baselines.perception.mid360_sparse_depth import (
    DEPTH_KEYS, MID360SparseDepthAdapter, MID360SparseDepthConfig, project_pointcloud,
)
from vlnce_baselines.waypoint_provider.mid360_sparse_depth_etp import MID360SparseDepthETPWaypointProvider
from vlnce_baselines.waypoint_provider.original_etp import OriginalETPWaypointProvider
from vlnce_baselines.waypoint_provider.types import WaypointOutput


def observations(batch=2, size=17, device='cpu', dtype=torch.float32):
    obs = OrderedDict(instruction=torch.tensor([[1, 2]] * batch, device=device))
    for i, key in enumerate(DEPTH_KEYS):
        obs[key.replace('depth', 'rgb')] = torch.full((batch, size, size, 3), i, dtype=torch.uint8, device=device)
        obs[key] = torch.full((batch, size, size, 1), 0.2 + i * .01, dtype=dtype, device=device)
    return obs


class SparseDepthTest(unittest.TestCase):
    def setUp(self):
        self.cfg = MID360SparseDepthConfig(azimuth_bin_degrees=10, elevation_bin_degrees=10)

    def test_adapter_contract_determinism_and_input_immutability(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for dtype in [torch.float32, torch.float64]:
                obs = observations(device=device, dtype=dtype)
                before = {k: v.clone() for k, v in obs.items()}
                adapter = MID360SparseDepthAdapter(self.cfg)
                first, diagnostics = adapter(obs)
                second, _ = adapter(obs)
                self.assertEqual(list(first), list(obs))
                for key in obs:
                    self.assertEqual((first[key].shape, first[key].dtype, first[key].device),
                                     (obs[key].shape, obs[key].dtype, obs[key].device))
                    self.assertTrue(torch.equal(first[key], second[key]))
                    self.assertTrue(torch.equal(obs[key], before[key]))
                    if key in DEPTH_KEYS:
                        self.assertFalse(torch.equal(first[key], obs[key]))
                    else:
                        self.assertIs(first[key], obs[key])
                self.assertEqual(len(diagnostics), 2)
                self.assertIn('first_frame_statistics', diagnostics[0])
                self.assertEqual(diagnostics[0]['valid_projected_pixels'], sum(diagnostics[0]['valid_projected_pixels_per_view'].values()))

    def test_front_and_positive_ninety_heading(self):
        for heading, point in [(0, [0, 1.25, -2]), (90, [-2, 1.25, 0])]:
            depth, mask = project_pointcloud([point], 5, 5, heading, self.cfg)
            self.assertEqual(mask.sum(), 1)
            self.assertTrue(mask[2, 2, 0])
            self.assertAlmostEqual(float(depth[2, 2, 0]), .2, places=6)
            np.testing.assert_array_equal(depth[~mask], 1.)

    def test_inverse_all_pixels_and_headings_z_not_euclidean(self):
        # Non-square, even-size camera exercises half-pixel center and fy=fx.
        metric = np.arange(24, dtype=np.float32).reshape(4, 6) / 10 + 2
        for heading in range(0, 360, 30):
            cloud = backproject_depth(metric, 90, 1.25, heading)
            depth, mask = project_pointcloud(cloud, 4, 6, heading, self.cfg)
            self.assertTrue(mask.all())
            np.testing.assert_allclose(depth[..., 0] * 10, metric, atol=1e-6)
        cloud = backproject_depth(metric, 90, 1.25)
        self.assertGreater(cloud[0, 1], 1.25)  # top row is +Y
        self.assertLess(cloud[0, 0], 0)  # left column is -X

    def test_z_buffer_and_configured_fill(self):
        cfg = MID360SparseDepthConfig(no_return_fill_normalized=.75)
        depth, mask = project_pointcloud([[0, 1.25, -4], [0, 1.25, -2], [0, 1.25, 1], [np.nan, 0, 0]], 5, 5, 0, cfg)
        self.assertAlmostEqual(float(depth[2, 2, 0]), .2)
        self.assertEqual(mask.sum(), 1)
        np.testing.assert_array_equal(depth[~mask], .75)

    def test_no_returns_and_saturated_sources(self):
        obs = observations(batch=1)
        for key in DEPTH_KEYS:
            obs[key].fill_(1)
        sparse, diag = MID360SparseDepthAdapter(self.cfg)(obs)
        self.assertEqual(diag[0]['filtered_cloud_points'], 0)
        self.assertEqual(diag[0]['no_return_fraction'], 1)
        for key in DEPTH_KEYS:
            self.assertTrue(torch.all(sparse[key] == 1))

    def test_policy_range_is_not_euclidean_cutoff(self):
        obs = observations(batch=1, size=5)
        for key in DEPTH_KEYS:
            obs[key].zero_()
        obs['depth'][0, 0, 0, 0] = .9  # 9 m z, >10 m Euclidean range
        sparse, diag = MID360SparseDepthAdapter(self.cfg)(obs)
        self.assertEqual(diag[0]['dense_cloud_points'], 1)
        self.assertEqual(diag[0]['filtered_cloud_points'], 1)
        self.assertAlmostEqual(sparse['depth'][0, 0, 0, 0].item(), .9, places=6)

    def test_non_binary_fill_is_exact_in_float64(self):
        cfg = MID360SparseDepthConfig(no_return_fill_normalized=.1)
        obs = observations(batch=1, dtype=torch.float64)
        for key in DEPTH_KEYS:
            obs[key].zero_()
        sparse, _ = MID360SparseDepthAdapter(cfg)(obs)
        for key in DEPTH_KEYS:
            self.assertTrue(torch.equal(sparse[key], torch.full_like(obs[key], .1)))

    def test_reject_range_confound_averaging_and_view_reordering(self):
        for values in [dict(maximum_range_m=3.5), dict(voxel_size_m=.05)]:
            with self.assertRaises(ValueError):
                MID360SparseDepthConfig(**values)
        obs = observations()
        obs.move_to_end('depth')
        with self.assertRaises(ValueError):
            MID360SparseDepthAdapter(self.cfg)(obs)

    def test_provider_delegates_exact_contract_and_sparse_embeddings(self):
        class RGB(nn.Module):
            def forward(self, obs):
                return torch.zeros(obs['rgb'].shape[0], 512, 1, 1)

        class Depth(nn.Module):
            def forward(self, obs):
                self.seen = obs['depth'].clone()
                self.embedding = obs['depth'].mean((1, 2, 3)).reshape(-1, 1, 1, 1).expand(-1, 128, 4, 4)
                return self.embedding

        class Predictor(nn.Module):
            def forward(self, rgb, depth):
                self.seen = depth
                logits = torch.zeros(rgb.shape[0] // 12, 120, 12)
                logits[:, 0, 0] = 10
                logits[:, 30, 3] = 9
                return logits

        depth, predictor = Depth().eval(), Predictor().eval()
        common = dict(rgb_encoder=RGB().eval(), depth_encoder=depth,
                      rgb_pool=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(2)),
                      depth_pool=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(2)),
                      waypoint_predictor=predictor, pano_angle_features=torch.zeros(12, 4),
                      pano_image_indexes=np.arange(12))
        provider = MID360SparseDepthETPWaypointProvider(self.cfg, **common)
        obs = observations()
        adapted, _ = MID360SparseDepthAdapter(self.cfg)(obs)
        actual = provider.predict(obs)
        self.assertIsInstance(actual, WaypointOutput)
        self.assertIs(predictor.seen, depth.embedding)
        for b in range(2):
            for i, key in enumerate(DEPTH_KEYS):
                self.assertTrue(torch.equal(depth.seen[b * 12 + (12-i) % 12], adapted[key][b]))
        expected = OriginalETPWaypointProvider(**common).predict(adapted)
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            a, e = actual[key], expected[key]
            if torch.is_tensor(e):
                self.assertTrue(torch.equal(a, e), key)
            elif isinstance(e, np.ndarray):
                np.testing.assert_array_equal(a, e)
            else:
                for av, ev in zip(a, e):
                    if torch.is_tensor(ev):
                        self.assertTrue(torch.equal(av, ev), key)
                    else:
                        np.testing.assert_array_equal(av, ev)
        with self.assertRaises(RuntimeError):
            provider.predict(obs, in_train=True)
        predictor.train()
        with self.assertRaises(RuntimeError):
            provider.predict(obs)


if __name__ == '__main__':
    unittest.main()
