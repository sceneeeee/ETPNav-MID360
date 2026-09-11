import sys
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_sparse_depth_proposer_collapse import (
    analyze_observation, capture_condition, diversity, logit_statistics,
    make_conditions, similarity, state_fingerprint,
)
from vlnce_baselines.perception.mid360_sparse_depth import (
    DEPTH_KEYS, MID360SparseDepthAdapter, MID360SparseDepthConfig,
)
from vlnce_baselines.waypoint_provider import original_etp


class RGBEncoder(nn.Module):
    def forward(self, obs):
        return obs['rgb'].float().mean((1, 2, 3)).reshape(12, 1, 1, 1).expand(12, 512, 1, 1)


class DepthEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.ones(1))

    def forward(self, obs):
        return (obs['depth'].mean((1, 2, 3)) * self.scale).reshape(12, 1, 1, 1).expand(12, 128, 4, 4)


class Predictor(nn.Module):
    def forward(self, rgb, depth):
        logits = torch.arange(1440).float().reshape(1, 120, 12) / 1000
        return logits + depth.mean()


class SparseForensicsTest(unittest.TestCase):
    def setUp(self):
        self.obs = OrderedDict()
        for i, key in enumerate(DEPTH_KEYS):
            self.obs[key.replace('depth', 'rgb')] = torch.full((1, 8, 8, 3), i + 1, dtype=torch.uint8)
            self.obs[key] = torch.linspace(.1 + i * .01, .8, 64).reshape(1, 8, 8, 1)
        self.obs['instruction'] = torch.tensor([[1, 2, 3]])
        self.adapter = MID360SparseDepthAdapter(MID360SparseDepthConfig.from_yaml(
            str(Path(__file__).resolve().parents[1] / 'configs/mid360_sparse_depth_zero_shot.yaml')))
        self.sensor = SimpleNamespace(NORMALIZE_DEPTH=True, MIN_DEPTH=0., MAX_DEPTH=10., HFOV=90., POSITION=[0, 1.25, 0])
        self.original = original_etp.OriginalETPWaypointProvider(
            RGBEncoder().eval(), DepthEncoder().eval(),
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(2)),
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(2)),
            Predictor().eval(), torch.zeros(12, 4), np.arange(12))

    def test_conditions_match_adapter_and_preserve_all_non_depth(self):
        before = {k: v.clone() for k, v in self.obs.items()}
        conditions, diagnostics = make_conditions(self.obs, self.adapter, self.sensor)
        expected, _ = MID360SparseDepthAdapter(self.adapter.config)(self.obs)
        for key in DEPTH_KEYS:
            self.assertTrue(torch.equal(conditions['SPARSE'][key], expected[key]))
            self.assertTrue(torch.all(conditions['ALL_FAR'][key] == 1).item())
        for condition in conditions.values():
            for key in self.obs:
                if key not in DEPTH_KEYS:
                    self.assertIs(condition[key], self.obs[key])
                self.assertTrue(torch.equal(self.obs[key], before[key]))
        self.assertAlmostEqual(diagnostics['valid_projected_fraction'] + diagnostics['no_return_fraction'], 1)

    def test_all_far_requires_audited_sensor(self):
        self.sensor.NORMALIZE_DEPTH = False
        with self.assertRaises(ValueError):
            make_conditions(self.obs, self.adapter, self.sensor)

    def test_same_models_frozen_state_and_raw_logits_before_original_nms(self):
        provider = SimpleNamespace(original=self.original, adapter=self.adapter)
        before = state_fingerprint(self.original.depth_encoder)
        with patch.object(original_etp, '_candidate_map', wraps=original_etp._candidate_map) as candidate_map:
            report, tensors = analyze_observation(self.obs, provider, self.sensor)
        self.assertEqual(candidate_map.call_count, 6)  # provider + offline replay, per condition
        for name, record in report['conditions'].items():
            guards = record['guards']
            self.assertEqual(guards['depth_encoder_object_id'], id(self.original.depth_encoder))
            self.assertEqual(guards['predictor_object_id'], id(self.original.waypoint_predictor))
            self.assertLess(guards['events'].index('raw_logits'), guards['events'].index('provider_return_after_nms'))
            self.assertTrue(guards['offline_nms_matches_provider'])
            self.assertGreater(float(tensors[name]['raw_logits'].max()), 1.)
        self.assertEqual(before, state_fingerprint(self.original.depth_encoder))
        self.assertEqual(len({r['guards']['rgb_sha256'] for r in report['conditions'].values()}), 1)

    def test_candidate_map_uses_exact_original_nms(self):
        raw = torch.randn(1, 120, 12)
        with patch.object(original_etp, 'nms', wraps=original_etp.nms) as nms:
            original_etp._candidate_map(raw, 1)
        nms.assert_called_once()
        packed = nms.call_args[0][0]
        probability = torch.softmax(raw.reshape(1, 1440), dim=1).reshape(1, 120, 12)
        self.assertTrue(torch.equal(packed[:, 0, 1:-1], probability))
        self.assertTrue(torch.equal(packed[:, 0, :1], probability[:, -1:]))
        self.assertEqual(nms.call_args[1], dict(max_predictions=5, sigma=(7.0, 5.0)))

    def test_mutating_buffer_is_detected_and_hooks_removed(self):
        def mutate(module, inputs, output):
            module.scale.add_(1)
        handle = self.original.depth_encoder.register_forward_hook(mutate)
        try:
            with self.assertRaisesRegex(AssertionError, 'parameters or buffers changed'):
                capture_condition(self.original, self.obs)
            self.assertEqual(len(self.original.waypoint_predictor._forward_hooks), 0)
        finally:
            handle.remove()

    def test_state_hash_includes_nonpersistent_buffers(self):
        module = nn.Module()
        module.register_buffer('scratch', torch.zeros(1), persistent=False)
        before = state_fingerprint(module)
        module.scratch.add_(1)
        self.assertNotEqual(before, state_fingerprint(module))

    def test_radial_peaks_and_tie_metrics(self):
        raw = torch.zeros(1, 120, 12)
        raw[:, 5::10, 2] = 4
        record = logit_statistics(raw, original_etp._candidate_map(raw, 1))
        self.assertEqual(record['cells_exactly_at_max'], 12)
        self.assertEqual(record['max_minus_second'], 0)
        self.assertEqual(record['periodic_radial2_peak_std'], 0)
        self.assertEqual(record['pre_nms_dominant_radial_index'], 2)
        self.assertTrue(record['post_nms_all_radial2'])
        self.assertEqual(record['top20_cells'][0]['distance_meters'], .75)
        # Preserve Original float32 softmax, including its rounding error.
        self.assertAlmostEqual(sum(record['radial_marginal']), 1., delta=5e-6)

    def test_similarity_and_diversity_known_values(self):
        a = torch.ones(12, 2)
        self.assertAlmostEqual(similarity(a, a)['cosine'], 1)
        self.assertEqual(similarity(a, a)['normalized_l2_difference'], 0)
        self.assertEqual(diversity(a)['feature_variance_across_views'], 0)
        self.assertAlmostEqual(diversity(a)['mean_pairwise_cosine'], 1)
        self.assertEqual(diversity(torch.eye(12))['mean_pairwise_cosine'], 0)


if __name__ == '__main__':
    unittest.main()
