import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from sparse_depth_feature_alignment import (
    ChannelMoments, EPSILON, FROZEN_IDS, HELD_OUT_IDS, FixedChannelAlignment,
    MID360SparseDepthFeatureAlignedETP, select_calibration_episodes, validate_calibration_ids,
)
from analyze_sparse_depth_proposer_collapse import state_fingerprint
import test_sparse_depth_forensics as forensic_fixtures
import run_sparse_depth_feature_alignment as runner


class RecordingPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.tensor(2.))
        self.last_depth = None

    def forward(self, rgb, depth):
        self.last_depth = depth.detach().clone()
        return torch.arange(1440, device=depth.device).float().reshape(1, 120, 12) / 1000 + depth.mean() * self.scale


class FeatureAlignmentTest(unittest.TestCase):
    def test_formula_broadcast_shape_dtype_device_no_parameters_or_mutation(self):
        alignment = FixedChannelAlignment([10, 20], [2, 4], [1, 3], [2, 4])
        self.assertEqual(list(alignment.parameters()), [])
        before = state_fingerprint(alignment)
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float16, torch.float32, torch.float64):
                x = torch.arange(24, device=device).to(dtype).reshape(3, 2, 2, 2)
                original = x.clone()
                y = alignment(x)
                expected = np.empty((3, 2, 2, 2))
                for n in range(3):
                    for c in range(2):
                        for h in range(2):
                            for w in range(2):
                                expected[n, c, h, w] = ((float(x[n, c, h, w]) - [1, 3][c]) /
                                    ([2, 4][c] + EPSILON) * [2, 4][c] + [10, 20][c])
                np.testing.assert_allclose(y.cpu().numpy(), expected, rtol=1e-3 if dtype == torch.float16 else 1e-6)
                self.assertEqual((y.shape, y.dtype, y.device), (x.shape, x.dtype, x.device))
                self.assertTrue(torch.equal(x, original))
                self.assertFalse(y.requires_grad)
        self.assertEqual(before, state_fingerprint(alignment))

    def test_zero_sigma_and_invalid_statistics(self):
        alignment = FixedChannelAlignment([7], [0], [3], [0])
        self.assertEqual(alignment(torch.full((2, 1, 2, 3), 3.)).unique().tolist(), [7.])
        with self.assertRaises(ValueError):
            FixedChannelAlignment([1], [-1], [1], [1])
        with self.assertRaises(ValueError):
            FixedChannelAlignment([1], [1, 2], [1], [1])
        with self.assertRaises(ValueError):
            alignment(torch.zeros(2, 2, 2, 3))
        with self.assertRaises(ValueError):
            alignment(torch.full((2, 1, 2, 3), float('nan')))

    def test_streaming_moments_reduce_observation_view_and_spatial_only(self):
        x = torch.arange(96).double().reshape(2, 3, 2, 2, 4) / 10
        moments = ChannelMoments()
        for observation in x:
            moments.update(observation)
        mean, sigma = moments.finish()
        expected = x.numpy()
        np.testing.assert_allclose(mean, expected.mean(axis=(0, 1, 3, 4)), atol=1e-12)
        np.testing.assert_allclose(sigma, expected.std(axis=(0, 1, 3, 4)), atol=1e-12)
        self.assertEqual(moments.count, 48)
        self.assertEqual(mean.shape, (2,))

    def test_calibration_rejects_every_frozen_id_and_wrong_split(self):
        for eid in FROZEN_IDS:
            with self.assertRaises(ValueError):
                validate_calibration_ids('val_seen', ['9999', eid])
        for split in ('val_unseen', 'test'):
            with self.assertRaises(ValueError):
                validate_calibration_ids(split, ['9999'])
        validate_calibration_ids('val_seen', ['1', '3'])
        validate_calibration_ids('train', ['1', '3'])
        with self.assertRaises(ValueError):
            validate_calibration_ids('val_seen', ['1', '1'])

    def test_selection_is_one_lowest_eligible_id_per_scene(self):
        episodes = [dict(episode_id=e, scene_id=s, success=score) for e, s, score in
                    [(53, 'a', 1), (56, 'a', 0), (54, 'a', 1), (2, 'b', 0), (3, 'b', 1)]]
        self.assertEqual([e['episode_id'] for e in select_calibration_episodes(episodes)], [54, 3])
        for e in episodes:
            e['success'] = 1 - e['success']
        self.assertEqual([e['episode_id'] for e in select_calibration_episodes(episodes[::-1])], [54, 3])

    def test_analysis_provider_preserves_original_paths_models_and_policy_features(self):
        fixture = forensic_fixtures.SparseForensicsTest()
        fixture.setUp()
        original = fixture.original
        original.waypoint_predictor = RecordingPredictor().eval()
        sparse_provider = SimpleNamespace(original=original, adapter=fixture.adapter)
        alignment = FixedChannelAlignment(np.ones(128), np.full(128, 2), np.zeros(128), np.ones(128))
        aligned = MID360SparseDepthFeatureAlignedETP(sparse_provider, alignment)
        self.assertIs(aligned.original.depth_encoder, original.depth_encoder)
        self.assertIs(aligned.original.waypoint_predictor, original.waypoint_predictor)
        modules = [original.depth_encoder, original.rgb_encoder, original.waypoint_predictor, alignment]
        before = [state_fingerprint(m) for m in modules]
        source = {k: v.clone() for k, v in fixture.obs.items()}
        sparse, _ = fixture.adapter(fixture.obs)
        dense_before = original.predict(fixture.obs)
        sparse_before = original.predict(sparse)
        raw_features = original.waypoint_predictor.last_depth.clone()
        result = aligned.predict(fixture.obs)
        self.assertTrue(torch.equal(original.waypoint_predictor.last_depth, alignment(raw_features)))
        self.assertTrue(torch.equal(result['pano_depth'], sparse_before['pano_depth']))
        self.assertTrue(torch.equal(result['pano_rgb'], sparse_before['pano_rgb']))
        dense_after = original.predict(fixture.obs)
        sparse_after = original.predict(sparse)
        for a, b in [(dense_before, dense_after), (sparse_before, sparse_after)]:
            for key in ('cand_angles', 'cand_distances'):
                self.assertEqual(a[key], b[key])
            self.assertTrue(torch.equal(a['pano_depth'], b['pano_depth']))
        self.assertEqual(before, [state_fingerprint(m) for m in modules])
        self.assertTrue(all(torch.equal(source[k], v) for k, v in fixture.obs.items()))
        with self.assertRaises(RuntimeError):
            aligned.predict(fixture.obs, in_train=True)
        original.depth_encoder.train()
        with self.assertRaises(RuntimeError):
            aligned.predict(fixture.obs)

    def test_capture_distinguishes_active_frontend_from_dead_classifier(self):
        core = nn.Module()
        core.visual_fc_depth = nn.Sequential(nn.Flatten(), nn.Linear(2, 2), nn.ReLU(True))
        core.waypoint_TRM = nn.Identity()
        core.vis_classifier = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 120))
        with torch.no_grad():
            core.visual_fc_depth[1].weight.copy_(torch.eye(2))
            core.visual_fc_depth[1].bias.zero_()
            core.vis_classifier[0].weight.copy_(-torch.eye(2))
            core.vis_classifier[0].bias.fill_(-1)
        features = torch.tensor([1., -2.]).repeat(12, 1)
        encoded = SimpleNamespace(depth_embedding=features, rgb_embedding=features)
        original = SimpleNamespace(waypoint_predictor=core, depth_encoder=nn.Identity())

        def predict(*args):
            front = core.visual_fc_depth(features)
            hidden = core.waypoint_TRM(front.reshape(1, 12, 2))
            logits = core.vis_classifier(hidden).reshape(1, 120, 12)
            return dict(cand_angles=[[0.]], cand_distances=[[.75]]), logits, torch.zeros_like(logits), features

        with patch.object(runner, 'encode', return_value=encoded), patch.object(runner, 'predict_encoded', side_effect=predict):
            result, captured, _ = runner.capture(original, {})
        self.assertEqual(result['visual_fc_depth_post_relu']['positive_count'], 12)
        self.assertEqual(result['visual_fc_depth_post_relu']['positive_fraction'], .5)
        self.assertEqual(result['classifier_post_relu']['positive_count'], 0)
        self.assertEqual(result['classifier_post_relu']['positive_fraction'], 0)
        # A negative value must survive the snapshot before the in-place front ReLU.
        self.assertEqual(captured['visual_fc_depth_pre_relu'][0, 1].item(), -2)
        self.assertTrue((captured['classifier_pre_relu'] < 0).all())
        self.assertFalse(any(m._forward_hooks for m in core.modules()))
        report = dict(research_question='Upstream alignment / downstream classifier rescue?',
                      observations={'53': dict(conditions={name: result for name in
                          ('DENSE', 'SPARSE', 'SPARSE_ALIGNED', 'ALL_FAR')})})
        with TemporaryDirectory() as directory:
            runner.write_activity_report(Path(directory), report)
            text = (Path(directory) / 'report.md').read_text()
        self.assertIn('classifier_post_relu positive_count and positive_fraction', text)
        self.assertIn('| SPARSE_ALIGNED | 50.000000 | 0 | 0.000000 |', text)
        # Reusing one ReLU for both sites would otherwise pass shape/value checks.
        core.vis_classifier[1] = core.visual_fc_depth[2]
        with patch.object(runner, 'encode', return_value=encoded):
            with self.assertRaisesRegex(AssertionError, 'ReLU sites must be distinct'):
                runner.capture(original, {})

    def test_completed_evidence_guards_if_available(self):
        root = ROOT / 'outputs/mid360_sparse_depth_feature_alignment'
        if not (root / 'analysis.json').exists():
            self.skipTest('static evidence not yet collected')
        manifest = json.loads((root / 'calibration_manifest.json').read_text())
        report = json.loads((root / 'analysis.json').read_text())
        validate_calibration_ids(manifest['calibration_split'], manifest['episode_ids'])
        self.assertEqual(set(report['observations']), set(HELD_OUT_IDS))
        self.assertEqual(report['navigation_steps'], 0)
        self.assertEqual(report['model_state_before'], report['model_state_after'])
        self.assertEqual(report['model_state_before'], manifest['model_state_after'])
        self.assertTrue(report['alignment_buffers_unchanged'])
        for observation in report['observations'].values():
            for name in ('DENSE', 'SPARSE', 'SPARSE_ALIGNED', 'ALL_FAR'):
                condition = observation['conditions'][name]
                for prefix in ('visual_fc_depth', 'classifier'):
                    self.assertEqual(condition[prefix + '_pre_relu']['positive_count'],
                                     condition[prefix + '_post_relu']['positive_count'])
                    self.assertEqual(condition[prefix + '_pre_relu']['positive_fraction'],
                                     condition[prefix + '_post_relu']['positive_fraction'])
                if name in ('SPARSE', 'ALL_FAR'):
                    self.assertGreater(condition['visual_fc_depth_post_relu']['positive_fraction'], 0)
                    self.assertEqual(condition['classifier_post_relu']['positive_fraction'], 0)
            self.assertEqual(report['primary_mechanism_metrics'],
                             ['classifier_post_relu.positive_count', 'classifier_post_relu.positive_fraction'])
            for name in ('DENSE', 'SPARSE', 'ALL_FAR'):
                self.assertTrue(observation['conditions'][name]['accepted_forensics_encoder_and_logits_bit_identical'])


if __name__ == '__main__':
    unittest.main()
