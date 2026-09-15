import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from sparse_depth_learned_adapter import (
    ResidualFeatureAdapter, feature_distillation_loss,
    save_adapter_checkpoint, load_adapter_checkpoint,
)
import run_sparse_depth_learned_adapter_m0 as runner
from sparse_depth_feature_alignment import encode, predict_encoded, FROZEN_IDS, HELD_OUT_IDS
from analyze_sparse_depth_proposer_collapse import state_fingerprint
import test_sparse_depth_feature_alignment as stage4_fixtures
import test_sparse_depth_forensics as forensic_fixtures


class LearnedAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(7)

    def test_zero_init_identity_shape_dtype_device_and_no_mutation(self):
        adapter = ResidualFeatureAdapter()
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 33024)
        self.assertEqual([type(m) for m in adapter.children()], [nn.Conv2d, nn.GELU, nn.Conv2d])
        self.assertEqual(torch.count_nonzero(adapter.conv2.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(adapter.conv2.bias).item(), 0)
        for dtype in (torch.float32, torch.float64):
            adapter.to(dtype=dtype)
            for x in (torch.randn(12, 128, 4, 4, dtype=dtype),
                      torch.zeros(12, 128, 4, 4, dtype=dtype),
                      torch.full((12, 128, 4, 4), 10000., dtype=dtype)):
                before = x.clone()
                for training in (True, False):
                    adapter.train(training)
                    result = adapter(x)
                    self.assertTrue(torch.equal(result, before))
                    self.assertTrue(torch.equal(x, before))
                    self.assertEqual((result.shape, result.dtype, result.device), (x.shape, x.dtype, x.device))
                    self.assertEqual(adapter.activation.training, training)

    def test_input_guards(self):
        adapter = ResidualFeatureAdapter()
        for x in (torch.zeros(128, 4, 4), torch.zeros(1, 127, 4, 4),
                  torch.zeros(1, 128, 3, 4), torch.zeros(0, 128, 4, 4),
                  torch.zeros(1, 128, 4, 4, dtype=torch.long),
                  torch.full((1, 128, 4, 4), float('nan')),
                  torch.full((1, 128, 4, 4), float('inf'))):
            with self.assertRaises(ValueError):
                adapter(x)
        for channels in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                ResidualFeatureAdapter(channels)

    def test_loss_definition_per_item_cosine_and_autograd(self):
        a = torch.randn(3, 2, 4, 4, requires_grad=True)
        dense = torch.randn_like(a)
        dense[0].zero_()
        values = feature_distillation_loss(a, dense)
        per_item = torch.stack([F.cosine_similarity(a[i].reshape(-1), dense[i].reshape(-1),
                                                   dim=0, eps=1e-8) for i in range(3)])
        self.assertTrue(torch.equal(values['cosine_loss'], (1 - per_item).mean()))
        self.assertTrue(torch.equal(values['mean_cosine_similarity'], per_item.mean()))
        self.assertTrue(torch.equal(values['smooth_l1'], F.smooth_l1_loss(a, dense, beta=1.0)))
        self.assertTrue(torch.equal(values['total'], values['smooth_l1'] + values['cosine_loss']))
        values['total'].backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        with self.assertRaises(ValueError):
            feature_distillation_loss(a, dense[:1])
        with self.assertRaises(ValueError):
            feature_distillation_loss(a, dense, cosine_weight=float('nan'))

    def test_isolated_backward_only_adapter_trainable(self):
        encoder = nn.Conv2d(128, 128, 1).eval()
        predictor = nn.Linear(128, 12).eval()
        for module in (encoder, predictor):
            for p in module.parameters():
                p.requires_grad_(False)
        with torch.no_grad():
            sparse = encoder(torch.randn(12, 128, 4, 4))
            dense = encoder(torch.randn(12, 128, 4, 4))
        snapshots = (sparse.clone(), dense.clone())
        before = [state_fingerprint(m) for m in (encoder, predictor)]
        adapter = ResidualFeatureAdapter()
        trainable = {id(p) for m in (encoder, predictor, adapter) for p in m.parameters() if p.requires_grad}
        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertEqual(trainable, optimized)
        self.assertEqual(trainable, {id(p) for p in adapter.parameters()})
        feature_distillation_loss(adapter(sparse), dense)['total'].backward()
        for p in adapter.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        # Zero conv2 blocks conv1's first gradient; this is expected, not a failure.
        self.assertGreater(adapter.conv2.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        for value, snapshot in zip((sparse, dense), snapshots):
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad)
            self.assertTrue(torch.equal(value, snapshot))
        self.assertEqual(before, [state_fingerprint(m) for m in (encoder, predictor)])
        self.assertTrue(all(p.grad is None for m in (encoder, predictor) for p in m.parameters()))

    def test_synthetic_overfit_and_checkpoint_exact_round_trip(self):
        sparse = torch.randn(12, 128, 4, 4, requires_grad=True)
        dense = (sparse.detach() * .6 + .3).requires_grad_()
        before = (sparse.detach().clone(), dense.detach().clone())
        adapter, report = runner.optimize_pair(dict(sparse=sparse, dense=dense), steps=100, seed=0)
        self.assertLess(report['final']['total'], report['initial']['total'] * .25)
        self.assertTrue(report['gradients_finite'])
        self.assertEqual(report['max_identity_error_at_initialization'], 0)
        for value, snapshot in zip((sparse, dense), before):
            self.assertIsNone(value.grad)
            self.assertTrue(torch.equal(value, snapshot))
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'adapter.pt'
            save_adapter_checkpoint(path, adapter, seed=0, step=100)
            restored, metadata = load_adapter_checkpoint(path, expected_channels=128)
            restored.eval()
            self.assertEqual((metadata['seed'], metadata['step']), (0, 100))
            self.assertTrue(torch.equal(adapter(sparse), restored(sparse)))
            with self.assertRaises(FileExistsError):
                save_adapter_checkpoint(path, adapter)
            with self.assertRaises(ValueError):
                load_adapter_checkpoint(path, expected_channels=64)
            payload = torch.load(path, map_location='cpu')
            payload['architecture']['activation'] = 'ReLU'
            bad = Path(directory) / 'bad.pt'
            torch.save(payload, bad)
            with self.assertRaisesRegex(ValueError, 'architecture'):
                load_adapter_checkpoint(bad)
        print('\nSynthetic M0: ' + json.dumps(report, sort_keys=True))

    def test_original_waypoint_identity_and_raw_policy_features(self):
        fixture = forensic_fixtures.SparseForensicsTest()
        fixture.setUp()
        original = fixture.original
        original.waypoint_predictor = stage4_fixtures.RecordingPredictor().eval()
        modules = (original.rgb_encoder, original.depth_encoder, original.waypoint_predictor)
        before = [state_fingerprint(m) for m in modules]
        sparse, _ = fixture.adapter(fixture.obs)
        with torch.no_grad():
            encoded = encode(original, sparse)
            raw_embedding, raw_policy = encoded.depth_embedding.clone(), encoded.depth_features.clone()
            baseline = predict_encoded(original, encoded, alignment=None)
            adapter = ResidualFeatureAdapter().eval()
            identity = predict_encoded(original, encoded, alignment=adapter)
            for i in (1, 2, 3):  # logits, Original NMS candidate map, predictor depth input
                self.assertTrue(torch.equal(baseline[i], identity[i]))
            self.assertGreater(torch.count_nonzero(baseline[2]).item(), 0)
            for key in ('cand_angles', 'cand_distances'):
                self.assertEqual(baseline[0][key], identity[0][key])
            self.assertTrue(torch.equal(identity[0]['pano_depth'], raw_policy))
            # Nonidentity probe proves predictor-only routing even after learning.
            adapter.conv2.bias.fill_(.2)
            changed = predict_encoded(original, encoded, alignment=adapter)
            self.assertFalse(torch.equal(changed[3], raw_embedding))
            self.assertTrue(torch.equal(encoded.depth_embedding, raw_embedding))
            self.assertTrue(torch.equal(encoded.depth_features, raw_policy))
            self.assertTrue(torch.equal(changed[0]['pano_depth'], raw_policy))
            for result in (identity, changed):
                for candidates, indexes in zip(result[0]['cand_depth'], result[0]['cand_img_idxes']):
                    self.assertTrue(torch.equal(candidates, raw_policy[0, indexes]))
        self.assertEqual(before, [state_fingerprint(m) for m in modules])

    def test_runner_id_bounds_and_pair_guards(self):
        stage4 = stage4_fixtures.runner
        for eid in FROZEN_IDS | set(HELD_OUT_IDS):
            for spelling in (eid, '00' + eid):
                with self.assertRaises(ValueError):
                    runner.validate_episode_id(spelling, stage4)
        self.assertEqual(runner.validate_episode_id('0137', stage4), '137')
        for steps, lr in ((0, .001), (1001, .001), (1, 0), (1, float('inf'))):
            with self.assertRaises(ValueError):
                runner.validate_optimization(steps, lr)
        with self.assertRaises(ValueError):
            runner.validate_pair(dict(dense=torch.zeros(12, 128, 4, 4), sparse=torch.zeros(1, 128, 4, 4)))

    def test_collection_closes_environment_and_checks_model_state(self):
        stage4 = stage4_fixtures.runner
        for mutate in (False, True):
            encoder = nn.Conv2d(128, 128, 1)
            predictor = nn.Linear(128, 12)
            policy = nn.ModuleList([encoder, predictor])
            trainer = SimpleNamespace(policy=policy)
            original = SimpleNamespace(depth_encoder=encoder, waypoint_predictor=predictor,
                                       rgb_encoder=nn.Identity())
            provider = SimpleNamespace(original=original)
            events = []
            envs = SimpleNamespace(close=lambda: events.append('close'))
            config = SimpleNamespace(TASK_CONFIG=SimpleNamespace(DATASET=SimpleNamespace(SCENES_DIR='data/scenes')))

            def reset(*args, **kwargs):
                events.append('reset')
                return {}, dict(episode_id='137', scene_id='data/scenes/test.glb')

            def collect(*args):
                events.append('features')
                self.assertEqual(events, ['reset', 'close', 'features'])
                self.assertTrue(all(not p.requires_grad for p in policy.parameters()))
                with torch.no_grad():
                    if mutate:
                        encoder.weight.add_(1)
                return dict(dense=torch.ones(12, 128, 4, 4), sparse=torch.zeros(12, 128, 4, 4))

            with TemporaryDirectory() as directory, \
                    patch.object(stage4, 'runtime', return_value=(None, config, trainer, envs, provider)) as runtime, \
                    patch.object(stage4, 'reset_batch', side_effect=reset) as reset_mock, \
                    patch.object(stage4, 'calibration_features', side_effect=collect) as collect_mock:
                selected = dict(episode_id='137', scene_id='test.glb')
                if mutate:
                    with self.assertRaisesRegex(RuntimeError, 'changed during collection'):
                        runner.collect_pair(stage4, selected, Path(directory))
                else:
                    pair, report = runner.collect_pair(stage4, selected, Path(directory))
                    self.assertTrue(report['original_states_unchanged'])
                    self.assertTrue(all(v.device.type == 'cpu' and not v.requires_grad for v in pair.values()))
                self.assertEqual(runtime.call_args[0][:2], ('val_seen', ['137']))
                self.assertTrue(runtime.call_args[1]['calibration_only'])
                self.assertEqual(reset_mock.call_count, 1)
                self.assertEqual(collect_mock.call_count, 1)
                self.assertEqual(events.count('close'), 1)


if __name__ == '__main__':
    unittest.main()
