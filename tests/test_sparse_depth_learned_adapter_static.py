import ast
import gzip
import inspect
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_sparse_depth_learned_adapter_static as runner
from sparse_depth_learned_adapter import ResidualFeatureAdapter, save_adapter_checkpoint
from vlnce_baselines.waypoint_provider import original_etp
import test_sparse_depth_forensics as fixtures


class BoundaryPredictor(nn.Module):
    """Active upstream ReLU; sparse classifier dead, dense classifier active."""
    def __init__(self):
        super().__init__()
        self.visual_fc_depth = nn.Sequential(nn.Flatten(), nn.Linear(2048, 2), nn.ReLU(True))
        self.waypoint_TRM = nn.Identity()
        self.vis_classifier = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 120))
        with torch.no_grad():
            self.visual_fc_depth[1].weight[0].fill_(1. / 2048)
            self.visual_fc_depth[1].weight[1].fill_(-1. / 2048)
            self.visual_fc_depth[1].bias.zero_()
            self.vis_classifier[0].weight.copy_(-torch.eye(2))
            self.vis_classifier[0].bias.copy_(torch.tensor([.7, -.1]))
            self.vis_classifier[2].weight[:, 0].copy_(torch.linspace(-2, 2, 120))
            self.vis_classifier[2].weight[:, 1].zero_()
            self.vis_classifier[2].bias.copy_(torch.arange(120).float() / 1000)
        self.inputs = []

    def forward(self, rgb, depth):
        if torch.is_grad_enabled():
            raise RuntimeError('inference must disable autograd')
        self.inputs.append((rgb.detach().clone(), depth.detach().clone()))
        front = self.visual_fc_depth(depth)
        hidden = self.waypoint_TRM(front.reshape(1, 12, 2))
        return self.vis_classifier(hidden).reshape(1, 120, 12)


class StaticAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(4)
        self.fixture = fixtures.SparseForensicsTest()
        self.fixture.setUp()
        # Coarse synthetic angular cells keep the tiny 8x8 fixture sparse.
        # The host runner always uses the accepted unchanged YAML configuration.
        self.fixture.adapter = fixtures.MID360SparseDepthAdapter(
            fixtures.MID360SparseDepthConfig(azimuth_bin_degrees=90., elevation_bin_degrees=90.))
        self.original = self.fixture.original
        self.original.waypoint_predictor = BoundaryPredictor()
        self.provider = SimpleNamespace(original=self.original, adapter=self.fixture.adapter)
        self.adapter = ResidualFeatureAdapter()
        for module in list(runner.original_modules(self.original).values()) + [self.adapter]:
            runner.freeze(module)

    def evaluate(self):
        return runner.evaluate_observation(self.fixture.obs, self.provider,
                                           self.fixture.sensor, self.adapter)

    def test_same_observation_two_encodings_raw_policy_and_original_nms_identity(self):
        source = {k: v.clone() for k, v in self.fixture.obs.items()}
        encoded = []
        real_encode = runner.stage4.encode

        def encode(*args):
            value = real_encode(*args)
            encoded.append((value, value.depth_embedding.clone(), value.depth_features.clone()))
            return value

        with patch.object(runner.stage4, 'encode', side_effect=encode) as encoding, \
                patch.object(original_etp, 'nms', wraps=original_etp.nms) as nms:
            record, tensors, outputs = self.evaluate()
        self.assertEqual(encoding.call_count, 2)
        self.assertIs(encoding.call_args_list[0][0][1], self.fixture.obs)
        sparse, _ = self.fixture.adapter(self.fixture.obs)
        for key in sparse:
            self.assertTrue(torch.equal(encoding.call_args_list[1][0][1][key], sparse[key]))
        self.assertEqual(nms.call_count, 3)
        for call in nms.call_args_list:
            self.assertEqual(call[1], dict(max_predictions=5, sigma=(7.0, 5.0)))
        for field in tensors['SPARSE']:
            self.assertTrue(torch.equal(tensors['SPARSE'][field], tensors['A1_ADAPTED'][field]), field)
        self.assertGreater(torch.count_nonzero(tensors['SPARSE']['candidate_map']).item(), 0)
        for name in ('cand_angles', 'cand_distances'):
            self.assertEqual(outputs['SPARSE'][name], outputs['A1_ADAPTED'][name])
        for value, embedding, policy in encoded:
            self.assertTrue(torch.equal(value.depth_embedding, embedding))
            self.assertTrue(torch.equal(value.depth_features, policy))
        for name in ('SPARSE', 'A1_ADAPTED'):
            self.assertTrue(torch.equal(outputs[name]['pano_depth'], encoded[1][2]))
        self.assertTrue(all(torch.equal(v, source[k]) for k, v in self.fixture.obs.items()))
        self.assertEqual(record['guards']['state_before'], record['guards']['state_after'])
        with torch.no_grad():
            dense_reference = self.original.predict(self.fixture.obs, in_train=False)
            sparse_reference = self.original.predict(sparse, in_train=False)
        for name, reference in [('DENSE', dense_reference), ('SPARSE', sparse_reference)]:
            for key in ('cand_angles', 'cand_distances'):
                self.assertEqual(reference[key], outputs[name][key])
        self.assertEqual(record['features']['delta_smooth_l1'], 0)
        self.assertEqual(record['features']['delta_cosine'], 0)

    def test_nonidentity_metrics_and_correct_classifier_boundary(self):
        with torch.no_grad():
            self.adapter.conv2.bias.fill_(-.45)
        record, tensors, outputs = self.evaluate()
        self.assertLess(record['features']['delta_smooth_l1'], 0)
        sparse, adapted = record['conditions']['SPARSE'], record['conditions']['A1_ADAPTED']
        self.assertGreater(sparse['visual_fc_depth_post_relu']['positive_count'], 0)
        self.assertEqual(sparse['classifier_post_relu']['positive_count'], 0)
        self.assertGreater(adapted['classifier_post_relu']['positive_count'], 0)
        self.assertTrue((tensors['SPARSE']['visual_fc_depth_pre_relu'][:, 1] < 0).all())
        self.assertTrue((tensors['SPARSE']['classifier_pre_relu'] < 0).all())
        self.assertFalse(torch.equal(tensors['SPARSE']['raw_logits'], tensors['A1_ADAPTED']['raw_logits']))
        self.assertTrue(torch.equal(tensors['SPARSE']['depth_embedding'], tensors['A1_ADAPTED']['depth_embedding']))
        self.assertTrue(torch.equal(tensors['SPARSE']['predictor_input'] - .45,
                                    tensors['A1_ADAPTED']['predictor_input']))
        self.assertTrue(torch.equal(outputs['SPARSE']['pano_depth'], outputs['A1_ADAPTED']['pano_depth']))
        self.assertTrue(all(torch.equal(rgb, self.original.waypoint_predictor.inputs[0][0])
                            for rgb, _ in self.original.waypoint_predictor.inputs))
        self.assertFalse(any(m._forward_hooks for m in self.original.waypoint_predictor.modules()))
        self.assertTrue(all(not t.requires_grad and t.grad_fn is None
                            for values in tensors.values() for t in values.values()))
        self.assertEqual(record['guards']['state_before'], record['guards']['state_after'])

    def test_feature_metrics_known_values_per_view_and_zero_norm(self):
        dense = torch.ones(2, 128, 4, 4)
        sparse = torch.zeros_like(dense)
        sparse[1].fill_(-1)
        result = runner.feature_metrics(sparse, dense, dense)
        self.assertAlmostEqual(result['SPARSE_vs_DENSE']['smooth_l1'], 1.)
        self.assertAlmostEqual(result['SPARSE_vs_DENSE']['mean_cosine_similarity'], -.5, places=6)
        self.assertEqual(result['A1_ADAPTED_vs_DENSE']['smooth_l1'], 0)
        self.assertAlmostEqual(result['A1_ADAPTED_vs_DENSE']['mean_cosine_similarity'], 1., places=6)
        self.assertAlmostEqual(result['delta_smooth_l1'], -1.)
        self.assertAlmostEqual(result['delta_cosine'], 1.5, places=6)

    def test_feature_improvement_does_not_imply_classifier_rescue(self):
        with torch.no_grad():
            self.adapter.conv2.bias.fill_(-.05)
        record, _, _ = self.evaluate()
        self.assertLess(record['features']['delta_smooth_l1'], 0)
        self.assertEqual(record['conditions']['A1_ADAPTED']['classifier_post_relu']['positive_count'], 0)
        self.assertIn('candidates', record['conditions']['A1_ADAPTED'])

    def test_candidate_summary_known_values_and_empty(self):
        record = dict(logits=dict(post_nms_candidates=[dict(distance_index=i) for i in [2, 2, 7]]))
        output = dict(cand_angles=[[0., np.pi, 2 * np.pi]], cand_distances=[[.75, .75, 2.]])
        values = runner.candidate_metrics(record, output)
        self.assertEqual(values['distinct_radial_bins'], 2)
        self.assertEqual(values['distinct_distances'], 2)
        self.assertEqual(values['median_distance_meters'], .75)
        self.assertAlmostEqual(values['mean_distance_meters'], 3.5 / 3)
        self.assertEqual(values['angles_degrees_counterclockwise'], [0, 180, 0])
        empty = runner.candidate_metrics(dict(logits=dict(post_nms_candidates=[])),
                                         dict(cand_angles=[[]], cand_distances=[[]]))
        self.assertIsNone(empty['median_distance_meters'])
        self.assertIsNone(empty['mean_distance_meters'])

    def test_frozen_modes_and_mutating_adapter_or_original_are_rejected(self):
        self.adapter.train()
        with self.assertRaisesRegex(RuntimeError, 'eval mode'):
            self.evaluate()
        runner.freeze(self.adapter)
        self.original.waypoint_predictor.vis_classifier[0].weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, 'frozen'):
            self.evaluate()
        runner.freeze(self.original.waypoint_predictor)
        for module, value in [(self.adapter, self.adapter.conv2.bias),
                              (self.original.depth_encoder, self.original.depth_encoder.scale)]:
            def mutate(m, inputs, output):
                value.add_(.01)
            handle = module.register_forward_hook(mutate)
            try:
                with self.assertRaisesRegex(RuntimeError, 'parameters/buffers changed'):
                    self.evaluate()
                self.assertFalse(any(m._forward_hooks for m in self.original.waypoint_predictor.modules()))
            finally:
                handle.remove()

    def test_shared_relu_site_is_rejected(self):
        core = self.original.waypoint_predictor
        core.vis_classifier[1] = core.visual_fc_depth[2]
        with self.assertRaisesRegex(AssertionError, 'ReLU sites must be distinct'):
            self.evaluate()

    def test_formal_id_guard_and_smoke_exclusions(self):
        exact = ['53', '821', '1024', '250', '25']
        self.assertEqual(runner.validate_ids('formal'), exact)
        self.assertEqual(runner.validate_ids('formal', exact), exact)
        self.assertEqual(runner.validate_ids('formal', ['00' + e for e in exact]), exact)
        for ids in ([], exact[:4], exact + ['137'], exact[:-1] + ['137'], exact[::-1], exact + ['25']):
            with self.assertRaises(ValueError):
                runner.validate_ids('formal', ids)
        for eid in runner.stage4.FROZEN_IDS:
            with self.assertRaises(ValueError):
                runner.validate_ids('nonformal-smoke', ['00' + eid])
        for ids in (None, [], ['137', '138'], ['1.0'], [True]):
            with self.assertRaises(ValueError):
                runner.validate_ids('nonformal-smoke', ids)
        self.assertEqual(runner.validate_ids('nonformal-smoke', ['0137']), ['137'])

    def test_checkpoint_metadata_strict_load_and_preflight_before_runtime(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'adapter.pt'
            save_adapter_checkpoint(checkpoint, self.adapter)
            restored, record = runner.load_frozen_adapter(checkpoint)
            self.assertEqual(record['sha256'], runner.stage4.sha(checkpoint))
            self.assertEqual(record['parameter_count'], 33024)
            runner.require_frozen({'adapter': restored})
            payload = torch.load(checkpoint, map_location='cpu')
            mutations = [lambda p: p.pop('architecture'),
                         lambda p: p['architecture'].update(activation='ReLU'),
                         lambda p: p.pop('loss'),
                         lambda p: p['loss'].update(smooth_l1_weight=2.),
                         lambda p: p['state_dict'].pop('conv2.bias'),
                         lambda p: p['state_dict']['conv1.weight'].fill_(float('nan'))]
            for index, mutate in enumerate(mutations):
                payload = torch.load(checkpoint, map_location='cpu')
                mutate(payload)
                bad = root / ('bad_{}.pt'.format(index))
                torch.save(payload, bad)
                with patch.object(runner.stage4, 'runtime') as runtime:
                    with self.assertRaises((ValueError, RuntimeError, KeyError)):
                        runner.run('formal', None, bad, root / 'formal_output')
                    runtime.assert_not_called()
                    self.assertFalse((root / 'formal_output').exists())
            with self.assertRaises(FileNotFoundError):
                runner.run('formal', None, root / 'missing.pt', root / 'formal_output')
            with self.assertRaises(FileExistsError):
                runner.run('formal', None, checkpoint, root)
            with self.assertRaisesRegex(ValueError, 'nonformal_'):
                runner.run('nonformal-smoke', ['137'], checkpoint, root / 'formal_output')

    def test_smoke_dataset_hash_unique_membership_and_missing_id(self):
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / 'synthetic.json.gz'
            for episodes in ([dict(episode_id='137', scene_id='synthetic.glb')],
                             [dict(episode_id='137'), dict(episode_id='0137')], []):
                with gzip.open(str(dataset), 'wt') as handle:
                    json.dump(dict(episodes=episodes), handle)
                with patch.object(runner.stage4, 'CALIBRATION_DATA', str(dataset)), \
                        patch.object(runner.stage4, 'CALIBRATION_SHA', runner.stage4.sha(dataset)):
                    if len(episodes) == 1:
                        selected, metadata = runner.select_dataset('nonformal-smoke', ['137'])
                        self.assertEqual(selected['137'], episodes[0])
                        self.assertEqual(metadata['sha256'], runner.stage4.sha(dataset))
                    else:
                        with self.assertRaisesRegex(ValueError, 'exactly once'):
                            runner.select_dataset('nonformal-smoke', ['137'])
                with patch.object(runner.stage4, 'CALIBRATION_DATA', str(dataset)), \
                        patch.object(runner.stage4, 'CALIBRATION_SHA', 'invalid'):
                    with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                        runner.select_dataset('nonformal-smoke', ['137'])

    def test_mocked_smoke_lifecycle_manifest_and_failure_reporting(self):
        # Synthetic reset only: never imports/constructs a real Habitat environment.
        for mutate in (False, True):
            with TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / 'adapter.pt'
                save_adapter_checkpoint(checkpoint, self.adapter)
                output = root / 'nonformal_synthetic'
                events = []
                envs = SimpleNamespace(close=lambda: events.append('close'))
                trainer = SimpleNamespace(policy=nn.ModuleList(list(runner.original_modules(self.original).values())),
                                          device=torch.device('cpu'))
                config = SimpleNamespace(IL=SimpleNamespace(is_requeue=False), TASK_CONFIG=SimpleNamespace(
                    DATASET=SimpleNamespace(SCENES_DIR=str(root)),
                    SIMULATOR=SimpleNamespace(DEPTH_SENSOR=self.fixture.sensor), SEED=0))
                scene = root / 'synthetic.glb'
                scene.write_bytes(b'synthetic scene fixture')
                def runtime(*args, **kwargs):
                    self.assertEqual(args[:2], ('val_seen', ['137']))
                    self.assertEqual(kwargs, dict(prepared_config=config))
                    trainer.optimizer = torch.optim.AdamW(trainer.policy.parameters(), lr=.001)
                    for name in ('step', 'zero_grad', 'load_state_dict', 'add_param_group', 'state_dict'):
                        setattr(trainer.optimizer, name, Mock(side_effect=AssertionError(
                            'optimizer API invoked after construction: ' + name)))
                    (output / 'runtime_config.yaml').write_text('synthetic: true')
                    return None, config, trainer, envs, self.provider
                def reset(*args):
                    events.append('reset')
                    self.assertFalse(torch.is_grad_enabled())
                    return self.fixture.obs, dict(episode_id='137', scene_id=str(scene), decision_index=0)
                real_sha = runner.stage4.sha
                def sha(path):
                    name = str(path)
                    if name in runner.stage4.EXPECTED:
                        return runner.stage4.EXPECTED[name]
                    if name == 'data/ddppo-models/gibson-2plus-resnet50.pth':
                        return 'synthetic-ddppo'
                    return real_sha(path)
                real_evaluate = runner.evaluate_observation
                def evaluate(*args):
                    result = real_evaluate(*args)
                    if mutate:
                        args[3].conv2.bias.add_(.1)
                    return result
                with patch.object(runner, 'select_dataset', return_value=(
                        {'137': dict(scene_id='synthetic.glb')}, dict(path='synthetic', sha256='synthetic'))), \
                        patch.object(runner.stage4, 'sha', side_effect=sha), \
                        patch.object(runner, 'source_hashes', return_value={'synthetic.py': 'synthetic'}), \
                        patch.object(torch.cuda, 'is_available', return_value=True), \
                        patch.dict(os.environ, {'LD_PRELOAD': ':'.join('/lib/x86_64-linux-gnu/' + p for p in
                            ['libGLdispatch.so.0', 'libGLX.so.0', 'libOpenGL.so.0'])}), \
                        patch.object(runner.stage4, 'runtime', side_effect=runtime), \
                        patch.object(runner.stage4, 'runtime_config', return_value=config), \
                        patch.object(runner.stage4, 'reset_batch', side_effect=reset), \
                        patch.object(runner, 'evaluate_observation', side_effect=evaluate), \
                        patch.object(torch.Tensor, 'backward', side_effect=AssertionError('backward forbidden')) as backward, \
                        patch.object(torch.autograd, 'backward', side_effect=AssertionError('backward forbidden')) as autograd_backward, \
                        patch.object(torch.autograd, 'grad', side_effect=AssertionError('grad forbidden')) as grad:
                    if mutate:
                        with self.assertRaisesRegex(RuntimeError, 'state changed'):
                            runner.run('nonformal-smoke', ['137'], checkpoint, output)
                    else:
                        runner.run('nonformal-smoke', ['137'], checkpoint, output)
                    backward.assert_not_called()
                    autograd_backward.assert_not_called()
                    grad.assert_not_called()
                for name in ('step', 'zero_grad', 'load_state_dict', 'add_param_group', 'state_dict'):
                    getattr(trainer.optimizer, name).assert_not_called()
                self.assertEqual(events, ['reset', 'close'])
                report = json.loads((output / 'NONFORMAL_report.json').read_text())
                self.assertEqual(report['status'], 'NONFORMAL_FAILED' if mutate else 'NONFORMAL_COMPLETE')
                self.assertEqual(report['stage'], 'M1_STATIC_NONFORMAL_SMOKE')
                self.assertFalse(report['formal_evaluation'])
                self.assertEqual(report['training_steps'], 0)
                self.assertTrue(report['legacy_optimizer_constructed'])
                self.assertEqual(report['navigation_steps'], 0)
                self.assertEqual(report['episode_ids'], ['137'])
                self.assertEqual(report['states_unchanged'], not mutate)
                self.assertFalse((output / 'report.json').exists())
                self.assertIn('no held-out evidence', (output / 'NONFORMAL_report.md').read_text())
                with np.load(str(output / 'capture_137.npz'), allow_pickle=False) as captured:
                    self.assertIn('A1_ADAPTED_classifier_post_relu', captured.files)

    def test_formal_markdown_has_required_labels_without_execution(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            report = dict(stage='M1_STATIC', formal_evaluation=True, training_steps=0,
                          navigation_steps=0, status='STARTED', episode_ids=runner.validate_ids('formal'),
                          adapter_checkpoint=dict(sha256='synthetic'), observations={})
            runner.write_report(output, report)
            text = (output / 'report.md').read_text()
            for value in ('stage = "M1_STATIC"', 'formal_evaluation = true', 'training_steps = 0',
                          'navigation_steps = 0', '53, 821, 1024, 250, 25', 'CASE A', 'CASE B', 'CASE C'):
                self.assertIn(value, text)

    def test_local_capture_matches_historical_stage4_capture(self):
        sparse, _ = self.fixture.adapter(self.fixture.obs)
        with torch.no_grad():
            self.adapter.conv2.bias.fill_(-.45)
            for batch, alignment in ((self.fixture.obs, None), (sparse, None), (sparse, self.adapter)):
                encoded = runner.stage4.encode(self.original, batch)
                expected, historical, reference = runner.stage4.capture(self.original, batch, alignment)
                actual, captured, output = runner.capture_encoded(self.original, encoded, alignment)
                self.assertEqual(actual, expected)
                for name, value in historical.items():
                    self.assertTrue(torch.equal(captured[name], value), name)
                self.assertTrue(torch.equal(captured['candidate_map'],
                    original_etp._candidate_map(historical['raw_logits'], encoded.batch_size)))
                for key in ('cand_angles', 'cand_distances'):
                    self.assertEqual(output[key], reference[key])
                self.assertTrue(torch.equal(output['pano_depth'], reference['pano_depth']))
                self.assertFalse(any(m._forward_hooks for m in self.original.waypoint_predictor.modules()))

    def test_requeue_rejected_before_legacy_runtime(self):
        config = SimpleNamespace(IL=SimpleNamespace(is_requeue=True))
        with TemporaryDirectory() as directory, \
                patch.object(runner.stage4, 'runtime_config', return_value=config), \
                patch.object(runner.stage4, 'runtime') as runtime:
            with self.assertRaisesRegex(ValueError, 'cannot resume training'):
                runner.collect_static(Path(directory), ['137'], {}, self.adapter, dict(split='val_seen'))
            runtime.assert_not_called()

    def test_no_training_or_navigation_calls_in_static_path(self):
        sources = [Path(runner.__file__).read_text()]
        for function in (runner.stage4.runtime, runner.stage4.runtime_config,
                         runner.stage4.reset_batch, runner.stage4.encode, runner.stage4.predict_encoded,
                         original_etp.encode_panorama, original_etp._candidate_map,
                         original_etp.construct_waypoint_output):
            sources.append(inspect.getsource(function))
        forbidden = {'step', 'backward', 'train', 'zero_grad', 'Adam', 'SGD',
                     'optimize_pair', 'calibrate', 'evaluate', 'rollout', '_training_candidates'}
        for source in sources:
            tree = ast.parse(source, feature_version=(3, 6)) if sys.version_info >= (3, 8) else ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
                    self.assertNotIn(name, forbidden)
                if isinstance(node, ast.Attribute):
                    self.assertNotEqual(node.attr, 'optim')


if __name__ == '__main__':
    unittest.main()
