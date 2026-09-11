"""CPU regression tests. Habitat construction/CUDA/model loading are mocked."""
import copy
import json
from collections import OrderedDict
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from habitat.config.default import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_sparse_depth_feature_alignment as runner
from alignment_calibration_recovery import CalibrationProgress
from sparse_depth_feature_alignment import calibration_depth_config, encode_calibration_depth, encode
from vlnce_baselines.evaluation_sensor_diagnostics import require_valid_sensor_observations
import test_sparse_depth_forensics as fixtures

ARCHIVE = ROOT / 'outputs/mid360_sparse_depth_alignment_calibration'
DEPTH = ['DEPTH_SENSOR'] + ['DEPTH_{}'.format(a) for a in range(30, 360, 30)]


def accepted_config():
    config = Config()
    config.merge_from_file(str(ARCHIVE / 'runtime_config.yaml'))
    config.freeze()
    return config


class DepthOnlyTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SparseForensicsTest()
        self.fixture.setUp()
        self.provider = SimpleNamespace(original=self.fixture.original, adapter=self.fixture.adapter)
        self.batch = OrderedDict((k, v) for k,v in self.fixture.obs.items() if not k.startswith('rgb'))

    def test_clone_filters_both_levels_preserves_every_other_config_value_and_disk(self):
        path = ARCHIVE / 'runtime_config.yaml'
        before = path.read_bytes()
        original = accepted_config()
        dump = original.dump()
        self.assertEqual(len(original.SENSORS), 24)
        self.assertEqual(len(original.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS), 24)
        result = calibration_depth_config(original)
        self.assertEqual(result.SENSORS, DEPTH)
        self.assertEqual(result.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS, DEPTH)
        self.assertFalse(any(s.startswith('RGB') for s in result.SENSORS))
        self.assertTrue(result.is_frozen())
        restored = result.clone()
        restored.defrost()
        restored.SENSORS = list(original.SENSORS)
        restored.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = list(original.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS)
        self.assertEqual(restored, original)
        self.assertEqual(original.dump(), dump)
        self.assertEqual(path.read_bytes(), before)

    def test_missing_reordered_or_mismatched_depth_config_rejected(self):
        for level in ('top', 'agent'):
            config = accepted_config().clone()
            config.defrost()
            target = config if level == 'top' else config.TASK_CONFIG.SIMULATOR.AGENT_0
            target.SENSORS = list(reversed(target.SENSORS))
            config.freeze()
            with self.assertRaisesRegex(ValueError, 'ordered Depth'):
                calibration_depth_config(config)

    def test_depth_encoder_packing_bit_exact_to_original_without_rgb_encoder_or_predictor(self):
        original = self.provider.original
        # Also test a nonnumeric insertion order: Original consumes dict order.
        for observations in (self.fixture.obs, OrderedDict(reversed(list(self.fixture.obs.items())))):
            expected = encode(original, observations).depth_embedding
            depth = OrderedDict((k,v) for k,v in observations.items() if not k.startswith('rgb'))
            with patch.object(original.rgb_encoder, 'forward', side_effect=AssertionError('RGB called')), \
                    patch.object(original.waypoint_predictor, 'forward', side_effect=AssertionError('predictor called')):
                actual = encode_calibration_depth(original, depth)
            self.assertTrue(torch.equal(actual, expected))
            self.assertFalse(actual.requires_grad)

    def test_shared_feature_path_uses_real_sparse_adapter_and_finite_cpu_moments(self):
        before = {k: v.clone() for k,v in self.batch.items()}
        features = runner.calibration_features(self.provider, self.batch)
        for name, value in features.items():
            moment = runner.ChannelMoments()
            moment.update(value)
            self.assertEqual(value.shape, (12, 128, 4, 4))
            self.assertEqual((moment.observations, moment.count), (1, 192))
            self.assertTrue(np.isfinite(moment.finish()).all())
        self.assertTrue(all(torch.equal(before[k], v) for k,v in self.batch.items()))

    def test_no_rgb_missing_extra_views_and_unfrozen_encoder_rejected(self):
        for bad in (dict(self.batch, rgb=torch.zeros(1)),
                    {k:v for k,v in self.batch.items() if k != 'depth_30'},
                    dict(self.batch, depth_extra=self.batch['depth'])):
            with self.assertRaisesRegex(RuntimeError, 'exactly 12 Depth'):
                runner.calibration_features(self.provider, bad)
        self.provider.original.depth_encoder.train()
        with self.assertRaisesRegex(RuntimeError, 'frozen Original'):
            encode_calibration_depth(self.provider.original, self.batch)

    def reset_fixture(self, raw=None):
        raw = {k: v[0].numpy() for k,v in self.batch.items()} if raw is None else raw
        envs = Mock()
        envs.reset.return_value = [raw]
        envs.current_episodes.return_value = [SimpleNamespace(
            episode_id='137', scene_id='data/scene_datasets/mp3d/B6ByNegPMKs/B6ByNegPMKs.glb')]
        envs.call_at.return_value = dict(position=[0,0,0], heading=0)
        tm = SimpleNamespace(require_valid_sensor_observations=require_valid_sensor_observations,
                             extract_instruction_tokens=lambda obs, *args, **kwargs: obs,
                             batch_obs=lambda obs, device: {k:torch.from_numpy(v).unsqueeze(0) for k,v in obs[0].items()},
                             apply_obs_transforms_batch=lambda batch, _:batch)
        trainer = SimpleNamespace(device='cpu', obs_transforms=[])
        return tm, trainer, envs

    def test_reset_validates_depth_and_reports_sensor_keys_before_features(self):
        for defect in (None, 'rgb', 'missing', 'zero', 'nan', 'all_far', 'out_of_range'):
            raw = {k:v[0].numpy().copy() for k,v in self.batch.items()}
            if defect == 'rgb':
                raw['rgb'] = np.ones((8,8,3), dtype=np.uint8)
            elif defect == 'missing':
                del raw['depth_30']
            elif defect is not None:
                raw['depth_30'].fill({'zero':0, 'nan':np.nan, 'all_far':1, 'out_of_range':2}[defect])
            tm, trainer, envs = self.reset_fixture(raw)
            with TemporaryDirectory() as directory:
                path = Path(directory) / 'sensors.json'
                if defect is None:
                    batch, record = runner.reset_batch(tm, accepted_config(), trainer, envs, path, calibration_only=True)
                    self.assertEqual(set(batch), set(self.batch))
                    diagnostic = json.loads(path.read_text())
                    self.assertEqual(diagnostic['runtime_depth_key_count'], 12)
                    self.assertEqual(diagnostic['runtime_rgb_key_count'], 0)
                    self.assertTrue(diagnostic['depth_valid'])
                else:
                    with self.assertRaises(RuntimeError):
                        runner.reset_batch(tm, accepted_config(), trainer, envs, path, calibration_only=True)
                self.assertTrue(path.exists())
                envs.reset.assert_called_once_with()

    def test_runtime_factory_only_changes_sensors_for_explicit_calibration(self):
        original = accepted_config()
        self.provider.original.depth_encoder.spatial_output = False
        policy = Mock()
        policy.net = SimpleNamespace(depth_encoder=self.provider.original.depth_encoder)
        trainer = SimpleNamespace(device='cpu', policy=policy,
                                  waypoint_predictor=self.provider.original.waypoint_predictor,
                                  _initialize_policy=Mock(), _evaluation_waypoint_provider=lambda:self.provider)
        envs = SimpleNamespace(num_envs=1, number_of_episodes=[1], observation_spaces=[{}],
                               action_spaces=[{}], close=Mock())
        tm = SimpleNamespace(RLTrainer=lambda config:trainer, construct_envs=Mock(return_value=envs),
                             get_env_class=lambda _:None, get_active_obs_transforms=lambda _:[],
                             apply_obs_transforms_obs_space=lambda space, _:space)
        for calibration in (False, True):
            with TemporaryDirectory() as directory, patch('vlnce_baselines.ss_trainer_ETP', tm), \
                    patch.object(torch.cuda, 'set_device'):
                output = Path(directory)
                accepted = output / 'runtime_config.yaml'
                accepted.write_text(original.dump())
                before = accepted.read_bytes()
                runner.runtime('val_seen', ['137'], output, prepared_config=original, calibration_only=calibration)
                passed = tm.construct_envs.call_args[0][0]
                expected = DEPTH if calibration else original.SENSORS
                self.assertEqual(passed.SENSORS, expected)
                self.assertEqual(passed.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS, expected)
                self.assertEqual(accepted.read_bytes(), before)
                self.assertEqual(len(original.SENSORS), 24)

    def test_actual_accepted_checkpoint_still_loads_read_only_and_changed_lock_rejects(self):
        path = ARCHIVE / 'running_checkpoint.json'
        before = path.read_bytes()
        state = json.loads(before)
        progress = CalibrationProgress.load(path, state['selection'], state['provenance'], state['feature_shape'])
        self.assertEqual((len(progress.selection), progress.next_index), (53, 11))
        self.assertEqual(progress.selection[11]['episode_id'], '137')
        changed = copy.deepcopy(state['provenance'])
        changed['calibration_runtime_config_sha256'] = 'new Depth-only config'
        with self.assertRaisesRegex(ValueError, 'provenance changed'):
            CalibrationProgress.load(path, state['selection'], changed, state['feature_shape'])
        self.assertEqual(path.read_bytes(), before)

    def test_smoke_archive_checks_real_selection_and_reports_only_intentional_source_delta(self):
        before = (ARCHIVE / 'running_checkpoint.json').read_bytes()
        progress, changed = runner.smoke_archive(ARCHIVE)
        self.assertEqual(progress.next_index, 11)
        self.assertEqual(set(changed), {'scripts/run_sparse_depth_feature_alignment.py',
                                       'scripts/sparse_depth_feature_alignment.py'})
        self.assertEqual((ARCHIVE / 'running_checkpoint.json').read_bytes(), before)

    def test_smoke_cli_rejects_archive_destinations_and_resume_before_runtime(self):
        for output, extra in ((ARCHIVE, []), (ARCHIVE / 'smoke', []),
                              (ROOT / 'outputs/unused-smoke-test', ['--resume']),
                              (ROOT / 'outputs/unused-smoke-test', ['--stop-after', '12'])):
            argv = ['runner', 'calibration-smoke', '--output', str(output)] + extra
            with patch.object(sys, 'argv', argv), patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(runner, 'runtime') as runtime, patch('sys.stderr'):
                with self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(error.exception.code, 2)
                runtime.assert_not_called()

    def test_smoke_one_reset_fresh_moments_no_checkpoint_writes_and_cleanup_on_encoder_failure(self):
        state = json.loads((ARCHIVE / 'running_checkpoint.json').read_text())
        for fail_encoder in (False, True):
            tm, trainer, envs = self.reset_fixture()
            trainer.policy = self.provider.original.depth_encoder
            trainer.waypoint_predictor = self.provider.original.waypoint_predictor
            before = runner.model_hashes(trainer)
            progress = CalibrationProgress(state['selection'], state['provenance'], state['feature_shape'])
            progress.records = copy.deepcopy(state['records'])
            for record in progress.records:
                record['model_state_after'] = before
            provenance = {k:progress.provenance[k] for k in
                          ('git_commit', 'torch_version', 'numpy_version', 'gpu', 'LD_PRELOAD')}
            provenance.update(navigation_steps=0, training_steps=0)
            with TemporaryDirectory() as directory:
                archive, output = Path(directory)/'archive', Path(directory)/'smoke'
                archive.mkdir()
                output.mkdir()
                checkpoint = archive / 'running_checkpoint.json'
                checkpoint.write_bytes(b'untouched accepted checkpoint sentinel')
                config = accepted_config().clone()
                config.defrost()
                config.MODEL.DEPTH_ENCODER.ddppo_checkpoint = str(checkpoint)
                config.freeze()
                (archive / 'runtime_config.yaml').write_text(config.dump())
                progress.provenance['depth_initialization_checkpoint_sha256'] = runner.sha(checkpoint)
                snapshot = copy.deepcopy(progress.to_state())
                def runtime(split, ids, destination, env_ids, prepared, calibration_only=False):
                    self.assertTrue(calibration_only)
                    self.assertEqual(env_ids, ['137'])
                    self.assertEqual(ids, [r['episode_id'] for r in state['selection']])
                    self.assertEqual(len(prepared.SENSORS), 24)
                    effective = calibration_depth_config(prepared)
                    (destination / 'calibration_runtime_config.yaml').write_text(effective.dump())
                    return tm, effective, trainer, envs, self.provider
                features = runner.calibration_features
                with patch.object(runner, 'runtime', side_effect=runtime), \
                        patch.object(runner, 'smoke_archive', return_value=(progress, {})), \
                        patch.object(runner, 'provenance', return_value=provenance), \
                        patch.object(CalibrationProgress, 'save', side_effect=AssertionError('checkpoint write')), \
                        patch.object(CalibrationProgress, 'add', side_effect=AssertionError('checkpoint advance')), \
                        patch.object(runner, 'calibration_features', side_effect=(RuntimeError('encoder failure')
                                     if fail_encoder else features)):
                    if fail_encoder:
                        with self.assertRaisesRegex(RuntimeError, 'encoder failure'):
                            runner.calibration_smoke(output, archive)
                    else:
                        runner.calibration_smoke(output, archive)
                report = json.loads((output/'smoke.json').read_text())
                self.assertTrue(report['archive_unchanged'])
                self.assertEqual(report['status'], 'STARTED' if fail_encoder else 'PASS')
                self.assertEqual(report['resets'], 1)
                self.assertEqual(report['checkpoint_writes'], 0)
                self.assertEqual(report['navigation_steps'], 0)
                self.assertEqual(report['training_steps'], 0)
                if not fail_encoder:
                    self.assertEqual(report['observations'], 1)
                    self.assertEqual(report['model_state_before'], report['model_state_after'])
                    self.assertTrue(report['same_frozen_models_as_checkpoint'])
                    self.assertTrue(all(m['observations'] == 1 and m['count'] == 192 for m in report['moments'].values()))
                self.assertEqual(snapshot, progress.to_state())
                self.assertFalse((output / 'running_checkpoint.json').exists())
                envs.close.assert_called_once_with()
                envs.reset.assert_called_once_with()
                envs.step.assert_not_called()


if __name__ == '__main__':
    unittest.main()
