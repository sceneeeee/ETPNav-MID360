"""Recovery tests use real Welford updates and atomic on-disk checkpoints."""
import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from sparse_depth_feature_alignment import ChannelMoments
from alignment_calibration_recovery import CalibrationProgress, SensorStability, atomic_json
import run_sparse_depth_feature_alignment as runner


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.selection = [dict(episode_id=e, scene_id=s) for e,s in [('95','a'),('26','b'),('10','c')]]
        self.provenance = dict(source_sha256={'a':'abc'}, config_sha256='def', checkpoint_sha256='ghi')
        self.shape = (12, 3, 2, 2)
        self.features = [torch.arange(144).double().reshape(self.shape) / (i+1) for i in range(3)]

    def progress(self):
        return CalibrationProgress(self.selection, self.provenance, self.shape)

    def add(self, progress, index):
        progress.add(self.selection[index], dict(dense=self.features[index], sparse=self.features[index]*.13))

    def test_welford_round_trip(self):
        for count in (0,1,3):
            original = ChannelMoments()
            for f in self.features[:count]:
                original.update(f)
            restored = ChannelMoments.from_state(json.loads(json.dumps(original.to_state())))
            self.assertEqual(original.to_state(), restored.to_state())

    def test_resumed_equals_uninterrupted_and_order_preserved(self):
        uninterrupted = self.progress()
        for i in range(3):
            self.add(uninterrupted, i)
        with TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.json'
            partial = self.progress()
            self.add(partial, 0)
            partial.save(path)
            resumed = CalibrationProgress.load(path, self.selection, self.provenance, self.shape)
            self.assertEqual(resumed.completed_ids, ['95'])
            self.assertEqual(resumed.next_index, 1)
            for i in range(1,3):
                self.add(resumed, i)
                resumed.save(path)
            self.assertEqual(resumed.completed_ids, ['95','26','10'])
            self.assertEqual(resumed.to_state(), uninterrupted.to_state())
            self.assertEqual(list(Path(directory).iterdir()), [path])
            for name in ('dense','sparse'):
                for a,b in zip(resumed.moments[name].finish(), uninterrupted.moments[name].finish()):
                    np.testing.assert_array_equal(a,b)

    def test_duplicate_wrong_order_scene_and_partial_pair_rejected_without_mutation(self):
        progress = self.progress()
        self.add(progress,0)
        before = copy.deepcopy(progress.to_state())
        for record in (self.selection[0], self.selection[2], dict(episode_id='26',scene_id='wrong')):
            with self.assertRaises(ValueError):
                progress.add(record, dict(dense=self.features[1],sparse=self.features[1]))
            self.assertEqual(before,progress.to_state())
        with self.assertRaises(ValueError):
            progress.add(self.selection[1],dict(dense=self.features[1],sparse=torch.zeros(12,3,2,3)))
        self.assertEqual(before,progress.to_state())

    def test_provenance_selection_shape_and_corruption_reject_resume(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.json'
            progress = self.progress()
            self.add(progress,0)
            progress.save(path)
            for key in self.provenance:
                bad = dict(self.provenance, **{key:'changed'})
                with self.assertRaisesRegex(ValueError,'provenance'):
                    CalibrationProgress.load(path,self.selection,bad,self.shape)
            with self.assertRaisesRegex(ValueError,'selection'):
                CalibrationProgress.load(path,self.selection[::-1],self.provenance,self.shape)
            with self.assertRaisesRegex(ValueError,'shape'):
                CalibrationProgress.load(path,self.selection,self.provenance,(12,3,2,3))
            state = progress.to_state()
            state['moments']['dense']['count'] += 1
            path.write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                CalibrationProgress.load(path,self.selection,self.provenance,self.shape)

    def test_recovery_introduces_no_trainable_parameters(self):
        progress = self.progress()
        for i in range(3):
            self.features[i].requires_grad_(True)
            self.add(progress,i)
        self.assertFalse(isinstance(progress,torch.nn.Module))
        for moment in progress.moments.values():
            self.assertIsInstance(moment.mean,np.ndarray)
            self.assertIsInstance(moment.m2,np.ndarray)
        self.assertTrue(all(f.grad is None for f in self.features))

    def test_lucky_and_inconsistent_sensor_frames_are_not_accepted(self):
        a = {'depth':dict(shape=[2,2,1], dtype='float32', sha256='a')}
        b = {'depth':dict(shape=[2,2,1], dtype='float32', sha256='b')}
        guard = SensorStability()
        for stats, valid in [(a,True),(a,False),(a,True),(b,True),(a,True)]:
            self.assertFalse(guard.observe(stats,valid))
        self.assertTrue(guard.observe(a,True))

    def test_atomic_failure_keeps_previous_checkpoint(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.json'
            atomic_json(path,dict(committed=1))
            with patch('alignment_calibration_recovery.os.replace',side_effect=OSError('crash before rename')):
                with self.assertRaises(OSError):
                    atomic_json(path,dict(committed=2))
            self.assertEqual(json.loads(path.read_text()),dict(committed=1))
            self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_completed_checkpoint_finalization_can_be_repeated_without_counting(self):
        progress = self.progress()
        for index in range(3):
            self.add(progress,index)
        before = copy.deepcopy(progress.to_state())
        manifest = dict(episode_ids=progress.completed_ids,scenes=['a','b','c'])
        with TemporaryDirectory() as directory:
            output = Path(directory)
            for _ in range(2):
                runner.finish_calibration(output,manifest,progress,{'policy':'frozen'},{'policy':'frozen'})
                with np.load(str(output/'feature_statistics.npz')) as values:
                    for name,moment in progress.moments.items():
                        mean,sigma=moment.finish()
                        np.testing.assert_array_equal(values['mu_'+name],mean)
                        np.testing.assert_array_equal(values['sigma_'+name],sigma)
            self.assertEqual(before,progress.to_state())
            self.assertFalse(list(output.glob('*.tmp')))

    def test_collector_closes_each_retry_and_fails_at_fixed_bound(self):
        class Environment:
            num_envs = 1
            number_of_episodes = [1]
            def __init__(self):
                self.closed = False
            def close(self):
                assert not self.closed
                self.closed = True
        config = SimpleNamespace(TASK_CONFIG=SimpleNamespace(DATASET=SimpleNamespace(SCENES_DIR='data')),
                                 ENV_NAME='test', clone=lambda:config)
        selected = dict(episode_id='26',scene_id='scene.glb')
        for valid_pattern, expected_count in [([False,True,True],3), ([True,False,True,False,True],5)]:
            envs, calls = [], []
            def construct(*args, **kwargs):
                self.assertEqual(kwargs['episodes_allowed'],['26'])
                env = Environment()
                envs.append(env)
                return env
            tm = SimpleNamespace(construct_envs=construct,get_env_class=lambda _:Environment)
            def reset(tm, config, trainer, env, path, calibration_only=False):
                self.assertTrue(calibration_only)
                self.assertFalse(env.closed)
                valid = valid_pattern[len(calls)]
                calls.append(env)
                record = dict(episode_id='26',scene_id='data/scene.glb',agent_pose=dict(position=[0,0,0],heading=0))
                atomic_json(path,dict(record,sensors={'depth':dict(shape=[2,2,1],dtype='float32',sha256='x')},
                                     rgb_valid=valid,depth_valid=valid))
                if not valid:
                    raise RuntimeError('invalid sensors')
                return {'depth':'unchanged'},record
            with TemporaryDirectory() as directory, patch.object(runner,'reset_batch',side_effect=reset):
                if expected_count == 3:
                    batch,record = runner.stable_calibration_batch(tm,config,None,None,Path(directory),0,selected,'test')
                    self.assertEqual(batch,{'depth':'unchanged'})
                    self.assertEqual(len(record['sensor_attempts']),3)
                else:
                    with self.assertRaisesRegex(RuntimeError,'five fresh lifecycles'):
                        runner.stable_calibration_batch(tm,config,None,None,Path(directory),0,selected,'test')
                self.assertEqual(len(list(Path(directory).glob('*.json'))),expected_count)
            self.assertEqual(len(envs),expected_count)
            self.assertTrue(all(e.closed for e in envs))


if __name__ == '__main__':
    unittest.main()
