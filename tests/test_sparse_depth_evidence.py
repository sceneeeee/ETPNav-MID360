import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collect_mid360_sparse_depth_zero_shot20 import collect, EPISODE_IDS, BASELINE
from vlnce_baselines.config.default import get_config
from vlnce_baselines.common.r2r_runtime import initialize_runtime_config, prepare_r2r_evaluation_environment
from vlnce_baselines.perception.mid360_sparse_depth import MID360SparseDepthConfig, DEPTH_KEYS
from vlnce_baselines.ss_trainer_ETP import RLTrainer


class SparseEvidenceTest(unittest.TestCase):
    def test_calibration_and_training_guard(self):
        runtime = get_config('run_r2r/iter_train.yaml')
        initialize_runtime_config(runtime, 0)
        prepare_r2r_evaluation_environment(runtime)
        cfg = MID360SparseDepthConfig.from_yaml('configs/mid360_sparse_depth_zero_shot.yaml')
        cfg.validate_runtime(runtime)
        runtime.defrost()
        runtime.TASK_CONFIG.SIMULATOR.DEPTH_90.ORIENTATION = [0, -1.570796, 0]
        with self.assertRaises(ValueError):
            cfg.validate_runtime(runtime)
        runtime.MODEL.WAYPOINT_PROVIDER = 'mid360_sparse_depth_etp'
        with self.assertRaises(RuntimeError):
            RLTrainer.train(SimpleNamespace(config=runtime))
        runtime.MODEL.WAYPOINT_PROVIDER = 'original_etp'
        self.assertIsNone(RLTrainer._evaluation_waypoint_provider(SimpleNamespace(config=runtime)))

    def test_factory_passes_original_component_identities(self):
        runtime = get_config('run_r2r/iter_train.yaml', ['MODEL.WAYPOINT_PROVIDER', 'mid360_sparse_depth_etp'])
        initialize_runtime_config(runtime, 0)
        prepare_r2r_evaluation_environment(runtime)
        net = SimpleNamespace(**{k: object() for k in ['rgb_encoder', 'depth_encoder', 'space_pool_rgb', 'space_pool_depth', 'pano_angle_fts', 'pano_img_idxes']})
        trainer = SimpleNamespace(config=runtime, policy=SimpleNamespace(net=net), waypoint_predictor=object())
        original = RLTrainer._evaluation_waypoint_provider(trainer).original
        for target, source in [('rgb_encoder', 'rgb_encoder'), ('depth_encoder', 'depth_encoder'), ('rgb_pool', 'space_pool_rgb'), ('depth_pool', 'space_pool_depth'), ('pano_angle_features', 'pano_angle_fts'), ('pano_image_indexes', 'pano_img_idxes')]:
            self.assertIs(getattr(original, target), getattr(net, source))
        self.assertIs(original.waypoint_predictor, trainer.waypoint_predictor)

    def test_collector_complete_evidence_and_reject_partial_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'mid360_sparse_depth_zero_shot20'
            raw.mkdir()
            def write(path, value):
                path.write_text(json.dumps(value))
            write(root / 'run_manifest.json', dict(episode_ids=EPISODE_IDS, provider='mid360_sparse_depth_etp'))
            metrics = dict(success=.5, spl=.4, distance_to_goal=4, oracle_success=.6, ndtw=.7, sdtw=.3, path_length=8, steps_taken=42)
            stats = {eid: metrics for eid in EPISODE_IDS}
            write(raw / 'stats_ep_ckpt_59_val_unseen_r0_w1.json', stats)
            row = dict(valid_projected_pixels_per_view={k: 1 for k in DEPTH_KEYS}, valid_projected_pixels=12,
                       total_depth_pixels=120, valid_projected_fraction=.1, filtered_cloud_points=6, no_return_fill_normalized=1.)
            telemetry = dict(summary=dict(cuda_memory=dict(peak_allocated_mib=1000)), episodes={eid: dict(scene_id='fixture', decision_latency_ms=[10, 30], waypoint_trace=[dict(sparse_depth=row), dict(sparse_depth=row)]) for eid in EPISODE_IDS})
            write(raw / 'telemetry_ep_ckpt_59_val_unseen_r0_w1.json', telemetry)
            write(raw / 'sensor_validity_ep_ckpt_59_val_unseen_r0_w1.json', {eid: dict(rgb_valid=True, depth_valid=True) for eid in EPISODE_IDS})
            outputs = collect(root)
            self.assertEqual(len(outputs), 5)
            summary = outputs['summary.json']
            self.assertAlmostEqual(summary['signed_delta']['SR'], .5 - BASELINE['SR'])
            self.assertAlmostEqual(summary['absolute_delta']['SR'], .3)
            self.assertEqual(summary['current']['mean_decision_latency_ms'], 20)
            self.assertEqual(summary['current']['p95_decision_latency_ms'], 30)
            sparse = outputs['sparse_depth_summary.json']
            self.assertAlmostEqual(sparse['valid_projected_fraction']['median'], .1)
            del stats['53']
            write(raw / 'stats_ep_ckpt_59_val_unseen_r0_w1.json', stats)
            with self.assertRaises(ValueError):
                collect(root)


if __name__ == '__main__':
    unittest.main()
