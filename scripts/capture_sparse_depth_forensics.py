#!/usr/bin/env python3
"""Capture exactly five reset observations, without invoking eval or rollout.

Run with etpnav_legacy Python after sourcing scripts/renderer_bootstrap.bash
and calling etp_apply_known_good_renderer_environment. Refuses existing output.
"""
import argparse
import inspect
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EPISODES = ['53', '821', '1024', '250', '25']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/mid360_sparse_depth_forensics')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSimDepthSensor
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.common.r2r_runtime import initialize_runtime_config, prepare_r2r_evaluation_environment
    from vlnce_baselines import ss_trainer_ETP as tm
    from analyze_sparse_depth_proposer_collapse import analyze_observation, state_fingerprint, fingerprint
    from sparse_depth_run_manifest import write_manifest, sha, CHECKPOINT

    assert torch.cuda.is_available(), 'require known-good GPU renderer environment'
    assert not any(k.startswith('ETP_') and 'FORENSICS' in k for k in os.environ), 'disable other forensic sinks'
    protected = ['vlnce_baselines', 'habitat_extensions', 'configs', 'run_r2r', 'run.py']
    subprocess.check_call(['git', 'diff', '--exit-code', 'b402412', '--'] + protected)
    write_manifest(root, EPISODES, [sys.executable, str(Path(__file__).resolve()), '--output', str(root)])
    formal_root = ROOT / 'outputs/mid360_sparse_depth_zero_shot20'
    evidence_before = {str(p.relative_to(formal_root)): sha(p) for p in sorted(formal_root.rglob('*')) if p.is_file()}
    formal = json.loads((formal_root / 'telemetry.json').read_text())['episodes']
    opts = ['SIMULATOR_GPU_IDS', '[0]', 'TORCH_GPU_IDS', '[0]', 'TORCH_GPU_ID', '0',
            'GPU_NUMBERS', '1', 'NUM_ENVIRONMENTS', '1', 'EVAL.SPLIT', 'val_unseen',
            'EVAL.EPISODE_IDS', repr(EPISODES), 'EVAL.EPISODE_COUNT', '5',
            'IL.ckpt_to_load', CHECKPOINT, 'EVAL.CKPT_PATH_DIR', CHECKPOINT,
            'RESULTS_DIR', str(root) + '/', 'IL.back_algo', 'control', 'MODEL.pretrained_path', 'None',
            'MODEL.WAYPOINT_PROVIDER', 'mid360_sparse_depth_etp',
            'MODEL.MID360_SPARSE_DEPTH_CONFIG_PATH', 'configs/mid360_sparse_depth_zero_shot.yaml',
            'VIDEO_OPTION', '[]', 'TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING', 'True']
    config = get_config('run_r2r/iter_train.yaml', opts)
    initialize_runtime_config(config, 0)
    prepare_r2r_evaluation_environment(config)
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_num_threads(1)
    trainer = tm.RLTrainer(config)
    torch.cuda.set_device(trainer.device)
    envs = tm.construct_envs(config, tm.get_env_class(config.ENV_NAME), episodes_allowed=EPISODES, auto_reset_done=False)
    report = dict(accepted_commit='b402412', episode_ids=EPISODES, navigation_steps=0,
                  observations={}, definitions=dict(std='population (ddof=0)',
                  cosine='flattened vectors; undefined for zero norm',
                  normalized_l2='||a-b||2 / max(||a||2, ||b||2, 1e-12)',
                  view_variance='mean over feature coordinates of population variance across 12 views',
                  top_cell_tie_order='stable angle-major order for reporting only; NMS uses Original torch.max',
                  angle='logit angle index increases clockwise; policy angle is (360 - 3*index) mod 360',
                  distances_m=[(i + 1) * .25 for i in range(12)]))
    try:
        assert sum(envs.number_of_episodes) == 5 and envs.num_envs == 1
        trainer.obs_transforms = tm.get_active_obs_transforms(config)
        space = tm.apply_obs_transforms_obs_space(envs.observation_spaces[0], trainer.obs_transforms)
        trainer._initialize_policy(config, True, space, envs.action_spaces[0])
        trainer.policy.eval()
        trainer.waypoint_predictor.eval()
        provider = trainer._evaluation_waypoint_provider()
        policy_net = trainer.policy.net.module if hasattr(trainer.policy.net, 'module') else trainer.policy.net
        assert provider.original.depth_encoder is policy_net.depth_encoder
        assert provider.original.waypoint_predictor is trainer.waypoint_predictor
        before = dict(policy=state_fingerprint(trainer.policy), predictor=state_fingerprint(trainer.waypoint_predictor))
        sensor = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR
        provider.adapter.config.validate_runtime(config)
        # Exercise installed Habitat's actual normalization at its maximum.
        depth_sensor = HabitatSimDepthSensor(sensor)
        normalized_max = depth_sensor.get_observation({'depth': np.full((256, 256), sensor.MAX_DEPTH, dtype=np.float32)})
        assert np.all(normalized_max == 1.0)
        report['sensor_audit'] = dict(min_depth_m=sensor.MIN_DEPTH, max_depth_m=sensor.MAX_DEPTH,
            normalize_depth=sensor.NORMALIZE_DEPTH, normalized_max_depth=1.0,
            installed_sensor_source=inspect.getfile(HabitatSimDepthSensor),
            normalization_source=inspect.getsource(HabitatSimDepthSensor.get_observation))
        (root / 'runtime_config.yaml').write_text(config.dump())
        (root / 'captured').mkdir()
        for _ in range(5):
            observations = envs.reset()
            episode = envs.current_episodes()[0]
            episode_id = str(episode.episode_id)
            assert episode_id in EPISODES and episode_id not in report['observations']
            tm.require_valid_sensor_observations(observations[0])
            raw_depth_hashes = {k: fingerprint(v) for k, v in observations[0].items() if 'depth' in k}
            pose = envs.call_at(0, 'get_agent_info')
            frozen = formal[episode_id]['waypoint_trace'][0]
            assert episode.scene_id == formal[episode_id]['scene_id']
            assert np.allclose(pose['position'], frozen['agent_position'], atol=1e-6, rtol=0)
            angle_delta = (pose['heading'] - frozen['agent_heading'] + np.pi) % (2 * np.pi) - np.pi
            assert abs(angle_delta) < 1e-6
            observations = tm.extract_instruction_tokens(observations, config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                         max_length=config.IL.max_text_len, pad_id=0)
            batch = tm.apply_obs_transforms_batch(tm.batch_obs(observations, trainer.device), trainer.obs_transforms)
            assert all(fingerprint(batch[k]) == h for k, h in raw_depth_hashes.items()), 'depth transformed'
            record, tensors = analyze_observation(batch, provider, sensor)
            record.update(episode_id=episode_id, decision_index=0, scene_id=episode.scene_id,
                          agent_pose=pose, formal_start_pose_matches=True, raw_depth_sha256=raw_depth_hashes)
            candidates = record['conditions']['SPARSE']['logits']['post_nms_candidates']
            record['formal_sparse_candidate_match'] = (
                len(candidates) == len(frozen['candidates']) and
                all(abs(c['distance_meters'] - f['distance']) < 1e-6 and
                    abs(c['angle_degrees_counterclockwise'] - np.degrees(f['angle']) % 360) < 1e-4
                    for c, f in zip(candidates, frozen['candidates'])))
            report['observations'][episode_id] = record
            # Small feature/logit artifacts, never depth or RGB image arrays.
            np.savez_compressed(str(root / 'captured' / (episode_id + '.npz')),
                                **{condition + '_' + name: value.cpu().numpy()
                                   for condition, captured in tensors.items()
                                   for name, value in captured.items() if torch.is_tensor(value)})
            with (root / 'captured' / (episode_id + '.json')).open('x') as handle:
                json.dump(record, handle, indent=2, allow_nan=False)
            print('CAPTURED', episode_id, 'formal candidates match:', record['formal_sparse_candidate_match'], flush=True)
        after = dict(policy=state_fingerprint(trainer.policy), predictor=state_fingerprint(trainer.waypoint_predictor))
        assert before == after
        assert set(report['observations']) == set(EPISODES)
        report['weights'] = dict(before=before, after=after, parameters_and_buffers_unchanged=True)
    finally:
        envs.close()
    evidence_after = {str(p.relative_to(formal_root)): sha(p) for p in sorted(formal_root.rglob('*')) if p.is_file()}
    assert evidence_before == evidence_after
    report['formal_evidence_unchanged'] = True
    report['formal_evidence_sha256'] = evidence_before
    subprocess.check_call(['git', 'diff', '--exit-code', 'b402412', '--'] + protected)
    with (root / 'analysis.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print('COMPLETE: five reset observations, 15 proposer passes, zero navigation steps', flush=True)


if __name__ == '__main__':
    main()
