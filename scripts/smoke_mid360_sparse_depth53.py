#!/usr/bin/env python3
"""Full episode 53, with a bounded first-decision, same-observation control.

Instrumentation lives only in this smoke entry point. Formal evaluation has no
extra dense inference or tensor fingerprint overhead. Never persists sensors.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))


def fingerprint(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def state_fingerprint(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(fingerprint(value).encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: require the known-good GPU renderer environment')
    from run import run_exp
    import vlnce_baselines.ss_trainer_ETP as trainer_module
    from vlnce_baselines.perception.mid360_sparse_depth import DEPTH_KEYS, MID360SparseDepthAdapter
    from vlnce_baselines.waypoint_provider.mid360_sparse_depth_etp import MID360SparseDepthETPWaypointProvider
    from vlnce_baselines.evaluation_telemetry import serialize_waypoint_candidates
    from sparse_depth_run_manifest import write_manifest

    exp_name = 'mid360_sparse_depth_smoke53'
    opts = [
        'SIMULATOR_GPU_IDS', '[0]', 'TORCH_GPU_IDS', '[0]', 'TORCH_GPU_ID', '0',
        'GPU_NUMBERS', '1', 'NUM_ENVIRONMENTS', '1', 'EVAL.SPLIT', 'val_unseen',
        'EVAL.EPISODE_IDS', "['53']", 'EVAL.EPISODE_COUNT', '1',
        'EVAL.CKPT_PATH_DIR', 'data/logs/checkpoints/release_r2r/ckpt.iter12000.pth',
        'RESULTS_DIR', str(root) + '/', 'EVAL.DETAILED_TELEMETRY', 'True',
        'IL.back_algo', 'control', 'MODEL.pretrained_path', 'None',
        'MODEL.WAYPOINT_PROVIDER', 'mid360_sparse_depth_etp',
        'MODEL.MID360_SPARSE_DEPTH_CONFIG_PATH', 'configs/mid360_sparse_depth_zero_shot.yaml',
        'VIDEO_OPTION', '[]', 'TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING', 'True',
    ]
    write_manifest(root, ['53'], ['run.py', '--exp_name', exp_name, '--run-type', 'eval',
                                '--exp-config', 'run_r2r/iter_train.yaml'] + opts)
    report = dict(episode_id='53', comparison='same first raw Habitat observation', training=False)
    def write():
        (root / 'first_decision_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    guard = trainer_module.require_valid_sensor_observations
    def capture_raw(obs):
        result = guard(obs)
        if 'raw_habitat_depth_sha256' not in report:
            report['raw_habitat_depth_sha256'] = {k: fingerprint(obs[k]) for k in DEPTH_KEYS}
        return result
    trainer_module.require_valid_sensor_observations = capture_raw

    factory = trainer_module.RLTrainer._evaluation_waypoint_provider
    holder = {}
    def capture_models(trainer):
        holder['trainer'] = trainer
        report['weights_before'] = dict(policy=state_fingerprint(trainer.policy), predictor=state_fingerprint(trainer.waypoint_predictor))
        return factory(trainer)
    trainer_module.RLTrainer._evaluation_waypoint_provider = capture_models
    predict = MID360SparseDepthETPWaypointProvider.predict
    def compare_first(provider, observations, in_train=False):
        if 'sparse_candidates' in report:
            return predict(provider, observations, in_train)
        before = {k: fingerprint(v) for k, v in observations.items()}
        assert all(before[k] == report['raw_habitat_depth_sha256'][k] for k in DEPTH_KEYS), 'raw Habitat depth differs from proposer source'
        adapted, _ = MID360SparseDepthAdapter(provider.adapter.config)(observations)
        evidence = {}
        active = ['dense']
        encoder_outputs = {}
        def depth_input(module, inputs):
            packed = inputs[0]['depth']
            source = observations if active[0] == 'dense' else adapted
            assert packed.shape[0] == 12
            for i, key in enumerate(DEPTH_KEYS):
                assert torch.equal(packed[(12-i) % 12], source[key][0]), 'encoder did not receive intended depth'
            evidence[active[0]] = dict(encoder_input_sha256=fingerprint(packed), encoder_input_verified=True)
        def depth_output(module, inputs, output):
            encoder_outputs[active[0]] = fingerprint(output)
        def predictor_input(module, inputs):
            actual = fingerprint(inputs[1])
            assert actual == encoder_outputs[active[0]], 'predictor input differs from depth encoder output'
            evidence[active[0]].update(predictor_depth_embedding_sha256=actual, predictor_embedding_verified=True)
        handles = [provider.original.depth_encoder.register_forward_pre_hook(depth_input),
                   provider.original.depth_encoder.register_forward_hook(depth_output),
                   provider.original.waypoint_predictor.register_forward_pre_hook(predictor_input)]
        try:
            # Preserve the formal run's RNG state around the additional control pass.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
                dense = provider.original.predict(observations, in_train=False)
            active[0] = 'sparse'
            sparse = predict(provider, observations, in_train=False)
        finally:
            for handle in handles:
                handle.remove()
        assert all(fingerprint(v) == before[k] for k, v in observations.items()), 'source observations mutated'
        assert evidence['dense']['encoder_input_sha256'] != evidence['sparse']['encoder_input_sha256']
        assert evidence['dense']['predictor_depth_embedding_sha256'] != evidence['sparse']['predictor_depth_embedding_sha256']
        assert torch.equal(dense['pano_rgb'], sparse['pano_rgb'])
        report.update(dense_candidates=serialize_waypoint_candidates(dense, 0),
                      sparse_candidates=serialize_waypoint_candidates(sparse, 0),
                      sparse_depth=provider.last_sparse_depth_diagnostics[0],
                      input_path_evidence=evidence, source_observations_unchanged=True,
                      pano_rgb_unchanged=True, raw_depth_equals_dense_encoder_source=True)
        write()
        return sparse
    MID360SparseDepthETPWaypointProvider.predict = compare_first
    try:
        run_exp(exp_name, 'run_r2r/iter_train.yaml', 'eval', opts, local_rank=0)
        trainer = holder['trainer']
        report['weights_after'] = dict(policy=state_fingerprint(trainer.policy), predictor=state_fingerprint(trainer.waypoint_predictor))
        assert report['weights_before'] == report['weights_after'], 'weights or buffers changed'
        stats_paths = list((root / exp_name).glob('stats_ep_ckpt_*_val_unseen_r0_w1.json'))
        assert len(stats_paths) == 1
        stats = json.loads(stats_paths[0].read_text())
        assert set(stats) == {'53'}
        report.update(status='PASS', weights_unchanged=True, episode_metrics=stats['53'],
                      latency_note='Smoke includes extra dense inference, hashes and checks; not formal performance evidence.')
        write()
    finally:
        MID360SparseDepthETPWaypointProvider.predict = predict
        trainer_module.RLTrainer._evaluation_waypoint_provider = factory
        trainer_module.require_valid_sensor_observations = guard


if __name__ == '__main__':
    main()
