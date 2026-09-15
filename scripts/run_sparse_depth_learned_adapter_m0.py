#!/usr/bin/env python3
"""Manual host-only one-reset feature collection, then isolated CPU optimization.

Source renderer_bootstrap.bash and call etp_apply_known_good_renderer_environment
before launching in etpnav_legacy. No navigation or formal evaluation is run.
"""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sparse_depth_learned_adapter import (
    ResidualFeatureAdapter, feature_distillation_loss,
    save_adapter_checkpoint, load_adapter_checkpoint,
)

FEATURE_SHAPE = (12, 128, 4, 4)
MAX_STEPS = 1000


def validate_episode_id(episode_id, stage4):
    # Canonicalize numeric IDs so leading zeros cannot bypass frozen-ID rules.
    try:
        episode_id = str(int(episode_id))
    except (ValueError, TypeError):
        raise ValueError('episode ID must be numeric')
    stage4.validate_calibration_ids('val_seen', [episode_id])
    if episode_id in stage4.HELD_OUT_IDS:
        raise ValueError('held-out observation cannot be used for M0')
    return episode_id


def select_episode(episode_id, stage4):
    episode_id = validate_episode_id(episode_id, stage4)
    if stage4.sha(stage4.CALIBRATION_DATA) != stage4.CALIBRATION_SHA:
        raise ValueError('val_seen dataset hash differs from accepted Stage 4 input')
    with gzip.open(stage4.CALIBRATION_DATA, 'rt') as handle:
        episodes = json.load(handle)['episodes']
    matches = [e for e in episodes if str(e['episode_id']) == episode_id]
    if len(matches) != 1:
        raise ValueError('episode must occur exactly once in accepted val_seen dataset')
    return dict(episode_id=episode_id, scene_id=matches[0]['scene_id'])


def validate_pair(features):
    if set(features) != {'dense', 'sparse'}:
        raise ValueError('expected exactly dense and sparse features')
    for value in features.values():
        if tuple(value.shape) != FEATURE_SHAPE or value.dtype != torch.float32:
            raise ValueError('M0 requires float32 [12,128,4,4] features')
        if not torch.isfinite(value).all():
            raise ValueError('feature pair must be finite')


def collect_pair(stage4, selected, output):
    """Reuse accepted Depth-only runtime, exactly one reset and encoding pair."""
    envs = None
    try:
        tm, config, trainer, envs, provider = stage4.runtime(
            'val_seen', [selected['episode_id']], output, calibration_only=True)
        # The full policy may contain trainable navigation layers at load time.
        # Freeze everything; only a separately created adapter will be optimized.
        for module in (trainer.policy, provider.original.rgb_encoder,
                       provider.original.depth_encoder, provider.original.waypoint_predictor):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
        modules = dict(depth_encoder=provider.original.depth_encoder,
                       predictor=provider.original.waypoint_predictor, policy=trainer.policy)
        before = {name: stage4.state_fingerprint(m) for name, m in modules.items()}
        batch, record = stage4.reset_batch(tm, config, trainer, envs,
                                          output / 'sensors.json', calibration_only=True)
        expected_scene = str(Path(config.TASK_CONFIG.DATASET.SCENES_DIR) / selected['scene_id'])
        if record['episode_id'] != selected['episode_id'] or record['scene_id'] != expected_scene:
            raise RuntimeError('reset returned an unexpected episode or scene')
        # Match the accepted one-observation smoke: close renderer before encoding.
        envs.close()
        envs = None
        features = stage4.calibration_features(provider, batch)
        validate_pair(features)
        if any(v.requires_grad or v.grad_fn is not None for v in features.values()):
            raise RuntimeError('collected features must be detached from ETPNav')
        after = {name: stage4.state_fingerprint(m) for name, m in modules.items()}
        if before != after:
            raise RuntimeError('Original model parameters or buffers changed during collection')
        # Own small CPU copies. No ETPNav models leave this collection scope.
        pair = {name: value.detach().cpu().clone() for name, value in features.items()}
        return pair, dict(observation=record, resets=1, observations=1,
                          environment_closed_before_optimization=True,
                          model_state_before=before, model_state_after=after,
                          original_models_frozen=True, original_states_unchanged=True)
    finally:
        if envs is not None:
            envs.close()


def validate_optimization(steps, learning_rate):
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError('steps must be between 1 and {}'.format(MAX_STEPS))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning rate must be positive and finite')


def optimize_pair(features, steps=100, learning_rate=1e-3, seed=0):
    """Bounded CPU smoke, detached owned inputs, Adam(adapter.parameters()) only."""
    validate_optimization(steps, learning_rate)
    validate_pair(features)
    pair = {name: value.detach().cpu().clone() for name, value in features.items()}
    snapshots = {name: value.clone() for name, value in pair.items()}
    torch.manual_seed(seed)
    adapter = ResidualFeatureAdapter()
    adapter.train()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate)

    def metrics():
        with torch.no_grad():
            return {name: value.item() for name, value in
                    feature_distillation_loss(adapter(pair['sparse']), pair['dense']).items()}

    with torch.no_grad():
        identity_error = (adapter(pair['sparse']) - pair['sparse']).abs().max().item()
    if identity_error != 0.0:
        raise RuntimeError('fresh float32 adapter failed exact identity')
    initial = metrics()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = feature_distillation_loss(adapter(pair['sparse']), pair['dense'])['total']
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite distillation loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in adapter.parameters()):
            raise RuntimeError('missing or nonfinite adapter gradient')
        optimizer.step()
    adapter.eval()
    final = metrics()
    if not all(math.isfinite(v) for v in final.values()):
        raise RuntimeError('nonfinite final metrics')
    if any(v.requires_grad or v.grad is not None or not torch.equal(v, snapshots[name])
           for name, v in pair.items()):
        raise RuntimeError('fixed feature pair changed during optimization')
    return adapter, dict(initial=initial, final=final, loss_reduced=final['total'] < initial['total'],
                         max_identity_error_at_initialization=identity_error,
                         feature_shapes={k: list(v.shape) for k, v in pair.items()},
                         adapter_parameter_count=sum(p.numel() for p in adapter.parameters()),
                         gradients_finite=True, fixed_features_unchanged=True,
                         optimization_device='cpu', optimizer='Adam', learning_rate=learning_rate,
                         loss_weights=dict(smooth_l1=1.0, cosine=1.0), seed=seed, training_steps=steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-id', required=True, help='one eligible val_seen numeric ID')
    parser.add_argument('--output', required=True, type=Path, help='fresh directory; overwrite refused')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-features', action='store_true')
    args = parser.parse_args()
    validate_optimization(args.steps, args.learning_rate)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('refusing existing output: {}'.format(output))
    os.chdir(ROOT)  # Existing Stage 4 config/asset paths are repository-relative.
    # Lazy import keeps --help and isolated optimization free of Habitat imports.
    import run_sparse_depth_feature_alignment as stage4
    selected = select_episode(args.episode_id, stage4)
    required_gl = ['/lib/x86_64-linux-gnu/' + name for name in
                   ('libGLdispatch.so.0', 'libGLX.so.0', 'libOpenGL.so.0')]
    if not all(path in os.environ.get('LD_PRELOAD', '').split(':') for path in required_gl):
        raise RuntimeError('apply scripts/renderer_bootstrap.bash before starting Python')
    checkpoint_hashes = {}
    for path in (stage4.CHECKPOINT, stage4.AUXILIARY):
        checkpoint_hashes[path] = stage4.sha(path)
        if checkpoint_hashes[path] != stage4.EXPECTED[path]:
            raise ValueError('accepted model checkpoint hash mismatch: ' + path)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    sources = ['scripts/sparse_depth_learned_adapter.py',
               'scripts/run_sparse_depth_learned_adapter_m0.py',
               'scripts/sparse_depth_feature_alignment.py',
               'scripts/run_sparse_depth_feature_alignment.py', 'scripts/renderer_bootstrap.bash']
    git = lambda *cmd: subprocess.check_output(['git'] + list(cmd), universal_newlines=True).strip()
    report = dict(stage='M0', navigation_steps=0, formal_evaluation=False, status='STARTED',
                  split='val_seen', selected=selected, command=sys.argv,
                  timestamp=datetime.now(timezone.utc).isoformat(),
                  branch=git('branch', '--show-current'), git_commit=git('rev-parse', 'HEAD'),
                  git_status=git('status', '--short'), torch_version=torch.__version__,
                  checkpoint_hashes=checkpoint_hashes, dataset_path=stage4.CALIBRATION_DATA,
                  dataset_sha256=stage4.CALIBRATION_SHA,
                  source_sha256={p: stage4.sha(p) for p in sources},
                  sparse_config_sha256=stage4.sha(stage4.CONFIG),
                  renderer_bootstrap='scripts/renderer_bootstrap.bash:etp_apply_known_good_renderer_environment',
                  LD_PRELOAD=os.environ.get('LD_PRELOAD', ''),
                  acceptance_scope='one real feature pair optimization sanity; no generalization claim')
    stage4.atomic_json(output / 'report.json', report)
    try:
        features, collection = collect_pair(stage4, selected, output)
        report.update(collection)
        report['runtime_config_sha256'] = stage4.sha(output / 'calibration_runtime_config.yaml')
        report['feature_sha256'] = {k: stage4.fingerprint(v) for k, v in features.items()}
        report['gpu_used_for_collection'] = torch.cuda.get_device_name(0)
        adapter, optimization = optimize_pair(features, args.steps, args.learning_rate, args.seed)
        report.update(optimization)
        checkpoint = output / 'adapter.pt'
        save_adapter_checkpoint(checkpoint, adapter, seed=args.seed, step=args.steps)
        reloaded, metadata = load_adapter_checkpoint(checkpoint, expected_channels=128)
        reloaded.eval()
        with torch.no_grad():
            equal = torch.equal(adapter(features['sparse']), reloaded(features['sparse']))
        report.update(checkpoint_metadata=metadata, checkpoint_sha256=stage4.sha(checkpoint),
                      checkpoint_output_exactly_equal=equal)
        if not equal:
            raise RuntimeError('checkpoint output round trip failed')
        if args.save_features:
            with (output / 'features.pt').open('xb') as handle:
                torch.save(features, handle)
        if not report['loss_reduced']:
            raise RuntimeError('bounded real-pair smoke did not reduce loss')
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAILED', error=str(error))
        raise
    finally:
        stage4.atomic_json(output / 'report.json', report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
