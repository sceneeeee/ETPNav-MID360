#!/usr/bin/env python3
"""Stage 5 M1 reset-only evidence for ONE explicitly supplied frozen A1.

Host execution only. Formal held-out use must wait for the independently frozen
training recipe/checkpoint. Importing this module does not initialize Habitat.
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_sparse_depth_feature_alignment as stage4
from sparse_depth_learned_adapter import load_adapter_checkpoint, feature_distillation_loss
from sparse_depth_run_manifest import DATASET

CONDITIONS = ('DENSE', 'SPARSE', 'A1_ADAPTED')
ENCODED_FIELDS = ('rgb_embedding', 'depth_embedding', 'rgb_features', 'depth_features')


def canonical_id(value):
    text = str(value)
    if not text or any(c not in '0123456789' for c in text):
        raise ValueError('episode IDs must be nonnegative decimal integers')
    return str(int(text))


def validate_ids(mode, episode_ids=None):
    if mode not in ('formal', 'nonformal-smoke'):
        raise ValueError('explicit formal or nonformal-smoke mode required')
    ids = list(stage4.HELD_OUT_IDS) if episode_ids is None and mode == 'formal' else episode_ids
    if not ids:
        raise ValueError('nonformal smoke requires one explicit eligible val_seen ID')
    ids = [canonical_id(e) for e in ids]
    if mode == 'formal':
        # Pin both membership and order; duplicates cannot pass this guard.
        if ids != ['53', '821', '1024', '250', '25']:
            raise ValueError('formal IDs must be exactly 53 821 1024 250 25, in that order')
    else:
        if len(ids) != 1:
            raise ValueError('nonformal smoke requires exactly one observation')
        stage4.validate_calibration_ids('val_seen', ids)
    return ids


def freeze(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def require_frozen(modules):
    for name, module in modules.items():
        if (any(m.training for m in module.modules()) or
                any(p.requires_grad or p.grad is not None for p in module.parameters())):
            raise RuntimeError(name + ' must be frozen, gradient-free, and in eval mode')


def original_modules(original):
    return {name: getattr(original, name) for name in
            ('rgb_encoder', 'depth_encoder', 'rgb_pool', 'depth_pool', 'waypoint_predictor')}


def fingerprints(modules):
    return {name: stage4.state_fingerprint(module) for name, module in modules.items()}


def load_frozen_adapter(path):
    # Existing strict reconstruction loader is the single checkpoint authority.
    before = stage4.sha(path)
    adapter, metadata = load_adapter_checkpoint(path, expected_channels=128, device='cpu')
    if metadata['dtype'] != 'torch.float32':
        raise ValueError('M1 requires float32 A1 matching the frozen Original encoder')
    if metadata['loss']['smooth_l1_weight'] != 1.0 or metadata['loss']['cosine_weight'] != 1.0:
        raise ValueError('M1 A1 contract requires equal 1.0/1.0 feature loss weights')
    if stage4.sha(path) != before:
        raise RuntimeError('adapter checkpoint changed while loading')
    freeze(adapter)
    require_frozen({'adapter': adapter})
    return adapter, dict(path=str(path), sha256=before, metadata=metadata,
                         parameter_count=sum(p.numel() for p in adapter.parameters()))


@torch.no_grad()
def feature_metrics(sparse, adapted, dense):
    def measure(value):
        loss = feature_distillation_loss(value, dense)
        return {k: loss[k].item() for k in ('smooth_l1', 'mean_cosine_similarity')}
    baseline, a1 = measure(sparse), measure(adapted)
    return dict(SPARSE_vs_DENSE=baseline, A1_ADAPTED_vs_DENSE=a1,
                delta_smooth_l1=a1['smooth_l1'] - baseline['smooth_l1'],
                delta_cosine=a1['mean_cosine_similarity'] - baseline['mean_cosine_similarity'])


def candidate_metrics(result, output):
    distances = output['cand_distances'][0]
    angles = output['cand_angles'][0]
    bins = [c['distance_index'] for c in result['logits']['post_nms_candidates']]
    return dict(count=len(distances), angles_radians_counterclockwise=angles,
                angles_degrees_counterclockwise=[math.degrees(a) % 360 for a in angles],
                distances_meters=distances, radial_bin_indexes=bins,
                distinct_radial_bins=len(set(bins)), distinct_distances=len(set(distances)),
                mean_distance_meters=float(np.mean(distances)) if distances else None,
                median_distance_meters=float(np.median(distances)) if distances else None)


@torch.no_grad()
def capture_encoded(original, encoded, alignment=None):
    """Stage 5-local variant of Stage 4 capture for one shared sparse encoding.

    Keep the historical hook sites and clone-before-inplace-ReLU semantics.
    All statistics and Original predictor/NMS dispatch stay in shared helpers;
    only hook orchestration and the captured candidate map are local to M1.
    """
    captured = dict(depth_embedding=encoded.depth_embedding.detach().clone())
    predictor = original.waypoint_predictor
    core = predictor.module if hasattr(predictor, 'module') else predictor
    assert isinstance(core.visual_fc_depth[1], torch.nn.Linear)
    assert isinstance(core.visual_fc_depth[2], torch.nn.ReLU)
    assert isinstance(core.vis_classifier[0], torch.nn.Linear)
    assert isinstance(core.vis_classifier[1], torch.nn.ReLU)
    assert core.visual_fc_depth[2] is not core.vis_classifier[1], 'ReLU sites must be distinct'
    def save(name):
        def hook(module, inputs, output):
            # Clone immediately: visual_fc_depth ReLU operates in-place.
            captured[name] = output.detach().clone()
        return hook
    handles = [module.register_forward_hook(save(name)) for name, module in
               [('visual_fc_depth_pre_relu', core.visual_fc_depth[1]), ('visual_fc_depth_post_relu', core.visual_fc_depth[2]),
                ('transformer_output', core.waypoint_TRM),
                ('classifier_pre_relu', core.vis_classifier[0]), ('classifier_post_relu', core.vis_classifier[1])]]
    try:
        output, logits, candidate_map, features = stage4.predict_encoded(original, encoded, alignment)
    finally:
        for handle in handles:
            handle.remove()
    for prefix in ('visual_fc_depth', 'classifier'):
        pre, post = captured[prefix + '_pre_relu'], captured[prefix + '_post_relu']
        assert torch.isfinite(pre).all() and torch.isfinite(post).all(), prefix
        assert pre.shape == post.shape and torch.equal(pre.clamp_min(0), post), prefix
    captured.update(predictor_input=features.detach().clone(), raw_logits=logits.detach().clone(),
                    candidate_map=candidate_map.detach().clone())
    result = dict(encoder=dict(statistics=stage4.statistics(encoded.depth_embedding), diversity=stage4.diversity(encoded.depth_embedding)),
                  predictor_input=dict(statistics=stage4.statistics(features), diversity=stage4.diversity(features)),
                  visual_fc_depth_pre_relu=stage4.activity(captured['visual_fc_depth_pre_relu']), visual_fc_depth_post_relu=stage4.activity(captured['visual_fc_depth_post_relu']),
                  classifier_pre_relu=stage4.activity(captured['classifier_pre_relu']),
                  classifier_post_relu=stage4.activity(captured['classifier_post_relu']),
                  logits=stage4.logit_statistics(logits, candidate_map),
                  candidate_count=len(output['cand_angles'][0]),
                  rgb_embedding_sha256=stage4.fingerprint(encoded.rgb_embedding),
                  depth_encoder_object_id=id(original.depth_encoder), predictor_object_id=id(predictor))
    result['logits']['distinct_candidate_radii'] = len(set(output['cand_distances'][0]))
    return result, captured, output


@torch.no_grad()
def evaluate_observation(batch, provider, sensor, adapter):
    """One source observation, two encodings, three Original predictor calls."""
    original = provider.original
    modules = original_modules(original)
    modules['adapter'] = adapter
    require_frozen(modules)
    before = fingerprints(modules)
    source = {k: stage4.fingerprint(v) for k, v in batch.items()}
    conditions, diagnostics = stage4.make_conditions(batch, provider.adapter, sensor)
    conditions['A1_ADAPTED'] = conditions['SPARSE']
    dense = stage4.encode(original, conditions['DENSE'])
    sparse = stage4.encode(original, conditions['SPARSE'])
    encodings = dict(DENSE=dense, SPARSE=sparse, A1_ADAPTED=sparse)
    snapshots = {name: {k: stage4.fingerprint(getattr(value, k)) for k in ENCODED_FIELDS}
                 for name, value in encodings.items()}
    results, tensors, outputs = {}, {}, {}
    for name in CONDITIONS:
        encoded = encodings[name]
        if tuple(encoded.depth_embedding.shape) != (12, 128, 4, 4):
            raise ValueError('M1 requires one observation with twelve [128,4,4] views')
        result, captured, output = capture_encoded(
            original, encoded, adapter if name == 'A1_ADAPTED' else None)
        if any(not torch.isfinite(t).all() for t in captured.values()):
            raise RuntimeError('nonfinite static capture')
        for key in ENCODED_FIELDS:
            if stage4.fingerprint(getattr(encoded, key)) != snapshots[name][key]:
                raise RuntimeError('raw encoding changed: ' + key)
        if not torch.equal(output['pano_depth'], encoded.depth_features):
            raise RuntimeError('panorama depth must stay raw')
        for candidates, indexes in zip(output['cand_depth'], output['cand_img_idxes']):
            if not torch.equal(candidates, encoded.depth_features[0, indexes]):
                raise RuntimeError('candidate depth must stay raw')
        result['input'] = stage4.input_statistics(
            conditions[name], diagnostics if name != 'DENSE' else None)
        result['candidates'] = candidate_metrics(result, output)
        results[name], tensors[name], outputs[name] = result, captured, output
    if source != {k: stage4.fingerprint(v) for k, v in batch.items()}:
        raise RuntimeError('source observation changed')
    if len({r['rgb_embedding_sha256'] for r in results.values()}) != 1:
        raise RuntimeError('conditions must preserve the same RGB encoding')
    require_frozen(modules)
    after = fingerprints(modules)
    if before != after:
        raise RuntimeError('adapter or Original parameters/buffers changed during evaluation')
    comparisons = {}
    fields = ('predictor_input', 'visual_fc_depth_pre_relu', 'visual_fc_depth_post_relu',
              'transformer_output', 'classifier_pre_relu', 'classifier_post_relu', 'raw_logits')
    for left, right in (('SPARSE', 'DENSE'), ('A1_ADAPTED', 'DENSE'), ('SPARSE', 'A1_ADAPTED')):
        comparison = {field: stage4.similarity(tensors[left][field], tensors[right][field])
                      for field in fields}
        comparison['classifier_zero_crossings'] = stage4.crossings(
            tensors[left]['classifier_pre_relu'], tensors[right]['classifier_pre_relu'])
        comparisons[left + '_vs_' + right] = comparison
    record = dict(conditions=results, comparisons=comparisons,
                  features=feature_metrics(tensors['SPARSE']['predictor_input'],
                                           tensors['A1_ADAPTED']['predictor_input'],
                                           tensors['DENSE']['predictor_input']),
                  guards=dict(same_sparse_encoding=True, raw_policy_depth_unchanged=True,
                              source_unchanged=True, original_and_adapter_states_unchanged=True,
                              state_before=before, state_after=after, all_modules_eval_frozen=True))
    return record, tensors, outputs


def select_dataset(mode, ids):
    path = DATASET if mode == 'formal' else stage4.CALIBRATION_DATA
    expected = stage4.EXPECTED[path] if mode == 'formal' else stage4.CALIBRATION_SHA
    if stage4.sha(path) != expected:
        raise ValueError('accepted dataset hash mismatch: ' + path)
    with gzip.open(path, 'rt') as handle:
        episodes = json.load(handle)['episodes']
    selected = {}
    for eid in ids:
        matches = [e for e in episodes if canonical_id(e['episode_id']) == eid]
        if len(matches) != 1:
            raise ValueError('ID must occur exactly once in accepted dataset: ' + eid)
        selected[eid] = matches[0]
    return selected, dict(path=path, sha256=expected)


def source_hashes():
    # Include tracked runtime/config dependencies and this new, possibly untracked runner.
    names = subprocess.check_output(['git', 'ls-files', '--', 'scripts', 'vlnce_baselines',
                                     'habitat_extensions', 'configs', 'run_r2r'],
                                    universal_newlines=True).splitlines()
    names.append(str(Path(__file__).resolve().relative_to(ROOT)))
    return {name: stage4.sha(name) for name in sorted(set(names))
            if Path(name).is_file() and Path(name).suffix in ('.py', '.yaml', '.bash')}


def report_paths(output, formal):
    prefix = '' if formal else 'NONFORMAL_'
    return output / (prefix + 'report.json'), output / (prefix + 'report.md')


def write_report(output, report):
    json_path, markdown_path = report_paths(output, report['formal_evaluation'])
    stage4.atomic_json(json_path, report)
    lines = ['# ' + report['stage'], '',
             'stage = "{}"; formal_evaluation = {}; training_steps = 0; navigation_steps = 0'.format(
                 report['stage'], str(report['formal_evaluation']).lower()),
             'Status: ' + report['status'], 'Episode IDs: ' + ', '.join(report['episode_ids']), '',
             'ONE supplied checkpoint SHA256: ' + report['adapter_checkpoint']['sha256'], '',
             'Feature rescue != classifier rescue != candidate-diversity rescue.',
             'Primary mechanism: vis_classifier[1] post-ReLU positive count/fraction.',
             'visual_fc_depth[2] is upstream diagnostic evidence only.',
             'Deltas are A1 minus SPARSE: negative SmoothL1 and positive cosine indicate improvement.', '']
    if not report['formal_evaluation']:
        lines += ['NONFORMAL PLUMBING SMOKE ONLY: no held-out evidence or navigation gate claim.', '']
    if report.get('error'):
        lines += ['Failure: ' + report['error'], '']
    for eid in report['episode_ids']:
        if eid not in report['observations']:
            continue
        observation = report['observations'][eid]
        lines += ['## Episode ' + eid, '',
                  '| Condition | SmoothL1 vs dense | Mean cosine vs dense | Upstream active % | Classifier positive count | Classifier active % | Entropy (nats) | Candidates | Distinct radii | Mean / median m |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for name in CONDITIONS:
            c = observation['conditions'][name]
            feature = observation['features'].get(name + '_vs_DENSE')
            candidates = c['candidates']
            lines.append('| {} | {} | {} | {:.6f} | {} | {:.6f} | {:.9f} | {} | {} | {} / {} |'.format(
                name, '{:.8f}'.format(feature['smooth_l1']) if feature else 'reference',
                '{:.8f}'.format(feature['mean_cosine_similarity']) if feature else 'reference',
                100 * c['visual_fc_depth_post_relu']['positive_fraction'],
                c['classifier_post_relu']['positive_count'], 100 * c['classifier_post_relu']['positive_fraction'],
                c['logits']['entropy_nats'], candidates['count'], candidates['distinct_radial_bins'],
                candidates['mean_distance_meters'], candidates['median_distance_meters']))
        lines += ['', 'Delta SmoothL1: {:.8f}; delta cosine: {:.8f}.'.format(
            observation['features']['delta_smooth_l1'], observation['features']['delta_cosine']), '']
    lines += ['## Interpretation pending scientific review', '',
              '- CASE A: feature similarity, classifier activity, and meaningful candidate diversity recover: candidate for later frozen-20.',
              '- CASE B: features improve but classifier remains collapsed: consider A2 predictor-aware distillation.',
              '- CASE C: feature similarity fails to generalize: investigate data, capacity, and placement before navigation.',
              '- Activity without meaningful diversity is a partial rescue.',
              'No numerical pass threshold, automatic case assignment, checkpoint selection, or next-stage trigger.']
    markdown_path.write_text('\n'.join(lines) + '\n')


@torch.no_grad()
def collect_static(output, ids, selected, adapter, report):
    """Reuse Stage 4 model/environment setup and reset capture; no rollout."""
    envs = None
    modules = None
    try:
        prepared = stage4.runtime_config(report['split'], ids, output)
        if prepared.IL.is_requeue:
            raise ValueError('static evaluation cannot resume training')
        tm, config, trainer, envs, provider = stage4.runtime(
            report['split'], ids, output, prepared_config=prepared)
        # The unchanged legacy loader constructs AdamW. M1 never calls it.
        report['legacy_optimizer_constructed'] = getattr(trainer, 'optimizer', None) is not None
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        modules = original_modules(provider.original)
        modules['policy'] = trainer.policy
        for module in modules.values():
            freeze(module)
        adapter.to(device=trainer.device)
        modules['adapter'] = adapter
        require_frozen(modules)
        report['model_state_before'] = fingerprints(modules)
        report['runtime_config_sha256'] = stage4.sha(output / 'runtime_config.yaml')
        report['determinism'] = dict(cudnn_benchmark=False, cudnn_deterministic=True,
                                     seed=config.TASK_CONFIG.SEED)
        for _ in ids:
            batch, observation = stage4.reset_batch(tm, config, trainer, envs,
                output / ('sensors_{}.json'.format(len(report['observations']))))
            eid = observation['episode_id']
            if eid not in ids or eid in report['observations']:
                raise RuntimeError('unexpected or duplicate reset episode: ' + eid)
            expected_scene = str(Path(config.TASK_CONFIG.DATASET.SCENES_DIR) / selected[eid]['scene_id'])
            if observation['scene_id'] != expected_scene:
                raise RuntimeError('reset scene differs from selected dataset episode')
            observation['scene_sha256'] = stage4.sha(expected_scene)
            metrics, tensors, _ = evaluate_observation(
                batch, provider, config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR, adapter)
            observation.update(metrics)
            capture_path = output / ('capture_' + eid + '.npz')
            with capture_path.open('xb') as handle:
                np.savez_compressed(handle, **{name + '_' + field: tensor.cpu().numpy()
                    for name, values in tensors.items() for field, tensor in values.items()})
            observation['capture_sha256'] = stage4.sha(capture_path)
            report['observations'][eid] = observation
            write_report(output, report)
        if set(report['observations']) != set(ids):
            raise RuntimeError('incomplete observation set')
    finally:
        if envs is not None:
            envs.close()
        if modules is not None:
            report['model_state_after'] = fingerprints(modules)
            report['states_unchanged'] = report['model_state_before'] == report['model_state_after']
            require_frozen(modules)
            if not report['states_unchanged']:
                raise RuntimeError('Original or adapter state changed across static run')


def run(mode, episode_ids, checkpoint, output):
    ids = validate_ids(mode, episode_ids)
    formal = mode == 'formal'
    output, checkpoint = Path(output).resolve(), Path(checkpoint).resolve()
    if output.exists():
        raise FileExistsError('refusing existing output: ' + str(output))
    if (not formal) != output.name.lower().startswith('nonformal_'):
        raise ValueError('smoke output basename must start nonformal_; formal output must not')
    adapter, checkpoint_record = load_frozen_adapter(checkpoint)
    selected, dataset = select_dataset(mode, ids)
    hashes = {path: stage4.sha(path) for path in (stage4.CHECKPOINT, stage4.AUXILIARY)}
    if any(value != stage4.EXPECTED[path] for path, value in hashes.items()):
        raise ValueError('accepted ETPNav checkpoint hash mismatch')
    hashes['data/ddppo-models/gibson-2plus-resnet50.pth'] = stage4.sha(
        'data/ddppo-models/gibson-2plus-resnet50.pth')
    if not torch.cuda.is_available():
        raise RuntimeError('host runner requires accepted CUDA/Habitat environment; use CPU unit tests in Codex')
    required_gl = ['/lib/x86_64-linux-gnu/' + name for name in
                   ('libGLdispatch.so.0', 'libGLX.so.0', 'libOpenGL.so.0')]
    if not all(path in os.environ.get('LD_PRELOAD', '').split(':') for path in required_gl):
        raise RuntimeError('apply scripts/renderer_bootstrap.bash before starting Python')
    if any(k.startswith('ETP_') and 'FORENSICS' in k for k in os.environ):
        raise RuntimeError('unset forensic side-output environment variables')
    git = lambda *args: subprocess.check_output(['git'] + list(args), universal_newlines=True).strip()
    status = git('status', '--short')
    report = dict(stage='M1_STATIC' if formal else 'M1_STATIC_NONFORMAL_SMOKE',
                  formal_evaluation=formal, training_steps=0, navigation_steps=0,
                  status='STARTED' if formal else 'NONFORMAL_STARTED', episode_ids=ids,
                  split='val_unseen' if formal else 'val_seen', observations={},
                  adapter_checkpoint=checkpoint_record, checkpoint_hashes=hashes, dataset=dataset,
                  source_sha256=source_hashes(), sparse_depth_config_sha256=stage4.sha(stage4.CONFIG),
                  git_commit=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                  git_dirty=bool(status), git_status=status, command=sys.argv,
                  timestamp=datetime.now(timezone.utc).isoformat(), python_version=sys.version,
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  LD_PRELOAD=os.environ.get('LD_PRELOAD', ''),
                  adapter_state_before=stage4.state_fingerprint(adapter),
                  primary_mechanism_metrics=['classifier_post_relu.positive_count',
                                             'classifier_post_relu.positive_fraction'],
                  known_complete_collapse_layer='vis_classifier[1]',
                  definitions=dict(conditions={
                      'DENSE': 'original dense observation -> frozen encoder -> frozen Original predictor',
                      'SPARSE': 'same observation -> deterministic MID360 transform -> frozen encoder -> frozen Original predictor',
                      'A1_ADAPTED': 'exact same SPARSE encoding -> frozen A1 -> frozen Original predictor'},
                      feature_cosine='mean of per-view flattened CHW cosine, eps=1e-8',
                      smooth_l1='mean NCHW, beta=1', delta='A1 minus SPARSE',
                      std='float64 population', entropy='Original softmax, nats',
                      raw_depth='encoded.depth_features, pano_depth and cand_depth remain raw'))
    output.mkdir(parents=True, exist_ok=False)
    write_report(output, report)
    try:
        collect_static(output, ids, selected, adapter, report)
        report['adapter_state_after'] = stage4.state_fingerprint(adapter)
        if report['adapter_state_before'] != report['adapter_state_after']:
            raise RuntimeError('adapter state changed from supplied checkpoint')
        if stage4.sha(checkpoint) != checkpoint_record['sha256']:
            raise RuntimeError('checkpoint file changed during run')
        report['status'] = 'COMPLETE' if formal else 'NONFORMAL_COMPLETE'
    except BaseException as error:
        report.update(status='FAILED' if formal else 'NONFORMAL_FAILED', error=str(error))
        raise
    finally:
        report['adapter_state_after'] = stage4.state_fingerprint(adapter)
        write_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['formal', 'nonformal-smoke'])
    parser.add_argument('--episode-ids', nargs='+', help='formal defaults to the fixed five; smoke requires one eligible val_seen ID')
    parser.add_argument('--adapter-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='fresh directory; smoke basename must start nonformal_')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    run(args.mode, args.episode_ids, args.adapter_checkpoint, args.output)


if __name__ == '__main__':
    main()
