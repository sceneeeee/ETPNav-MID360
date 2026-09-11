#!/usr/bin/env python3
"""Independent calibration, then a separate five-reset static mechanism test.

Research question: Can upstream alignment of sparse depth-encoder feature
statistics restore downstream classifier-ReLU activity and waypoint diversity?
The known complete-collapse point is vis_classifier[1], not visual_fc_depth[2].
Primary metrics: classifier_post_relu positive_count and positive_fraction.
visual_fc_depth ReLU activity is an upstream diagnostic only. Activity rescue
with diversity recovery supports an upstream mismatch mechanism; changed front-
end activity without classifier rescue means moment alignment is insufficient.
Classifier activity without meaningful logit/candidate diversity recovery is
only a partial mechanism rescue, not resolution of the proposal domain shift.

Run in etpnav_legacy with the accepted renderer bootstrap. Never calls env.step,
rollout, evaluator, optimizer.step, or backward. Refuses existing destinations.
"""
import argparse
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from analyze_sparse_depth_proposer_collapse import (
    diversity, fingerprint, input_statistics, logit_statistics, make_conditions,
    similarity, state_fingerprint, statistics,
)
from sparse_depth_feature_alignment import (
    ChannelMoments, EPSILON, FORMULA, FROZEN_IDS, HELD_OUT_IDS,
    FixedChannelAlignment, MID360SparseDepthFeatureAlignedETP, encode,
    predict_encoded, select_calibration_episodes, validate_calibration_ids,
    calibration_depth_config, calibration_depth_keys, encode_calibration_depth,
)
from sparse_depth_run_manifest import AUXILIARY, CHECKPOINT, CONFIG, EXPECTED, sha
from alignment_calibration_recovery import CalibrationProgress, SensorStability, atomic_json

CALIBRATION_DATA = 'data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/val_seen/val_seen_bertidx.json.gz'
CALIBRATION_SHA = '83b231674ae3a18cb02b3d65847377e978f45cc35682f807c03becc38b09410c'
FORENSICS = ROOT / 'outputs/mid360_sparse_depth_forensics'
PROTECTED = ['vlnce_baselines', 'habitat_extensions', 'configs', 'run_r2r', 'run.py']


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def protected_hashes():
    roots = [FORENSICS, ROOT / 'outputs/mid360_sparse_depth_zero_shot20']
    files = [p for root in roots for p in root.rglob('*') if p.is_file()]
    files += [ROOT / 'scripts/analyze_gmin_current_node_swap.py',
              ROOT / 'scripts/analyze_gmin_graph_survival_probe.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def provenance():
    subprocess.check_call(['git', 'diff', '--exit-code', 'b402412', '--'] + PROTECTED)
    for path, expected in EXPECTED.items():
        assert sha(path) == expected, path
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                timestamp=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                checkpoint_hashes={path: sha(path) for path in (CHECKPOINT, AUXILIARY)},
                sparse_depth_config_path=CONFIG, sparse_depth_config_sha256=sha(CONFIG),
                source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                               [Path(__file__).resolve(), ROOT / 'scripts/sparse_depth_feature_alignment.py']},
                torch_version=torch.__version__, numpy_version=np.__version__,
                gpu=torch.cuda.get_device_name(0), LD_PRELOAD=os.environ.get('LD_PRELOAD', ''),
                navigation_steps=0, training_steps=0)


def runtime_config(split, ids, output):
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.common.r2r_runtime import initialize_runtime_config, prepare_r2r_evaluation_environment
    opts = ['SIMULATOR_GPU_IDS', '[0]', 'TORCH_GPU_IDS', '[0]', 'TORCH_GPU_ID', '0',
            'GPU_NUMBERS', '1', 'NUM_ENVIRONMENTS', '1', 'EVAL.SPLIT', split,
            'EVAL.EPISODE_IDS', repr(ids), 'EVAL.EPISODE_COUNT', str(len(ids)),
            'IL.ckpt_to_load', CHECKPOINT, 'EVAL.CKPT_PATH_DIR', CHECKPOINT,
            'RESULTS_DIR', str(output) + '/', 'IL.back_algo', 'control', 'MODEL.pretrained_path', 'None',
            'MODEL.WAYPOINT_PROVIDER', 'mid360_sparse_depth_etp',
            'MODEL.MID360_SPARSE_DEPTH_CONFIG_PATH', CONFIG,
            'VIDEO_OPTION', '[]', 'TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING', 'True']
    config = get_config('run_r2r/iter_train.yaml', opts)
    initialize_runtime_config(config, 0)
    prepare_r2r_evaluation_environment(config)
    config.defrost()
    config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = ids
    # Policy.from_config installs this same value; pin it before hashing the config.
    config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
    config.freeze()
    return config


def runtime(split, ids, output, env_ids=None, prepared_config=None, calibration_only=False):
    from vlnce_baselines import ss_trainer_ETP as tm
    config = prepared_config if prepared_config is not None else runtime_config(split, ids, output)
    if calibration_only:
        config = calibration_depth_config(config)
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_num_threads(1)
    trainer = tm.RLTrainer(config)
    torch.cuda.set_device(trainer.device)
    env_ids = ids if env_ids is None else env_ids
    envs = tm.construct_envs(config.clone(), tm.get_env_class(config.ENV_NAME), episodes_allowed=env_ids, auto_reset_done=False)
    try:
        assert envs.num_envs == 1 and sum(envs.number_of_episodes) == len(env_ids)
        trainer.obs_transforms = tm.get_active_obs_transforms(config)
        space = tm.apply_obs_transforms_obs_space(envs.observation_spaces[0], trainer.obs_transforms)
        trainer._initialize_policy(config, True, space, envs.action_spaces[0])
        trainer.policy.eval()
        trainer.waypoint_predictor.eval()
        provider = trainer._evaluation_waypoint_provider()
        net = trainer.policy.net.module if hasattr(trainer.policy.net, 'module') else trainer.policy.net
        assert provider.original.depth_encoder is net.depth_encoder
        assert provider.original.waypoint_predictor is trainer.waypoint_predictor
        assert not net.depth_encoder.spatial_output
        provider.adapter.config.validate_runtime(config)
        filename = 'calibration_runtime_config.yaml' if calibration_only else 'runtime_config.yaml'
        (output / filename).write_text(config.dump())
        return tm, config, trainer, envs, provider
    except BaseException:
        envs.close()
        raise


def reset_batch(tm, config, trainer, envs, diagnostic_path=None, calibration_only=False):
    obs = envs.reset()
    episode = envs.current_episodes()[0]
    raw_statistics = None
    if diagnostic_path is not None or calibration_only:
        from diagnose_alignment_sensor_resets import sensor_statistics, validity
        raw_statistics = sensor_statistics(obs[0])
        sensor_validity = validity(raw_statistics)
        if calibration_only:
            sensor_validity = dict(depth_valid=sensor_validity['depth_valid'],
                                   no_rgb_observations=not any(k.lower().startswith('rgb') for k in obs[0]))
        if diagnostic_path is not None:
            atomic_json(diagnostic_path, dict(episode_id=str(episode.episode_id), scene_id=episode.scene_id,
                        agent_pose=envs.call_at(0, 'get_agent_info'), sensors=raw_statistics,
                        runtime_observation_keys=list(obs[0]),
                        runtime_rgb_key_count=sum(k.lower().startswith('rgb') for k in obs[0]),
                        runtime_depth_key_count=sum(k.startswith('depth') for k in obs[0]), **sensor_validity))
    if calibration_only:
        calibration_depth_keys(obs[0])
    tm.require_valid_sensor_observations(obs[0])
    if raw_statistics is not None and not all(sensor_validity.values()):
        raise RuntimeError('invalid finite/nonempty sensor views; see raw sensor diagnostic')
    hashes = {k: fingerprint(v) for k, v in obs[0].items() if 'depth' in k}
    pose = envs.call_at(0, 'get_agent_info')
    obs = tm.extract_instruction_tokens(obs, config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                        max_length=config.IL.max_text_len, pad_id=0)
    batch = tm.apply_obs_transforms_batch(tm.batch_obs(obs, trainer.device), trainer.obs_transforms)
    if calibration_only:
        calibration_depth_keys(batch)
    assert all(fingerprint(batch[k]) == h for k, h in hashes.items())
    return batch, dict(episode_id=str(episode.episode_id), scene_id=episode.scene_id,
                       agent_pose=pose, raw_depth_sha256=hashes, decision_index=0)


def model_hashes(trainer):
    return dict(policy=state_fingerprint(trainer.policy), predictor=state_fingerprint(trainer.waypoint_predictor))


@torch.no_grad()
def calibration_features(provider, batch):
    """Shared full-calibration/smoke seam: Depth adapter, frozen encoder, no predictor."""
    calibration_depth_keys(batch)
    source = {k: fingerprint(v) for k, v in batch.items()}
    sparse, _ = provider.adapter(batch)
    for key in batch:
        if 'depth' not in key:
            assert sparse[key] is batch[key]
    features = dict(dense=encode_calibration_depth(provider.original, batch),
                    sparse=encode_calibration_depth(provider.original, sparse))
    assert source == {k: fingerprint(v) for k, v in batch.items()}
    return features


def smoke_archive(archive):
    """Read-only validation of the accepted 11/53 prefix; never migrate its lock."""
    state = json.loads((archive / 'running_checkpoint.json').read_text())
    saved = json.loads((archive / 'selection.json').read_text())
    assert sha(CALIBRATION_DATA) == CALIBRATION_SHA
    with gzip.open(CALIBRATION_DATA, 'rt') as handle:
        episodes = select_calibration_episodes(json.load(handle)['episodes'])
    selection = [dict(episode_id=str(e['episode_id']), scene_id=e['scene_id']) for e in episodes]
    progress = CalibrationProgress.load(archive / 'running_checkpoint.json', selection,
                                        saved['recovery_provenance'], (12, 128, 4, 4))
    if len(selection) != 53 or progress.next_index != 11 or selection[11] != dict(
            episode_id='137', scene_id='mp3d/B6ByNegPMKs/B6ByNegPMKs.glb'):
        raise ValueError('smoke requires the accepted 11/53 prefix and observation #12 / episode 137')
    if saved['episode_ids'] != [r['episode_id'] for r in selection]:
        raise ValueError('saved deterministic selection changed')
    lock = state['provenance']
    if sha(archive / 'runtime_config.yaml') != lock['runtime_config_sha256']:
        raise ValueError('accepted runtime config changed')
    for record in progress.records:
        for attempt in record['sensor_attempts']:
            if sha(archive / attempt['path']) != attempt['sha256']:
                raise ValueError('saved sensor attempt provenance changed')
    # Record the intentional experiment-source delta; all other source/config
    # fingerprints stay strict. This permits a smoke, never a resumed checkpoint.
    changed = {}
    for path, expected in lock['source_and_config_sha256'].items():
        actual = sha(path)
        if actual != expected:
            if path not in ('scripts/run_sparse_depth_feature_alignment.py',
                            'scripts/sparse_depth_feature_alignment.py'):
                raise ValueError('unexpected source/config change: ' + path)
            changed[path] = dict(accepted_sha256=expected, smoke_sha256=actual)
    for path, expected in lock['checkpoint_hashes'].items():
        if sha(path) != expected:
            raise ValueError('model checkpoint changed: ' + path)
    return progress, changed


@torch.no_grad()
def calibration_smoke(output, archive):
    """Exactly one reset/sample, using fresh moments; the archive is read-only.

    Deliberately does not use the two-reset stability collector: this smoke tests
    one model observation, and cannot commit or authorize a calibration sample.
    """
    from habitat.config.default import Config
    archive_before = {str(p.relative_to(archive)): sha(p) for p in archive.rglob('*') if p.is_file()}
    progress, changed = smoke_archive(archive)
    config = Config()
    config.merge_from_file(str(archive / 'runtime_config.yaml'))
    config.freeze()
    assert len(config.SENSORS) == len(config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS) == 24
    assert sha(config.MODEL.DEPTH_ENCODER.ddppo_checkpoint) == progress.provenance['depth_initialization_checkpoint_sha256']
    report = provenance()
    report.update(status='STARTED', archive=str(archive), archive_sha256_before=archive_before,
                  accepted_recovery_provenance=progress.provenance, intentional_source_changes=changed,
                  next_calibration_index=11, calibration_index=12, selection=progress.selection[11],
                  observations=0, resets=0, checkpoint_writes=0,
                  acceptance_scope='one-observation encoder smoke; not calibration stability acceptance')
    # Preserve the model/runtime environment checks independently of the changed sources.
    for key in ('git_commit', 'torch_version', 'numpy_version', 'gpu', 'LD_PRELOAD'):
        if report[key] != progress.provenance[key]:
            raise ValueError('smoke runtime differs from accepted provenance: ' + key)
    atomic_json(output / 'smoke.json', report)
    ids = [r['episode_id'] for r in progress.selection]
    envs = None
    try:
        tm, config, trainer, envs, provider = runtime('val_seen', ids, output, ['137'], config,
                                                    calibration_only=True)
        before = model_hashes(trainer)
        assert all(r['model_state_after'] == before for r in progress.records)
        batch, record = reset_batch(tm, config, trainer, envs, output / 'sensors.json', calibration_only=True)
        report.update(resets=1, observation=record, model_state_before=before)
        assert record['episode_id'] == '137'
        assert record['scene_id'] == str(Path(config.TASK_CONFIG.DATASET.SCENES_DIR) / progress.selection[11]['scene_id'])
        envs.close()
        envs = None
        report['environment_closed'] = True
        atomic_json(output / 'smoke.json', report)
        features = calibration_features(provider, batch)
        moments = {name: ChannelMoments() for name in features}
        for name, value in features.items():
            assert tuple(value.shape) == (12, 128, 4, 4)
            moments[name].update(value)
            assert moments[name].observations == 1 and moments[name].count == 192
        after = model_hashes(trainer)
        assert before == after
        assert all(not value.requires_grad for value in features.values())
        report.update(observations=1, model_loads=1, model_state_after=after,
                      parameters_and_buffers_unchanged=True, same_frozen_models_as_checkpoint=True,
                      depth_encoder_frozen=True, depth_encoder_eval=True,
                      features={name:dict(statistics(value), sha256=fingerprint(value)) for name,value in features.items()},
                      moments={name:moment.to_state() for name,moment in moments.items()},
                      calibration_runtime_config_sha256=sha(output / 'calibration_runtime_config.yaml'))
    finally:
        if envs is not None:
            envs.close()
        archive_after = {str(p.relative_to(archive)): sha(p) for p in archive.rglob('*') if p.is_file()}
        report.update(archive_sha256_after=archive_after, archive_unchanged=archive_before == archive_after)
        atomic_json(output / 'smoke.json', report)
        assert archive_before == archive_after, 'accepted calibration archive changed'
    report['status'] = 'PASS'
    atomic_json(output / 'smoke.json', report)
    sensors = json.loads((output / 'sensors.json').read_text())
    print('SMOKE PASS', 'episode 137 / B6ByNegPMKs / observation #12',
          'Depth', sensors['runtime_depth_key_count'], 'RGB', sensors['runtime_rgb_key_count'],
          'depth_valid', sensors['depth_valid'], 'one observation; zero navigation/training; archive unchanged', flush=True)


def stable_calibration_batch(tm, config, trainer, first_env, output, index, selected, invocation):
    """At most five fresh resets; two consecutive valid exact matches are required.

    No features are encoded and no moments updated until stability is established.
    Every attempt survives failed validation and subsequent process resumes.
    """
    stability, attempts = SensorStability(), []
    envs = first_env
    expected_scene = str(Path(config.TASK_CONFIG.DATASET.SCENES_DIR) / selected['scene_id'])
    for attempt in range(1, 6):
        path = output / ('sensors_{}_i{:03d}_a{}.json'.format(invocation, index+1, attempt))
        try:
            if envs is None:
                envs = tm.construct_envs(config.clone(), tm.get_env_class(config.ENV_NAME),
                                         episodes_allowed=[selected['episode_id']], auto_reset_done=False)
            assert envs.num_envs == 1 and sum(envs.number_of_episodes) == 1
            batch, record = reset_batch(tm, config, trainer, envs, path, calibration_only=True)
            error = None
        except RuntimeError as failure:
            if not path.exists():
                atomic_json(path, dict(error=str(failure), stage='construct_or_reset', calibration_index=index+1,
                                      episode_id=selected['episode_id'], scene_id=expected_scene))
                raise
            error = str(failure)
        finally:
            if envs is not None:
                envs.close()
                envs = None
        raw = json.loads(path.read_text())
        assert raw['episode_id'] == selected['episode_id'] and raw['scene_id'] == expected_scene
        raw.update(strict_validation_error=error, environment_closed=True, attempt=attempt,
                   calibration_index=index+1, invocation=invocation)
        atomic_json(path, raw)
        attempts.append(dict(path=path.name, sha256=sha(path), valid=error is None))
        if stability.observe(raw['sensors'], error is None):
            record.update(sensor_attempts=attempts, resets=attempt, stable_consecutive_frames=2,
                          sensor_stability='two consecutive valid fresh lifecycles with bit-identical sensors')
            return batch, record
        print('SENSOR RETRY', index+1, selected['episode_id'], attempt, error or 'awaiting exact confirmation', flush=True)
    raise RuntimeError('sensor stability not established in five fresh lifecycles; calibration stopped without counting episode')


@torch.no_grad()
def calibrate(output, resume=False, stop_after=None):
    assert sha(CALIBRATION_DATA) == CALIBRATION_SHA
    with gzip.open(CALIBRATION_DATA, 'rt') as handle:
        episodes = json.load(handle)['episodes']
    selected = select_calibration_episodes(episodes)
    ids = [str(e['episode_id']) for e in selected]
    scenes = sorted({e['scene_id'] for e in selected})
    with gzip.open(next(p for p in EXPECTED if 'val_unseen' in p), 'rt') as handle:
        unseen = json.load(handle)['episodes']
    frozen_scenes = {e['scene_id'] for e in unseen if str(e['episode_id']) in FROZEN_IDS}
    assert not set(scenes).intersection(frozen_scenes)
    manifest = provenance()
    manifest.update(calibration_split='val_seen', dataset_path=CALIBRATION_DATA,
                    dataset_sha256=CALIBRATION_SHA, dataset_observations_available=len(episodes),
                    dataset_scenes_available=len({e['scene_id'] for e in episodes}),
                    episode_ids=ids, scenes=scenes, number_of_observations=len(ids), number_of_scenes=len(scenes),
                    sampling='one observation at lowest eligible numeric episode ID per scene, sorted scene order; all scenes; '
                             'bounded exact-reset stability confirmation does not add feature samples',
                    excluded_frozen_numeric_ids=sorted(FROZEN_IDS, key=int), frozen_scene_overlap=[],
                    uses_navigation_outcomes=False, formula=FORMULA, epsilon=EPSILON,
                    std_definition='population; float64 streaming parallel Welford', observations={})
    selection = [dict(episode_id=str(e['episode_id']), scene_id=e['scene_id']) for e in selected]
    config = runtime_config('val_seen', ids, output)
    effective_config = calibration_depth_config(config)
    source_files = subprocess.check_output(['git', 'ls-files', '--'] + PROTECTED).decode().splitlines()
    source_files += ['scripts/run_sparse_depth_feature_alignment.py', 'scripts/sparse_depth_feature_alignment.py',
                     'scripts/alignment_calibration_recovery.py', 'scripts/diagnose_alignment_sensor_resets.py',
                     'scripts/renderer_bootstrap.bash', 'scripts/analyze_sparse_depth_proposer_collapse.py',
                     'scripts/sparse_depth_run_manifest.py']
    lock = dict(git_commit=manifest['git_commit'], source_and_config_sha256={p:sha(p) for p in sorted(source_files)},
                runtime_config_sha256=hashlib.sha256(config.dump().encode()).hexdigest(),
                calibration_runtime_config_sha256=hashlib.sha256(effective_config.dump().encode()).hexdigest(),
                dataset_sha256=CALIBRATION_SHA, checkpoint_hashes=manifest['checkpoint_hashes'],
                depth_initialization_checkpoint_sha256=sha(config.MODEL.DEPTH_ENCODER.ddppo_checkpoint),
                torch_version=manifest['torch_version'], numpy_version=manifest['numpy_version'],
                gpu=manifest['gpu'], LD_PRELOAD=manifest['LD_PRELOAD'], formula=FORMULA, epsilon=EPSILON,
                collection_method='fresh single-episode environments; shared frozen model; twelve Depth views and no RGB; '
                                  'at most five attempts; require two consecutive valid bit-identical sensor panoramas')
    feature_shape = (12, 128, 4, 4)
    checkpoint = output / 'running_checkpoint.json'
    if resume:
        # All provenance/selection/shape checks precede environment construction and model loading.
        progress = CalibrationProgress.load(checkpoint, selection, lock, feature_shape)
        saved = json.loads((output / 'selection.json').read_text())
        if saved['episode_ids'] != ids or saved['recovery_provenance'] != lock:
            raise ValueError('saved selection/provenance changed')
        manifest = saved
        if sha(output / 'runtime_config.yaml') != lock['runtime_config_sha256']:
            raise ValueError('saved runtime config changed')
        for record in progress.records:
            for attempt in record['sensor_attempts']:
                if sha(output / attempt['path']) != attempt['sha256']:
                    raise ValueError('saved sensor attempt provenance changed')
    else:
        progress = CalibrationProgress(selection, lock, feature_shape)
        manifest.update(selection=selection, recovery_provenance=lock)
        write_json(output / 'selection.json', manifest)
        (output / 'runtime_config.yaml').write_text(config.dump())
        progress.save(checkpoint)
    limit = len(ids) if stop_after is None else stop_after
    if progress.next_index == len(ids) and limit == len(ids):
        # A crash after the last observation must not require another reset/count.
        states = [r['model_state_after'] for r in progress.records]
        if any(state != states[0] for state in states):
            raise ValueError('completed observations have inconsistent frozen model states')
        finish_calibration(output, manifest, progress, states[0], states[-1])
        return
    if not 0 < limit <= len(ids) or progress.next_index >= limit:
        raise ValueError('stop-after must exceed the completed prefix and not exceed the fixed selection')
    start_index = progress.next_index
    invocation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    tm, config, trainer, envs, provider = runtime('val_seen', ids, output, [ids[start_index]], config,
                                               calibration_only=True)
    before = model_hashes(trainer)
    if progress.records and any(r['model_state_after'] != before for r in progress.records):
        envs.close()
        raise ValueError('loaded model state differs from completed observations')
    moments = progress.moments
    try:
        assert not any(p.requires_grad for p in provider.original.depth_encoder.parameters())
        assert feature_shape == (12,) + tuple(provider.original.depth_encoder.visual_encoder.output_shape)
        for index in range(start_index, limit):
            first_env, envs = envs, None
            batch, record = stable_calibration_batch(tm, config, trainer, first_env, output, index, selection[index], invocation)
            eid = record['episode_id']
            assert eid == ids[index]
            assert record['scene_id'] == str(Path(config.TASK_CONFIG.DATASET.SCENES_DIR) / selection[index]['scene_id'])
            record['runtime_scene_id'] = record['scene_id']
            record['scene_id'] = selection[index]['scene_id']
            record.update(calibration_index=index+1, environment_closed=True,
                          invocation=invocation,
                          depth_encoder_object_id=id(provider.original.depth_encoder), model_loads_this_invocation=1)
            validate_calibration_ids('val_seen', [eid])
            features = calibration_features(provider, batch)
            shape = tuple(features['dense'].shape)
            record['model_state_after'] = model_hashes(trainer)
            assert record['model_state_after'] == before
            record['features'] = {name:dict(shape=list(value.shape), finite=bool(torch.isfinite(value).all()),
                                          sha256=fingerprint(value)) for name,value in features.items()}
            progress.add(record, features)
            progress.save(checkpoint)
            moments = progress.moments
            assert all(m.observations == index+1 and m.count == (index+1)*192 for m in moments.values())
            print('CALIBRATED', index + 1, '/', len(ids), eid, record['scene_id'], shape, flush=True)
        after = model_hashes(trainer)
        assert before == after
    finally:
        if envs is not None:
            envs.close()
    atomic_json(output / 'collection_status.json', dict(complete=progress.next_index == len(ids),
                intended_observations=len(ids), completed_episode_ids=progress.completed_ids,
                completed_scenes=[r['scene_id'] for r in progress.records], next_calibration_index=progress.next_index,
                resumed_from_index=start_index, collected_this_invocation=progress.next_index-start_index,
                model_loads_this_invocation=1, parameters_and_buffers_unchanged=True,
                model_state_before=before, model_state_after=after, navigation_steps=0, training_steps=0,
                moments={k:dict(observations=m.observations, count=m.count, shape=list(m.shape)) for k,m in moments.items()}))
    if progress.next_index < len(ids):
        print('BOUNDED STOP', progress.next_index, 'of', len(ids), 'selection unchanged; no final statistics', flush=True)
        return
    finish_calibration(output, manifest, progress, before, after)


def finish_calibration(output, manifest, progress, before, after):
    """Idempotent finalization from an already complete provenance-checked checkpoint."""
    ids, scenes, moments = manifest['episode_ids'], manifest['scenes'], progress.moments
    assert progress.next_index == len(ids)
    manifest['observations'] = {r['episode_id']:r for r in progress.records}
    values = {}
    for name, moment in moments.items():
        values['mu_' + name], values['sigma_' + name] = moment.finish()
    temporary = output / 'feature_statistics.npz.tmp'
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(output / 'feature_statistics.npz'))
    manifest.update(feature_shape=list(moments['dense'].shape),
                    axis_semantics=['views (clockwise Original packing)', 'channels', 'height', 'width'],
                    reduction_axes_per_observation=[0, 2, 3], reduction_axes_stacked_observations=[0, 1, 3, 4],
                    retained_axis='channel (128); spatial coordinates never rearranged or pooled by transform',
                    number_of_feature_samples_per_condition=len(ids) * moments['dense'].shape[0],
                    number_of_scalar_samples_per_channel_per_condition=moments['dense'].count,
                    spatial_embeddings_appended=False, parameters_and_buffers_unchanged=True,
                    model_state_before=before, model_state_after=after,
                    feature_statistics_sha256=sha(output / 'feature_statistics.npz'))
    atomic_json(output / 'calibration_manifest.json', manifest)
    gains = values['sigma_dense'] / (values['sigma_sparse'] + EPSILON)
    summary = dict(number_of_observations=len(ids), number_of_scenes=len(scenes),
                   feature_shape=manifest['feature_shape'], statistics={k: statistics(v) for k, v in values.items()},
                   gain_statistics=statistics(gains), gain_per_channel=gains.tolist(),
                   near_zero_sparse_sigma_channels=int((values['sigma_sparse'] < EPSILON).sum()),
                   per_channel={k: v.tolist() for k, v in values.items()},
                   formula=FORMULA, epsilon=EPSILON, held_out_data_used=False)
    atomic_json(output / 'summary.json', summary)


def activity(value):
    x = value.detach()
    return dict(statistics=statistics(x), positive_count=int((x > 0).sum()),
                positive_fraction=float((x > 0).double().mean()),
                all_zero_views=int((x.reshape(12, -1) == 0).all(dim=1).sum()))


def crossings(left, right):
    a, b = left > 0, right > 0
    return dict(nonpositive_to_positive_count=int((~a & b).sum()),
                positive_to_nonpositive_count=int((a & ~b).sum()),
                sign_disagreement_fraction=float((a != b).double().mean()))


@torch.no_grad()
def capture(original, batch, alignment=None):
    encoded = encode(original, batch)
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
        output, logits, candidate_map, features = predict_encoded(original, encoded, alignment)
    finally:
        for handle in handles:
            handle.remove()
    for prefix in ('visual_fc_depth', 'classifier'):
        pre, post = captured[prefix + '_pre_relu'], captured[prefix + '_post_relu']
        assert torch.isfinite(pre).all() and torch.isfinite(post).all(), prefix
        assert pre.shape == post.shape and torch.equal(pre.clamp_min(0), post), prefix
    captured.update(predictor_input=features.detach().clone(), raw_logits=logits.detach().clone())
    result = dict(encoder=dict(statistics=statistics(encoded.depth_embedding), diversity=diversity(encoded.depth_embedding)),
                  predictor_input=dict(statistics=statistics(features), diversity=diversity(features)),
                  visual_fc_depth_pre_relu=activity(captured['visual_fc_depth_pre_relu']), visual_fc_depth_post_relu=activity(captured['visual_fc_depth_post_relu']),
                  classifier_pre_relu=activity(captured['classifier_pre_relu']),
                  classifier_post_relu=activity(captured['classifier_post_relu']),
                  logits=logit_statistics(logits, candidate_map),
                  candidate_count=len(output['cand_angles'][0]),
                  rgb_embedding_sha256=fingerprint(encoded.rgb_embedding),
                  depth_encoder_object_id=id(original.depth_encoder), predictor_object_id=id(predictor))
    result['logits']['distinct_candidate_radii'] = len(set(output['cand_distances'][0]))
    return result, captured, output


@torch.no_grad()
def evaluate(output, calibration):
    manifest = json.loads((calibration / 'calibration_manifest.json').read_text())
    validate_calibration_ids(manifest['calibration_split'], manifest['episode_ids'])
    assert manifest['dataset_sha256'] == CALIBRATION_SHA
    assert sha(calibration / 'feature_statistics.npz') == manifest['feature_statistics_sha256']
    assert manifest['sparse_depth_config_sha256'] == sha(CONFIG)
    assert manifest['epsilon'] == EPSILON and manifest['formula'] == FORMULA
    alignment = FixedChannelAlignment.from_npz(calibration / 'feature_statistics.npz')
    alignment_before = state_fingerprint(alignment)
    assert not list(alignment.parameters())
    report = provenance()
    assert report['checkpoint_hashes'] == manifest['checkpoint_hashes']
    report.update(episode_ids=HELD_OUT_IDS, observations={},
                  calibration_manifest_sha256=sha(calibration / 'calibration_manifest.json'),
                  calibration_feature_statistics_sha256=sha(calibration / 'feature_statistics.npz'),
                  provider=MID360SparseDepthFeatureAlignedETP.name,
                  research_question='Can upstream alignment of sparse depth-encoder feature statistics restore downstream classifier-ReLU activity and waypoint diversity?',
                  primary_mechanism_metrics=['classifier_post_relu.positive_count',
                                             'classifier_post_relu.positive_fraction'],
                  known_complete_collapse_layer='vis_classifier[1]',
                  definitions=dict(
                      visual_fc_depth_pre_relu='visual_fc_depth[1] Linear output',
                      visual_fc_depth_post_relu='visual_fc_depth[2] ReLU output; upstream diagnostic',
                      classifier_pre_relu='vis_classifier[0] Linear output',
                      classifier_post_relu='vis_classifier[1] ReLU output; primary collapse/rescue metric',
                      std='population float64',
                      normalized_l2='||a-b||2 / max(||a||2, ||b||2, 1e-12)',
                      cosine='flattened vectors; null for zero norm',
                      view_variance='mean coordinate population variance across 12 views',
                      zero_crossings='elementwise <=0 to >0, and >0 to <=0; pair name gives direction',
                      entropy='Original float32 softmax over 1440 cells, nats; collapsed bias is nearly uniform',
                      candidate_angles='counterclockwise degrees modulo 360'))
    # Load frozen calibration fully BEFORE the first held-out reset.
    tm, config, trainer, envs, provider = runtime('val_unseen', HELD_OUT_IDS, output)
    before = model_hashes(trainer)
    assert before == manifest['model_state_after']
    historical = json.loads((FORENSICS / 'analysis.json').read_text())
    (output / 'captured').mkdir()
    try:
        for _ in HELD_OUT_IDS:
            batch, record = reset_batch(tm, config, trainer, envs)
            eid = record['episode_id']
            assert eid in HELD_OUT_IDS and eid not in report['observations']
            frozen = historical['observations'][eid]
            assert record['scene_id'] == frozen['scene_id']
            assert record['raw_depth_sha256'] == frozen['raw_depth_sha256']
            assert np.allclose(record['agent_pose']['position'], frozen['agent_pose']['position'], atol=1e-6, rtol=0)
            assert abs(record['agent_pose']['heading'] - frozen['agent_pose']['heading']) < 1e-6
            conditions, diagnostics = make_conditions(batch, provider.adapter, config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR)
            conditions['SPARSE_ALIGNED'] = conditions['SPARSE']
            record['conditions'], tensors = {}, {}
            for name in ('DENSE', 'SPARSE', 'SPARSE_ALIGNED', 'ALL_FAR'):
                aligned = alignment if name == 'SPARSE_ALIGNED' else None
                result, tensors[name], actual = capture(provider.original, conditions[name], aligned)
                result['input'] = input_statistics(conditions[name], diagnostics if name.startswith('SPARSE') else None)
                if aligned is None:
                    with np.load(str(FORENSICS / 'captured' / (eid + '.npz')), allow_pickle=False) as saved:
                        for field in ('depth_embedding', 'raw_logits'):
                            assert np.array_equal(tensors[name][field].cpu().numpy(), saved[name + '_' + field]), (eid, name, field)
                        assert np.array_equal(tensors[name]['visual_fc_depth_post_relu'].cpu().numpy(),
                                              saved[name + '_predictor_depth_fc']), (eid, name, 'visual_fc_depth')
                    accepted = historical['predictor_internal_trace']['observations'][eid]['conditions'][name]
                    assert result['classifier_post_relu']['positive_count'] == accepted['blocks']['classifier_post_relu']['nonzero_count']
                    if name in ('SPARSE', 'ALL_FAR'):
                        assert result['visual_fc_depth_post_relu']['positive_count'] > 0
                        assert result['classifier_post_relu']['positive_count'] == 0
                    result['accepted_forensics_encoder_and_logits_bit_identical'] = True
                # Also compare the public providers to the explicit analysis seam.
                if name == 'DENSE':
                    reference = provider.original.predict(batch, in_train=False)
                elif name == 'SPARSE':
                    reference = provider.predict(batch, in_train=False)
                elif aligned is not None:
                    reference = MID360SparseDepthFeatureAlignedETP(provider, alignment).predict(batch)
                else:
                    reference = provider.original.predict(conditions[name], in_train=False)
                assert actual['cand_angles'] == reference['cand_angles']
                assert actual['cand_distances'] == reference['cand_distances']
                assert torch.equal(actual['pano_depth'], reference['pano_depth'])
                result['provider_candidates_and_policy_depth_match'] = True
                record['conditions'][name] = result
            for field in ('rgb_embedding_sha256', 'depth_encoder_object_id', 'predictor_object_id'):
                assert len({c[field] for c in record['conditions'].values()}) == 1
            assert torch.equal(tensors['SPARSE']['depth_embedding'], tensors['SPARSE_ALIGNED']['depth_embedding'])
            assert torch.equal(tensors['SPARSE']['raw_logits'], tensors['ALL_FAR']['raw_logits'])
            record['comparisons'] = {}
            for a, b in [('DENSE', 'SPARSE'), ('DENSE', 'SPARSE_ALIGNED'), ('SPARSE', 'ALL_FAR'), ('SPARSE', 'SPARSE_ALIGNED')]:
                comparison = {field: similarity(tensors[a][field], tensors[b][field]) for field in
                              ('depth_embedding', 'predictor_input', 'visual_fc_depth_pre_relu', 'visual_fc_depth_post_relu',
                               'transformer_output', 'classifier_pre_relu', 'classifier_post_relu', 'raw_logits')}
                comparison['visual_fc_depth_zero_crossings'] = crossings(tensors[a]['visual_fc_depth_pre_relu'], tensors[b]['visual_fc_depth_pre_relu'])
                comparison['classifier_zero_crossings'] = crossings(tensors[a]['classifier_pre_relu'], tensors[b]['classifier_pre_relu'])
                record['comparisons'][a + '_vs_' + b] = comparison
            np.savez_compressed(str(output / 'captured' / (eid + '.npz')),
                                **{name + '_' + field: tensor.cpu().numpy() for name, values in tensors.items() for field, tensor in values.items()})
            report['observations'][eid] = record
            print('EVALUATED', eid, {k: dict(visual_fc_depth_relu=v['visual_fc_depth_post_relu'],
                  classifier_relu=v['classifier_post_relu'], distinct_candidate_radii=v['logits']['distinct_candidate_radii']) for k, v in record['conditions'].items()}, flush=True)
        after = model_hashes(trainer)
        assert before == after == manifest['model_state_after']
        assert alignment_before == state_fingerprint(alignment)
    finally:
        envs.close()
    report.update(model_state_before=before, model_state_after=after,
                  parameters_and_buffers_unchanged=True, alignment_has_no_trainable_parameters=True,
                  alignment_buffers_unchanged=True, same_frozen_models_as_calibration=True,
                  held_out_reset_count=len(report['observations']))
    assert set(report['observations']) == set(HELD_OUT_IDS)
    write_json(output / 'analysis.json', report)
    write_activity_report(output, report)
    write_json(output / 'calibration_manifest.json', manifest)
    write_json(output / 'calibration_summary.json', json.loads((calibration / 'summary.json').read_text()))


def write_activity_report(output, report):
    lines = ['# Upstream feature alignment / downstream classifier activity', '',
             report['research_question'], '',
             'Primary mechanism metrics are classifier_post_relu positive_count and positive_fraction.',
             'visual_fc_depth ReLU activity is an upstream diagnostic. Classifier activity alone',
             'does not establish recovery of logit or waypoint diversity.', '',
             'Interpretation: classifier activity plus diversity recovery supports upstream mismatch',
             'as a sufficient driver of the downstream dead-ReLU regime in these observations.',
             'Front-end changes with a still-dead classifier mean moment alignment is insufficient.',
             'Classifier activity without meaningful diversity recovery is only a partial rescue.', '']
    for eid, observation in report['observations'].items():
        lines += ['## Episode ' + eid, '',
                  '| Condition | visual_fc_depth ReLU active % | classifier ReLU positive count | classifier ReLU active % | logit entropy (nats) | distinct candidate radii |',
                  '| --- | --- | --- | --- | --- | --- |']
        for name in ('DENSE', 'SPARSE', 'SPARSE_ALIGNED', 'ALL_FAR'):
            c = observation['conditions'][name]
            lines.append('| {} | {:.6f} | {} | {:.6f} | {:.9f} | {} |'.format(
                name, 100 * c['visual_fc_depth_post_relu']['positive_fraction'],
                c['classifier_post_relu']['positive_count'],
                100 * c['classifier_post_relu']['positive_fraction'],
                c['logits']['entropy_nats'], c['logits']['distinct_candidate_radii']))
        lines.append('')
    with (output / 'report.md').open('x') as handle:
        handle.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['calibrate', 'evaluate', 'calibration-smoke'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--calibration', default='outputs/mid360_sparse_depth_alignment_calibration')
    parser.add_argument('--resume', action='store_true', help='Verify provenance and resume the saved selection prefix')
    parser.add_argument('--stop-after', type=int, help='Stop at this absolute selection prefix; retain all 53 selected IDs')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    assert torch.cuda.is_available(), 'requires accepted CUDA renderer environment'
    assert not any(k.startswith('ETP_') and 'FORENSICS' in k for k in os.environ)
    output = Path(args.output).resolve()
    archive = Path(args.calibration).resolve()
    if args.mode == 'calibration-smoke' and (output == archive or archive in output.parents):
        parser.error('smoke output must be fresh and outside the accepted calibration archive')
    if args.mode != 'calibrate' and (args.resume or args.stop_after is not None):
        parser.error('recovery and bounded stops are calibration-only')
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    before = protected_hashes()
    if args.mode == 'calibrate':
        calibrate(output, resume=args.resume, stop_after=args.stop_after)
    elif args.mode == 'calibration-smoke':
        calibration_smoke(output, archive)
    else:
        evaluate(output, archive)
    assert before == protected_hashes(), 'accepted evidence or unrelated scripts changed'
    subprocess.check_call(['git', 'diff', '--exit-code', 'b402412', '--'] + PROTECTED)
    atomic_json(output / 'preservation.json', dict(accepted_source_unchanged=True,
               accepted_evidence_and_unrelated_scripts_unchanged=True, sha256=before))
    print('COMPLETE', args.mode, 'zero navigation/training steps', flush=True)


if __name__ == '__main__':
    main()
