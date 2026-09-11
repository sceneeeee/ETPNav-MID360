#!/usr/bin/env python3
"""Bounded raw sensor forensics: five fresh lifecycles, then ten sequential resets.

No model construction, encoding, prediction, navigation, or training.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np


def sensor_statistics(observation):
    result = {}
    for key, value in observation.items():
        if not (key == 'rgb' or key.startswith('rgb_') or key == 'depth' or key.startswith('depth_')):
            continue
        array = np.ascontiguousarray(value)
        x = array.astype(np.float64)
        finite = np.isfinite(x)
        values = x[finite]
        record = dict(key=key, shape=list(array.shape), dtype=str(array.dtype),
                      min=float(values.min()) if values.size else None,
                      max=float(values.max()) if values.size else None,
                      mean=float(values.mean()) if values.size else None,
                      std=float(values.std()) if values.size else None,
                      nonzero_fraction=float(np.mean(array != 0)),
                      finite_fraction=float(finite.mean()),
                      sha256=hashlib.sha256(array.tobytes()).hexdigest())
        if key.startswith('rgb'):
            pixels = np.any(array != 0, axis=-1)
            record.update(nonzero_pixel_count=int(pixels.sum()), nonzero_pixel_fraction=float(pixels.mean()))
        else:
            record['fraction_at_normalized_1'] = float(np.mean(array == 1.0))
        result[key] = record
    return result


def validity(stats):
    rgb = [s for k, s in stats.items() if k.startswith('rgb')]
    depth = [s for k, s in stats.items() if k.startswith('depth')]
    return dict(rgb_valid=len(rgb) == 12 and all(s['finite_fraction'] == 1 and s['nonzero_fraction'] > 0 for s in rgb),
                depth_valid=len(depth) == 12 and all(s['finite_fraction'] == 1 and s['nonzero_fraction'] > 0 and
                    0 <= s['min'] <= s['max'] <= 1 and s['fraction_at_normalized_1'] < 1 for s in depth))


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def make_ordered_env(config, ids):
    """Diagnostic-only factory: pin the saved selection order before construction."""
    import habitat_extensions  # Register VLN-CE dataset in the fresh worker.
    import vlnce_baselines  # Register the accepted environment class.
    from habitat import make_dataset
    from habitat_baselines.common.environments import get_env_class
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    byid = {str(e.episode_id):e for e in dataset.episodes}
    dataset.episodes = [byid[eid] for eid in ids]
    env = get_env_class(config.ENV_NAME)(config=config, dataset=dataset)
    env.seed(config.TASK_CONFIG.SEED)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--additional-fresh-index', type=int,
                        help='Append three independent fresh lifecycles for a discovered failing selection index')
    parser.add_argument('--stability-index', type=int, help='Bounded five-attempt stability probe of an exact episode')
    parser.add_argument('--ordered-first-ten', action='store_true', help='Append ten sequential resets in saved selection order')
    args = parser.parse_args()
    if args.additional_fresh_index is None and args.stability_index is None and not args.ordered_first_ten:
        args.output.mkdir(parents=True, exist_ok=False)
    from habitat.config.default import Config
    from vlnce_baselines.common.env_utils import construct_envs
    from habitat_baselines.common.environments import get_env_class
    from vlnce_baselines.evaluation_sensor_diagnostics import require_valid_sensor_observations
    import torch
    config = Config()
    config.merge_from_file(str(args.archive / 'runtime_config.yaml'))
    config.freeze()
    selection = json.loads((args.archive / 'selection.json').read_text())
    ids = selection['episode_ids']
    with gzip.open(ROOT / selection['dataset_path'], 'rt') as f:
        byid = {str(e['episode_id']): e for e in json.load(f)['episodes']}
    report = dict(archive=str(args.archive), selection_sha256=hashlib.sha256((args.archive/'selection.json').read_bytes()).hexdigest(),
                  runtime_config_sha256=hashlib.sha256((args.archive/'runtime_config.yaml').read_bytes()).hexdigest(),
                  selection=[dict(calibration_index=i+1, episode_id=eid, scene_id=byid[eid]['scene_id']) for i,eid in enumerate(ids)],
                  fresh=[], sequential=[], navigation_steps=0, training_steps=0, model_loads=0,
                  LD_PRELOAD=os.environ.get('LD_PRELOAD'), command=sys.argv)
    if args.additional_fresh_index is not None or args.stability_index is not None or args.ordered_first_ten:
        report = json.loads((args.output / 'analysis.json').read_text())
    def create(allowed):
        random.seed(config.TASK_CONFIG.SEED)
        np.random.seed(config.TASK_CONFIG.SEED)
        torch.manual_seed(config.TASK_CONFIG.SEED)
        return construct_envs(config.clone(), get_env_class(config.ENV_NAME), episodes_allowed=allowed, auto_reset_done=False)
    def capture(envs, reset_index, lifecycle):
        obs = envs.reset()[0]
        episode = envs.current_episodes()[0]
        eid = str(episode.episode_id)
        record = dict(reset_index=reset_index, calibration_index=ids.index(eid)+1, episode_id=eid,
                      scene_id=episode.scene_id, agent_pose=envs.call_at(0, 'get_agent_info'), lifecycle=lifecycle,
                      sensors=sensor_statistics(obs))
        record.update(validity(record['sensors']))
        # Persist raw scalar summaries BEFORE invoking the strict validator.
        dump(args.output / ('raw_' + lifecycle + '_' + str(reset_index) + '.json'), record)
        try:
            require_valid_sensor_observations(obs)
            record['strict_validation_error'] = None
        except RuntimeError as error:
            record['strict_validation_error'] = str(error)
        print('SENSOR', lifecycle, reset_index, 'selected', record['calibration_index'], eid,
              episode.scene_id, 'RGB', record['rgb_valid'], 'DEPTH', record['depth_valid'], flush=True)
        return record
    if args.ordered_first_ten:
        from habitat import VectorEnv
        ordered = config.clone()
        ordered.defrost()
        ordered.TASK_CONFIG.DATASET.EPISODES_ALLOWED = ids[:10]
        ordered.freeze()
        envs = VectorEnv(make_env_fn=make_ordered_env, env_fn_args=((ordered, ids[:10]),), auto_reset_done=False)
        report['sequential_selection_order'] = []
        try:
            for index in range(1,11):
                record = capture(envs, index, 'sequential_selection_order')
                assert record['calibration_index'] == index
                report['sequential_selection_order'].append(record)
                dump(args.output / 'analysis.json', report)
        finally:
            envs.close()
        report['sequential_selection_order_environment_closed'] = True
        dump(args.output / 'analysis.json', report)
        return
    plan = [(8,1),(9,1),(9,2),(9,3),(10,1)] if args.additional_fresh_index is None else [
        (args.additional_fresh_index, repeat) for repeat in (1,2,3)]
    stability = None
    if args.stability_index is not None:
        from alignment_calibration_recovery import SensorStability
        stability = SensorStability()
        plan = [(args.stability_index, repeat) for repeat in range(1,6)]
    for index, repeat in plan:
        lifecycle = '{}_{}_{}'.format('stability' if stability else 'fresh', index, repeat)
        envs = create([ids[index-1]])
        try:
            assert sum(envs.number_of_episodes) == 1
            record = capture(envs, 1, lifecycle)
            assert record['episode_id'] == ids[index-1]
            report['fresh'].append(record)
        finally:
            envs.close()
        report['fresh'][-1]['environment_closed'] = True
        dump(args.output / 'analysis.json', report)
        if stability and stability.observe(record['sensors'], record['rgb_valid'] and record['depth_valid']):
            break
    if stability:
        report['bounded_stability_probe'] = dict(calibration_index=args.stability_index,
                attempts=repeat, max_attempts=5, stable=stability.stable,
                acceptance='two consecutive valid fresh resets with identical hashes in all RGB and Depth views')
        dump(args.output / 'analysis.json', report)
    if args.additional_fresh_index is not None or args.stability_index is not None:
        return
    # Preserve the original full allowed set and iterator, but consume only ten resets.
    envs = create(ids)
    try:
        for index in range(1,11):
            report['sequential'].append(capture(envs, index, 'sequential'))
            dump(args.output / 'analysis.json', report)
    finally:
        envs.close()
    report['sequential_environment_closed'] = True
    dump(args.output / 'analysis.json', report)


if __name__ == '__main__':
    main()
