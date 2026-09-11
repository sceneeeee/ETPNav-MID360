#!/usr/bin/env python3
"""CPU-only evidence audit and fail-closed static alignment report. No model imports."""
import argparse
import csv
import gzip
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from sparse_depth_run_manifest import EXPECTED, sha

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / 'outputs/mid360_sparse_depth_alignment_calibration_depth_only'
OLD = ROOT / 'outputs/mid360_sparse_depth_alignment_calibration'
SMOKE = ROOT / 'outputs/alignment_depth_only_smoke_137_ygzWHJ/evidence/smoke.json'
FORENSICS = ROOT / 'outputs/mid360_sparse_depth_forensics'
IDS = ['53', '821', '1024', '250', '25']
CONDITIONS = ['DENSE', 'SPARSE', 'SPARSE_ALIGNED', 'ALL_FAR']
FROZEN = set('53 1685 821 1024 1477 1636 675 164 1375 1679 250 2 475 1356 252 1439 1016 1285 1478 25'.split())
STAT_SHA = 'a3c4e9fc315c2b5ef0d3bddc62353055c52e8fe140bb11fb9a784a34ada31a2f'


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*')) if p.is_file()}


def verify_hashes(values):
    for path, expected in values.items():
        require(sha(ROOT / path) == expected, 'SHA256 mismatch: ' + path)


def validate_calibration():
    """Read existing evidence only; independently reconstruct selection and moments."""
    manifest = read(CALIBRATION / 'calibration_manifest.json')
    selection = read(CALIBRATION / 'selection.json')
    checkpoint = read(CALIBRATION / 'running_checkpoint.json')
    summary = read(CALIBRATION / 'summary.json')
    status = read(CALIBRATION / 'collection_status.json')
    preservation = read(CALIBRATION / 'preservation.json')
    lock = selection['recovery_provenance']
    require(manifest['recovery_provenance'] == checkpoint['provenance'] == lock, 'provenance disagreement')
    verify_hashes(lock['source_and_config_sha256'])
    verify_hashes(lock['checkpoint_hashes'])
    verify_hashes(EXPECTED)
    verify_hashes(preservation['sha256'])
    verify_hashes(manifest['source_sha256'])
    require(preservation['accepted_source_unchanged'] and
            preservation['accepted_evidence_and_unrelated_scripts_unchanged'], 'preservation failed')
    for filename, key in [('runtime_config.yaml', 'runtime_config_sha256'),
                          ('calibration_runtime_config.yaml', 'calibration_runtime_config_sha256')]:
        require(sha(CALIBRATION / filename) == lock[key], filename + ' changed')
    config = yaml.safe_load((CALIBRATION / 'calibration_runtime_config.yaml').read_text())
    sensors = ['DEPTH_SENSOR'] + ['DEPTH_' + str(a) for a in range(30, 360, 30)]
    require(config['SENSORS'] == config['TASK_CONFIG']['SIMULATOR']['AGENT_0']['SENSORS'] == sensors,
            'effective runtime must have exactly twelve Depth sensors, zero RGB')
    require(sha(ROOT / config['MODEL']['DEPTH_ENCODER']['ddppo_checkpoint']) ==
            lock['depth_initialization_checkpoint_sha256'], 'depth initialization changed')
    require(sha(ROOT / manifest['sparse_depth_config_path']) == manifest['sparse_depth_config_sha256'], 'adapter changed')
    dataset = ROOT / manifest['dataset_path']
    require(sha(dataset) == manifest['dataset_sha256'] == lock['dataset_sha256'] ==
            '83b231674ae3a18cb02b3d65847377e978f45cc35682f807c03becc38b09410c', 'dataset changed')
    with gzip.open(str(dataset), 'rt') as handle:
        episodes = json.load(handle)['episodes']
    lowest = {}
    for episode in sorted(episodes, key=lambda e: int(e['episode_id'])):
        if str(episode['episode_id']) not in FROZEN:
            lowest.setdefault(episode['scene_id'], str(episode['episode_id']))
    intended = [dict(episode_id=lowest[s], scene_id=s) for s in sorted(lowest)]
    ids = [r['episode_id'] for r in intended]
    scenes = [r['scene_id'] for r in intended]
    require(len(ids) == len(set(ids)) == len(scenes) == 53, 'expected 53 unique observations/scenes')
    require(manifest['calibration_split'] == selection['calibration_split'] == 'val_seen', 'wrong split')
    require(manifest['selection'] == selection['selection'] == checkpoint['selection'] == intended, 'selection differs')
    require(manifest['episode_ids'] == selection['episode_ids'] == checkpoint['completed_episode_ids'] ==
            status['completed_episode_ids'] == ids, 'completed IDs differ')
    require(status['complete'] and status['next_calibration_index'] == checkpoint['next_calibration_index'] == 53,
            'checkpoint/status not exactly complete')
    require(status['completed_scenes'] == manifest['scenes'] == scenes, 'scene order differs')
    with gzip.open(str(ROOT / next(p for p in EXPECTED if 'val_unseen' in p)), 'rt') as handle:
        unseen = json.load(handle)['episodes']
    require(not set(scenes).intersection(e['scene_id'] for e in unseen if str(e['episode_id']) in FROZEN),
            'frozen scene leakage')
    require(manifest['uses_navigation_outcomes'] is False and summary['held_out_data_used'] is False,
            'outcome/held-out leakage')
    for record in (manifest, selection, status):
        require(record['navigation_steps'] == record['training_steps'] == 0, 'non-static collection')
    require(manifest['model_state_before'] == manifest['model_state_after'] == status['model_state_before'] ==
            status['model_state_after'], 'model state changed')
    require(manifest['parameters_and_buffers_unchanged'] and status['parameters_and_buffers_unchanged'],
            'model preservation failed')
    require(len(checkpoint['records']) == len(manifest['observations']) == 53, 'record count differs')
    keys = {'depth'} | {'depth_' + str(a) for a in range(30, 360, 30)}
    attempts_seen = []
    for index, (record, expected) in enumerate(zip(checkpoint['records'], intended), 1):
        eid = expected['episode_id']
        require(record == manifest['observations'][eid] and all(record[k] == v for k, v in expected.items()), 'record differs')
        require(record['calibration_index'] == index and record['decision_index'] == 0 and
                record['environment_closed'] and record['model_state_after'] == manifest['model_state_before'],
                'record static/model/index guarantee failed')
        require(record['stable_consecutive_frames'] == 2 and record['resets'] == len(record['sensor_attempts']),
                'stability record inconsistent')
        accepted = []
        for attempt in record['sensor_attempts']:
            path = CALIBRATION / attempt['path']
            attempts_seen.append(path.name)
            require(sha(path) == attempt['sha256'], 'sensor evidence changed')
            sensor = read(path)
            require(sensor['episode_id'] == eid and sensor['scene_id'] == record['runtime_scene_id'], 'sensor pose identity')
            require(sensor['runtime_depth_key_count'] == 12 and sensor['runtime_rgb_key_count'] == 0 and
                    set(sensor['sensors']) == keys and sensor['no_rgb_observations'], 'unexpected runtime sensors')
            if attempt['valid']:
                require(sensor['depth_valid'] and sensor['strict_validation_error'] is None, 'invalid accepted frame')
                require(all(v['finite_fraction'] == 1 and v['nonzero_fraction'] > 0 for v in sensor['sensors'].values()),
                        'nonfinite/empty Depth frame')
                accepted.append({k: (v['shape'], v['dtype'], v['sha256']) for k, v in sensor['sensors'].items()})
            else:
                accepted = []
        require(len(accepted) >= 2 and accepted[-1] == accepted[-2], 'missing consecutive exact stable frames')
        require(record['raw_depth_sha256'] == {k: v[2] for k, v in accepted[-1].items()}, 'accepted panorama mismatch')
        require(all(v['shape'] == [12, 128, 4, 4] and v['finite'] for v in record['features'].values()), 'feature shape/finite')
    require(set(attempts_seen) == {p.name for p in CALIBRATION.glob('sensors_*.json')}, 'unaccounted sensor evidence')
    require(sha(CALIBRATION / 'feature_statistics.npz') == manifest['feature_statistics_sha256'] == STAT_SHA,
            'statistics hash differs from reviewed Depth-only artifact')
    with np.load(str(CALIBRATION / 'feature_statistics.npz'), allow_pickle=False) as saved:
        require(set(saved.files) == {'mu_dense', 'sigma_dense', 'mu_sparse', 'sigma_sparse'}, 'NPZ fields')
        for condition in ('dense', 'sparse'):
            moment = checkpoint['moments'][condition]
            require(moment['count'] == 10176 and moment['observations'] == 53 and moment['shape'] == [12, 128, 4, 4],
                    'moment counters/shape')
            require(status['moments'][condition] == {k: moment[k] for k in ('count', 'observations', 'shape')}, 'status moments')
            mean, m2 = np.asarray(moment['mean']), np.asarray(moment['m2'])
            require(np.isfinite(mean).all() and np.isfinite(m2).all() and (m2 >= 0).all(), 'invalid Welford state')
            for name, expected in [('mu_' + condition, mean), ('sigma_' + condition, np.sqrt(m2 / 10176))]:
                require(saved[name].dtype == np.float64 and saved[name].shape == (128,) and
                        np.array_equal(saved[name], expected) and np.array_equal(saved[name], summary['per_channel'][name]),
                        'NPZ/Welford/summary inconsistent: ' + name)
        gains = saved['sigma_dense'] / (saved['sigma_sparse'] + 1e-6)
        require(np.array_equal(gains, summary['gain_per_channel']), 'gain summary differs')
        require(summary['near_zero_sparse_sigma_channels'] == int((saved['sigma_sparse'] < 1e-6).sum()), 'zero sigma summary')
    require(manifest['number_of_observations'] == summary['number_of_observations'] == 53 and
            manifest['number_of_scenes'] == summary['number_of_scenes'] == 53 and
            manifest['number_of_scalar_samples_per_channel_per_condition'] == 10176 and
            manifest['number_of_feature_samples_per_condition'] == 636, 'manifest counts')
    require(manifest['epsilon'] == summary['epsilon'] == 1e-6 and manifest['formula'] == summary['formula'] ==
            '(sparse - mu_sparse) / (sigma_sparse + 1e-6) * sigma_dense + mu_dense', 'alignment definition')
    smoke = read(SMOKE)
    require(smoke['status'] == 'PASS' and smoke['archive_unchanged'] and
            tree(OLD) == smoke['archive_sha256_before'] == smoke['archive_sha256_after'], 'old archive mutated since smoke')
    require(read(OLD / 'running_checkpoint.json')['next_calibration_index'] == 11, 'old prefix changed')
    subprocess.check_call(['git', '--no-pager', 'diff', '--exit-code', 'b402412', '--',
                           'vlnce_baselines', 'habitat_extensions', 'configs', 'run_r2r', 'run.py'], cwd=str(ROOT))
    historical = validate_history()
    return dict(status='PASS', calibration=str(CALIBRATION), feature_statistics_sha256=manifest['feature_statistics_sha256'],
                completed=53, intended=53, samples_per_channel=10176, channels=128,
                depth_sensors=12, rgb_sensors=0, sensor_attempts=len(attempts_seen),
                navigation_steps=0, training_steps=0, model_and_source_preservation='PASS',
                old_archive_unchanged_since=str(SMOKE), unrelated_gmin_hashes='PASS',
                historical_evidence=historical, new_gpu_baseline_regression='PENDING HOST RUN',
                review_source_sha256={p: sha(ROOT / p) for p in
                    ('scripts/review_depth_only_alignment.py', 'scripts/run_depth_only_alignment_static_manual.bash')},
                calibration_tree_sha256=tree(CALIBRATION), old_archive_tree_sha256=tree(OLD))


def validate_history():
    history = read(FORENSICS / 'analysis.json')
    result = {}
    for eid in IDS:
        with np.load(str(FORENSICS / 'captured' / (eid + '.npz')), allow_pickle=False) as saved:
            require(np.array_equal(saved['SPARSE_raw_logits'], saved['ALL_FAR_raw_logits']), eid + ': historical logits differ')
            fractions = {}
            for name in ('DENSE', 'SPARSE', 'ALL_FAR'):
                front = saved[name + '_predictor_depth_fc']
                fraction = float((front > 0).mean())
                trace = history['predictor_internal_trace']['observations'][eid]['conditions'][name]
                post = trace['blocks']['classifier_post_relu']
                if name != 'DENSE':
                    require(fraction > 0, 'historical frontend is completely dead')
                    require(post['nonzero_count'] == 0 and np.prod(post['statistics']['shape']) == 9216 and
                            trace['classifier_equals_final_bias'], 'historical collapse differs')
                    logits = history['observations'][eid]['conditions'][name]['logits']
                    require(logits['post_nms_all_radial2'] and logits['cells_exactly_at_max'] == 12 and
                            all(c['distance_meters'] == .75 for c in logits['post_nms_candidates']), 'historical tied radii differ')
                fractions[name] = dict(visual_fc_depth_post_relu_positive_fraction=fraction,
                                       classifier_post_relu_positive_count=post['nonzero_count'])
            result[eid] = fractions
    return result


def candidate_signature(condition):
    return [(c['angle_index'], c['distance_index']) for c in condition['logits']['post_nms_candidates']]


def validate_evaluation(output):
    """Every baseline gate completes before any aligned interpretation/report is made."""
    report = read(output / 'analysis.json')
    manifest = read(CALIBRATION / 'calibration_manifest.json')
    history = read(FORENSICS / 'analysis.json')
    require(set(report['observations']) == set(IDS) and report['held_out_reset_count'] == 5, 'wrong held-out set')
    require(report['calibration_feature_statistics_sha256'] == manifest['feature_statistics_sha256'] and
            report['calibration_manifest_sha256'] == sha(CALIBRATION / 'calibration_manifest.json'), 'wrong calibration')
    require(report['navigation_steps'] == report['training_steps'] == 0, 'non-static experiment')
    require(report['model_state_before'] == report['model_state_after'] == manifest['model_state_after'] and
            all(report[k] for k in ('parameters_and_buffers_unchanged', 'alignment_buffers_unchanged',
                                    'alignment_has_no_trainable_parameters', 'same_frozen_models_as_calibration')),
            'model/alignment preservation failed')
    preservation = read(output / 'preservation.json')
    require(preservation['accepted_source_unchanged'] and preservation['accepted_evidence_and_unrelated_scripts_unchanged'],
            'runtime preservation failed')
    verify_hashes(preservation['sha256'])
    for eid in IDS:
        observation = report['observations'][eid]
        require(observation['decision_index'] == 0 and set(observation['conditions']) == set(CONDITIONS), 'non-static/missing condition')
        with np.load(str(output / 'captured' / (eid + '.npz')), allow_pickle=False) as actual, \
                np.load(str(FORENSICS / 'captured' / (eid + '.npz')), allow_pickle=False) as accepted:
            for name in ('DENSE', 'SPARSE', 'ALL_FAR'):
                condition = observation['conditions'][name]
                for field, old_field in [('depth_embedding', 'depth_embedding'), ('raw_logits', 'raw_logits'),
                                         ('visual_fc_depth_post_relu', 'predictor_depth_fc')]:
                    require(np.array_equal(actual[name + '_' + field], accepted[name + '_' + old_field]),
                            'HISTORICAL REGRESSION: ' + eid + '/' + name + '/' + field)
                require(candidate_signature(condition) == candidate_signature(history['observations'][eid]['conditions'][name]),
                        'HISTORICAL REGRESSION: candidate angles/radii')
                expected_count = history['predictor_internal_trace']['observations'][eid]['conditions'][name]['blocks']['classifier_post_relu']['nonzero_count']
                require(condition['classifier_post_relu']['positive_count'] == expected_count, 'HISTORICAL REGRESSION: classifier')
            require(np.array_equal(actual['SPARSE_raw_logits'], actual['ALL_FAR_raw_logits']), 'Sparse/All-Far logits differ')
            for name in CONDITIONS:
                c = observation['conditions'][name]
                for prefix in ('visual_fc_depth', 'classifier'):
                    pre, post = [actual[name + '_' + prefix + '_' + side + '_relu'] for side in ('pre', 'post')]
                    require(np.isfinite(pre).all() and np.array_equal(np.maximum(pre, 0), post), 'ReLU capture invalid')
                    recomputed_positive_count = int((post > 0).sum())
                    require(
                        c[prefix + '_post_relu']['positive_count'] == recomputed_positive_count,
                        'activity count mismatch',
                    )
                    recomputed_positive_fraction = recomputed_positive_count / float(post.size)
                    require(
                        abs(
                            c[prefix + '_post_relu']['positive_fraction']
                            - recomputed_positive_fraction
                        ) <= 1e-15,
                        'activity fraction mismatch',
                    )
                require(c['candidate_count'] == len(candidate_signature(c)), 'candidate count mismatch')
    return report


def compact_rows(report):
    rows = []
    for eid in IDS:
        for name in CONDITIONS:
            c = report['observations'][eid]['conditions'][name]
            candidates = c['logits']['post_nms_candidates']
            radii = [v['distance_meters'] for v in candidates]
            angles = [v['angle_degrees_counterclockwise'] for v in candidates]
            row = dict(observation=eid, condition=name,
                       visual_fc_depth_post_relu_positive_fraction=c['visual_fc_depth_post_relu']['positive_fraction'],
                       classifier_post_relu_positive_fraction=c['classifier_post_relu']['positive_fraction'],
                       logit_entropy=c['logits']['entropy_nats'], candidate_count=c['candidate_count'],
                       distinct_candidate_radii=len(set(radii)), candidate_radii=radii, candidate_angles=angles,
                       distinct_candidate_angles=len(set(angles)),
                       candidate_radial_bins=[v['distance_index'] for v in candidates],
                       candidate_radial_bin_counts=c['logits']['post_nms_radial_counts'])
            for site in ('visual_fc_depth_pre_relu', 'classifier_pre_relu'):
                for field in ('min', 'max', 'mean', 'std', 'finite_fraction'):
                    row[site + '_' + field] = c[site]['statistics'][field]
            rows.append(row)
    return rows


def aggregate_rows(rows):
    result = {}
    for name in CONDITIONS:
        group = [r for r in rows if r['condition'] == name]
        metrics = {key: dict(mean=float(np.mean([r[key] for r in group])),
                            min=min(r[key] for r in group), max=max(r[key] for r in group)) for key in
                   ('visual_fc_depth_post_relu_positive_fraction', 'classifier_post_relu_positive_fraction',
                    'logit_entropy', 'candidate_count', 'distinct_candidate_radii', 'distinct_candidate_angles')}
        metrics.update(observations=5,
                       classifier_active_observations=sum(r['classifier_post_relu_positive_fraction'] > 0 for r in group),
                       candidate_angle_counts=dict(sorted(Counter(a for r in group for a in r['candidate_angles']).items())),
                       candidate_radius_counts=dict(sorted(Counter(a for r in group for a in r['candidate_radii']).items())),
                       candidate_radial_bin_counts=np.sum([r['candidate_radial_bin_counts'] for r in group], axis=0).tolist())
        result[name] = metrics
    paired = []
    for eid in IDS:
        sparse, aligned = [next(r for r in rows if r['observation'] == eid and r['condition'] == name)
                           for name in ('SPARSE', 'SPARSE_ALIGNED')]
        active = aligned['classifier_post_relu_positive_fraction'] > 0
        # Conservative operational gate: radius variety plus non-tied angular output.
        diverse = (aligned['distinct_candidate_radii'] > 1 and aligned['distinct_candidate_angles'] > 1 and
                   aligned['candidate_angles'] != sparse['candidate_angles'])
        paired.append(dict(observation=eid, gate=('A' if diverse else 'B') if active else 'C',
                           classifier_activity_delta=aligned['classifier_post_relu_positive_fraction'] - sparse['classifier_post_relu_positive_fraction'],
                           entropy_delta=aligned['logit_entropy'] - sparse['logit_entropy'],
                           distinct_radii_delta=aligned['distinct_candidate_radii'] - sparse['distinct_candidate_radii']))
    return dict(conditions=result, paired_sparse_aligned_minus_sparse=paired,
                gate_counts=dict(Counter(p['gate'] for p in paired)),
                overall_gate=paired[0]['gate'] if len({p['gate'] for p in paired}) == 1 else 'MIXED; review per observation',
                diversity_gate='>1 candidate radius, >1 candidate angle, and angle sequence differs from Sparse; conservative descriptive gate, not a significance test')


def write_reports(output, report):
    rows = compact_rows(report)
    aggregate = aggregate_rows(rows)
    write(output / 'compact_results.json', rows)
    write(output / 'aggregate.json', aggregate)
    with (output / 'compact_results.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows)
    columns = list(rows[0])[:9]
    lines = ['# Static Depth-only calibrated alignment', '',
             'Historical Dense / Sparse / All-Far regression: PASS (all five observations).',
             'Primary metric: classifier_post_relu.positive_fraction. Fractions are 0–1; entropy is nats;',
             'radii are meters; angles are counterclockwise degrees modulo 360; radial bins are zero-based.',
             'The historical first complete collapse is vis_classifier post-ReLU = 0/9216.',
             'Sparse visual_fc_depth post-ReLU remains active at roughly 23–24%.', '',
             '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join('{:.8g}'.format(row[c]) if isinstance(row[c], float) else str(row[c]) for c in columns) + ' |')
    lines += ['', '## Five-observation comparison', '',
              '| Condition | Mean classifier activity | Active observations | Mean entropy | Mean candidates | Mean distinct radii | Mean distinct angles |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for name, metrics in aggregate['conditions'].items():
        lines.append('| {} | {:.8g} | {}/5 | {:.8g} | {:.8g} | {:.8g} | {:.8g} |'.format(
            name, metrics['classifier_post_relu_positive_fraction']['mean'], metrics['classifier_active_observations'],
            metrics['logit_entropy']['mean'], metrics['candidate_count']['mean'],
            metrics['distinct_candidate_radii']['mean'], metrics['distinct_candidate_angles']['mean']))
    lines += ['', 'Gate counts: ' + str(aggregate['gate_counts']) + '. Overall: ' + aggregate['overall_gate'] + '.',
              aggregate['diversity_gate'] + '.', '',
              'A: classifier activity and waypoint diversity recover → supports upstream feature-domain mismatch as a major mechanism; next: lightweight learned adapter.',
              'B: classifier activity recovers but the diversity gate fails → partial/ambiguous output rescue; inspect/fine-tune waypoint front-end. Review distributions before calling outputs collapsed.',
              'C: classifier activity remains zero → first/second-moment alignment is insufficient; next: learned adapter/front-end adaptation.',
              'Mixed outcomes require per-observation review; no statistical significance claim from five observations.',
              'No navigation, training, parameter updates, GraphMap rollout, or planner rollout. Do not run navigation yet.', '',
              'Pre-ReLU statistics are in compact_results.csv/json and analysis.json; pooled angle/radius/bin counts and paired deltas are in aggregate.json.']
    with (output / 'static_report.md').open('x') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['validate', 'report', 'preserve'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output != CALIBRATION and CALIBRATION not in output.parents and output != OLD and OLD not in output.parents,
            'output must be outside calibration evidence')
    if args.mode == 'preserve':
        before = read(output / 'preflight/calibration_validation.json')
        require(tree(CALIBRATION) == before['calibration_tree_sha256'], 'new calibration mutated during experiment')
        require(tree(OLD) == before['old_archive_tree_sha256'], 'old archive mutated during experiment')
        verify_hashes(read(CALIBRATION / 'preservation.json')['sha256'])
        verify_hashes(before['review_source_sha256'])
        write(output / 'input_preservation.json', dict(status='PASS', new_calibration_unchanged=True,
              old_calibration_unchanged=True, accepted_evidence_and_gmin_unchanged=True))
        print('PASS INPUT_PRESERVATION NEW/OLD CALIBRATION; ACCEPTED EVIDENCE; GMIN', flush=True)
        return
    audit = validate_calibration()
    if args.mode == 'validate':
        output.mkdir(parents=True, exist_ok=False)
        write(output / 'calibration_validation.json', audit)
        print('PASS CALIBRATION 53/53; DEPTH 12 RGB 0; STATISTICS; MODEL/SOURCE; OLD ARCHIVE; GMIN', flush=True)
        print('PASS HISTORICAL_EVIDENCE 5/5; HOST REGRESSION PENDING', flush=True)
    else:
        report = validate_evaluation(output)
        write(output / 'regression_validation.json', dict(status='PASS', observations=IDS,
              baseline_conditions=['DENSE', 'SPARSE', 'ALL_FAR'], encoder_frontend_logits_bit_identical=True,
              candidate_angles_and_radii_identical=True, sparse_all_far_logits_identical=True,
              calibration_feature_statistics_sha256=audit['feature_statistics_sha256']))
        write_reports(output, report)
        print('PASS HISTORICAL_REGRESSION 5/5; PASS STATIC_FROZEN; PASS REPORT 20 ROWS', flush=True)


if __name__ == '__main__':
    main()
