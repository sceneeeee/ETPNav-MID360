#!/usr/bin/env python3
"""Collect completed frozen-20 evidence; never starts evaluation or tunes parameters."""
import argparse
import json
from pathlib import Path

import numpy as np

from collect_original_baseline20_evidence import EPISODE_IDS

BASELINE = dict(SR=.8, SPL=.6983661897804773, NE=2.9216403543949125, OS=.8)


def load_one(root, pattern):
    files = list(root.glob(pattern))
    if len(files) != 1:
        raise ValueError('expected exactly one {} in {}'.format(pattern, root))
    return json.loads(files[0].read_text())


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('missing or nonfinite scalar evidence')
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                min=float(values.min()), max=float(values.max()))


def collect(root):
    root = Path(root)
    manifest = json.loads((root / 'run_manifest.json').read_text())
    if manifest['episode_ids'] != EPISODE_IDS or manifest['provider'] != 'mid360_sparse_depth_etp':
        raise ValueError('formal manifest does not match the frozen experiment')
    raw = root / 'mid360_sparse_depth_zero_shot20'
    stats = load_one(raw, 'stats_ep_ckpt_*_val_unseen_r0_w1.json')
    telemetry = load_one(raw, 'telemetry_ep_ckpt_*_val_unseen_r0_w1.json')
    sensors = load_one(raw, 'sensor_validity_ep_ckpt_*_val_unseen_r0_w1.json')
    if list(stats) != EPISODE_IDS or list(telemetry['episodes']) != EPISODE_IDS:
        raise ValueError('frozen episode order mismatch or incomplete evaluation')
    if set(sensors) != set(EPISODE_IDS) or not all(s['rgb_valid'] and s['depth_valid'] for s in sensors.values()):
        raise ValueError('sensor validity incomplete or invalid')
    rows, latencies, episodes = [], [], []
    for eid in EPISODE_IDS:
        trace = telemetry['episodes'][eid]
        decisions = trace['waypoint_trace']
        if not decisions or len(decisions) != len(trace['decision_latency_ms']):
            raise ValueError('incomplete decision telemetry: ' + eid)
        for decision in decisions:
            row = decision['sparse_depth']
            counts = row['valid_projected_pixels_per_view']
            if len(counts) != 12 or sum(counts.values()) != row['valid_projected_pixels']:
                raise ValueError('invalid per-view counts')
            fraction = row['valid_projected_pixels'] / row['total_depth_pixels']
            if not 0 <= fraction <= 1 or abs(fraction - row['valid_projected_fraction']) > 1e-10:
                raise ValueError('invalid projected fraction')
            rows.append(row)
        latencies.extend(trace['decision_latency_ms'])
        episodes.append(dict(episode_id=eid, scene_id=trace['scene_id'], **stats[eid]))
    mapping = dict(SR='success', SPL='spl', NE='distance_to_goal', OS='oracle_success',
                   nDTW='ndtw', SDTW='sdtw', mean_trajectory_length='path_length', mean_steps='steps_taken')
    current = {name: statistics([stats[eid][key] for eid in EPISODE_IDS])['mean'] for name, key in mapping.items()}
    current.update(mean_decision_latency_ms=statistics(latencies)['mean'],
                   p95_decision_latency_ms=float(np.percentile(latencies, 95)),
                   peak_vram_mib=telemetry['summary']['cuda_memory']['peak_allocated_mib'])
    delta = {key: current[key] - value for key, value in BASELINE.items()}
    summary = dict(current=current, frozen_original_baseline=BASELINE,
                   signed_delta=delta, absolute_delta={k: abs(v) for k, v in delta.items()},
                   episode_count=20, decision_count=len(rows), sensor_validity_complete=True)
    per_view = {}
    for key in rows[0]['valid_projected_pixels_per_view']:
        per_view[key] = dict(
            projected_pixels=statistics([r['valid_projected_pixels_per_view'][key] for r in rows]),
            projected_fraction=statistics([r['valid_projected_pixels_per_view'][key] / (r['total_depth_pixels'] / 12) for r in rows]))
    sparse_summary = dict(valid_projected_fraction=statistics([r['valid_projected_fraction'] for r in rows]),
                          mean_filtered_point_count=statistics([r['filtered_cloud_points'] for r in rows])['mean'],
                          per_view=per_view, aggregation='each adapted decision batch item equally weighted',
                          no_return_fill_normalized=sorted(set(r['no_return_fill_normalized'] for r in rows)))
    return {'summary.json': summary, 'episodes.json': episodes, 'sparse_depth_summary.json': sparse_summary,
            'telemetry.json': telemetry, 'sensor_validity.json': sensors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    outputs = collect(args.output)
    for name, value in outputs.items():
        (Path(args.output) / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
