#!/usr/bin/env python3
"""Analyze the first frozen panel; read-only with respect to all run evidence."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_geometry_mru_bnms20 import EPISODE_IDS, read_run, metrics
from vlnce_baselines.models.graph_utils import GraphMap


def summary(values):
    v = np.asarray(values, dtype=float)
    return (dict(count=len(v), mean=float(v.mean()), median=float(np.median(v)),
                 p95=float(np.percentile(v, 95)), minimum=float(v.min()), maximum=float(v.max()))
            if len(v) else dict(count=0, mean=None, median=None, p95=None, minimum=None, maximum=None))


def exact_cells(trace):
    return [(int(round((2 * math.pi - c['angle']) * 120 / (2 * math.pi))) % 120,
             int(round(c['distance'] / .25)) - 1) for c in trace['candidates']]


def distribution(run):
    traces = [t for ep in run['telemetry']['episodes'].values() for t in ep['waypoint_trace']]
    cells = [c for t in traces for c in exact_cells(t)]
    distances = [c['distance'] for t in traces for c in t['candidates']]
    counts = [len(t['candidates']) for t in traces]
    return dict(total_candidates=len(cells), candidate_count_per_decision=summary(counts),
        distance_m=summary(distances),
        selected_waypoint_distance_m=summary([t['selected_waypoint']['distance'] for t in traces if t['selected_waypoint']]),
        radial_histogram=np.bincount([r for a, r in cells], minlength=12).tolist(),
        angular_histogram=np.bincount([a for a, r in cells], minlength=120).tolist(),
        distinct_radial_bins_per_decision=summary([len(set(r for a, r in exact_cells(t))) for t in traces]),
        zero_candidate_decisions=sum(c == 0 for c in counts), zero_candidate_frequency=float(np.mean(np.asarray(counts) == 0)),
        duplicate_image_view_decisions=sum(len(t['candidates']) > len(set(c['panorama_image_index'] for c in t['candidates'])) for t in traces))


def replay_graph(episode):
    """Replay actual GraphMap insertions with recorded poses and dummy features.

    Features do not affect GraphMap localization/topology. Never run a planner.
    Verify ghost/node counts against historical telemetry at every decision.
    """
    graph = GraphMap(False, .5, True, 0.)
    previous = None
    rows = []
    for index, trace in enumerate(episode['waypoint_trace']):
        pos = np.asarray(trace['agent_position'])
        angles = (trace['agent_heading'] + np.asarray([c['angle'] for c in trace['candidates']])) % (2 * math.pi)
        distances = np.asarray([c['distance'] for c in trace['candidates']])
        positions = np.empty((len(distances), 3))
        positions[:, 0] = pos[0] - distances * np.sin(angles)
        positions[:, 1] = pos[1]
        positions[:, 2] = pos[2] - distances * np.cos(angles)
        vp = str(len(graph.node_pos))
        ids = [vp + '_' + str(i) for i in range(len(distances))]
        before_ghost_count = graph.ghost_cnt
        graph.update_graph(previous, index + 1, vp, pos, torch.zeros(2), ids, list(positions),
                           torch.zeros(len(ids), 2), None)
        assert len(graph.node_pos) == trace['visited_node_count'], (index, 'node count')
        assert len(graph.ghost_aug_pos) == trace['ghost_node_count'], (index, 'ghost count')
        suppressed = []
        for j, cpos in enumerate(positions):
            node = graph._localize(cpos, graph.node_pos)
            if node is not None:
                suppressed.append(dict(candidate_index=j, node=node, distance_to_node=float(np.linalg.norm(cpos - graph.node_pos[node])),
                                       radial_distance=trace['candidates'][j]['distance']))
        selected = trace['selected_graph_node_or_ghost_id']
        ghost_pos = None
        front = None
        front_pos = None
        if selected is not None:
            ghost_pos = np.asarray(graph.ghost_aug_pos[selected]).tolist()
            _, front = graph.front_to_ghost_dist(selected)
            front_pos = np.asarray(graph.node_pos[front]).tolist()
            previous = front
        row = dict(step_index=index, candidate_positions=positions.tolist(),
                   candidate_cells=exact_cells(trace), selected_ghost_id=selected,
                   selected_ghost_position=ghost_pos, execution_front_node=front,
                   execution_front_position=front_pos, nodes=len(graph.node_pos), ghosts=len(graph.ghost_aug_pos),
                   graph_edges=graph.graph_nx.number_of_edges(), new_ghosts=graph.ghost_cnt-before_ghost_count,
                   candidates_localized_to_visited_nodes=suppressed,
                   candidate_count=len(distances),
                   duplicate_views=len(distances) - len(set(c['panorama_image_index'] for c in trace['candidates'])),
                   stop=trace['stop'], no_ghosts_from_graph=len(graph.ghost_pos) == 0,
                   step_limit_from_index=index == 14,
                   planner_stop=trace.get('planner_stop'),
                   stop_vp=trace.get('stop_vp'), stop_pos=trace.get('stop_pos'),
                   distance_to_goal_after_execution=trace.get('distance_to_goal'))
        # Current traces provide independent measurements for replay validation.
        if trace.get('graph_insertion') is not None:
            assert row['new_ghosts'] == trace['graph_insertion']['new_ghost_count']
            np.testing.assert_allclose(positions, np.asarray(trace['graph_insertion']['candidate_positions']).reshape(-1, 3), atol=1e-6)
        rows.append(row)
        if selected is not None:
            graph.delete_ghost(selected)
    assert graph.ghost_cnt == episode['metrics']['ghost_cnt']
    return rows


def first_difference(left, right, projector):
    for i, (a, b) in enumerate(zip(left, right)):
        va, vb = projector(a), projector(b)
        if va != vb:
            return dict(step_index=i, baseline=va, gmin=vb)
    if len(left) != len(right):
        return dict(step_index=min(len(left), len(right)), baseline='ended' if len(left) < len(right) else 'continues',
                    gmin='ended' if len(right) < len(left) else 'continues')
    return None


def rounded_position(row):
    p = row['selected_ghost_position']
    return None if p is None else [round(x, 5) for x in p]


def comparisons(label, baseline, gmin, replay):
    categories = {a + '/' + b: [] for a in ['baseline_success', 'baseline_fail'] for b in ['gmin_success', 'gmin_fail']}
    episodes = []
    for eid in EPISODE_IDS:
        old_success = bool(baseline['stats'][eid]['success']); new_success = bool(gmin['stats'][eid]['success'])
        category = ('baseline_success' if old_success else 'baseline_fail') + '/' + ('gmin_success' if new_success else 'gmin_fail')
        categories[category].append(eid)
        old, new = baseline['telemetry']['episodes'][eid], gmin['telemetry']['episodes'][eid]
        ot, nt = old['waypoint_trace'], new['waypoint_trace']
        ro, rn = replay[label][eid], replay['GMin'][eid]
        record = dict(episode_id=eid, category=category, changed_outcome=old_success != new_success,
            baseline_metrics=baseline['stats'][eid], gmin_metrics=gmin['stats'][eid],
            first_candidate_set_divergence=first_difference(ot, nt, exact_cells),
            first_selected_ghost_id_divergence=first_difference(ro, rn, lambda r: r['selected_ghost_id']),
            first_selected_ghost_position_divergence=first_difference(ro, rn, rounded_position),
            first_decision_pose_divergence=first_difference(ot, nt, lambda t: [*[round(x, 5) for x in t['agent_position']], round(t['agent_heading'], 5)]),
            first_low_level_trajectory_divergence=first_difference(old['trajectory'], new['trajectory'], lambda p: [round(x, 5) for x in p]),
            first_combined_stop_divergence=first_difference(ot, nt, lambda t: t['stop']),
            first_no_ghost_divergence=first_difference(ro, rn, lambda t: t['no_ghosts_from_graph']),
            baseline_final=ro[-1], gmin_final=rn[-1],
            initial_baseline_graph=ro[0], initial_gmin_graph=rn[0],
            gmin_decisions_creating_no_new_ghosts=[r['step_index'] for r in rn if r['new_ghosts'] == 0],
            gmin_duplicate_view_decisions=[r['step_index'] for r in rn if r['duplicate_views']],
            gmin_candidates_localized_to_nodes=sum(len(r['candidates_localized_to_visited_nodes']) for r in rn),
            causal_limit='After the first pose divergence, observations differ. Replayed topology is exact-count validated; proposer/planner causality remains observational.')
        episodes.append(record)
    return dict(baseline=label, categories=categories, episodes=episodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='outputs/geometry_gmin_frozen20_valid_renderer_attempt2')
    args = parser.parse_args(); root = Path(args.root)
    if not (root / 'EVALUATION_COMPLETE').exists(): raise RuntimeError('evaluation incomplete')
    out = root / 'analysis'; out.mkdir(exist_ok=True)
    if any(out.iterdir()): raise RuntimeError('refusing to overwrite analysis artifacts')
    folders = dict(Original='original_baseline20_valid_renderer', BNMS='geometry_v0_bounded_nms20_valid_renderer', MRU='geometry_mru_bnms20_valid_renderer')
    runs = {label: read_run(Path('outputs')/folder/folder, label) for label, folder in folders.items()}
    runs['GMin'] = read_run(root / 'geometry_gmin_frozen20_valid_renderer', 'G-Min')
    source = json.loads((root / 'run_manifest.json').read_text())['source_sha256']
    for path, digest in source.items():
        if Path(path).is_file():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, 'Frozen source changed: ' + path
    replay = {label: {eid: replay_graph(ep) for eid, ep in run['telemetry']['episodes'].items()} for label, run in runs.items()}
    gmin_traces = [t for ep in runs['GMin']['telemetry']['episodes'].values() for t in ep['waypoint_trace']]
    for trace in gmin_traces:
        assert trace['gmin']['candidate_exact_cells'] == [list(c) for c in exact_cells(trace)]
        assert len(trace['candidates']) <= 5
        assert trace['stop'] == (trace['planner_stop'] or trace['step_limit'] or trace['no_ghosts'])
        assert trace['gmin']['observation_availability']['available'] or not trace['candidates']
    timed_keys = sorted(set(k for t in gmin_traces for k in t['geometry_timing_ms']))
    output = dict(metrics={k: metrics(v) for k, v in runs.items()}, distributions={k: distribution(v) for k, v in runs.items()},
        comparisons={k: comparisons(k, runs[k], runs['GMin'], replay) for k in ['Original', 'BNMS', 'MRU']},
        performance={k: v['telemetry']['summary'] for k, v in runs.items()},
        gmin_components_ms={k: summary([t['geometry_timing_ms'][k] for t in gmin_traces]) for k in timed_keys},
        gmin_stop_conditions={k: sum(t[k] for t in gmin_traces) for k in ['planner_stop', 'step_limit', 'no_ghosts']},
        graph_replay='PASS: actual GraphMap; all historical/current node/ghost counts and cumulative ghosts match',
        frozen_source_unchanged=True)
    for label, rr in replay.items():
        entries = [r for ep in rr.values() for r in ep]
        output['distributions'][label]['new_ghosts_per_decision'] = summary([r['new_ghosts'] for r in entries])
        output['distributions'][label]['candidates_localized_to_visited_nodes'] = sum(len(r['candidates_localized_to_visited_nodes']) for r in entries)
        output['distributions'][label]['front_to_selected_ghost_distance_m'] = summary([
            np.linalg.norm(np.asarray(r['selected_ghost_position']) - r['execution_front_position'])
            for r in entries if r['selected_ghost_position'] is not None])
    (out / 'results.json').write_text(json.dumps(output, indent=2) + '\n')
    (out / 'graph_replay.json').write_text(json.dumps(replay, indent=2) + '\n')
    for label in ['Original', 'BNMS', 'MRU']:
        print(label, output['comparisons'][label]['categories'])
    print('G-Min metrics', output['metrics']['GMin'])
    print('G-Min distribution', output['distributions']['GMin'])
    print('G-Min component timing', output['gmin_components_ms'])
    print('G-Min stops', output['gmin_stop_conditions'])
    print('Analysis saved to', out)


if __name__ == '__main__':
    main()
