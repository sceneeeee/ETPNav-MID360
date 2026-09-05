#!/usr/bin/env python3
"""Same saved state, real encoders/trainer/GraphMap; never creates an environment."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from test_waypoint_provider_equivalence import (
    tensor_fingerprint, compare_fingerprints, check_config, load_policy,
    panorama_observations, DEFAULT_CONFIG, DEFAULT_CHECKPOINT, DEFAULT_DDPPO, DEFAULT_WAYPOINT,
)
from vlnce_baselines.waypoint_provider.mid360_geometry import MID360GeometryConfig, MID360GeometryWaypointProvider
from vlnce_baselines.waypoint_provider.types import validate_waypoint_output
from vlnce_baselines.models.graph_utils import GraphMap
from vlnce_baselines.ss_trainer_ETP import RLTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observation', default='outputs/example_scene/fixed_12view_policy_observation.npz',
                        choices=['outputs/example_scene/fixed_12view_policy_observation.npz'],
                        help='The fixed asset associated with the pre-refactor byte-level golden.')
    parser.add_argument('--output', default='outputs/gmin_stage1_validation/saved_state')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('saved-state golden gate requires validated CUDA path')
    device = torch.device('cuda:0')
    policy_args = SimpleNamespace(config=DEFAULT_CONFIG, checkpoint=DEFAULT_CHECKPOINT,
                                 ddppo_checkpoint=DEFAULT_DDPPO, waypoint_checkpoint=DEFAULT_WAYPOINT, episodes=1)
    report = {'observation': args.observation, 'environment_created': False, 'environment_steps': 0}
    cfg = check_config(policy_args, report)
    policy, predictor = load_policy(cfg, policy_args, device, report)
    obs = panorama_observations(device, args.observation)
    obs_before = {k: v.clone() for k, v in obs.items()}
    net = policy.net
    common = dict(rgb_encoder=net.rgb_encoder, depth_encoder=net.depth_encoder,
                  rgb_pool=net.space_pool_rgb, depth_pool=net.space_pool_depth,
                  pano_angle_features=net.pano_angle_fts, pano_image_indexes=net.pano_img_idxes)
    providers = {
        'Original': None,
        'BNMS': MID360GeometryWaypointProvider(config=MID360GeometryConfig.from_yaml('configs/mid360_waypoint.yaml'), **common),
        'GMin': MID360GeometryWaypointProvider(config=MID360GeometryConfig.from_yaml('configs/mid360_waypoint_gmin.yaml'), **common),
    }
    trainer = SimpleNamespace(envs=SimpleNamespace(num_envs=1))
    outputs, records, arrays = {}, {}, {}
    keys = ['cand_rgb', 'cand_depth', 'cand_angle_fts', 'pano_rgb', 'pano_depth', 'pano_angle_fts']
    with torch.no_grad():
        for label, provider in providers.items():
            torch.manual_seed(17)
            if provider is None:
                with patch('vlnce_baselines.perception.gmin_candidates.select_gmin_candidates',
                           side_effect=AssertionError('Original invoked G-Min')):
                    output = net(mode='waypoint', waypoint_predictor=predictor, observations=obs, in_train=False)
            else:
                output = provider.predict(obs)
            validate_waypoint_output(output)
            outputs[label] = output
            for key in ['pano_rgb', 'pano_depth', 'pano_angle_fts']:
                assert torch.equal(output[key], outputs['Original'][key]), key
            np.testing.assert_array_equal(output['pano_img_idxes'], outputs['Original']['pano_img_idxes'])
            for feature, panorama in [('cand_rgb', 'pano_rgb'), ('cand_depth', 'pano_depth')]:
                assert torch.equal(output[feature][0], output[panorama][0, output['cand_img_idxes'][0]])
            fingerprints = {k: tensor_fingerprint(output[k]) for k in keys}
            inputs = RLTrainer._vp_feature_variable(trainer, output)
            count = len(output['cand_angles'][0])
            unique_views = len(set(output['cand_img_idxes'][0].tolist()))
            assert inputs['view_lens'].item() == count + 12 - unique_views
            assert int((inputs['nav_types'] == 1).sum()) == count
            embeddings, masks = net(mode='panorama', **inputs)
            assert torch.isfinite(embeddings).all()
            average = (embeddings * masks.unsqueeze(2)).sum(1) / masks.sum(1, keepdim=True)
            graph = GraphMap(False, cfg.IL.loc_noise, cfg.MODEL.merge_ghost, 0.)
            position, orientation = np.zeros(3), [0, 0, 0, 1]
            vp, ids, positions = graph.identify_node(position, orientation,
                                                    output['cand_angles'][0], output['cand_distances'][0])
            graph.update_graph(None, 1, vp, position, average[0], ids, positions,
                               embeddings[0][inputs['nav_types'][0] == 1], None)
            trainer.gmaps = [graph]
            nav = RLTrainer._nav_gmap_variable(trainer, [vp], [position], [orientation])
            assert torch.isfinite(nav['gmap_img_fts']).all()
            assert torch.isfinite(nav['gmap_pos_fts']).all()
            angle_bins = [int(round((2 * math.pi - a) * 120 / (2 * math.pi))) % 120 for a in output['cand_angles'][0]]
            radius_bins = [int(round(d / .25)) - 1 for d in output['cand_distances'][0]]
            record = dict(candidate_count=count, angle_bins=angle_bins, radial_bins=radius_bins,
                          angles=output['cand_angles'][0], distances=output['cand_distances'][0],
                          image_indexes=output['cand_img_idxes'][0].tolist(), fingerprints=fingerprints,
                          vp_inputs={k: tensor_fingerprint(v) for k, v in inputs.items()},
                          panorama_transformer=tensor_fingerprint(embeddings),
                          panorama_mask=masks.cpu().tolist(), node_count=len(graph.node_pos),
                          ghost_count=len(graph.ghost_pos), no_vp_left=nav['no_vp_left'],
                          candidate_positions=np.asarray(positions).tolist(),
                          ghost_positions={k: np.asarray(v).tolist() for k, v in graph.ghost_aug_pos.items()},
                          contract='PASS')
            if provider:
                debug = provider.last_debug[0]
                record['admissibility_count'] = int((debug['traversability'].score > 0).sum())
                record['observation_availability'] = debug.get('observation_availability')
                assert all(debug['traversability'].score[a, r] > 0 for a, r in zip(angle_bins, radius_bins))
            if label == 'Original':
                golden = json.loads(Path('tests/fixtures/example_scene_waypoint_golden.json').read_text())
                assert output['cand_angles'][0] == golden['candidate_angles']
                assert output['cand_distances'][0] == golden['candidate_distances']
                assert output['cand_img_idxes'][0].tolist() == golden['candidate_img_idxes']
                compare_fingerprints(fingerprints, golden['fingerprints'], 1e-5)
                record['pre_refactor_golden_bytes'] = 'PASS'
            records[label] = record
            for key in keys:
                value = output[key][0] if key.startswith('cand_') else output[key]
                arrays[label + '_' + key] = value.cpu().numpy()
            for key, value in inputs.items():
                arrays[label + '_vp_' + key] = value.cpu().numpy()
            arrays[label + '_panorama_transformer'] = embeddings.cpu().numpy()
        # Compare BNMS against the exact dirty-tree provider captured before editing.
        before = Path('outputs/gmin_stage1_validation/before/vlnce_baselines/waypoint_provider/mid360_geometry.py')
        spec = importlib.util.spec_from_file_location('pre_gmin_geometry', str(before))
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        baseline = old.MID360GeometryWaypointProvider(config=old.MID360GeometryConfig.from_yaml('configs/mid360_waypoint.yaml'), **common).predict(obs)
        for key in keys:
            assert tensor_fingerprint(baseline[key]) == tensor_fingerprint(outputs['BNMS'][key])
        for key in ['cand_angles', 'cand_distances']:
            assert baseline[key] == outputs['BNMS'][key]
        np.testing.assert_array_equal(baseline['cand_img_idxes'][0], outputs['BNMS']['cand_img_idxes'][0])
        report['BNMS_pre_edit_same_state_bytes'] = 'PASS'
        for key in obs:
            assert torch.equal(obs[key], obs_before[key])
    report.update(status='PASS', providers=records, waypoint_predictor_resident=all(p.is_cuda for p in predictor.parameters()),
                  model_memory_released='NONE')
    np.savez_compressed(str(out / 'features.npz'), **arrays)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: {f: v[f] for f in ['candidate_count', 'angle_bins', 'radial_bins', 'image_indexes', 'ghost_count']}
                      for k, v in records.items()}, indent=2))
    print('SAVED-STATE AND ORIGINAL GOLDEN: PASS')


if __name__ == '__main__':
    main()
