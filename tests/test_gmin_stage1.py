"""Controlled G-Min coverage, observation and downstream regression gates."""
import ast
import inspect
import itertools
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from vlnce_baselines.perception.gmin_candidates import circular_sectors, select_gmin_candidates
from vlnce_baselines.perception.local_traversability import compute_local_traversability
from vlnce_baselines.waypoint_provider.mid360_geometry import (
    MID360GeometryConfig, MID360GeometryWaypointProvider, gmin_observation_availability,
)
from vlnce_baselines.waypoint_provider.original_etp import (
    OriginalETPWaypointProvider, construct_waypoint_output, encode_panorama,
)
from vlnce_baselines.waypoint_provider.types import validate_waypoint_output
from vlnce_baselines.models.graph_utils import GraphMap
from vlnce_baselines.ss_trainer_ETP import RLTrainer


def observations(value=1.0):
    result = OrderedDict()
    for angle in range(0, 360, 30):
        suffix = '' if angle == 0 else '_{}'.format(angle)
        result['rgb' + suffix] = torch.full((1, 8, 8, 3), angle // 30 + 1, dtype=torch.uint8)
        result['depth' + suffix] = torch.full((1, 8, 8, 1), value)
    return result


class RGB(nn.Module):
    def forward(self, obs):
        return obs['rgb'][:, 0, 0, 0].float().view(-1, 1, 1, 1).expand(-1, 512, 1, 1)


class Depth(nn.Module):
    def forward(self, obs):
        return obs['depth'][:, 0, 0, 0].view(-1, 1, 1, 1).expand(-1, 128, 4, 4)


class Predictor(nn.Module):
    def forward(self, rgb, depth):
        logits = torch.arange(1440).float().reshape(1, 120, 12) / 100
        return logits


def shared():
    return dict(rgb_encoder=RGB(), depth_encoder=Depth(),
                rgb_pool=nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(2)),
                depth_pool=nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(2)),
                pano_angle_features=torch.zeros(12, 4),
                pano_image_indexes=np.arange(12, dtype=np.int64))


def config():
    return MID360GeometryConfig.from_yaml('configs/mid360_waypoint_gmin.yaml')


def cells(mask, budget=5):
    a, r = select_gmin_candidates(mask, budget)
    return list(zip(a.tolist(), r.tolist()))


class GMinExtractionTest(unittest.TestCase):
    def test_open_space_exact_order_and_radial_diversity(self):
        mask = np.ones((120, 12), dtype=bool)
        self.assertEqual(cells(mask), [(0, 11), (15, 0), (30, 5), (60, 8), (90, 2)])
        self.assertEqual(cells(mask, 1), [(0, 11)])
        self.assertEqual(len(set(r for a, r in cells(mask))), 5)

    def test_angular_circularity_and_sector_interior_tie(self):
        mask = np.zeros((120, 12), dtype=bool)
        mask[[118, 119, 0, 1]] = True
        labels = circular_sectors(mask[:, 0])
        self.assertEqual(labels[119], labels[0])
        self.assertEqual(cells(mask, 1), [(0, 11)])
        self.assertEqual(len(set(labels[mask[:, 0]])), 1)

    def test_unrepresented_sector_precedes_separation(self):
        mask = np.zeros((120, 12), dtype=bool)
        mask[0:101] = True
        mask[110] = True
        result = cells(mask, 2)
        self.assertEqual(result, [(50, 11), (110, 0)])

    def test_radial_boundary_never_suppresses(self):
        mask = np.zeros((120, 12), dtype=bool)
        mask[0] = True
        self.assertEqual(cells(mask, 2), [(0, 0), (0, 11)])

    def test_prefix_validation_shape_dtype_and_budget(self):
        for bad in [np.ones((12, 120), bool), np.ones((120, 12), float)]:
            with self.assertRaises(ValueError):
                select_gmin_candidates(bad)
        mask = np.zeros((120, 12), bool)
        mask[0, 1] = True
        with self.assertRaisesRegex(ValueError, 'radial-prefix'):
            select_gmin_candidates(mask)
        for budget in [0, 6, True, 2.5]:
            with self.assertRaises(ValueError):
                select_gmin_candidates(np.ones((120, 12), bool), budget)

    def test_empty_typed(self):
        for array in select_gmin_candidates(np.zeros((120, 12), bool)):
            self.assertEqual(array.dtype, np.int64)
            self.assertEqual(array.shape, (0,))

    def test_repeated_radii_deduplicate_and_skip(self):
        mask = np.zeros((120, 12), bool)
        mask[10, 0] = True
        self.assertEqual(cells(mask), [(10, 0)])
        mask[:, 0] = True
        self.assertEqual(cells(mask), [(0, 0), (15, 0), (30, 0), (60, 0), (90, 0)])

    def test_random_prefix_masks_repeatability_admissibility_maximum(self):
        rng = np.random.RandomState(17)
        for _ in range(35):
            mask = np.arange(12)[None, :] < rng.randint(0, 13, (120, 1))
            for budget in range(1, 6):
                before = mask.copy()
                result = cells(mask, budget)
                self.assertEqual(result, cells(mask.copy(), budget))
                self.assertEqual(result, sorted(set(result)))
                self.assertLessEqual(len(result), budget)
                self.assertTrue(all(mask[a, r] for a, r in result))
                np.testing.assert_array_equal(mask, before)

    def test_no_oracle_or_v1_dependency(self):
        import vlnce_baselines.perception.gmin_candidates as helper
        tree = ast.parse(inspect.getsource(helper))
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertEqual(imports, ['numpy'])
        self.assertEqual(list(inspect.signature(select_gmin_candidates).parameters),
                         ['admissibility', 'max_candidates'])

    def test_blocked_front_corridor_and_resolved_opening(self):
        cfg = config().traversability()
        front = np.array([[x, 1.0, -1.0] for x in np.arange(-3, 3.01, .05)])
        corridor = np.array([[x, 1.0, z] for x in [-.45, .45]
                             for z in np.arange(-3, 3.01, .05)])
        opening = front[np.abs(front[:, 0]) > .35]
        for cloud in [front, corridor, opening]:
            mask = compute_local_traversability(cloud, cfg).score > 0
            result = cells(mask)
            self.assertTrue(result)
            self.assertTrue(all(mask[a, r] for a, r in result))
        front_mask = compute_local_traversability(front, cfg).score > 0
        self.assertFalse(front_mask[0, 3])
        corridor_result = cells(compute_local_traversability(corridor, cfg).score > 0)
        self.assertTrue(all(abs(np.sin(a * np.pi / 60) * (r + 1) * .25) < .45
                            for a, r in corridor_result))
        opening_mask = compute_local_traversability(opening, cfg).score > 0
        self.assertTrue(opening_mask[0, 11])
        self.assertTrue(any(a == 0 and r >= 4 for a, r in cells(opening_mask)))


class GMinProviderTest(unittest.TestCase):
    def test_distant_valid_cloud_and_contract(self):
        provider = MID360GeometryWaypointProvider(config=config(), **shared())
        output = provider.predict(observations())
        validate_waypoint_output(output)
        self.assertEqual(provider.last_debug[0]['filtered_pointcloud'].shape, (0, 3))
        self.assertEqual(len(output['cand_angles'][0]), 5)
        for i, image in enumerate(output['cand_img_idxes'][0]):
            self.assertTrue(torch.equal(output['cand_rgb'][0][i], output['pano_rgb'][0][image]))
            self.assertTrue(torch.equal(output['cand_depth'][0][i], output['pano_depth'][0][image]))
        self.assertEqual(output['cand_distances'][0], [3., .25, 1.5, 2.25, .75])
        self.assertEqual(output['cand_img_idxes'][0].tolist(), [0, 10, 9, 6, 3])

    def test_no_sampled_evidence_returns_empty_with_reason(self):
        provider = MID360GeometryWaypointProvider(config=config(), **shared())
        obs = observations(0.)
        for key in obs:
            if 'depth' in key:
                obs[key][:, 1, 1] = 1.0  # unsampled pixels cannot open the gate
        output = provider.predict(obs)
        self.assertEqual(output['cand_distances'][0], [])
        validate_waypoint_output(output)
        self.assertEqual(provider.last_debug[0]['observation_availability']['reason'],
                         'no_sampled_depth_above_minimum_range')

    def test_invalid_depth_fails_before_encoder(self):
        invalid = []
        missing = observations(); del missing['depth_330']; invalid.append(missing)
        malformed = observations(); malformed['depth_30'] = torch.zeros(1, 8, 8); invalid.append(malformed)
        for value in [float('nan'), float('inf'), -.001, 1.001]:
            obs = observations(); obs['depth_60'][0, 1, 1, 0] = value; invalid.append(obs)
        for obs in invalid:
            provider = MID360GeometryWaypointProvider(config=config(), **shared())
            with patch('vlnce_baselines.waypoint_provider.mid360_geometry.encode_panorama',
                       side_effect=AssertionError('encoder must not run')):
                with self.assertRaises(ValueError):
                    provider.predict(obs)

    def test_panorama_noninterference_original_isolation_and_repeat(self):
        obs = observations()
        original = OriginalETPWaypointProvider(waypoint_predictor=Predictor(), **shared())
        with patch('vlnce_baselines.perception.gmin_candidates.select_gmin_candidates',
                   side_effect=AssertionError('original invoked G-Min')):
            expected = original.predict(obs)
            repeated = original.predict(obs)
        for key in expected:
            if key.startswith('cand_') and torch.is_tensor(expected[key][0]):
                self.assertTrue(torch.equal(expected[key][0], repeated[key][0]))
        provider = MID360GeometryWaypointProvider(config=config(), **shared())
        actual = provider.predict(obs)
        for key in ['pano_rgb', 'pano_depth', 'pano_angle_fts']:
            self.assertTrue(torch.equal(expected[key], actual[key]))
        np.testing.assert_array_equal(expected['pano_img_idxes'], actual['pano_img_idxes'])

    def test_config_preserves_physical_provenance_and_forbids_confounders(self):
        cfg = config()
        baseline = MID360GeometryConfig.from_yaml('configs/mid360_waypoint.yaml')
        self.assertEqual({k: v for k, v in cfg.values.items() if k != 'candidate_extraction'}, baseline.values)
        self.assertEqual(cfg.robot_radius_status, 'HABITAT_DEFAULT_AGENT_TEST_ONLY')
        for key, value in [('radial_utility', 'mru'), ('radial_utility', 'v0_linear'),
                           ('maximum_candidate_distance_m', 3.), ('mode', 'v1'),
                           ('maximum_candidates', 6)]:
            values = dict(cfg.values); values[key] = value
            with self.assertRaises(ValueError):
                MID360GeometryConfig(values)


class GMinDownstreamTest(unittest.TestCase):
    def test_graph_snapshot_before_first_update_and_after_consumption(self):
        tree = ast.parse(Path('vlnce_baselines/ss_trainer_ETP.py').read_text())
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'graph_before' for t in n.targets))
        graph = GraphMap(False, .5, True, 0.)
        env = dict(self=SimpleNamespace(gmaps=[graph]), i=0, np=np, int=int)
        self.assertEqual(eval(compile(ast.Expression(assignment.value), '', 'eval'), env),
                         {'nodes': {}, 'ghosts': {}, 'ghost_cnt': 0})
        pos, ori = np.zeros(3), [0, 0, 0, 1]
        vp, ids, positions = graph.identify_node(pos, ori, [0.], [2.])
        graph.update_graph(None, 1, vp, pos, torch.ones(8), ids, positions, torch.ones(1, 8), None)
        ghost = next(iter(graph.ghost_pos))
        graph.delete_ghost(ghost)
        snapshot = eval(compile(ast.Expression(assignment.value), '', 'eval'), env)
        self.assertEqual(snapshot['ghosts'], {})
        self.assertEqual(snapshot['ghost_cnt'], 1)

    def test_duplicate_image_indexes_keep_candidate_tokens(self):
        args = shared()
        encoded = encode_panorama(observations(), **{k: args[k] for k in
                                  ['rgb_encoder', 'depth_encoder', 'rgb_pool', 'depth_pool']})
        output = construct_waypoint_output(encoded, [torch.tensor([0, 1, 2])],
                                          [torch.tensor([0, 5, 11])],
                                          args['pano_angle_features'], args['pano_image_indexes'])
        trainer = SimpleNamespace(envs=SimpleNamespace(num_envs=1))
        with patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self):
            inputs = RLTrainer._vp_feature_variable(trainer, output)
        self.assertEqual(inputs['view_lens'].tolist(), [14])
        self.assertEqual(inputs['nav_types'][0, :3].tolist(), [1, 1, 1])
        self.assertTrue(torch.equal(inputs['rgb_fts'][0, :3], output['cand_rgb'][0]))
        self.assertTrue(torch.equal(inputs['loc_fts'][0, :3], output['cand_angle_fts'][0]))

    def test_zero_candidates_historical_ghosts_and_no_ghost_stop_input(self):
        for historical in [False, True]:
            graph = GraphMap(False, .5, True, 0.)
            pos, ori, embed = np.zeros(3), [0, 0, 0, 1], torch.ones(8)
            angles, distances = ([0.], [2.]) if historical else ([], [])
            vp, ids, positions = graph.identify_node(pos, ori, angles, distances)
            graph.update_graph(None, 1, vp, pos, embed, ids, positions,
                               torch.ones(len(ids), 8), None)
            prev = vp
            ghosts = list(graph.ghost_pos)
            vp, ids, positions = graph.identify_node(pos, ori, [], [])
            graph.update_graph(prev, 2, vp, pos, embed, ids, positions, torch.empty(0, 8), None)
            trainer = SimpleNamespace(envs=SimpleNamespace(num_envs=1), gmaps=[graph])
            with patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self):
                inputs = RLTrainer._nav_gmap_variable(trainer, [vp], [pos], [ori])
            self.assertEqual(inputs['no_vp_left'], [not historical])
            self.assertEqual(list(graph.ghost_pos), ghosts)
            self.assertEqual(inputs['gmap_vp_ids'][0][0], None)
            self.assertTrue(all(g in inputs['gmap_vp_ids'][0] for g in ghosts))

    def test_stop_telemetry_preserves_all_conditions(self):
        tree = ast.parse(Path('vlnce_baselines/ss_trainer_ETP.py').read_text())
        record = next(n for n in ast.walk(tree) if isinstance(n, ast.Dict)
                      and any(isinstance(k, ast.Str) and k.s == 'planner_stop' for k in n.keys))
        expressions = {k.s: v for k, v in zip(record.keys, record.values) if isinstance(k, ast.Str)}
        # A forced stop must still record the planner's nonzero ghost choice.
        env = dict(planner_selected_vp='g7', planner_ghost_position=np.array([1., 0., -2.]),
                   np=np, str=str)
        self.assertEqual(eval(compile(ast.Expression(expressions['selected_ghost_id']), '', 'eval'), env), 'g7')
        self.assertEqual(eval(compile(ast.Expression(expressions['selected_ghost_position']), '', 'eval'), env), [1., 0., -2.])
        for semantic, limit, empty in itertools.product([False, True], repeat=3):
            env = dict(action_index=0 if semantic else 1, stepk=9 if limit else 0,
                       self=SimpleNamespace(max_len=10), no_vp_left=[empty], i=0, bool=bool)
            for name, expected in [('planner_stop', semantic), ('step_limit', limit), ('no_ghosts', empty)]:
                self.assertEqual(eval(compile(ast.Expression(expressions[name]), '', 'eval'), env), expected)
        # Existing execution stop predicate is still the same three-way OR.
        predicates = [n.test for n in ast.walk(tree) if isinstance(n, ast.If)
                      and isinstance(n.test, ast.BoolOp) and 'cpu_a_t' in ast.dump(n.test)]
        self.assertEqual(len(predicates), 1)
        predicate = predicates[0]
        for semantic, limit, empty in itertools.product([False, True], repeat=3):
            env = dict(cpu_a_t=[0 if semantic else 1], stepk=9 if limit else 0,
                       self=SimpleNamespace(max_len=10), no_vp_left=[empty], i=0)
            self.assertEqual(eval(compile(ast.Expression(predicate), '', 'eval'), env), semantic or limit or empty)

    def test_executor_failure_propagates_before_success_accounting(self):
        tree = ast.parse(Path('vlnce_baselines/ss_trainer_ETP.py').read_text())
        parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'step'
                    and ast.dump(n.func.value) == ast.dump(ast.parse('self.envs', mode='eval').body))
        ancestor = call
        while ancestor in parent:
            ancestor = parent[ancestor]
            self.assertNotIsInstance(ancestor, ast.Try)
        class FailedExecutor:
            def step(self, actions):
                raise RuntimeError('executor failure')
        with self.assertRaisesRegex(RuntimeError, 'executor failure'):
            eval(compile(ast.Expression(call), '', 'eval'),
                 dict(self=SimpleNamespace(envs=FailedExecutor()), env_actions=[]))


if __name__ == '__main__':
    unittest.main()
