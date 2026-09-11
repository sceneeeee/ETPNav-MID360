import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import review_depth_only_alignment as review


def report_fixture():
    stats = dict(min=-2., max=1., mean=-.5, std=.2, finite_fraction=1.)
    candidate = dict(angle_index=5, distance_index=2, distance_meters=.75,
                     angle_degrees_counterclockwise=345.)
    condition = dict(visual_fc_depth_post_relu=dict(positive_fraction=.235),
                     classifier_post_relu=dict(positive_fraction=0.), candidate_count=1,
                     visual_fc_depth_pre_relu=dict(statistics=stats), classifier_pre_relu=dict(statistics=stats),
                     logits=dict(entropy_nats=7.2, post_nms_candidates=[candidate],
                                 post_nms_radial_counts=[0, 0, 1] + [0] * 9))
    return dict(observations={eid: dict(conditions={name: copy.deepcopy(condition) for name in review.CONDITIONS})
                              for eid in review.IDS})


class DepthOnlyReviewTest(unittest.TestCase):
    def test_report_has_twenty_rows_units_and_pre_relu_stats(self):
        report = report_fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            review.write_reports(output, report)
            rows = json.loads((output / 'compact_results.json').read_text())
            self.assertEqual(len(rows), 20)
            self.assertEqual(rows[0]['candidate_angles'], [345.])
            self.assertEqual(rows[0]['candidate_radii'], [.75])
            self.assertEqual(rows[0]['classifier_pre_relu_min'], -2.)
            self.assertIn('0/9216', (output / 'static_report.md').read_text())
            self.assertEqual(review.read(output / 'aggregate.json')['gate_counts'], {'C': 5})

    def test_activity_and_diversity_are_separate_gates_and_mixed_is_not_hidden(self):
        report = report_fixture()
        first = report['observations']['53']['conditions']['SPARSE_ALIGNED']
        first['classifier_post_relu']['positive_fraction'] = .1
        second = report['observations']['821']['conditions']['SPARSE_ALIGNED']
        second['classifier_post_relu']['positive_fraction'] = .2
        second['candidate_count'] = 2
        second['logits']['post_nms_candidates'] += [dict(angle_index=20, distance_index=4,
            distance_meters=1.25, angle_degrees_counterclockwise=300.)]
        second['logits']['post_nms_radial_counts'][4] = 1
        aggregate = review.aggregate_rows(review.compact_rows(report))
        self.assertEqual(aggregate['gate_counts'], {'A': 1, 'B': 1, 'C': 3})
        self.assertTrue(aggregate['overall_gate'].startswith('MIXED'))
        self.assertAlmostEqual(aggregate['conditions']['SPARSE_ALIGNED']['classifier_post_relu_positive_fraction']['mean'], .06)
        self.assertEqual(aggregate['conditions']['SPARSE_ALIGNED']['candidate_radial_bin_counts'][4], 1)

    def test_historical_sparse_all_far_bit_difference_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'captured').mkdir()
            (root / 'analysis.json').write_text((review.FORENSICS / 'analysis.json').read_text())
            with np.load(str(review.FORENSICS / 'captured/53.npz'), allow_pickle=False) as saved:
                values = {k: saved[k].copy() for k in saved.files}
            values['SPARSE_raw_logits'].flat[0] += .001
            np.savez(str(root / 'captured/53.npz'), **values)
            with patch.object(review, 'FORENSICS', root):
                with self.assertRaisesRegex(ValueError, 'historical logits differ'):
                    review.validate_history()

    def test_regression_failure_produces_no_alignment_interpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, 'argv', ['review', 'report', '--output', directory]), \
                    patch.object(review, 'validate_calibration', return_value={}), \
                    patch.object(review, 'validate_evaluation', side_effect=ValueError('HISTORICAL REGRESSION')):
                with self.assertRaisesRegex(ValueError, 'HISTORICAL REGRESSION'):
                    review.main()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_static_entry_has_no_navigation_training_or_rollout_calls(self):
        source = ast.parse((ROOT / 'scripts/run_sparse_depth_feature_alignment.py').read_text())
        names = {'evaluate', 'capture', 'reset_batch', 'runtime', 'model_hashes'}
        functions = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in functions}, names)
        forbidden = {'step', 'backward', 'train', 'rollout', '_eval_checkpoint', 'GraphMap', '_train_interval'}
        for function in functions:
            for node in ast.walk(function):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, 'attr', getattr(node.func, 'id', ''))
                    self.assertNotIn(name, forbidden, function.name)
            if function.name in ('evaluate', 'capture'):
                self.assertTrue(any(isinstance(d, ast.Call) and getattr(d.func, 'attr', '') == 'no_grad'
                                    for d in function.decorator_list))


if __name__ == '__main__':
    unittest.main()
