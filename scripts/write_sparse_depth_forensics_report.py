#!/usr/bin/env python3
"""Assemble the bounded forensic report from captured numerical evidence only."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def number(value):
    return '{:.9g}'.format(value)


def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(number(v) if isinstance(v, float) else str(v) for v in row) + ' |')
    return '\n'.join(lines) + '\n'


def write_report(root):
    target = root / 'report.md'
    assert not target.exists(), 'refusing to overwrite completed report'
    data = json.loads((root / 'analysis.json').read_text())
    trace = json.loads((root / 'predictor_trace.json').read_text())
    ids = data['episode_ids']
    obs = data['observations']
    shared = obs[ids[0]]['conditions']['SPARSE']['logits']
    # This report's interpretation requires these measured properties. Refuse
    # to carry a conclusion into a future capture with different observations.
    for eid in ids:
        record = obs[eid]
        assert record['formal_start_pose_matches'] and record['formal_sparse_candidate_match']
        assert record['comparisons']['SPARSE_vs_ALL_FAR']['raw_logits']['exactly_equal']
        for condition in ['SPARSE', 'ALL_FAR']:
            logits = record['conditions'][condition]['logits']
            assert logits == shared
            blocks = trace['observations'][eid]['conditions'][condition]
            assert blocks['classifier_equals_final_bias']
            assert blocks['blocks']['classifier_pre_relu']['statistics']['max'] < 0
            assert blocks['blocks']['classifier_post_relu']['nonzero_count'] == 0
            assert blocks['replay_vs_captured_gpu_logits']['exactly_equal']
        assert trace['observations'][eid]['conditions']['DENSE']['replay_vs_captured_gpu_logits']['exactly_equal']
    assert data['weights']['parameters_and_buffers_unchanged'] and data['formal_evidence_unchanged']
    assert data['navigation_steps'] == 0
    classification = dict(
        primary='PREDICTOR_COLLAPSE', first_complete_collapse='BinaryDistPredictor_TRM.vis_classifier[1] (ReLU)',
        supported_labels=['NO_RETURN_DOMINANCE', 'ENCODER_COLLAPSE', 'PREDICTOR_COLLAPSE'],
        encoder_collapse_scope='partial: most across-view variance lost, but features remain structured and distinct from All-Far',
        nms_collapse_supported=False,
        nms_role='deterministic tie selection among pre-existing bias-only radial-2 maxima; no diverse meaningful peaks destroyed',
        no_return_scope='input is closer to All-Far than Dense; logits exactly All-Far; encoder features only modestly closer, not equivalent',
        pre_nms_radial_marginal_dominant_index=shared['pre_nms_dominant_radial_index'],
        pre_nms_global_max_radial_index=shared['pre_nms_global_max_radial_index'],
        recommended_next_experiment='bounded no-return-aware input representation control, holding encoder, predictor and NMS frozen',
        implemented_next_experiment=False)
    data['classification'] = classification
    data['predictor_internal_trace'] = trace
    report = ['# MID360 sparse-depth proposer collapse forensics\n']
    add = report.append
    add('## A. Aim\n\nLocate the first layer at which sparse-depth proposal collapse appears. '
        'The first **complete** loss of observation-dependent output occurs inside the waypoint predictor: '
        '`vis_classifier[1]`, its ReLU, zeros all 9,216 classifier hidden activations for Sparse and All-Far. '
        'The final linear layer consequently emits only its learned bias. Earlier, the depth encoder already loses most view variance, '
        'but retains measurable scene and view structure. Original NMS selects among the resulting exact ties.\n\n'
        'Context supplied by the frozen formal evaluation: Original SR 0.80, sparse SR 0.00, 299 decisions, '
        '1,495 candidates all at 0.75 m, and 283/299 decisions with the recurring five-angle pattern. '
        'Those are existing results, not performance measurements from this study.\n')
    add('## B. Conditions\n\nFirst decisions only for episodes **53, 821, 1024, 250, 25**. '
        'Exactly five Habitat resets and 15 full proposer passes; **zero navigation steps**. '
        'A further 15 inexpensive predictor-block replays use the saved depth-FC activations, without Habitat. '
        'All replays reproduce captured GPU logits bit-for-bit. No later decisions were needed.\n\n'
        '- **DENSE:** original Habitat 12-view depth.\n'
        '- **SPARSE:** existing `MID360SparseDepthAdapter` using the accepted YAML, byte-identical protected source to `b402412`.\n'
        '- **ALL_FAR:** every depth pixel exactly 1.0. Installed `HabitatSimDepthSensor.get_observation` was exercised at '
        'MAX_DEPTH and returned exactly 1.0. Runtime MIN_DEPTH=0 m, MAX_DEPTH=10 m, NORMALIZE_DEPTH=True; '
        '`normalized=(metric-min)/(max-min)`. Thus 1.0 means policy maximum/far depth.\n\n'
        'Every condition uses the same non-depth tensor objects, including all RGB and instruction tensors. '
        'Packed encoder RGB hashes agree. The same frozen encoder and predictor objects serve every condition and episode. '
        'Raw reset depth hashes equal the proposer-source hashes; start scene, position and heading match formal telemetry. '
        'All five Sparse candidate lists match their formal first-decision lists.\n\n'
        'Exact Original path: `original_etp.encode_panorama` → depth encoder → waypoint predictor → '
        '`original_etp._candidate_map` → `waypoint_pred.utils.nms`. Hooks capture logits at predictor return, '
        'before the provider calls its candidate map. The saved raw GPU logits are also sent through the existing '
        '`_candidate_map`; its offline candidates exactly match the provider. No NMS reimplementation.\n\n'
        'Statistics use float64 reductions and population standard deviations. Softmax/NMS use Original float32 runtime operations. '
        'Normalized L2 means `||a-b|| / max(||a||, ||b||, 1e-12)`; it can exceed 1. '
        'View variance is the mean feature-coordinate population variance across 12 flattened view embeddings.\n')
    add('## C. Input statistics\n\nNormalized depth, before the encoder; all finite fractions are 1. '
        'Projected-return fractions come from the actual adapter hit counts, not an inferred non-fill mask. '
        'The ALL_FAR rows apply identically to every episode. Complete per-view min/max/mean/std/finite fractions and '
        'Sparse hit/no-return fractions are retained in `analysis.json`. No depth arrays are printed or saved.\n')
    rows = []
    for eid in ids:
        for name in ['DENSE', 'SPARSE']:
            v = obs[eid]['conditions'][name]['input']
            rows.append([eid, name, v['min'], v['max'], v['mean'], v['std'], v['finite_fraction'],
                         v['fraction_equal_1'], v.get('projected_return_fraction', '—'), v.get('no_return_fraction', '—')])
    rows.append(['all', 'ALL_FAR', 1., 1., 1., 0., 1., 1., '—', '—'])
    add(table(['Episode', 'Condition', 'Min', 'Max', 'Mean', 'Std', 'Finite', '=1 fraction', 'Projected', 'No-return'], rows))
    add('Per-view means, standard deviations and fill equality fractions (view headings are sensor headings, not predictor angle indices):\n')
    for eid in ids:
        add('### Episode ' + eid + '\n')
        rows = []
        for angle in range(0, 360, 30):
            key = 'depth' if angle == 0 else 'depth_' + str(angle)
            row = [angle]
            for name in ['DENSE', 'SPARSE', 'ALL_FAR']:
                v = obs[eid]['conditions'][name]['input']['per_view'][key]
                row += [v['mean'], v['std'], v['fraction_equal_1']]
            rows.append(row)
        add(table(['View °', 'D mean', 'D std', 'D =1', 'S mean', 'S std', 'S =1', 'F mean', 'F std', 'F =1'], rows))
    add('## D. Depth encoder comparison\n\nActual captured output is **[12, 128, 4, 4]**, directly verified against '
        'the predictor depth input. In this runtime it equals the visual encoder output; there are no appended fixed spatial '
        'embedding channels. Each view is compared as a 2,048-element vector. All features are finite.\n')
    rows = []
    for eid in ids:
        for name in ['DENSE', 'SPARSE', 'ALL_FAR']:
            d = obs[eid]['conditions'][name]['depth_encoder']
            s, v = d['statistics'], d['diversity']
            rows.append([eid, name, s['mean'], s['std'], s['l2_norm'], s['finite_fraction'],
                         v['mean_pairwise_cosine'], v['min_pairwise_cosine'], v['max_pairwise_cosine'], v['feature_variance_across_views']])
    add(table(['Episode', 'Condition', 'Mean', 'Std', 'L2 norm', 'Finite', 'Pair cos mean', 'Pair cos min', 'Pair cos max', 'View variance'], rows))
    rows = []
    for eid in ids:
        r = obs[eid]
        ds, sf = [r['comparisons'][p]['depth_embedding'] for p in ['DENSE_vs_SPARSE', 'SPARSE_vs_ALL_FAR']]
        ratio = r['conditions']['SPARSE']['depth_encoder']['diversity']['feature_variance_across_views'] / r['conditions']['DENSE']['depth_encoder']['diversity']['feature_variance_across_views']
        rows.append([eid, ds['cosine'], ds['normalized_l2_difference'], sf['cosine'], sf['normalized_l2_difference'], ratio])
    add(table(['Episode', 'D↔S cosine', 'D↔S norm L2', 'S↔F cosine', 'S↔F norm L2', 'Sparse/Dense view variance'], rows))
    add('Sparse view variance falls by **71.4–80.3%**. Mean pairwise cosine rises from 0.187–0.236 '
        'to 0.762–0.831. This supports **partial encoder diversity collapse**, not twelve indistinguishable embeddings. '
        'Sparse↔All-Far encoder cosine is only 0.377–0.383, with normalized L2 around 1.10: '
        'the encoder outputs are not equivalent to All-Far.\n')
    add('## E. Raw waypoint-logit comparison\n\nRaw maps have shape **[1,120,12]** before softmax and NMS. '
        'Sparse and All-Far maps are bit-identical across all five observations; one shared row describes all ten maps.\n')
    rows = []
    for eid, name, v in [(e, 'DENSE', obs[e]['conditions']['DENSE']['logits']) for e in ids] + [('all', 'SPARSE / ALL_FAR', shared)]:
        rows.append([eid, name, v['min'], v['max'], v['mean'], v['std'], v['entropy_nats'], v['entropy_fraction_of_uniform']])
    add(table(['Episode', 'Condition', 'Min', 'Max', 'Mean', 'Std', 'Entropy (nats)', 'Entropy / ln(1440)'], rows))
    rows = []
    for eid in ids:
        ds, sf = [obs[eid]['comparisons'][p]['raw_logits'] for p in ['DENSE_vs_SPARSE', 'SPARSE_vs_ALL_FAR']]
        rows.append([eid, ds['cosine'], ds['normalized_l2_difference'], sf['cosine'], sf['normalized_l2_difference'], sf['max_absolute_difference']])
    add(table(['Episode', 'D↔S cosine', 'D↔S norm L2', 'S↔F cosine', 'S↔F norm L2', 'S↔F max absolute difference'], rows))
    add('Uniform entropy is ln(1440) = ' + number(math.log(1440)) + ' nats. '
        'Sparse/All-Far entropy is nearly uniform, yet the small nonzero logit differences still determine argmax. '
        'All 120 angular marginals and top 20 distinct angular maxima per condition are in `analysis.json`. '
        'The shared angular marginal repeats this ten-bin vector twelve times (indices 0–9, then 10–19, etc.):\n\n'
        '`[' + ', '.join(number(x) for x in shared['angular_marginal'][:10]) + ']`\n')
    add('### Predictor internal boundary\n\nA separate GPU replay fed saved `visual_fc_depth` activations through '
        'the exact frozen `waypoint_TRM` and `vis_classifier` modules. It reproduced all 15 captured logits exactly. '
        'The Original predictor forward uses depth features; RGB is accepted but its RGB feature branch is commented out. '
        'The block replay therefore needed no RGB input or replacement. Parameters and buffers remained unchanged.\n')
    rows = []
    for eid in ids:
        for name in ['DENSE', 'SPARSE', 'ALL_FAR']:
            t = trace['observations'][eid]['conditions'][name]
            b = t['blocks']
            fc = obs[eid]['conditions'][name]['predictor_depth_fc']['diversity']
            rows.append([eid, name, fc['mean_pairwise_cosine'], b['transformer_output']['diversity']['feature_variance_across_views'],
                         b['classifier_pre_relu']['statistics']['max'], b['classifier_post_relu']['nonzero_count'], t['classifier_equals_final_bias']])
    add(table(['Episode', 'Condition', 'Depth FC pair cos', 'Transformer view variance', 'Max pre-ReLU', 'Nonzero / 9216 after ReLU', 'Head equals bias'], rows))
    add('Sparse FC activations still differ across views. Transformer variance remains nonzero (0.000237–0.000476). '
        'All classifier pre-ReLU values are negative: even the largest is between −0.198 and −0.0140 for Sparse. '
        '**The classifier ReLU is the first measured exact erasure of all remaining structure.** '
        'Its 9,216 outputs are zero, so the final linear output equals its 120-element bias repeated for 12 views, '
        'followed by the unchanged five-angle-bin offset. Dense retains 229–531 nonzero hidden activations. '
        'The bias-only map explains the observed 30° periodicity without invoking NMS as its origin.\n')
    add('### Top 20 raw cells\n\nAngle indices are clockwise, 3° per bin; reported policy angles are counterclockwise '
        '`(360−3*index) mod 360`. Distances use `(distance_index+1)*0.25 m`. Stable angle-major ordering '
        'is used only to display exact ties; it does not change Original NMS.\n')
    for label, logits in [('Dense episode ' + e, obs[e]['conditions']['DENSE']['logits']) for e in ids] + [('Sparse and All-Far — shared by all five episodes', shared)]:
        add('#### ' + label + '\n')
        add(table(['Rank', 'Angle index', 'Distance index', 'CW °', 'Policy CCW °', 'Distance m', 'Raw logit', 'Probability'],
                  [[i + 1, c['angle_index'], c['distance_index'], c['angle_degrees_clockwise'], c['angle_degrees_counterclockwise'],
                    c['distance_meters'], c['raw_logit'], c['softmax_probability']] for i, c in enumerate(logits['top20_cells'])]))
    add('## F. Pre-NMS vs Post-NMS\n\n### Radial marginal probability\n\nColumns are all 12 radial distances in metres.\n')
    add(table(['Observation'] + [number((i + 1) * .25) for i in range(12)],
              [[e + ' DENSE'] + obs[e]['conditions']['DENSE']['logits']['radial_marginal'] for e in ids] +
              [['all SPARSE / ALL_FAR'] + shared['radial_marginal']]))
    add('**0.75 m is not the dominant radial marginal before NMS.** Its mass is '
        + number(shared['radial_marginal'][2]) + '; the maximum marginal is 1.25 m (index 4), mass '
        + number(shared['radial_marginal'][4]) + '. Nevertheless, **all twelve global maximum cells already '
        'lie at 0.75 m before NMS**. Each has probability ' + number(shared['top20_cells'][0]['softmax_probability']) + '. '
        'Thus radial-2 dominance exists in the cell maxima before NMS, while 100% radial-2 concentration in '
        'the selected candidate list occurs after NMS. A marginal alone would miss this distinction.\n\n'
        '### Angular ties\n\nSparse/All-Far maxima occur at angle indices **5,15,25,…,115**: CW '
        '15°,45°,…,345°, equivalently all twelve CCW directions spaced by 30° with a 15° offset. '
        'They are already equal raw-logit peaks before NMS. The five formal angles are a subset of those peaks.\n')
    rows = []
    for label, l in [(e + ' DENSE', obs[e]['conditions']['DENSE']['logits']) for e in ids] + [('all SPARSE / ALL_FAR', shared)]:
        rows.append([label, l['max_minus_second'], l['max_minus_median'], l['periodic_radial2_peak_std'],
                     l['cells_exactly_at_max'], l['cells_within_1e_6_of_max'], l['repeat_30_degree_max_abs_difference']])
    add(table(['Observation', 'Max−second', 'Max−median', 'Std of 12 periodic radial-2 cells', 'Exact max ties', 'Within 1e−6', '30° repeat max difference'], rows))
    add('For Sparse/All-Far, all twelve maxima are **exactly** tied (peak range 0, peak std 0, max−second 0), '
        'not merely close within a reporting tolerance. The Original NMS loop calls `torch.max` on the flattened '
        'suppressed map; the observed selection follows the earliest tied surviving cells.\n\n'
        '### Exact Original post-NMS candidates\n\nEach entry is `(angle index, policy CCW degrees, distance metres)`, '
        'in provider order. Sparse/All-Far share the same list for every episode.\n')
    rows = []
    for label, l in [(e + ' DENSE', obs[e]['conditions']['DENSE']['logits']) for e in ids] + [('all SPARSE', shared), ('all ALL_FAR', shared)]:
        rows.append([label, '; '.join('({}, {}°, {} m)'.format(c['angle_index'], number(c['angle_degrees_counterclockwise']),
                                                             number(c['distance_meters'])) for c in l['post_nms_candidates'])])
    add(table(['Observation', 'Selected candidates'], rows))
    add('Some Dense outputs contain four candidates after the Original wrapped-map NMS and crop. '
        'Those are the exact provider results, not padded or forced to five.\n')
    add('## G. All-Far control\n\n**Yes: All-Far exactly reproduces Sparse behavior in all five observations.** '
        'Every raw logit is identical (cosine 1, normalized L2 0, max absolute difference 0). '
        'Both yield five candidates at 0.75 m and policy CCW angles 345°,315°,285°,255°,225° '
        '(the same set as 225°,255°,285°,315°,345°). The input and encoder tensors are not identical; '
        'the exact equivalence emerges inside the predictor classifier.\n')
    add('## H. Failure classification\n\n'
        '- **PREDICTOR_COLLAPSE — primary, directly established.** Structured sparse features enter the predictor; '
        'its classifier ReLU zeros all hidden activations, producing fixed bias-only logits.\n'
        '- **ENCODER_COLLAPSE — partial diversity loss.** Sparse encoder view variance loses 71.4–80.3% relative to Dense. '
        'The remaining features are not identical across views and are not very close to All-Far (cosine ≈0.38).\n'
        '- **NO_RETURN_DOMINANCE — supported at the input/output level, with qualification.** '
        '55.7–56.2% of sparse pixels are no-return fill. Sparse input is closer to All-Far than Dense '
        '(S↔F cosine 0.812–0.839 vs D↔S 0.649–0.711; normalized L2 0.550–0.586 vs 0.821–0.879). '
        'Encoder similarity is only modestly closer to All-Far, while predictor logits become exactly All-Far. '
        'The control demonstrates behavioral equivalence, not that fill alone has been isolated from sparsity.\n'
        '- **NMS_COLLAPSE — not supported as the source under the requested definition.** '
        'There are no meaningful observation-dependent diverse sparse peaks left for NMS to destroy. '
        'NMS resolves existing exact bias ties into a fixed subset and makes the selected list exclusively radial-2.\n\n'
        'Evidence-supported chain: no-return-heavy sparse representation → partial encoder diversity loss → '
        'stronger predictor homogenization → complete classifier ReLU collapse → bias-only periodic logits → '
        'deterministic Original NMS tie subset. The complete collapse first appears in **C, the waypoint predictor**, '
        'with earlier degradation in **B**, rather than first appearing in **D**.\n')
    add('## I. Recommended NEXT experiment\n\nChoose a **bounded no-return-aware input representation control**: '
        'separate missing-ray information from measured far surfaces, while keeping the frozen encoder, predictor and '
        'Original NMS fixed initially. Evaluate recovery of classifier activations, view diversity and non-bias logits '
        'on these same observations before any navigation benchmark. This is motivated by the All-Far equivalence and '
        'the domain shift visible before the classifier. It is a proposed causal test, not a proven remedy. '
        'The present evidence does not yet justify choosing encoder or predictor fine-tuning over a representation intervention; '
        'an NMS-only change would merely choose different peaks of a fixed bias map. **No next experiment was implemented.**\n')
    add('## J. Limitations\n\nFive first-decision observations are a small bounded sample; this study does not establish '
        'navigation performance, explain every one of the 299 formal decisions, or characterize the 16 noncanonical angular cases. '
        'No later episode-53 observations were collected. The adapter combines missingness, rasterization and spatial sampling; '
        'the All-Far control does not isolate the causal contribution of each. Partial encoder collapse denotes measured '
        'variance loss, not complete information loss or a proof of what geometry can still be decoded. The first exact '
        'collapse is located among the measured blocks; individual encoder convolutional layers were not probed. '
        'The auxiliary block replay used the same GPU because an initial CPU replay differed numerically from GPU logits '
        '(maximum observed absolute error 0.000521); only the bit-identical GPU replay supports the sublayer conclusion.\n\n'
        'No full evaluation, navigation parameter changes, training, fine-tuning, geometry proposals, GraphMap changes, '
        'planner changes, checkpoint writes, or formal-evidence writes occurred. Whole-policy and predictor parameter/buffer '
        'hashes agree before and after. All formal evidence file hashes agree.\n\n'
        'Artifacts: `analysis.json` contains full requested statistics, marginals, top cells, guards and internal trace; '
        '`captured/*.npz` contains only embeddings and logits; `captured/*.json` contains per-observation scalar evidence; '
        '`predictor_trace.json`, `runtime_config.yaml` and `run_manifest.json` record the frozen execution.\n\n'
        'Reproduction (only into a new output directory): source `scripts/renderer_bootstrap.bash`, call '
        '`etp_apply_known_good_renderer_environment`, then use `etpnav_legacy` Python to run '
        '`scripts/capture_sparse_depth_forensics.py --output NEW_DIRECTORY`, '
        '`scripts/trace_saved_sparse_predictor.py --output NEW_DIRECTORY`, and '
        '`scripts/write_sparse_depth_forensics_report.py --output NEW_DIRECTORY`. '
        'The capture entry point never calls `eval`, `rollout`, or `envs.step`.\n')
    data['artifact_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'analysis.json'}
    with target.open('x') as handle:
        handle.write('\n'.join(report))
    # This is our newly created analysis file, never a formal result file.
    (root / 'analysis.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/mid360_sparse_depth_forensics')
    write_report(Path(parser.parse_args().output))
