#!/usr/bin/env python3
"""Trace frozen predictor blocks on captured depth-FC activations, same GPU.

No images, simulator, encoder, RGB replacement, or navigation. The actual
waypoint_TRM and vis_classifier modules consume saved visual_fc_depth outputs.
Exact agreement with the captured GPU logits is required for every pass.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/mid360_sparse_depth_forensics')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    import numpy as np
    import torch
    from analyze_sparse_depth_proposer_collapse import statistics, diversity, similarity, state_fingerprint
    from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
    from sparse_depth_run_manifest import AUXILIARY, EXPECTED, sha

    root = Path(args.output)
    destination = root / 'predictor_trace.json'
    assert not destination.exists()
    assert sha(AUXILIARY) == EXPECTED[AUXILIARY]
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    model = BinaryDistPredictor_TRM(device=device)
    model.load_state_dict(torch.load(AUXILIARY, map_location='cpu')['predictor']['state_dict'])
    model.to(device).eval()
    before = state_fingerprint(model)
    analysis = json.loads((root / 'analysis.json').read_text())
    report = dict(device=str(device), source='saved GPU predictor_depth_fc activations', observations={})
    for episode in analysis['episode_ids']:
        records, tensors = {}, {}
        with np.load(str(root / 'captured' / (episode + '.npz'))) as saved:
            for condition in ['DENSE', 'SPARSE', 'ALL_FAR']:
                fc = torch.from_numpy(saved[condition + '_predictor_depth_fc']).to(device).reshape(1, 12, -1)
                captured = {}
                def save(name):
                    def hook(module, inputs, output):
                        captured[name] = output.detach().clone()
                    return hook
                handles = [model.waypoint_TRM.register_forward_hook(save('transformer_output')),
                           model.vis_classifier[0].register_forward_hook(save('classifier_pre_relu')),
                           model.vis_classifier[1].register_forward_hook(save('classifier_post_relu')),
                           model.vis_classifier[2].register_forward_hook(save('classifier_output'))]
                try:
                    with torch.no_grad():
                        # Exact calls and tensor dimensions audited in BinaryDistPredictor_TRM.forward.
                        related = model.waypoint_TRM(fc, attention_mask=model.mask.repeat(1, 1, 1, 1))
                        head = model.vis_classifier(related).reshape(1, 120, 12)
                        offset = model.HEATMAP_OFFSET
                        logits = torch.cat((head[:, offset:], head[:, :offset]), dim=1)
                finally:
                    for handle in handles:
                        handle.remove()
                reference = torch.from_numpy(saved[condition + '_raw_logits']).to(device)
                difference = similarity(logits, reference)
                assert torch.equal(logits, reference), difference
                records[condition] = dict(blocks={}, replay_vs_captured_gpu_logits=difference)
                for name, value in captured.items():
                    records[condition]['blocks'][name] = dict(statistics=statistics(value), diversity=diversity(value),
                                                              nonzero_count=int(torch.count_nonzero(value)))
                bias = model.vis_classifier[2].bias.reshape(1, 1, -1).expand(1, 12, -1)
                records[condition]['classifier_equals_final_bias'] = bool(torch.equal(captured['classifier_output'], bias))
                tensors[condition] = captured
        comparisons = {}
        for a, b in [('DENSE', 'SPARSE'), ('SPARSE', 'ALL_FAR')]:
            comparisons[a + '_vs_' + b] = {name: similarity(tensors[a][name], tensors[b][name]) for name in tensors[a]}
        report['observations'][episode] = dict(conditions=records, comparisons=comparisons)
    assert before == state_fingerprint(model)
    report.update(parameters_and_buffers_unchanged=True, state_sha256=before,
                  same_predictor_object_all_conditions=True, predictor_object_id=id(model))
    with destination.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print('COMPLETE: frozen predictor block replay from saved activations, no Habitat calls')


if __name__ == '__main__':
    main()
