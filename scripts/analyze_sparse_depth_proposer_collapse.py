#!/usr/bin/env python3
"""Read-only, same-observation proposer probes; no evaluation or navigation loop.

Statistics use float64 population standard deviation. Inference and NMS retain
the frozen runtime dtype/device. Full depth images are never serialized.
"""
import hashlib
import math

import numpy as np
import torch

from vlnce_baselines.perception.mid360_sparse_depth import DEPTH_KEYS
from vlnce_baselines.waypoint_provider import original_etp


def fingerprint(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def state_fingerprint(module):
    """Include all parameters and buffers, including nonpersistent buffers."""
    digest = hashlib.sha256()
    for kind, values in [('parameter', module.named_parameters()),
                         ('buffer', module.named_buffers())]:
        for name, value in values:
            digest.update((kind + name + str(value.shape) + str(value.dtype)).encode())
            digest.update(fingerprint(value).encode())
    return digest.hexdigest()


def array(value):
    return value.detach().cpu().double().numpy() if torch.is_tensor(value) else np.asarray(value, dtype=np.float64)


def statistics(value):
    values = array(value)
    finite = np.isfinite(values)
    good = values[finite]
    return dict(shape=list(values.shape), min=float(good.min()) if good.size else None,
                max=float(good.max()) if good.size else None,
                mean=float(good.mean()) if good.size else None,
                std=float(good.std()) if good.size else None,
                l2_norm=float(np.linalg.norm(good)), finite_fraction=float(finite.mean()))


def similarity(left, right):
    a, b = array(left).ravel(), array(right).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return dict(cosine=float(np.dot(a, b) / (na * nb)) if na * nb else None,
                normalized_l2_difference=float(np.linalg.norm(a - b) / max(na, nb, 1e-12)),
                max_absolute_difference=float(np.max(np.abs(a - b))),
                exactly_equal=bool(np.array_equal(a, b)))


def diversity(value):
    views = array(value).reshape(12, -1)
    norms = np.linalg.norm(views, axis=1)
    cos = np.dot(views, views.T) / np.maximum(np.outer(norms, norms), 1e-24)
    pairs = cos[np.triu_indices(12, 1)]
    variance = float(views.var(axis=0).mean())
    return dict(mean_pairwise_cosine=float(pairs.mean()), min_pairwise_cosine=float(pairs.min()),
                max_pairwise_cosine=float(pairs.max()), feature_variance_across_views=variance,
                variance_over_mean_square=variance / max(float((views ** 2).mean()), 1e-24),
                zero_norm_views=int((norms == 0).sum()))


def make_conditions(observations, adapter, sensor):
    adapter.config.validate_sensor(sensor)
    assert sensor.NORMALIZE_DEPTH and adapter.config.no_return_fill_normalized == 1.0
    assert sensor.MIN_DEPTH + 1.0 * (sensor.MAX_DEPTH - sensor.MIN_DEPTH) == sensor.MAX_DEPTH
    sparse, diagnostics = adapter(observations)
    far = observations.copy()
    for key in DEPTH_KEYS:
        far[key] = torch.full_like(observations[key], 1.0)
    conditions = dict(DENSE=observations, SPARSE=sparse, ALL_FAR=far)
    for condition in conditions.values():
        for key in observations:
            if key not in DEPTH_KEYS:
                assert condition[key] is observations[key], 'non-depth observation replaced'
    return conditions, diagnostics[0]


def input_statistics(observations, diagnostics=None):
    views = torch.cat([observations[k] for k in DEPTH_KEYS])
    result = statistics(views)
    result['fraction_equal_1'] = float((views == 1).double().mean())
    result['per_view'] = {}
    for key in DEPTH_KEYS:
        value = observations[key]
        record = statistics(value)
        record['fraction_equal_1'] = float((value == 1).double().mean())
        if diagnostics is not None:
            count = diagnostics['valid_projected_pixels_per_view'][key]
            record['projected_return_fraction'] = count / value.numel()
            record['no_return_fraction'] = 1 - count / value.numel()
        result['per_view'][key] = record
    if diagnostics is not None:
        result.update(projected_return_fraction=diagnostics['valid_projected_fraction'],
                      no_return_fraction=diagnostics['no_return_fraction'], adapter_diagnostics=diagnostics)
    return result


def cell(a, d, logits, probabilities):
    return dict(angle_index=int(a), distance_index=int(d), angle_degrees_clockwise=float(a * 3),
                angle_degrees_counterclockwise=float((360 - a * 3) % 360),
                distance_meters=float((d + 1) * original_etp.DISTANCE_STEP_M),
                raw_logit=float(logits[a, d]), softmax_probability=float(probabilities[a, d]))


def logit_statistics(raw_logits, candidate_map):
    logits = array(raw_logits).reshape(120, 12)
    # Same float32 softmax as Original; convert only subsequent statistics.
    probabilities = array(torch.softmax(raw_logits.reshape(1, -1), dim=1)).reshape(120, 12)
    ranked = np.argsort(-logits.ravel(), kind='stable')
    top = [cell(*np.unravel_index(i, logits.shape), logits, probabilities) for i in ranked[:20]]
    selected = [cell(a, d, logits, probabilities) for a, d in np.argwhere(array(candidate_map).reshape(120, 12) != 0)]
    angular_peaks = logits.max(axis=1)
    peak_angles = np.argsort(-angular_peaks, kind='stable')[:20]
    periodic_peaks = logits[5::10, 2]
    sorted_logits = logits.ravel()[ranked]
    radial = probabilities.sum(axis=0)
    result = statistics(logits)
    result.update(entropy_nats=float(-(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum()),
                  entropy_fraction_of_uniform=float(-(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum() / math.log(1440)),
                  radial_marginal=radial.tolist(), angular_marginal=probabilities.sum(axis=1).tolist(),
                  top20_cells=top, top20_angular_peaks=[cell(a, logits[a].argmax(), logits, probabilities) for a in peak_angles],
                  max_minus_second=float(sorted_logits[0] - sorted_logits[1]),
                  max_minus_median=float(sorted_logits[0] - np.median(logits)),
                  cells_exactly_at_max=int((logits == logits.max()).sum()),
                  cells_within_1e_6_of_max=int((logits.max() - logits <= 1e-6).sum()),
                  cells_within_1e_4_of_max=int((logits.max() - logits <= 1e-4).sum()),
                  periodic_radial2_peak_std=float(periodic_peaks.std()),
                  periodic_radial2_peak_range=float(np.ptp(periodic_peaks)),
                  periodic_radial2_peaks=[cell(a, 2, logits, probabilities) for a in range(5, 120, 10)],
                  angular_peak_std=float(angular_peaks.std()),
                  repeat_30_degree_max_abs_difference=float(np.max(np.abs(logits - np.roll(logits, 10, axis=0)))),
                  pre_nms_dominant_radial_index=int(radial.argmax()),
                  pre_nms_global_max_radial_index=int(top[0]['distance_index']),
                  angular_bins_with_radial2_argmax=int((logits.argmax(axis=1) == 2).sum()),
                  post_nms_candidates=selected,
                  post_nms_all_radial2=all(c['distance_index'] == 2 for c in selected),
                  post_nms_radial_counts=np.bincount([c['distance_index'] for c in selected], minlength=12).tolist())
    return result


def capture_condition(original, observations):
    """Hooks see untouched encoder output and raw predictor output BEFORE NMS.

    Replay captured logits through the existing candidate map and assert exact
    agreement with the provider result. No replacement models or NMS algorithm.
    """
    captured, events = {}, []
    predictor = original.waypoint_predictor
    core = predictor.module if hasattr(predictor, 'module') else predictor
    modules = dict(depth_encoder=original.depth_encoder, predictor=predictor, rgb_encoder=original.rgb_encoder)
    before = {k: state_fingerprint(v) for k, v in modules.items()}
    assert all(not m.training for root in modules.values() for m in root.modules())
    source = {k: fingerprint(v) for k, v in observations.items()}
    def save(name):
        def hook(module, inputs, output):
            captured[name] = output.detach().clone()
            events.append(name)
        return hook
    def check_predictor(module, inputs):
        assert torch.equal(inputs[1], captured['depth_embedding'])
        events.append('predictor_input_verified')
    def check_packed(module, inputs):
        packed = inputs[0]
        for index, key in enumerate(DEPTH_KEYS):
            dest = (12 - index) % 12
            assert torch.equal(packed['depth'][dest], observations[key][0])
            assert torch.equal(packed['rgb'][dest], observations[key.replace('depth', 'rgb')][0])
        captured['packed_rgb_sha256'] = fingerprint(packed['rgb'])
    handles = [original.depth_encoder.register_forward_pre_hook(check_packed),
               original.depth_encoder.register_forward_hook(save('depth_embedding')),
               predictor.register_forward_pre_hook(check_predictor), predictor.register_forward_hook(save('raw_logits'))]
    for name, module in [('visual_channels', getattr(original.depth_encoder, 'visual_encoder', None)),
                         ('predictor_depth_fc', getattr(core, 'visual_fc_depth', None))]:
        if module is not None:
            handles.append(module.register_forward_hook(save(name)))
    try:
        with torch.no_grad():
            output = original.predict(observations, in_train=False)
            events.append('provider_return_after_nms')
            raw = captured['raw_logits']
            assert tuple(raw.shape) == (1, 120, 12)
            candidate_map = original_etp._candidate_map(raw, 1)
    finally:
        for handle in handles:
            handle.remove()
    assert events.index('raw_logits') < events.index('provider_return_after_nms')
    assert source == {k: fingerprint(v) for k, v in observations.items()}
    assert before == {k: state_fingerprint(v) for k, v in modules.items()}, 'parameters or buffers changed'
    selected = candidate_map[0].nonzero()
    assert np.allclose(output['cand_distances'][0], ((selected[:, 1] + 1) * original_etp.DISTANCE_STEP_M).cpu().numpy(), atol=0, rtol=0)
    expected_angles = (2 * math.pi - selected[:, 0].float() / 120 * 2 * math.pi).tolist()
    assert output['cand_angles'][0] == expected_angles
    result = dict(depth_encoder=dict(statistics=statistics(captured['depth_embedding']), diversity=diversity(captured['depth_embedding'])),
                  logits=logit_statistics(raw, candidate_map),
                  guards=dict(rgb_sha256=captured['packed_rgb_sha256'], source_unchanged=True,
                              parameters_and_buffers_unchanged=True, predictor_depth_input_verified=True,
                              offline_nms_matches_provider=True, events=events,
                              depth_encoder_object_id=id(original.depth_encoder), predictor_object_id=id(predictor)))
    for name in ['visual_channels', 'predictor_depth_fc']:
        if name in captured:
            result[name] = dict(statistics=statistics(captured[name]), diversity=diversity(captured[name]))
    return result, captured


def analyze_observation(observations, provider, sensor):
    source = {k: fingerprint(v) for k, v in observations.items()}
    conditions, diagnostics = make_conditions(observations, provider.adapter, sensor)
    result, tensors = dict(conditions={}), {}
    for name, batch in conditions.items():
        record, tensors[name] = capture_condition(provider.original, batch)
        record['input'] = input_statistics(batch, diagnostics if name == 'SPARSE' else None)
        result['conditions'][name] = record
    guards = [record['guards'] for record in result['conditions'].values()]
    for field in ['rgb_sha256', 'depth_encoder_object_id', 'predictor_object_id']:
        assert len({guard[field] for guard in guards}) == 1
    result['comparisons'] = {}
    for left, right in [('DENSE', 'SPARSE'), ('SPARSE', 'ALL_FAR'), ('DENSE', 'ALL_FAR')]:
        comparisons = {'input': similarity(torch.cat([conditions[left][k] for k in DEPTH_KEYS]),
                                          torch.cat([conditions[right][k] for k in DEPTH_KEYS]))}
        for name in ['depth_embedding', 'visual_channels', 'predictor_depth_fc', 'raw_logits']:
            if name in tensors[left]:
                comparisons[name] = similarity(tensors[left][name], tensors[right][name])
        result['comparisons'][left + '_vs_' + right] = comparisons
    assert source == {k: fingerprint(v) for k, v in observations.items()}
    return result, tensors
