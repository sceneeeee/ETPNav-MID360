"""Static, non-learned experiment seam; never registered with the evaluator."""
import numpy as np
import torch
from torch import nn

from vlnce_baselines.waypoint_provider import original_etp


FROZEN_IDS = frozenset('53 1685 821 1024 1477 1636 675 164 1375 1679 250 2 475 1356 252 1439 1016 1285 1478 25'.split())
HELD_OUT_IDS = ['53', '821', '1024', '250', '25']
EPSILON = 1e-6
FORMULA = '(sparse - mu_sparse) / (sigma_sparse + 1e-6) * sigma_dense + mu_dense'


def validate_calibration_ids(split, episode_ids):
    ids = [str(e) for e in episode_ids]
    if split not in ('val_seen', 'train') or not ids or len(set(ids)) != len(ids):
        raise ValueError('calibration requires unique episodes from val_seen or train')
    if FROZEN_IDS.intersection(ids):
        raise ValueError('calibration includes a frozen val_unseen-20 numeric ID')


def select_calibration_episodes(episodes):
    """One reset per scene: lowest eligible numeric ID; no outcome fields read."""
    selected = {}
    for episode in sorted(episodes, key=lambda e: int(e['episode_id'])):
        if str(episode['episode_id']) not in FROZEN_IDS:
            selected.setdefault(episode['scene_id'], episode)
    result = [selected[scene] for scene in sorted(selected)]
    validate_calibration_ids('val_seen', [e['episode_id'] for e in result])
    return result


class ChannelMoments:
    """Float64 parallel Welford accumulation over N,H,W; retain only C values."""
    def __init__(self):
        self.count = 0
        self.observations = 0
        self.shape = None
        self.mean = None
        self.m2 = None

    def update(self, features):
        x = features.detach().cpu().double().numpy()
        if x.ndim != 4 or not np.isfinite(x).all():
            raise ValueError('expected finite N,C,H,W features')
        if self.shape is not None and tuple(x.shape) != self.shape:
            raise ValueError('feature shape changed during calibration')
        self.shape = tuple(x.shape)
        values = x.transpose(1, 0, 2, 3).reshape(x.shape[1], -1)
        count = values.shape[1]
        mean = values.mean(axis=1)
        m2 = ((values - mean[:, None]) ** 2).sum(axis=1)
        if self.count:
            delta = mean - self.mean
            total = self.count + count
            self.m2 += m2 + delta ** 2 * self.count * count / total
            self.mean += delta * count / total
        else:
            self.mean, self.m2 = mean, m2
        self.count += count
        self.observations += 1

    def finish(self):
        if not self.count:
            raise ValueError('empty calibration')
        return self.mean.copy(), np.sqrt(np.maximum(self.m2 / self.count, 0))

    def to_state(self):
        """JSON float round trips preserve the float64 Welford accumulators."""
        return dict(count=self.count, observations=self.observations,
                    shape=None if self.shape is None else list(self.shape),
                    mean=None if self.mean is None else self.mean.tolist(),
                    m2=None if self.m2 is None else self.m2.tolist())

    @classmethod
    def from_state(cls, state):
        result = cls()
        count, observations = state['count'], state['observations']
        if type(count) is not int or type(observations) is not int or min(count, observations) < 0:
            raise ValueError('invalid Welford counters')
        if observations == 0:
            if state != result.to_state():
                raise ValueError('invalid empty Welford state')
            return result
        shape = state['shape']
        if not isinstance(shape, list) or len(shape) != 4 or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('invalid Welford shape')
        if count != observations * shape[0] * shape[2] * shape[3]:
            raise ValueError('Welford count disagrees with observations/shape')
        mean, m2 = (np.asarray(state[k], dtype=np.float64) for k in ('mean', 'm2'))
        if any(a.shape != (shape[1],) or not np.isfinite(a).all() for a in (mean, m2)) or (m2 < 0).any():
            raise ValueError('invalid Welford channel statistics')
        result.count, result.observations, result.shape = count, observations, tuple(shape)
        result.mean, result.m2 = mean.copy(), m2.copy()
        return result


class FixedChannelAlignment(nn.Module):
    """Fixed buffers only. Broadcasting preserves view and spatial coordinates."""
    def __init__(self, mu_dense, sigma_dense, mu_sparse, sigma_sparse):
        super().__init__()
        shape = None
        for name, value in [('mu_dense', mu_dense), ('sigma_dense', sigma_dense),
                            ('mu_sparse', mu_sparse), ('sigma_sparse', sigma_sparse)]:
            value = torch.as_tensor(value, dtype=torch.float64).detach().clone()
            if value.ndim != 1 or not value.numel() or not torch.isfinite(value).all():
                raise ValueError('statistics must be finite nonempty channel vectors')
            if shape is not None and value.shape != shape:
                raise ValueError('statistics have inconsistent channel counts')
            if name.startswith('sigma') and (value < 0).any():
                raise ValueError('standard deviations must be nonnegative')
            shape = value.shape
            self.register_buffer(name, value.reshape(1, -1, 1, 1))

    @classmethod
    def from_npz(cls, path):
        with np.load(str(path), allow_pickle=False) as saved:
            return cls(**{name: saved[name] for name in
                          ('mu_dense', 'sigma_dense', 'mu_sparse', 'sigma_sparse')})

    @torch.no_grad()
    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.mu_dense.shape[1]:
            raise ValueError('expected N,C,H,W with calibrated channel count')
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError('expected finite floating point features')
        # Compute half/bfloat16 inputs in float32 so epsilon cannot underflow.
        dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
        stats = [getattr(self, name).to(device=features.device, dtype=dtype) for name in
                 ('mu_dense', 'sigma_dense', 'mu_sparse', 'sigma_sparse')]
        md, sd, ms, ss = stats
        result = ((features.to(dtype) - ms) / (ss + EPSILON) * sd + md).to(features.dtype)
        if not torch.isfinite(result).all():
            raise ValueError('alignment output overflowed runtime dtype')
        return result


def encode(original, observations):
    return original_etp.encode_panorama(observations, original.rgb_encoder, original.depth_encoder,
                                       original.rgb_pool, original.depth_pool)


def calibration_depth_config(config):
    """Clone the accepted config; only calibration opts into Depth-only sensors."""
    expected = ['DEPTH_SENSOR'] + ['DEPTH_{}'.format(a) for a in range(30, 360, 30)]
    sensors = [s for s in config.SENSORS if s.startswith('DEPTH')]
    agent_sensors = [s for s in config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS if s.startswith('DEPTH')]
    if sensors != expected or agent_sensors != expected:
        raise ValueError('calibration requires the accepted twelve ordered Depth sensors at both config levels')
    result = config.clone()
    result.defrost()
    # construct_envs overwrites AGENT_0.SENSORS from the top-level list.
    result.SENSORS = list(sensors)
    result.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = list(sensors)
    result.freeze()
    return result


def calibration_depth_keys(observations):
    keys = [k for k in observations if k.startswith('depth')]
    expected = ['depth'] + ['depth_{}'.format(a) for a in range(30, 360, 30)]
    if set(keys) != set(expected) or any(k.lower().startswith('rgb') for k in observations):
        raise RuntimeError('calibration requires exactly 12 Depth observations and no RGB observations')
    return keys


@torch.no_grad()
def encode_calibration_depth(original, observations):
    """Original's exact insertion-order/reverse-index packing, without RGB work."""
    keys = calibration_depth_keys(observations)
    encoder = original.depth_encoder
    if any(m.training for m in encoder.modules()) or any(p.requires_grad for p in encoder.parameters()):
        raise RuntimeError('calibration requires the frozen Original depth encoder in eval mode')
    depth_batch = torch.zeros_like(observations['depth']).repeat(12, 1, 1, 1)
    for angle_count, key in enumerate(keys):
        value = observations[key]
        if value.shape != observations['depth'].shape:
            raise ValueError('calibration Depth view shapes disagree')
        for batch_index in range(value.size(0)):
            depth_batch[(12 - angle_count) % 12 + batch_index * 12] = value[batch_index]
    return encoder({'depth': depth_batch})


def predict_encoded(original, encoded, alignment=None):
    """Align encoder output upstream of the original visual_fc_depth and transformer.

    Tests whether upstream moments rescue the later vis_classifier ReLU collapse.
    Policy/GraphMap features stay raw; all predictor layers and NMS are original.
    """
    depth = encoded.depth_embedding
    if alignment is not None:
        depth = alignment(depth)
    logits = original.waypoint_predictor(encoded.rgb_embedding, depth)
    candidate_map = original_etp._candidate_map(logits, encoded.batch_size)
    cells = [candidate_map[i].nonzero() for i in range(encoded.batch_size)]
    output = original_etp.construct_waypoint_output(
        encoded, [c[:, 0] for c in cells], [c[:, 1] for c in cells],
        original.pano_angle_features, original.pano_image_indexes)
    return output, logits, candidate_map, depth


class MID360SparseDepthFeatureAlignedETP:
    """Independently selectable static provider, wrapping accepted sparse adapter.

    This is intentionally an analysis-only class, absent from trainer factories.
    No hooks or replacements are installed on shared Original modules.
    """
    name = 'mid360_sparse_depth_feature_aligned_etp'

    def __init__(self, sparse_provider, alignment):
        self.sparse_provider = sparse_provider
        self.original = sparse_provider.original
        self.alignment = alignment

    @torch.no_grad()
    def predict(self, observations, in_train=False):
        if in_train:
            raise RuntimeError('feature alignment is a static evaluation-only experiment')
        for name in ('rgb_encoder', 'depth_encoder', 'waypoint_predictor'):
            if any(m.training for m in getattr(self.original, name).modules()):
                raise RuntimeError('all Original modules must be in eval mode')
        sparse, _ = self.sparse_provider.adapter(observations)
        encoded = encode(self.original, sparse)
        return predict_encoded(self.original, encoded, self.alignment)[0]
