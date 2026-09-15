"""Stage 5 A1 feature-only distillation. This module needs only PyTorch."""
import math

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = 'residual_1x1_gelu_1x1_v1'
COSINE_EPS = 1e-8


def _features(value, channels=None):
    if (not torch.is_tensor(value) or value.ndim != 4 or
            any(n <= 0 for n in value.shape) or tuple(value.shape[2:]) != (4, 4)):
        raise ValueError('expected nonempty N,C,4,4 features')
    if channels is not None and value.shape[1] != channels:
        raise ValueError('unexpected feature channel count')
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError('expected finite floating point features')


def _weights(smooth_l1_weight, cosine_weight):
    values = (smooth_l1_weight, cosine_weight)
    if any(not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
        raise ValueError('loss weights must be finite, nonnegative, and not both zero')


class ResidualFeatureAdapter(nn.Module):
    """x + Conv1x1(GELU(Conv1x1(x))); use .to() for dtype/device as usual.

    The second convolution starts at exactly zero. No input mutation, spatial
    mixing, normalization, or special train/eval behavior is introduced.
    """
    def __init__(self, channels=128):
        super().__init__()
        if type(channels) is not int or channels <= 0:
            raise ValueError('channels must be a positive integer')
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, features):
        _features(features, self.channels)
        if features.device != self.conv1.weight.device or features.dtype != self.conv1.weight.dtype:
            raise ValueError('move adapter to the input device and dtype with .to()')
        result = features + self.conv2(self.activation(self.conv1(features)))
        if not torch.isfinite(result).all():
            raise ValueError('adapter output is nonfinite (possible arithmetic overflow)')
        return result


def feature_distillation_loss(adapted, dense, smooth_l1_weight=1.0, cosine_weight=1.0):
    """Mean SmoothL1 (beta=1) + mean per-N (1 - cosine), equal M0 weights.

    Cosine uses C,H,W flattened per item, dim=1, eps=1e-8. Zero vectors
    consequently have cosine zero. All returned scalars retain autograd;
    callers must detach/freeze their collected feature pair before optimization.
    """
    _features(adapted)
    _features(dense)
    if adapted.shape != dense.shape or adapted.dtype != dense.dtype or adapted.device != dense.device:
        raise ValueError('adapted and dense shape, dtype, and device must match')
    _weights(smooth_l1_weight, cosine_weight)
    smooth_l1 = F.smooth_l1_loss(adapted, dense, reduction='mean', beta=1.0)
    cosine = F.cosine_similarity(adapted.flatten(1), dense.flatten(1), dim=1, eps=COSINE_EPS)
    cosine_loss = (1.0 - cosine).mean()
    return dict(total=smooth_l1_weight * smooth_l1 + cosine_weight * cosine_loss,
                smooth_l1=smooth_l1, cosine_loss=cosine_loss,
                mean_cosine_similarity=cosine.mean())


def architecture_metadata(channels):
    return dict(name=ARCHITECTURE, channels=channels, kernel_size=1,
                activation='GELU', bias=True, residual=True, spatial_shape=[4, 4])


def save_adapter_checkpoint(path, adapter, smooth_l1_weight=1.0, cosine_weight=1.0,
                            seed=None, step=None):
    """Save adapter tensors and reconstruction metadata only; refuse overwrite."""
    _weights(smooth_l1_weight, cosine_weight)
    state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    payload = dict(format_version=1, architecture=architecture_metadata(adapter.channels),
                   loss=dict(smooth_l1_weight=smooth_l1_weight, cosine_weight=cosine_weight,
                             smooth_l1_beta=1.0, cosine_eps=COSINE_EPS,
                             cosine_reduction='flatten_CHW_per_N_then_mean'),
                   dtype=str(adapter.conv1.weight.dtype), state_dict=state, seed=seed, step=step)
    with open(path, 'xb') as handle:
        torch.save(payload, handle)


def load_adapter_checkpoint(path, expected_channels=None, device='cpu'):
    """Load a trusted local checkpoint (legacy torch.load uses pickle)."""
    payload = torch.load(path, map_location='cpu')
    arch = payload.get('architecture', {})
    channels = arch.get('channels')
    if (payload.get('format_version') != 1 or type(channels) is not int or channels <= 0 or
            arch != architecture_metadata(channels) or
            (expected_channels is not None and channels != expected_channels)):
        raise ValueError('unsupported adapter architecture metadata')
    loss = payload.get('loss', {})
    if (set(loss) != {'smooth_l1_weight', 'cosine_weight', 'smooth_l1_beta', 'cosine_eps', 'cosine_reduction'} or
            loss['smooth_l1_beta'] != 1.0 or loss['cosine_eps'] != COSINE_EPS or
            loss['cosine_reduction'] != 'flatten_CHW_per_N_then_mean'):
        raise ValueError('unsupported loss metadata')
    _weights(loss['smooth_l1_weight'], loss['cosine_weight'])
    dtypes = {str(t): t for t in (torch.float16, torch.float32, torch.float64, torch.bfloat16)}
    if payload.get('dtype') not in dtypes:
        raise ValueError('unsupported checkpoint dtype')
    dtype = dtypes[payload['dtype']]
    state = payload['state_dict']
    if any(v.dtype != dtype or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('invalid checkpoint tensors')
    adapter = ResidualFeatureAdapter(channels).to(dtype=dtype)
    adapter.load_state_dict(state, strict=True)
    return adapter.to(device=device), {k: v for k, v in payload.items() if k != 'state_dict'}
