"""Evaluation-only sensor adapter delegating all proposal behavior to Original."""
import torch

from vlnce_baselines.perception.mid360_sparse_depth import MID360SparseDepthAdapter
from vlnce_baselines.waypoint_provider.base import WaypointProvider
from vlnce_baselines.waypoint_provider.original_etp import OriginalETPWaypointProvider


class MID360SparseDepthETPWaypointProvider(WaypointProvider):
    def __init__(self, config, **original_components):
        self.adapter = MID360SparseDepthAdapter(config)
        self.original = OriginalETPWaypointProvider(**original_components)
        self.last_sparse_depth_diagnostics = []

    def predict(self, observations, in_train=False):
        if in_train:
            raise RuntimeError("MID360-like sparse-depth control is evaluation only; no training")
        for name in ('rgb_encoder', 'depth_encoder', 'waypoint_predictor'):
            if getattr(getattr(self.original, name), 'training', False):
                raise RuntimeError("{} must be in eval mode for zero-shot control".format(name))
        with torch.no_grad():
            adapted, self.last_sparse_depth_diagnostics = self.adapter(observations)
            return self.original.predict(adapted, in_train=False)
