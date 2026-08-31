import unittest
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from vlnce_baselines.waypoint_provider.original_etp import (
    OriginalETPWaypointProvider,
)


class FakeRGBEncoder(nn.Module):
    def forward(self, observations):
        count = observations["rgb"].shape[0]
        return torch.zeros(count, 512, 1, 1)


class FakeDepthEncoder(nn.Module):
    def forward(self, observations):
        count = observations["depth"].shape[0]
        return torch.zeros(count, 128, 4, 4)


class FakePredictor(nn.Module):
    def forward(self, rgb_embedding, depth_embedding):
        logits = torch.zeros(1, 120, 12)
        logits[0, 0, 0] = 10.0
        logits[0, 30, 3] = 9.0
        return logits


class OriginalETPWaypointProviderTest(unittest.TestCase):
    def test_training_augmentation_returns_tensor_candidates(self):
        observations = OrderedDict()
        for angle in range(0, 360, 30):
            suffix = "" if angle == 0 else "_{}".format(angle)
            observations["rgb{}".format(suffix)] = torch.zeros(
                1, 2, 2, 3, dtype=torch.uint8
            )
            observations["depth{}".format(suffix)] = torch.zeros(
                1, 2, 2, 1
            )
        provider = OriginalETPWaypointProvider(
            rgb_encoder=FakeRGBEncoder(),
            depth_encoder=FakeDepthEncoder(),
            rgb_pool=nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(2)),
            depth_pool=nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(2)),
            waypoint_predictor=FakePredictor(),
            pano_angle_features=torch.zeros(12, 4),
            pano_image_indexes=np.arange(12, dtype=np.int64),
        )

        output = provider.predict(observations, in_train=True)

        self.assertEqual(output["cand_rgb"][0].ndim, 2)
        self.assertEqual(output["cand_depth"][0].ndim, 2)
        self.assertEqual(
            len(output["cand_angles"][0]), len(output["cand_distances"][0])
        )


if __name__ == "__main__":
    unittest.main()
