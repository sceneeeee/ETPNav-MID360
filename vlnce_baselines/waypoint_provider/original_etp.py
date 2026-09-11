from copy import deepcopy
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from vlnce_baselines.models.utils import angle_feature_torch
from vlnce_baselines.waypoint_pred.utils import nms
from vlnce_baselines.waypoint_provider.base import WaypointProvider
from vlnce_baselines.waypoint_provider.types import WaypointOutput
from vlnce_baselines.original_forensics import capture as capture_forensics


NUM_ANGLES = 120
NUM_IMAGES = 12
NUM_DISTANCES = 12
DISTANCE_STEP_M = 0.25


class EncodedPanorama(object):
    def __init__(
        self,
        batch_size,
        rgb_embedding,
        depth_embedding,
        rgb_features,
        depth_features,
    ):
        self.batch_size = batch_size
        self.rgb_embedding = rgb_embedding
        self.depth_embedding = depth_embedding
        self.rgb_features = rgb_features
        self.depth_features = depth_features


def encode_panorama(
    observations,
    rgb_encoder,
    depth_encoder,
    rgb_pool,
    depth_pool,
):
    """Encode the original CCW sensor mapping while preserving ETP ordering."""
    batch_size = observations["rgb"].shape[0]
    depth_batch = torch.zeros_like(observations["depth"]).repeat(
        NUM_IMAGES, 1, 1, 1
    )
    rgb_batch = torch.zeros_like(observations["rgb"]).repeat(
        NUM_IMAGES, 1, 1, 1
    )

    angle_count = 0
    for key, value in observations.items():
        if "depth" not in key:
            continue
        for batch_index in range(value.size(0)):
            reverse_index = (NUM_IMAGES - angle_count) % NUM_IMAGES
            destination = reverse_index + batch_index * NUM_IMAGES
            depth_batch[destination] = value[batch_index]
            rgb_batch[destination] = observations[
                key.replace("depth", "rgb")
            ][batch_index]
        angle_count += 1
    if angle_count != NUM_IMAGES:
        raise ValueError(
            "ETP waypoint input requires 12 ordered depth views, received {}".format(
                angle_count
            )
        )

    packed = {"depth": depth_batch, "rgb": rgb_batch}
    depth_embedding = depth_encoder(packed)
    rgb_embedding = rgb_encoder(packed)

    rgb_reshape = rgb_embedding.reshape(batch_size, NUM_IMAGES, 512, 1, 1)
    depth_reshape = depth_embedding.reshape(
        batch_size, NUM_IMAGES, 128, 4, 4
    )
    rgb_counter_clockwise = torch.cat(
        (rgb_reshape[:, 0:1, :], torch.flip(rgb_reshape[:, 1:, :], [1])),
        dim=1,
    )
    depth_counter_clockwise = torch.cat(
        (
            depth_reshape[:, 0:1, :],
            torch.flip(depth_reshape[:, 1:, :], [1]),
        ),
        dim=1,
    )
    return EncodedPanorama(
        batch_size=batch_size,
        rgb_embedding=rgb_embedding,
        depth_embedding=depth_embedding,
        rgb_features=rgb_pool(rgb_counter_clockwise),
        depth_features=depth_pool(depth_counter_clockwise),
    )


def _candidate_map(waypoint_heatmap_logits, batch_size):
    probabilities = torch.softmax(
        waypoint_heatmap_logits.reshape(
            batch_size, NUM_ANGLES * NUM_DISTANCES
        ),
        dim=1,
    ).reshape(batch_size, NUM_ANGLES, NUM_DISTANCES)
    wrapped = torch.cat(
        (probabilities[:, -1:, :], probabilities, probabilities[:, :1, :]),
        dim=1,
    )
    return nms(
        wrapped.unsqueeze(1), max_predictions=5, sigma=(7.0, 5.0)
    ).squeeze(1)[:, 1:-1, :]


def _training_candidates(waypoint_heatmap_logits, candidate_map, batch_size):
    heatmap_offset = 5
    regional = torch.cat(
        (
            waypoint_heatmap_logits[:, -heatmap_offset:, :],
            waypoint_heatmap_logits[:, :-heatmap_offset, :],
        ),
        dim=1,
    ).reshape(batch_size, 12, 10, 12)
    batch_angle_indexes = []
    batch_distance_indexes = []
    for batch_index in range(batch_size):
        angle_indexes = candidate_map[batch_index].nonzero()[:, 0]
        image_indexes = ((angle_indexes.cpu().numpy() + 5) // 10)
        image_indexes[image_indexes == 12] = 0
        heat = regional[batch_index][image_indexes].view(
            image_indexes.size, -1
        )
        actions = torch.distributions.Categorical(F.softmax(heat, 1)).sample()
        sampled_angles = []
        sampled_distances = []
        for image_index, action in zip(image_indexes, actions):
            angle_pointer = (image_index - 1) * 10 + 5 if image_index != 0 else 0
            sampled_angles.append(action // 12 + angle_pointer)
            sampled_distances.append(action % 12)
        batch_angle_indexes.append(sampled_angles)
        batch_distance_indexes.append(sampled_distances)
    return batch_angle_indexes, batch_distance_indexes


def construct_waypoint_output(
    encoded,
    batch_angle_indexes,
    batch_distance_indexes,
    pano_angle_features,
    pano_image_indexes,
):
    cand_rgb = []
    cand_depth = []
    cand_angle_features = []
    cand_image_indexes = []
    cand_angles = []
    cand_distances = []
    for batch_index in range(encoded.batch_size):
        angle_indexes = batch_angle_indexes[batch_index]
        distance_indexes = batch_distance_indexes[batch_index]
        if not torch.is_tensor(angle_indexes):
            angle_indexes = torch.tensor(angle_indexes)
        if not torch.is_tensor(distance_indexes):
            distance_indexes = torch.tensor(distance_indexes)
        clockwise_angles = angle_indexes.cpu().float() / NUM_ANGLES * 2 * math.pi
        counter_clockwise_angles = (
            2 * math.pi
            - angle_indexes.float() / NUM_ANGLES * 2 * math.pi
        )
        cand_angle_features.append(angle_feature_torch(clockwise_angles))
        cand_angles.append(counter_clockwise_angles.tolist())
        cand_distances.append(
            ((distance_indexes + 1) * DISTANCE_STEP_M).tolist()
        )

        image_indexes = 12 - (angle_indexes.cpu().numpy() + 5) // 10
        image_indexes[image_indexes == 12] = 0
        cand_image_indexes.append(image_indexes)
        cand_rgb.append(encoded.rgb_features[batch_index, image_indexes, ...])
        cand_depth.append(
            encoded.depth_features[batch_index, image_indexes, ...]
        )

    return WaypointOutput(
        cand_rgb=cand_rgb,
        cand_depth=cand_depth,
        cand_angle_fts=cand_angle_features,
        cand_img_idxes=cand_image_indexes,
        cand_angles=cand_angles,
        cand_distances=cand_distances,
        pano_rgb=encoded.rgb_features,
        pano_depth=encoded.depth_features,
        pano_angle_fts=deepcopy(pano_angle_features),
        pano_img_idxes=deepcopy(pano_image_indexes),
    )


class OriginalETPWaypointProvider(WaypointProvider):
    """The official learned ETP proposal behavior behind a stable seam."""

    def __init__(
        self,
        rgb_encoder,
        depth_encoder,
        rgb_pool,
        depth_pool,
        waypoint_predictor,
        pano_angle_features,
        pano_image_indexes,
    ):
        self.rgb_encoder = rgb_encoder
        self.depth_encoder = depth_encoder
        self.rgb_pool = rgb_pool
        self.depth_pool = depth_pool
        self.waypoint_predictor = waypoint_predictor
        self.pano_angle_features = pano_angle_features
        self.pano_image_indexes = pano_image_indexes

    def predict(self, observations, in_train=False):
        encoded = encode_panorama(
            observations=observations,
            rgb_encoder=self.rgb_encoder,
            depth_encoder=self.depth_encoder,
            rgb_pool=self.rgb_pool,
            depth_pool=self.depth_pool,
        )
        logits = self.waypoint_predictor(
            encoded.rgb_embedding, encoded.depth_embedding
        )
        candidate_map = _candidate_map(logits, encoded.batch_size)
        if in_train:
            angle_indexes, distance_indexes = _training_candidates(
                logits, candidate_map, encoded.batch_size
            )
        else:
            angle_indexes = [
                candidate_map[index].nonzero()[:, 0]
                for index in range(encoded.batch_size)
            ]
            distance_indexes = [
                candidate_map[index].nonzero()[:, 1]
                for index in range(encoded.batch_size)
            ]
        output = construct_waypoint_output(
            encoded=encoded,
            batch_angle_indexes=angle_indexes,
            batch_distance_indexes=distance_indexes,
            pano_angle_features=self.pano_angle_features,
            pano_image_indexes=self.pano_image_indexes,
        )
        # The manual forensic runner supplies this path.  Capture is opt-in and
        # runs before evaluator telemetry serializes candidates.
        capture_forensics(os.environ.get("ETP_ORIGINAL_FORENSICS_RAW_PATH"),
                          observations, logits, output, DISTANCE_STEP_M)
        return output
