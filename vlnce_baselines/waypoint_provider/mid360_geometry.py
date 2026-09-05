import math
import os
import time

import numpy as np
import torch
import yaml

from vlnce_baselines.perception.geometry_provider import (
    DepthReconstructedGeometryProvider,
    GeometryProvider,
)
from vlnce_baselines.perception.mid360_like_filter import MID360LikeFilter
from vlnce_baselines.perception.local_traversability import (
    TraversabilityConfig,
    compute_local_traversability,
)
from vlnce_baselines.waypoint_provider.base import WaypointProvider
from vlnce_baselines.waypoint_provider.original_etp import (
    construct_waypoint_output,
    encode_panorama,
)


GEOMETRY_ANGLE_BINS = 120
GEOMETRY_RADIAL_BINS = 12
# Preserve the generic NMS call's sigma=(7, 5) axis order: x is radial
# (columns) and y is angle (rows).
GEOMETRY_NMS_RADIAL_SIGMA = 7
GEOMETRY_NMS_ANGULAR_SIGMA = 5
GEOMETRY_RADIAL_UTILITY_V0_LINEAR = "v0_linear"
GEOMETRY_RADIAL_UTILITY_MRU = "mru"
GMIN_EXTRACTION = "gmin_coverage_v0"


def gmin_observation_availability(observations, config):
    """Validate depth input, then test the frontend's stride-sampled evidence.

    Distant valid surfaces count even when no local obstacle points survive.
    Invalid samples anywhere in a view fail before visual encoding.
    """
    expected = ["depth" if a == 0 else "depth_{}".format(a)
                for a in range(0, 360, 30)]
    actual = [key for key in observations if "depth" in key]
    if set(actual) != set(expected):
        raise ValueError("G-Min requires all twelve expected depth views")
    shape = tuple(observations["depth"].shape)
    if len(shape) != 4 or shape[-1] != 1 or min(shape[:3]) < 1:
        raise ValueError("G-Min depth shape must be BxHxWx1 with nonempty axes")
    if "rgb" not in observations or observations["rgb"].shape[0] != shape[0]:
        raise ValueError("G-Min RGB/depth batch shapes disagree")
    counts = np.zeros(shape[0], dtype=np.int64)
    for key in expected:
        value = observations[key]
        if tuple(value.shape) != shape:
            raise ValueError("G-Min depth view shapes must match")
        depth = value.detach().cpu().numpy()
        if not np.isfinite(depth).all():
            raise ValueError("G-Min depth must contain only finite values")
        if np.any(depth < 0) or np.any(depth > 1):
            raise ValueError("G-Min normalized depth must be in [0,1]")
        sampled = depth[:, ::config.depth_pixel_stride,
                        ::config.depth_pixel_stride, 0]
        metric = sampled * (config.policy_depth_maximum_m
                            - config.policy_depth_minimum_m)
        metric = metric + config.policy_depth_minimum_m
        counts += (metric > config.minimum_range_m).sum(axis=(1, 2))
    return [{"available": bool(count),
             "reason": ("sampled_depth_above_minimum_range" if count else
                        "no_sampled_depth_above_minimum_range"),
             "sampled_depth_above_minimum_count": int(count)} for count in counts]


def _synchronize_observation_device(observations):
    value = observations["depth"]
    if value.is_cuda:
        torch.cuda.synchronize(value.device)


def apply_radial_utility(traversability_score, radial_utility):
    """Map Geometry V0 traversability to its pre-NMS radial score.

    Geometry V0 establishes a prefix of positive traversable bins and exact
    zeros for blocked bins.  MRU preserves that prefix semantics while making
    the sole controlled scoring change ``x * exp(1 - x)``, where
    ``x = 2d / 3``.  In particular, the valid 3.0 m bin remains positive; this
    is a utility curve, never a candidate-distance cutoff.
    """
    score = np.asarray(traversability_score, dtype=np.float32)
    if score.shape != (GEOMETRY_ANGLE_BINS, GEOMETRY_RADIAL_BINS):
        raise ValueError("geometry score must have shape 120x12")
    if radial_utility == GEOMETRY_RADIAL_UTILITY_V0_LINEAR:
        return score.copy()
    if radial_utility != GEOMETRY_RADIAL_UTILITY_MRU:
        raise ValueError("unknown Geometry radial utility: {}".format(radial_utility))

    distances_m = (
        np.arange(1, GEOMETRY_RADIAL_BINS + 1, dtype=np.float32) * 0.25
    )
    x = 2.0 * distances_m / 3.0
    utility = x * np.exp(1.0 - x)
    # Positive source scores identify precisely the Geometry V0 traversable
    # prefix.  Multiplication retains exact float zero for every blocked cell.
    return np.where(score > 0.0, utility[None, :], 0.0).astype(np.float32)


def select_geometry_candidates(score, max_candidates=5):
    """Select Geometry V0 candidates with circular angle and bounded radius.

    This deliberately does not change the generic ETP NMS implementation.
    Heading is periodic, whereas the radial score axis is physical distance and
    must terminate at bins zero and 11 rather than wrap between them.
    """
    score_tensor = torch.from_numpy(np.asarray(score, dtype=np.float32))
    if score_tensor.shape != (GEOMETRY_ANGLE_BINS, GEOMETRY_RADIAL_BINS):
        raise ValueError("geometry score must have shape 120x12")
    remaining = score_tensor.clone()
    selected = []
    angle_grid = torch.arange(GEOMETRY_ANGLE_BINS).view(-1, 1)
    radial_grid = torch.arange(GEOMETRY_RADIAL_BINS).view(1, -1)

    for _ in range(max_candidates):
        value, flat_index = torch.max(remaining.reshape(-1), dim=0)
        # This matches generic NMS's externally visible behavior: zero and
        # negative score cells never become waypoint candidates.
        if not torch.isfinite(value) or value <= 0:
            break
        angle_index = int(flat_index.item() // GEOMETRY_RADIAL_BINS)
        radial_index = int(flat_index.item() % GEOMETRY_RADIAL_BINS)
        selected.append((angle_index, radial_index))

        angular_difference = torch.abs(angle_grid - angle_index)
        angular_difference = torch.min(
            angular_difference, GEOMETRY_ANGLE_BINS - angular_difference
        )
        radial_difference = torch.abs(radial_grid - radial_index)
        suppress = torch.logical_and(
            angular_difference <= GEOMETRY_NMS_ANGULAR_SIGMA,
            radial_difference <= GEOMETRY_NMS_RADIAL_SIGMA,
        )
        remaining[suppress] = -np.inf

    if not selected:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    # Generic NMS returns ``selected.nonzero()``, whose row-major ordering is
    # independent of greedy selection rank. Preserve that output contract.
    indexes = np.asarray(sorted(selected), dtype=np.int64)
    return indexes[:, 0], indexes[:, 1]


class MID360GeometryConfig(object):
    REQUIRED_KEYS = (
        "angular_bins",
        "distance_bins",
        "distance_step_m",
        "maximum_distance_m",
        "maximum_candidates",
        "camera_height_m",
        "horizontal_fov_degrees",
        "policy_depth_minimum_m",
        "policy_depth_maximum_m",
        "depth_pixel_stride",
        "minimum_range_m",
        "maximum_range_m",
        "vertical_fov_min_degrees",
        "vertical_fov_max_degrees",
        "azimuth_bin_degrees",
        "elevation_bin_degrees",
        "voxel_size_m",
        "robot_radius_m",
        "robot_radius_status",
        "clearance_margin_m",
        "obstacle_height_min_m",
        "obstacle_height_max_m",
        "grid_resolution_m",
    )

    def __init__(self, values):
        missing = [key for key in self.REQUIRED_KEYS if key not in values]
        if missing:
            raise ValueError("MID360 geometry config is missing: {}".format(missing))
        self.values = dict(values)
        for key, value in self.values.items():
            setattr(self, key, value)
        self.maximum_candidate_distance_m = self.values.get(
            "maximum_candidate_distance_m", self.maximum_distance_m
        )
        self.radial_utility = self.values.get(
            "radial_utility", GEOMETRY_RADIAL_UTILITY_V0_LINEAR
        )
        self.candidate_extraction = self.values.get("candidate_extraction", "bounded_nms")
        if self.candidate_extraction not in ("bounded_nms", GMIN_EXTRACTION):
            raise ValueError("unknown Geometry candidate_extraction")
        if self.candidate_extraction == GMIN_EXTRACTION:
            if (isinstance(self.maximum_candidates, bool)
                    or not isinstance(self.maximum_candidates, int)
                    or not 1 <= self.maximum_candidates <= 5):
                raise ValueError("G-Min maximum_candidates must be an integer in 1..5")
            if ("radial_utility" in values
                    or "maximum_candidate_distance_m" in values
                    or any("v1" in str(key).lower() for key in values)
                    or "mode" in values):
                raise ValueError("G-Min forbids radial utility, distance caps and V1 mode")
            if not (np.isfinite(self.policy_depth_minimum_m)
                    and np.isfinite(self.policy_depth_maximum_m)
                    and 0 <= self.policy_depth_minimum_m < self.policy_depth_maximum_m):
                raise ValueError("G-Min metric depth contract is invalid")
        contracted_range = self.distance_bins * self.distance_step_m
        if abs(contracted_range - self.maximum_distance_m) > 1e-6:
            raise ValueError("distance bins and maximum distance disagree")
        if self.angular_bins != 120 or self.distance_bins != 12:
            raise ValueError("ETP geometry contract requires a 120x12 score map")
        if self.distance_step_m <= 0 or self.maximum_distance_m <= 0:
            raise ValueError("candidate distance parameters must be positive")
        if (
            self.maximum_candidate_distance_m <= 0
            or self.maximum_candidate_distance_m > self.maximum_distance_m
            or abs(
                self.maximum_candidate_distance_m / self.distance_step_m
                - round(self.maximum_candidate_distance_m / self.distance_step_m)
            ) > 1e-6
        ):
            raise ValueError("maximum_candidate_distance_m must be an ETP distance bin in range")
        if self.maximum_candidates < 1:
            raise ValueError("maximum_candidates must be positive")
        if self.radial_utility not in (
            GEOMETRY_RADIAL_UTILITY_V0_LINEAR,
            GEOMETRY_RADIAL_UTILITY_MRU,
        ):
            raise ValueError("radial_utility must be v0_linear or mru")
        if self.depth_pixel_stride < 1:
            raise ValueError("depth_pixel_stride must be positive")
        if self.minimum_range_m < 0 or self.maximum_range_m <= self.minimum_range_m:
            raise ValueError("sensor range bounds are invalid")
        if self.robot_radius_m <= 0 or self.clearance_margin_m < 0:
            raise ValueError("footprint parameters are invalid")
        if self.grid_resolution_m <= 0 or self.voxel_size_m < 0:
            raise ValueError("geometry resolutions are invalid")
        if self.obstacle_height_max_m < self.obstacle_height_min_m:
            raise ValueError("obstacle height bounds are invalid")
        if self.vertical_fov_max_degrees <= self.vertical_fov_min_degrees:
            raise ValueError("vertical FoV bounds are invalid")

        footprint_extent = (
            self.maximum_distance_m
            + self.robot_radius_m
            + self.clearance_margin_m
        )
        vertical_extent = max(
            abs(self.obstacle_height_min_m - self.camera_height_m),
            abs(self.obstacle_height_max_m - self.camera_height_m),
        )
        required_sensor_range = math.hypot(footprint_extent, vertical_extent)
        if self.maximum_range_m + 1e-6 < required_sensor_range:
            raise ValueError(
                "sensor range must cover candidate footprint and obstacle-height "
                "extent: {} < {}".format(
                    self.maximum_range_m, required_sensor_range
                )
            )

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as handle:
            values = yaml.safe_load(handle)
        return cls(values)

    def traversability(self):
        return TraversabilityConfig(
            angular_bins=self.angular_bins,
            distance_bins=self.distance_bins,
            distance_step_m=self.distance_step_m,
            robot_radius_m=self.robot_radius_m,
            clearance_margin_m=self.clearance_margin_m,
            obstacle_height_min_m=self.obstacle_height_min_m,
            obstacle_height_max_m=self.obstacle_height_max_m,
            grid_resolution_m=self.grid_resolution_m,
        )


class MID360GeometryWaypointProvider(WaypointProvider):
    """Replace only learned proposal generation with local geometry."""

    def __init__(
        self,
        rgb_encoder,
        depth_encoder,
        rgb_pool,
        depth_pool,
        pano_angle_features,
        pano_image_indexes,
        config,
        geometry_provider=None,
    ):
        self.rgb_encoder = rgb_encoder
        self.depth_encoder = depth_encoder
        self.rgb_pool = rgb_pool
        self.depth_pool = depth_pool
        self.pano_angle_features = pano_angle_features
        self.pano_image_indexes = pano_image_indexes
        self.config = config
        if geometry_provider is None:
            geometry_provider = DepthReconstructedGeometryProvider(config)
        if not isinstance(geometry_provider, GeometryProvider):
            raise TypeError("geometry_provider must implement GeometryProvider")
        self.geometry_provider = geometry_provider
        self.lidar_filter = MID360LikeFilter(
            minimum_range_m=config.minimum_range_m,
            maximum_range_m=config.maximum_range_m,
            vertical_fov_min_degrees=config.vertical_fov_min_degrees,
            vertical_fov_max_degrees=config.vertical_fov_max_degrees,
            azimuth_bin_degrees=config.azimuth_bin_degrees,
            elevation_bin_degrees=config.elevation_bin_degrees,
            voxel_size_m=config.voxel_size_m,
            sensor_height_m=config.camera_height_m,
        )
        self.traversability_config = config.traversability()
        self.last_debug = []
        self.last_timings = []

    def predict(self, observations, in_train=False):
        if in_train:
            raise ValueError(
                "MID360GeometryWaypointProvider V0 is evaluation-only"
            )
        is_gmin = self.config.candidate_extraction == GMIN_EXTRACTION
        availability = None
        if is_gmin:
            _synchronize_observation_device(observations)
            gate_started = time.perf_counter()
            availability = gmin_observation_availability(observations, self.config)
            availability_ms = (time.perf_counter() - gate_started) * 1000.0
        started = time.perf_counter()
        encoded = encode_panorama(
            observations=observations,
            rgb_encoder=self.rgb_encoder,
            depth_encoder=self.depth_encoder,
            rgb_pool=self.rgb_pool,
            depth_pool=self.depth_pool,
        )
        if is_gmin:
            _synchronize_observation_device(observations)
        visual_encoding_ms = (time.perf_counter() - started) * 1000.0
        batch_angle_indexes = []
        batch_distance_indexes = []
        debug = []
        timings = []
        for batch_index in range(encoded.batch_size):
            started = time.perf_counter()
            geometry = self.geometry_provider.get_geometry(
                observations, batch_index
            )
            pointcloud_ms = (time.perf_counter() - started) * 1000.0
            merged_cloud = geometry.pointcloud
            started = time.perf_counter()
            filtered_cloud = self.lidar_filter(merged_cloud)
            filter_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            traversability = compute_local_traversability(
                filtered_cloud, self.traversability_config
            )
            occupancy_traversability_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            if is_gmin:
                from vlnce_baselines.perception.gmin_candidates import select_gmin_candidates
                score = traversability.score > 0
                # Availability gates extraction, not the shared geometry map.
                mask = score if availability[batch_index]["available"] else np.zeros_like(score)
                angle_indexes, distance_indexes = select_gmin_candidates(
                    mask, max_candidates=self.config.maximum_candidates)
            else:
                score = apply_radial_utility(
                    traversability.score, self.config.radial_utility
                )
                maximum_bin = int(round(
                    self.config.maximum_candidate_distance_m / self.config.distance_step_m
                ))
                score[:, maximum_bin:] = -np.inf
                angle_indexes, distance_indexes = select_geometry_candidates(
                    score,
                    max_candidates=self.config.maximum_candidates,
                )
            candidate_construction_ms = (time.perf_counter() - started) * 1000.0
            batch_angle_indexes.append(torch.from_numpy(angle_indexes))
            batch_distance_indexes.append(torch.from_numpy(distance_indexes))
            debug.append(
                {
                    "merged_pointcloud": merged_cloud,
                    "filtered_pointcloud": filtered_cloud,
                    "traversability": traversability,
                    # This is the exact 120x12 input to Geometry V0 NMS. It is
                    # retained in memory only; the evaluation driver may
                    # opt-in to serialize selected decisions for diagnosis.
                    # Never expose this to policy or candidate construction.
                    "pre_nms_score": score,
                    "angle_indexes": angle_indexes,
                    "distance_indexes": distance_indexes,
                    "geometry_source": geometry.source,
                    "headings_degrees": geometry.headings_degrees,
                }
            )
            timings.append({
                'shared_visual_encoding': visual_encoding_ms,
                'depth_back_projection': pointcloud_ms,
                'mid360_like_filtering': filter_ms,
                'occupancy_inflation_traversability': occupancy_traversability_ms,
                'nms_candidate_construction': candidate_construction_ms,
            })
            if is_gmin:
                debug[-1].update({
                    'candidate_extraction': GMIN_EXTRACTION,
                    'observation_availability': availability[batch_index],
                    'admissibility_count': int(score.sum()),
                })
                timings[-1].pop('nms_candidate_construction')
                timings[-1].update({
                    'gmin_extraction': candidate_construction_ms,
                    'observation_availability': availability_ms,
                    'geometry_frontend': pointcloud_ms + filter_ms + occupancy_traversability_ms,
                })
        output = construct_waypoint_output(
            encoded=encoded,
            batch_angle_indexes=batch_angle_indexes,
            batch_distance_indexes=batch_distance_indexes,
            pano_angle_features=self.pano_angle_features,
            pano_image_indexes=self.pano_image_indexes,
        )
        self.last_debug = debug
        self.last_timings = timings
        return output


def default_mid360_config_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "configs",
        "mid360_waypoint.yaml",
    )
