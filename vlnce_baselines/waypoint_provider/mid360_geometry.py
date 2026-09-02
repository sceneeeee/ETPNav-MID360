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
from vlnce_baselines.waypoint_pred.utils import nms
from vlnce_baselines.waypoint_provider.base import WaypointProvider
from vlnce_baselines.waypoint_provider.original_etp import (
    construct_waypoint_output,
    encode_panorama,
)


def select_geometry_candidates(score, max_candidates=5):
    """Apply the original ETP circular NMS contract to a geometry score."""
    score_tensor = torch.from_numpy(np.asarray(score, dtype=np.float32))
    if score_tensor.shape != (120, 12):
        raise ValueError("geometry score must have shape 120x12")
    wrapped = torch.cat(
        (score_tensor[-1:, :], score_tensor, score_tensor[:1, :]), dim=0
    )
    selected = nms(
        wrapped.unsqueeze(0).unsqueeze(0),
        max_predictions=max_candidates,
        sigma=(7.0, 5.0),
    ).squeeze(0).squeeze(0)[1:-1, :]
    indexes = selected.nonzero()
    return (
        indexes[:, 0].cpu().numpy().astype(np.int64),
        indexes[:, 1].cpu().numpy().astype(np.int64),
    )


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
        contracted_range = self.distance_bins * self.distance_step_m
        if abs(contracted_range - self.maximum_distance_m) > 1e-6:
            raise ValueError("distance bins and maximum distance disagree")
        if self.angular_bins != 120 or self.distance_bins != 12:
            raise ValueError("ETP geometry contract requires a 120x12 score map")
        if self.distance_step_m <= 0 or self.maximum_distance_m <= 0:
            raise ValueError("candidate distance parameters must be positive")
        if self.maximum_candidates < 1:
            raise ValueError("maximum_candidates must be positive")
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
        started = time.perf_counter()
        encoded = encode_panorama(
            observations=observations,
            rgb_encoder=self.rgb_encoder,
            depth_encoder=self.depth_encoder,
            rgb_pool=self.rgb_pool,
            depth_pool=self.depth_pool,
        )
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
            angle_indexes, distance_indexes = select_geometry_candidates(
                traversability.score,
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
