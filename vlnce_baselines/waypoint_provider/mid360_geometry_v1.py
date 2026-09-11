"""Opt-in end-to-end Geometry V1 waypoint provider.

V1 starts at normalized raw depth so ray identity, censoring, and view
provenance survive.  It intentionally does not construct a dense radial score
or call legacy NMS.
"""
import math
import os
import time

import numpy as np
import torch

from vlnce_baselines.perception.depth_to_rays import depth_panorama_to_rays
from vlnce_baselines.perception.geometry_v1_candidates import (
    extract_geometry_v1_candidates,
)
from vlnce_baselines.perception.ray_carving_representation import RayCarvingGrid
from vlnce_baselines.waypoint_provider.base import WaypointProvider
from vlnce_baselines.waypoint_provider.mid360_geometry import MID360GeometryConfig
from vlnce_baselines.waypoint_provider.original_etp import (
    construct_waypoint_output,
    encode_panorama,
)


GEOMETRY_V1_MODE = "mid360_geometry_v1"
GEOMETRY_V1_DESIGN_ID = "height-aware-ray-grid/hybrid-extraction-v1"


class MID360GeometryV1Config(MID360GeometryConfig):
    """Frozen V1 configuration using the inherited physical geometry values."""

    def __init__(self, values):
        if "radial_utility" in values:
            raise ValueError("Geometry V1 does not accept a radial utility")
        super().__init__(values)

    def grid_height_bounds_m(self):
        lower = self.camera_height_m + self.maximum_range_m * math.sin(
            math.radians(self.vertical_fov_min_degrees)
        )
        upper = self.camera_height_m + self.maximum_range_m * math.sin(
            math.radians(self.vertical_fov_max_degrees)
        )
        resolution = self.grid_resolution_m
        return (
            math.floor(min(0.0, lower) / resolution) * resolution,
            math.ceil(max(self.camera_height_m, upper) / resolution) * resolution,
        )


class MID360GeometryV1WaypointProvider(WaypointProvider):
    """Generate ETP candidates from frozen V1 evidence and extraction."""

    def __init__(self, rgb_encoder, depth_encoder, rgb_pool, depth_pool,
                 pano_angle_features, pano_image_indexes, config):
        self.rgb_encoder = rgb_encoder
        self.depth_encoder = depth_encoder
        self.rgb_pool = rgb_pool
        self.depth_pool = depth_pool
        self.pano_angle_features = pano_angle_features
        self.pano_image_indexes = pano_image_indexes
        self.config = config
        if not isinstance(config, MID360GeometryV1Config):
            raise TypeError("config must be MID360GeometryV1Config")
        self.last_debug = []
        self.last_timings = []

    def _grid(self):
        y_min, y_max = self.config.grid_height_bounds_m()
        return RayCarvingGrid(
            extent_xy_m=self.config.maximum_range_m,
            y_min_m=y_min,
            y_max_m=y_max,
            resolution_m=self.config.grid_resolution_m,
            obstacle_height_min_m=self.config.obstacle_height_min_m,
            obstacle_height_max_m=self.config.obstacle_height_max_m,
        )

    def predict(self, observations, in_train=False):
        if in_train:
            raise ValueError("MID360GeometryV1WaypointProvider is evaluation-only")
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
            ray_result = depth_panorama_to_rays(
                observations, batch_index, self.config
            )
            raw_ray_generation_ms = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            grid = self._grid()
            grid.integrate_many(ray_result.rays)
            safety_invalid = grid.safety_invalid(
                self.config.robot_radius_m + self.config.clearance_margin_m
            )
            ray_carving_ms = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            candidates, polar = extract_geometry_v1_candidates(
                grid,
                safety_invalid=safety_invalid,
                maximum_candidates=self.config.maximum_candidates,
            )
            candidate_extraction_ms = (time.perf_counter() - started) * 1000.0
            angle_indexes = np.asarray(
                [candidate.angle_index for candidate in candidates],
                dtype=np.int64,
            )
            distance_indexes = np.asarray(
                [candidate.distance_index for candidate in candidates],
                dtype=np.int64,
            )
            batch_angle_indexes.append(torch.from_numpy(angle_indexes))
            batch_distance_indexes.append(torch.from_numpy(distance_indexes))
            debug.append({
                "design_id": GEOMETRY_V1_DESIGN_ID,
                "ray_result": ray_result,
                "grid": grid,
                "safety_invalid": safety_invalid,
                "polar_evidence": polar,
                "candidates": candidates,
                "candidate_kinds": [candidate.kind for candidate in candidates],
            })
            timings.append({
                "shared_visual_encoding": visual_encoding_ms,
                "raw_ray_generation": raw_ray_generation_ms,
                "ray_carving_representation": ray_carving_ms,
                "candidate_extraction": candidate_extraction_ms,
            })

        started = time.perf_counter()
        output = construct_waypoint_output(
            encoded=encoded,
            batch_angle_indexes=batch_angle_indexes,
            batch_distance_indexes=batch_distance_indexes,
            pano_angle_features=self.pano_angle_features,
            pano_image_indexes=self.pano_image_indexes,
        )
        feature_mapping_ms = (time.perf_counter() - started) * 1000.0
        for record in timings:
            record["feature_mapping"] = feature_mapping_ms
        self.last_debug = debug
        self.last_timings = timings
        return output


def default_mid360_v1_config_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "configs", "mid360_waypoint_v1.yaml",
    )
