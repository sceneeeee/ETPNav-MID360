"""Point-cloud source contract for geometry waypoint proposal.

The frozen V0 implementation reconstructs this cloud from the legacy depth
panorama.  V1 simulators and the robot will implement the same contract from
a native range source instead.
"""
from abc import ABCMeta, abstractmethod

import numpy as np

from vlnce_baselines.perception.depth_to_pointcloud import (
    depth_panorama_to_pointcloud,
)

PANORAMA_VIEW_COUNT = 12


class GeometryFrame(object):
    """One base-frame point cloud and minimal provenance for a policy batch."""

    def __init__(self, pointcloud, source, headings_degrees=None):
        points = np.asarray(pointcloud, dtype=np.float32)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("geometry pointcloud must have shape Nx3")
        if not np.all(np.isfinite(points)):
            raise ValueError("geometry pointcloud must contain only finite points")
        self.pointcloud = np.ascontiguousarray(points)
        self.source = str(source)
        self.headings_degrees = headings_degrees


class GeometryProvider(object, metaclass=ABCMeta):
    """Supply a base-frame point cloud for one batch item.

    Implementations own acquisition, calibration, and world-to-base transforms.
    The waypoint provider only filters and scores the returned Nx3 point cloud.
    """

    @abstractmethod
    def get_geometry(self, observations, batch_index):
        """Return a :class:`GeometryFrame` for ``batch_index``."""
        raise NotImplementedError


class DepthReconstructedGeometryProvider(GeometryProvider):
    """V0 geometry source: reconstruct a point cloud from 12 metric depths."""

    def __init__(self, config):
        self.config = config

    def get_geometry(self, observations, batch_index):
        depth_views = []
        headings = []
        for key, value in observations.items():
            if "depth" not in key:
                continue
            suffix = key[len("depth") :]
            headings.append(float(suffix[1:]) if suffix else 0.0)
            normalized = value[batch_index].detach().cpu().numpy()
            depth_range = (
                self.config.policy_depth_maximum_m
                - self.config.policy_depth_minimum_m
            )
            depth_views.append(
                normalized * depth_range + self.config.policy_depth_minimum_m
            )
        if len(depth_views) != PANORAMA_VIEW_COUNT:
            raise ValueError(
                "V0 depth reconstruction requires 12 depth views, received {}".format(
                    len(depth_views)
                )
            )
        pointcloud = depth_panorama_to_pointcloud(
            depth_views,
            headings_degrees=headings,
            horizontal_fov_degrees=self.config.horizontal_fov_degrees,
            camera_height_m=self.config.camera_height_m,
            minimum_depth_m=self.config.minimum_range_m,
            maximum_depth_m=self.config.maximum_range_m,
            pixel_stride=self.config.depth_pixel_stride,
        )
        return GeometryFrame(
            pointcloud=pointcloud,
            source="v0_depth_reconstruction",
            headings_degrees=headings,
        )
