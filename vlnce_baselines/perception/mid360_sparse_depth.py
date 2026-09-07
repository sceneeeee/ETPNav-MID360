"""Deterministic MID360-like sparse depth, not Livox scan physics.

The range control is camera z-depth, matching Habitat's policy support. The
shared filter uses Euclidean range, so its enclosing sphere is derived from
that z limit and the camera corners; it must not cut off off-axis 0--10 m data.
"""
import math

import numpy as np
import torch
import yaml

from vlnce_baselines.perception.depth_to_pointcloud import depth_panorama_to_pointcloud
from vlnce_baselines.perception.mid360_like_filter import MID360LikeFilter

DEPTH_KEYS = tuple("depth" if a == 0 else "depth_{}".format(a) for a in range(0, 360, 30))


class MID360SparseDepthConfig(object):
    DEFAULTS = dict(
        camera_height_m=1.25, horizontal_fov_degrees=90.0,
        policy_depth_minimum_m=0.0, policy_depth_maximum_m=10.0,
        source_pixel_stride=1, minimum_range_m=0.0, maximum_range_m=10.0,
        vertical_fov_min_degrees=-90.0, vertical_fov_max_degrees=90.0,
        azimuth_bin_degrees=0.5, elevation_bin_degrees=0.5,
        voxel_size_m=0.0, no_return_fill_normalized=1.0,
    )

    def __init__(self, **values):
        unknown = set(values) - set(self.DEFAULTS)
        if unknown:
            raise ValueError("unknown sparse-depth config fields: {}".format(sorted(unknown)))
        self.__dict__.update(self.DEFAULTS)
        self.__dict__.update(values)
        if not all(np.isfinite(v) for v in self.__dict__.values()):
            raise ValueError("sparse-depth config must be finite")
        if not 0 <= self.policy_depth_minimum_m < self.policy_depth_maximum_m:
            raise ValueError("invalid policy depth range")
        if (self.minimum_range_m != self.policy_depth_minimum_m or
                self.maximum_range_m != self.policy_depth_maximum_m):
            raise ValueError("zero-shot control must preserve the policy z-depth range")
        if not 0 < self.horizontal_fov_degrees < 180:
            raise ValueError("invalid HFOV")
        if not -90 <= self.vertical_fov_min_degrees < self.vertical_fov_max_degrees <= 90:
            raise ValueError("invalid elevation support")
        if self.azimuth_bin_degrees <= 0 or self.elevation_bin_degrees <= 0:
            raise ValueError("angular cells must be positive")
        if self.source_pixel_stride < 1 or int(self.source_pixel_stride) != self.source_pixel_stride:
            raise ValueError("source_pixel_stride must be a positive integer")
        self.source_pixel_stride = int(self.source_pixel_stride)
        if self.voxel_size_m != 0:
            raise ValueError("voxel averaging synthesizes returns; use voxel_size_m=0 for this control")
        if not 0 <= self.no_return_fill_normalized <= 1:
            raise ValueError("normalized fill must be in [0, 1]")

    @classmethod
    def from_yaml(cls, path):
        with open(path) as handle:
            return cls(**yaml.safe_load(handle))

    def validate_sensor(self, sensor):
        if not sensor.NORMALIZE_DEPTH:
            raise ValueError("sparse-depth control requires normalized Habitat Depth")
        expected = (self.policy_depth_minimum_m, self.policy_depth_maximum_m,
                    self.horizontal_fov_degrees)
        if tuple(float(v) for v in (sensor.MIN_DEPTH, sensor.MAX_DEPTH, sensor.HFOV)) != expected:
            raise ValueError("sparse-depth config does not match Habitat depth calibration")
        if not np.allclose(sensor.POSITION, [0, self.camera_height_m, 0], atol=1e-7):
            raise ValueError("sparse-depth camera translation does not match Habitat")

    def validate_runtime(self, runtime):
        """Reject camera/transform changes that invalidate the inverse projection."""
        simulator = runtime.TASK_CONFIG.SIMULATOR
        transforms = runtime.RL.POLICY.OBS_TRANSFORMS
        if tuple(transforms.ENABLED_TRANSFORMS) != ('CenterCropperPerSensor',):
            raise ValueError("sparse-depth control requires the audited observation transforms")
        crops = dict(transforms.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS)
        for key, heading in zip(DEPTH_KEYS, range(0, 360, 30)):
            sensor = getattr(simulator, 'DEPTH_SENSOR' if heading == 0 else key.upper())
            self.validate_sensor(sensor)
            if not np.allclose(sensor.ORIENTATION, [0, math.radians(heading), 0], atol=1e-7):
                raise ValueError("sparse-depth camera yaw/pitch/roll mismatch")
            if tuple(crops[key]) != (sensor.HEIGHT, sensor.WIDTH):
                raise ValueError("depth cropping would change the audited camera intrinsics")


def project_pointcloud(points, height, width, heading_degrees, config, output_dtype=np.float32):
    """Inverse of backproject_depth; return normalized HxWx1 and a hit mask.

    Optical center: ((W-1)/2, (H-1)/2); fx=fy=W/(2*tan(HFOV/2)).
    Undo positive Habitat yaw, remove camera height, then use z=-camera_z.
    Rasterize to one nearest pixel only. Z-buffer compares camera z, not range.
    """
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
    angle = math.radians(heading_degrees)
    c, s = math.cos(angle), math.sin(angle)
    x = c * cloud[:, 0] - s * cloud[:, 2]
    y = cloud[:, 1] - config.camera_height_m
    z = -(s * cloud[:, 0] + c * cloud[:, 2])
    good = (z > config.minimum_range_m) & (z <= config.maximum_range_m)
    x, y, z = x[good], y[good], z[good]
    focal = width / (2 * math.tan(math.radians(config.horizontal_fov_degrees) / 2))
    u = focal * x / z + (width - 1) / 2
    v = -focal * y / z + (height - 1) / 2
    inside = (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)
    columns = np.floor(u[inside] + 0.5).astype(np.int64)
    rows = np.floor(v[inside] + 0.5).astype(np.int64)
    buffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(buffer, rows * width + columns, z[inside])
    valid = np.isfinite(buffer)
    normalized = np.full(height * width, config.no_return_fill_normalized, dtype=output_dtype)
    normalized[valid] = ((buffer[valid] - config.policy_depth_minimum_m) /
                         (config.policy_depth_maximum_m - config.policy_depth_minimum_m))
    return normalized.reshape(height, width, 1), valid.reshape(height, width, 1)


def depth_statistics(values):
    finite = np.isfinite(values)
    selected = values[finite]
    return dict(min=float(selected.min()) if selected.size else None,
                max=float(selected.max()) if selected.size else None,
                mean=float(selected.mean()) if selected.size else None,
                finite_fraction=float(finite.mean()))


class MID360SparseDepthAdapter(object):
    def __init__(self, config):
        self.config = config
        self.first_frame_statistics = None

    def __call__(self, observations):
        keys = [key for key in observations if "depth" in key]
        # Original packs by insertion order; reject ambiguous ordering/cached features.
        if tuple(keys) != DEPTH_KEYS:
            raise ValueError("expected depth views ordered 0,30,...,330 without cached depth features")
        reference = observations['depth']
        if reference.ndim != 4 or reference.shape[-1] != 1 or not reference.is_floating_point():
            raise ValueError("Depth must be floating BxHxWx1")
        for key in keys:
            value = observations[key]
            if (value.shape != reference.shape or value.dtype != reference.dtype or
                    value.device != reference.device):
                raise ValueError("all depth views must have identical shape, dtype and device")
        batch_size, height, width, _ = reference.shape
        cfg = self.config
        adapted = observations.copy()
        for key in keys:
            adapted[key] = torch.empty_like(observations[key])
        diagnostics = []
        tangent = math.tan(math.radians(cfg.horizontal_fov_degrees) / 2)
        enclosing_range = cfg.maximum_range_m * math.sqrt(1 + tangent ** 2 + (height / width * tangent) ** 2)
        sampler = MID360LikeFilter(
            minimum_range_m=cfg.minimum_range_m, maximum_range_m=enclosing_range,
            vertical_fov_min_degrees=cfg.vertical_fov_min_degrees,
            vertical_fov_max_degrees=cfg.vertical_fov_max_degrees,
            azimuth_bin_degrees=cfg.azimuth_bin_degrees,
            elevation_bin_degrees=cfg.elevation_bin_degrees,
            voxel_size_m=cfg.voxel_size_m, sensor_height_m=cfg.camera_height_m)
        for b in range(batch_size):
            dense = np.stack([observations[k][b].detach().cpu().numpy() for k in keys])
            if np.any(np.isfinite(dense) & ((dense < 0) | (dense > 1))):
                raise ValueError("source depth is outside normalized [0,1]")
            metric = dense.astype(np.float32) * (cfg.policy_depth_maximum_m - cfg.policy_depth_minimum_m) + cfg.policy_depth_minimum_m
            # Saturated max depth is censored, not a known 3D surface.
            metric[(dense >= 1) | (dense <= 0)] = np.nan
            cloud = depth_panorama_to_pointcloud(
                metric, range(0, 360, 30), cfg.horizontal_fov_degrees, cfg.camera_height_m,
                minimum_depth_m=cfg.minimum_range_m, maximum_depth_m=cfg.maximum_range_m,
                pixel_stride=cfg.source_pixel_stride)
            filtered = sampler(cloud)
            counts, sparse_views = {}, []
            for key, heading in zip(keys, range(0, 360, 30)):
                sparse, mask = project_pointcloud(filtered, height, width, heading, cfg, output_dtype=dense.dtype)
                adapted[key][b] = torch.as_tensor(sparse, device=reference.device, dtype=reference.dtype)
                counts[key] = int(mask.sum())
                sparse_views.append(sparse)
            total = 12 * height * width
            count = sum(counts.values())
            record = dict(dense_cloud_points=int(len(cloud)), filtered_cloud_points=int(len(filtered)),
                          valid_projected_pixels=count, total_depth_pixels=total,
                          valid_projected_fraction=count / total,
                          valid_projected_pixels_per_view=counts,
                          no_return_fill_normalized=float(cfg.no_return_fill_normalized),
                          no_return_fraction=1 - count / total,
                          euclidean_filter_envelope_m=enclosing_range)
            if self.first_frame_statistics is None:
                sparse_stats = depth_statistics(np.stack(sparse_views))
                sparse_stats.update(no_return_fraction=1 - count / total, valid_return_fraction=count / total)
                self.first_frame_statistics = dict(dense_depth=depth_statistics(dense), sparse_depth=sparse_stats)
                record['first_frame_statistics'] = self.first_frame_statistics
            diagnostics.append(record)
        return adapted, diagnostics
