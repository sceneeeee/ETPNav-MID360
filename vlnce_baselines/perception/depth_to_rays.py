"""Calibrated raw-depth adapter for the opt-in Geometry V1 provider.

Habitat's depth image stores camera-axis (z) depth, not Euclidean ray range.
This adapter deliberately runs on the normalized panorama tensors before the
legacy Nx3 pipeline discards ray status and provenance.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from vlnce_baselines.perception.ray_carving_representation import RayObservation


PANORAMA_HEADINGS_DEGREES = tuple(range(0, 360, 30))


@dataclass(frozen=True)
class RayAdapterResult:
    rays: tuple
    sampled_count: int
    hit_count: int
    censored_count: int
    invalid_count: int
    outside_vertical_fov_count: int
    input_shapes: tuple


def _depth_key(heading_degrees):
    return "depth" if heading_degrees == 0 else "depth_{}".format(heading_degrees)


def _depth_image(value, batch_index):
    if torch.is_tensor(value):
        value = value[batch_index].detach().cpu().numpy()
    else:
        value = np.asarray(value)[batch_index]
    image = np.asarray(value, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError("depth must have batched BxHxW or BxHxWx1 shape")
    return image


def _classify_ray(normalized_depth, euclidean_range_m,
                  vertical_angle_degrees, config):
    if not np.isfinite(normalized_depth) or normalized_depth <= 0.0:
        return "invalid", 0.0
    if (vertical_angle_degrees < config.vertical_fov_min_degrees or
            vertical_angle_degrees > config.vertical_fov_max_degrees):
        return "outside_vertical_fov", euclidean_range_m
    if euclidean_range_m < config.minimum_range_m:
        return "invalid", euclidean_range_m
    # NORMALIZE_DEPTH maps the renderer maximum exactly to one.  A physical
    # surface at that exact z-depth is indistinguishable from clipping, so V1
    # conservatively records only a clear usable-range prefix.
    renderer_censored = normalized_depth >= 1.0
    if renderer_censored or euclidean_range_m > config.maximum_range_m:
        return "censored", float(config.maximum_range_m)
    return "hit", euclidean_range_m


def depth_panorama_to_rays(observations, batch_index, config):
    """Convert twelve normalized Habitat z-depth views into calibrated rays.

    Rays are deterministically reduced to the inherited MID360-like azimuth /
    elevation lattice.  A nearest in-range hit wins a cell; otherwise a
    range-censored ray preserves the certified free prefix.  Invalid and
    out-of-vertical-FoV samples are counted but never carve evidence.
    """
    missing = [_depth_key(h) for h in PANORAMA_HEADINGS_DEGREES
               if _depth_key(h) not in observations]
    if missing:
        raise ValueError("V1 ray adapter is missing depth views: {}".format(missing))
    depth_range_m = (
        config.policy_depth_maximum_m - config.policy_depth_minimum_m
    )
    if depth_range_m <= 0.0:
        raise ValueError("policy depth range must be positive")

    sampled_count = invalid_count = outside_count = 0
    input_shapes = []
    cells = {}
    for view_index, heading_degrees in enumerate(PANORAMA_HEADINGS_DEGREES):
        image = _depth_image(observations[_depth_key(heading_degrees)], batch_index)
        input_shapes.append(tuple(image.shape))
        height, width = image.shape
        focal = width / (
            2.0 * math.tan(math.radians(config.horizontal_fov_degrees) / 2.0)
        )
        center_x = (width - 1.0) / 2.0
        center_y = (height - 1.0) / 2.0
        heading = math.radians(heading_degrees)
        cosine, sine = math.cos(heading), math.sin(heading)

        for row in range(0, height, config.depth_pixel_stride):
            for column in range(0, width, config.depth_pixel_stride):
                sampled_count += 1
                normalized = float(image[row, column])
                camera_direction = np.asarray([
                    (column - center_x) / focal,
                    -(row - center_y) / focal,
                    -1.0,
                ], dtype=np.float64)
                ray_norm = float(np.linalg.norm(camera_direction))
                camera_direction /= ray_norm
                direction = np.asarray([
                    cosine * camera_direction[0] + sine * camera_direction[2],
                    camera_direction[1],
                    -sine * camera_direction[0] + cosine * camera_direction[2],
                ], dtype=np.float64)
                vertical_angle = math.degrees(math.atan2(
                    direction[1], math.hypot(direction[0], direction[2])
                ))
                metric_z = normalized * depth_range_m + config.policy_depth_minimum_m
                euclidean_range = metric_z * ray_norm
                status, retained_range = _classify_ray(
                    normalized, euclidean_range, vertical_angle, config
                )
                if status == "invalid":
                    invalid_count += 1
                    continue
                if status == "outside_vertical_fov":
                    outside_count += 1
                    continue

                azimuth = (math.degrees(math.atan2(
                    direction[0], -direction[2]
                )) + 360.0) % 360.0
                angular_cell = (
                    int(math.floor(azimuth / config.azimuth_bin_degrees)),
                    int(math.floor(
                        (vertical_angle - config.vertical_fov_min_degrees) /
                        config.elevation_bin_degrees
                    )),
                )
                ray = RayObservation(
                    origin_xyz=(0.0, float(config.camera_height_m), 0.0),
                    direction_xyz=tuple(direction.tolist()),
                    range_m=float(retained_range),
                    kind=status,
                    view_index=view_index,
                    pixel_row=row,
                    pixel_column=column,
                    heading_degrees=float(heading_degrees),
                    vertical_angle_degrees=float(vertical_angle),
                    measured_z_depth_m=float(metric_z),
                )
                # Deterministic nearest-first selection matches the existing
                # angular point filter for physical hits.  Provenance resolves
                # exact overlaps between adjacent panorama views.
                key = (ray.range_m, 0 if ray.kind == "hit" else 1,
                       ray.view_index, ray.pixel_row, ray.pixel_column)
                previous = cells.get(angular_cell)
                if previous is None or key < previous[0]:
                    cells[angular_cell] = (key, ray)

    rays = tuple(value[1] for _, value in sorted(cells.items()))
    hit_count = sum(ray.kind == "hit" for ray in rays)
    censored_count = sum(ray.kind == "censored" for ray in rays)
    return RayAdapterResult(
        rays=rays,
        sampled_count=sampled_count,
        hit_count=hit_count,
        censored_count=censored_count,
        invalid_count=invalid_count,
        outside_vertical_fov_count=outside_count,
        input_shapes=tuple(input_shapes),
    )
