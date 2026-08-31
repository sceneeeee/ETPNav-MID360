import math

import numpy as np


def _depth_image(depth):
    array = np.asarray(depth, dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("depth must have shape HxW or HxWx1")
    return array


def backproject_depth(
    depth,
    horizontal_fov_degrees,
    camera_height_m,
    heading_degrees=0.0,
    minimum_depth_m=0.0,
    maximum_depth_m=float("inf"),
    pixel_stride=1,
):
    """Back-project metric z-depth into Habitat's agent/base coordinate frame.

    Habitat cameras look along ``-Z`` with ``+Y`` up. Positive sensor yaw is a
    left/counter-clockwise rotation when viewed from above.
    """
    image = _depth_image(depth)
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be positive")
    height, width = image.shape
    focal = width / (
        2.0 * math.tan(math.radians(horizontal_fov_degrees) / 2.0)
    )
    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    rows, columns = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
    sampled_depth = image[::pixel_stride, ::pixel_stride]
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth > minimum_depth_m)
        & (sampled_depth <= maximum_depth_m)
    )
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    distance = sampled_depth[valid]
    x = (columns[valid].astype(np.float32) - center_x) * distance / focal
    y = (
        -(rows[valid].astype(np.float32) - center_y) * distance / focal
        + camera_height_m
    )
    z = -distance

    heading = math.radians(heading_degrees)
    cosine = math.cos(heading)
    sine = math.sin(heading)
    rotated_x = cosine * x + sine * z
    rotated_z = -sine * x + cosine * z
    return np.stack((rotated_x, y, rotated_z), axis=1).astype(
        np.float32, copy=False
    )


def depth_panorama_to_pointcloud(
    depth_views,
    headings_degrees,
    horizontal_fov_degrees,
    camera_height_m,
    minimum_depth_m=0.0,
    maximum_depth_m=float("inf"),
    pixel_stride=1,
):
    if len(depth_views) != len(headings_degrees):
        raise ValueError("depth view and heading counts must match")
    clouds = [
        backproject_depth(
            depth,
            horizontal_fov_degrees=horizontal_fov_degrees,
            camera_height_m=camera_height_m,
            heading_degrees=heading,
            minimum_depth_m=minimum_depth_m,
            maximum_depth_m=maximum_depth_m,
            pixel_stride=pixel_stride,
        )
        for depth, heading in zip(depth_views, headings_degrees)
    ]
    nonempty = [cloud for cloud in clouds if cloud.size]
    if not nonempty:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(nonempty, axis=0)
