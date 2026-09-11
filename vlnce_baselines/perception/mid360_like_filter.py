import numpy as np


def voxel_downsample(points, voxel_size_m):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.size == 0 or voxel_size_m <= 0:
        return points.copy()
    voxel_indexes = np.floor(points / voxel_size_m).astype(np.int64)
    unique_voxels, inverse = np.unique(
        voxel_indexes, axis=0, return_inverse=True
    )
    sums = np.zeros((len(unique_voxels), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique_voxels)).astype(np.float64)
    for axis in range(3):
        sums[:, axis] = np.bincount(
            inverse,
            weights=points[:, axis].astype(np.float64),
            minlength=len(unique_voxels),
        )
    return (sums / counts[:, np.newaxis]).astype(np.float32)


def angular_downsample(
    points, sensor_height_m, azimuth_bin_degrees, elevation_bin_degrees
):
    """Keep the nearest deterministic return in each angular cell."""
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if cloud.size == 0:
        return cloud.copy()
    relative_y = cloud[:, 1] - sensor_height_m
    horizontal_range = np.hypot(cloud[:, 0], cloud[:, 2])
    range_3d = np.sqrt(horizontal_range ** 2 + relative_y ** 2)
    azimuth = (
        np.degrees(np.arctan2(cloud[:, 0], -cloud[:, 2])) + 360.0
    ) % 360.0
    elevation = np.degrees(np.arctan2(relative_y, horizontal_range))
    azimuth_index = np.floor(azimuth / azimuth_bin_degrees).astype(np.int64)
    elevation_index = np.floor(elevation / elevation_bin_degrees).astype(np.int64)
    order = np.lexsort(
        (
            cloud[:, 2],
            cloud[:, 1],
            cloud[:, 0],
            range_3d,
            elevation_index,
            azimuth_index,
        )
    )
    ordered_cells = np.stack(
        (azimuth_index[order], elevation_index[order]), axis=1
    )
    _, first_indexes = np.unique(ordered_cells, axis=0, return_index=True)
    return cloud[order[first_indexes]]


class MID360LikeFilter(object):
    """Deterministic engineering approximation, not Livox scan physics."""

    def __init__(
        self,
        minimum_range_m,
        maximum_range_m,
        vertical_fov_min_degrees,
        vertical_fov_max_degrees,
        azimuth_bin_degrees,
        elevation_bin_degrees,
        voxel_size_m,
        sensor_height_m=0.0,
    ):
        self.minimum_range_m = minimum_range_m
        self.maximum_range_m = maximum_range_m
        self.vertical_fov_min_degrees = vertical_fov_min_degrees
        self.vertical_fov_max_degrees = vertical_fov_max_degrees
        self.azimuth_bin_degrees = azimuth_bin_degrees
        self.elevation_bin_degrees = elevation_bin_degrees
        self.voxel_size_m = voxel_size_m
        self.sensor_height_m = sensor_height_m
        if azimuth_bin_degrees <= 0 or elevation_bin_degrees <= 0:
            raise ValueError("angular bin sizes must be positive")

    def __call__(self, points):
        cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if cloud.size == 0:
            return cloud.copy()
        relative_y = cloud[:, 1] - self.sensor_height_m
        horizontal_range = np.hypot(cloud[:, 0], cloud[:, 2])
        range_3d = np.sqrt(horizontal_range ** 2 + relative_y ** 2)
        vertical_angle = np.degrees(np.arctan2(relative_y, horizontal_range))
        valid = (
            np.all(np.isfinite(cloud), axis=1)
            & (range_3d >= self.minimum_range_m)
            & (range_3d <= self.maximum_range_m)
            & (vertical_angle >= self.vertical_fov_min_degrees)
            & (vertical_angle <= self.vertical_fov_max_degrees)
        )
        angular = angular_downsample(
            cloud[valid],
            sensor_height_m=self.sensor_height_m,
            azimuth_bin_degrees=self.azimuth_bin_degrees,
            elevation_bin_degrees=self.elevation_bin_degrees,
        )
        return voxel_downsample(angular, self.voxel_size_m)
