import math

import numpy as np


class TraversabilityConfig(object):
    def __init__(
        self,
        angular_bins,
        distance_bins,
        distance_step_m,
        robot_radius_m,
        clearance_margin_m,
        obstacle_height_min_m,
        obstacle_height_max_m,
        grid_resolution_m,
    ):
        self.angular_bins = angular_bins
        self.distance_bins = distance_bins
        self.distance_step_m = distance_step_m
        self.robot_radius_m = robot_radius_m
        self.clearance_margin_m = clearance_margin_m
        self.obstacle_height_min_m = obstacle_height_min_m
        self.obstacle_height_max_m = obstacle_height_max_m
        self.grid_resolution_m = grid_resolution_m
        if angular_bins != 120 or distance_bins != 12:
            raise ValueError("ETP geometry contract requires a 120x12 score map")
        if not math.isclose(distance_step_m, 0.25):
            raise ValueError("ETP geometry contract requires 0.25 m distance bins")
        if robot_radius_m <= 0:
            raise ValueError("robot_radius_m is required and must be positive")
        if clearance_margin_m < 0:
            raise ValueError("clearance_margin_m must be non-negative")
        if grid_resolution_m <= 0:
            raise ValueError("grid_resolution_m must be positive")
        if obstacle_height_max_m < obstacle_height_min_m:
            raise ValueError("obstacle height bounds are invalid")

    @property
    def maximum_distance_m(self):
        return self.distance_bins * self.distance_step_m


class TraversabilityResult(object):
    def __init__(self, score, occupancy, inflated_occupancy, grid_resolution_m):
        self.score = score
        self.occupancy = occupancy
        self.inflated_occupancy = inflated_occupancy
        self.grid_resolution_m = grid_resolution_m


def _inflate_occupancy(occupancy, radius_cells):
    if radius_cells <= 0 or not np.any(occupancy):
        return occupancy.copy()
    inflated = np.zeros_like(occupancy)
    occupied_rows, occupied_columns = np.nonzero(occupancy)
    size = occupancy.shape[0]
    for row_offset in range(-radius_cells, radius_cells + 1):
        for column_offset in range(-radius_cells, radius_cells + 1):
            if row_offset ** 2 + column_offset ** 2 > radius_cells ** 2:
                continue
            rows = occupied_rows + row_offset
            columns = occupied_columns + column_offset
            valid = (
                (rows >= 0) & (rows < size) & (columns >= 0) & (columns < size)
            )
            inflated[rows[valid], columns[valid]] = True
    return inflated


def _ray_is_free(occupancy, center, resolution, clockwise_angle, distance):
    sample_count = max(2, int(math.ceil(distance / (resolution * 0.5))) + 1)
    radial_samples = np.linspace(0.0, distance, sample_count)
    x = np.sin(clockwise_angle) * radial_samples
    z = -np.cos(clockwise_angle) * radial_samples
    columns = np.rint(x / resolution).astype(np.int64) + center
    rows = np.rint(z / resolution).astype(np.int64) + center
    inside = (
        (rows >= 0)
        & (rows < occupancy.shape[0])
        & (columns >= 0)
        & (columns < occupancy.shape[1])
    )
    return bool(np.all(inside) and not np.any(occupancy[rows, columns]))


def compute_local_traversability(points, config):
    inflation_m = config.robot_radius_m + config.clearance_margin_m
    extent_m = config.maximum_distance_m + inflation_m
    center = int(math.ceil(extent_m / config.grid_resolution_m))
    grid_size = center * 2 + 1
    occupancy = np.zeros((grid_size, grid_size), dtype=np.bool_)

    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if cloud.size:
        obstacle = cloud[
            (cloud[:, 1] >= config.obstacle_height_min_m)
            & (cloud[:, 1] <= config.obstacle_height_max_m)
        ]
        columns = np.rint(
            obstacle[:, 0] / config.grid_resolution_m
        ).astype(np.int64) + center
        rows = np.rint(obstacle[:, 2] / config.grid_resolution_m).astype(
            np.int64
        ) + center
        valid = (
            (rows >= 0)
            & (rows < grid_size)
            & (columns >= 0)
            & (columns < grid_size)
        )
        occupancy[rows[valid], columns[valid]] = True

    radius_cells = int(math.ceil(inflation_m / config.grid_resolution_m))
    inflated = _inflate_occupancy(occupancy, radius_cells)
    score = np.zeros(
        (config.angular_bins, config.distance_bins), dtype=np.float32
    )
    for angle_index in range(config.angular_bins):
        angle = angle_index / config.angular_bins * 2.0 * math.pi
        blocked = False
        for distance_index in range(config.distance_bins):
            distance = (distance_index + 1) * config.distance_step_m
            if blocked or not _ray_is_free(
                inflated,
                center,
                config.grid_resolution_m,
                angle,
                distance,
            ):
                blocked = True
                continue
            score[angle_index, distance_index] = (
                distance / config.maximum_distance_m
            )
    return TraversabilityResult(
        score=score,
        occupancy=occupancy,
        inflated_occupancy=inflated,
        grid_resolution_m=config.grid_resolution_m,
    )
