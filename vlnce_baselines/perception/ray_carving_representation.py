"""Isolated, CPU-only epistemic ray evidence for Geometry V1 study.

This module is deliberately not imported by an active waypoint provider.  It
operates on already-calibrated rays, preserving the fact that a return may be
a hit, range-censored, or invalid.  Its 2-D collision view is a conservative
projection of height-resolved voxel evidence; it is not an occupancy grid.
"""
from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np


class EpistemicState(IntEnum):
    UNKNOWN = 0
    OBSERVED_FREE = 1
    OCCUPIED = 2


@dataclass(frozen=True)
class RayObservation:
    """A single first-return ray in base coordinates.

    ``kind`` is one of ``hit``, ``censored``, or ``invalid``.  A censored ray
    has a known clear prefix ending at ``range_m`` but no physical endpoint.
    Invalid rays contribute no evidence.  Directions must be unit length.
    """
    origin_xyz: tuple
    direction_xyz: tuple
    range_m: float
    kind: str
    view_index: int = 0
    pixel_row: int = -1
    pixel_column: int = -1
    heading_degrees: float = 0.0
    vertical_angle_degrees: float = 0.0
    measured_z_depth_m: float = float("nan")


class RayCarvingGrid(object):
    """Local 3-D evidence grid with non-destructive free/hit/censor masks."""

    def __init__(self, extent_xy_m, y_min_m, y_max_m, resolution_m,
                 obstacle_height_min_m, obstacle_height_max_m):
        if extent_xy_m <= 0 or resolution_m <= 0 or y_max_m <= y_min_m:
            raise ValueError("invalid grid geometry")
        self.extent_xy_m = float(extent_xy_m)
        self.y_min_m, self.y_max_m = float(y_min_m), float(y_max_m)
        self.resolution_m = float(resolution_m)
        self.obstacle_height_min_m = float(obstacle_height_min_m)
        self.obstacle_height_max_m = float(obstacle_height_max_m)
        self.nxy = int(math.ceil(2.0 * extent_xy_m / resolution_m)) + 1
        self.ny = int(math.ceil((y_max_m - y_min_m) / resolution_m)) + 1
        shape = (self.ny, self.nxy, self.nxy)  # y, z(row), x(column)
        self.free_evidence = np.zeros(shape, dtype=np.bool_)
        self.occupied_evidence = np.zeros(shape, dtype=np.bool_)
        self.range_censored = np.zeros(shape, dtype=np.bool_)

    def _index(self, point):
        x, y, z = point
        return (int(np.rint((y - self.y_min_m) / self.resolution_m)),
                int(np.rint((z + self.extent_xy_m) / self.resolution_m)),
                int(np.rint((x + self.extent_xy_m) / self.resolution_m)))

    def _inside(self, index):
        y, z, x = index
        return 0 <= y < self.ny and 0 <= z < self.nxy and 0 <= x < self.nxy

    def integrate(self, ray):
        self.integrate_many((ray,))

    def integrate_many(self, rays):
        rays = tuple(rays)
        active = []
        for ray in rays:
            if ray.kind not in ("hit", "censored", "invalid"):
                raise ValueError("unknown ray kind")
            if ray.kind == "invalid":
                continue
            origin = np.asarray(ray.origin_xyz, dtype=np.float64)
            direction = np.asarray(ray.direction_xyz, dtype=np.float64)
            if origin.shape != (3,) or direction.shape != (3,) or ray.range_m <= 0:
                raise ValueError("invalid ray")
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or abs(norm - 1.0) > 1e-5:
                raise ValueError("ray direction must be unit length")
            active.append((ray, origin, direction))
        if not active:
            return

        origins = np.stack([item[1] for item in active], axis=0)
        directions = np.stack([item[2] for item in active], axis=0)
        ranges = np.asarray([item[0].range_m for item in active], dtype=np.float64)
        sample_counts = np.maximum(
            1, np.ceil(ranges / (self.resolution_m * 0.5)).astype(np.int64)
        )
        steps = np.arange(int(sample_counts.max()), dtype=np.float64)[None, :]
        valid_samples = steps < sample_counts[:, None]
        distances = steps * (ranges / sample_counts)[:, None]
        points = origins[:, None, :] + directions[:, None, :] * distances[:, :, None]

        iy = np.rint((points[:, :, 1] - self.y_min_m) /
                     self.resolution_m).astype(np.int64)
        iz = np.rint((points[:, :, 2] + self.extent_xy_m) /
                     self.resolution_m).astype(np.int64)
        ix = np.rint((points[:, :, 0] + self.extent_xy_m) /
                     self.resolution_m).astype(np.int64)
        endpoints = origins + directions * ranges[:, None]
        endpoint_y = np.rint((endpoints[:, 1] - self.y_min_m) /
                             self.resolution_m).astype(np.int64)
        endpoint_z = np.rint((endpoints[:, 2] + self.extent_xy_m) /
                             self.resolution_m).astype(np.int64)
        endpoint_x = np.rint((endpoints[:, 0] + self.extent_xy_m) /
                             self.resolution_m).astype(np.int64)

        inside = (valid_samples & (iy >= 0) & (iy < self.ny) &
                  (iz >= 0) & (iz < self.nxy) &
                  (ix >= 0) & (ix < self.nxy))
        not_endpoint = ~((iy == endpoint_y[:, None]) &
                         (iz == endpoint_z[:, None]) &
                         (ix == endpoint_x[:, None]))
        free = inside & not_endpoint
        self.free_evidence[iy[free], iz[free], ix[free]] = True

        endpoint_inside = ((endpoint_y >= 0) & (endpoint_y < self.ny) &
                           (endpoint_z >= 0) & (endpoint_z < self.nxy) &
                           (endpoint_x >= 0) & (endpoint_x < self.nxy))
        hit = endpoint_inside & np.asarray(
            [item[0].kind == "hit" for item in active], dtype=bool
        )
        censored = endpoint_inside & np.asarray(
            [item[0].kind == "censored" for item in active], dtype=bool
        )
        self.occupied_evidence[endpoint_y[hit], endpoint_z[hit], endpoint_x[hit]] = True
        self.range_censored[
            endpoint_y[censored], endpoint_z[censored], endpoint_x[censored]
        ] = True

    def _integrate_reference(self, ray):
        """Readable single-ray definition retained for semantic comparison."""
        if ray.kind not in ("hit", "censored", "invalid"):
            raise ValueError("unknown ray kind")
        if ray.kind == "invalid":
            return
        origin = np.asarray(ray.origin_xyz, dtype=np.float64)
        direction = np.asarray(ray.direction_xyz, dtype=np.float64)
        if origin.shape != (3,) or direction.shape != (3,) or ray.range_m <= 0:
            raise ValueError("invalid ray")
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or abs(norm - 1.0) > 1e-5:
            raise ValueError("ray direction must be unit length")
        # Half-cell sampling is deterministic and covers every crossed voxel;
        # the hit endpoint is excluded, preserving first-hit occlusion.
        samples = max(1, int(math.ceil(ray.range_m / (self.resolution_m * 0.5))))
        endpoint_index = self._index(origin + direction * ray.range_m)
        for distance in np.linspace(0.0, ray.range_m, samples, endpoint=False):
            index = self._index(origin + direction * distance)
            if self._inside(index) and index != endpoint_index:
                self.free_evidence[index] = True
        if self._inside(endpoint_index):
            if ray.kind == "hit":
                self.occupied_evidence[endpoint_index] = True
            else:
                self.range_censored[endpoint_index] = True

    def voxel_state(self):
        """Return a lossy display state; evidence masks retain conflicts."""
        state = np.full(self.free_evidence.shape, EpistemicState.UNKNOWN, dtype=np.uint8)
        state[self.free_evidence] = EpistemicState.OBSERVED_FREE
        state[self.occupied_evidence] = EpistemicState.OCCUPIED
        return state

    def collision_column_state(self):
        """Conservative 2-D state for the configured collision-height column.

        A cell is free only if every represented voxel intersecting the height
        band has direct free evidence and none has an occupied return.  A high
        or low ray therefore cannot certify a full robot-height column.
        """
        centers = self.y_min_m + np.arange(self.ny) * self.resolution_m
        levels = (centers >= self.obstacle_height_min_m) & (centers <= self.obstacle_height_max_m)
        if not np.any(levels):
            raise ValueError("collision height band has no voxel levels")
        free = np.all(self.free_evidence[levels], axis=0)
        occupied = np.any(self.occupied_evidence[levels], axis=0)
        result = np.full((self.nxy, self.nxy), EpistemicState.UNKNOWN, dtype=np.uint8)
        result[free & ~occupied] = EpistemicState.OBSERVED_FREE
        result[occupied] = EpistemicState.OCCUPIED
        return result

    def safety_invalid(self, radius_m=0.0):
        """Return a separate footprint mask; never changes evidence arrays."""
        occupied = self.collision_column_state() == EpistemicState.OCCUPIED
        cells = int(math.ceil(radius_m / self.resolution_m))
        if cells <= 0:
            return occupied.copy()
        result = np.zeros_like(occupied)
        rows, columns = np.nonzero(occupied)
        for dr in range(-cells, cells + 1):
            for dc in range(-cells, cells + 1):
                if dr * dr + dc * dc > cells * cells:
                    continue
                rr, cc = rows + dr, columns + dc
                valid = (rr >= 0) & (rr < self.nxy) & (cc >= 0) & (cc < self.nxy)
                result[rr[valid], cc[valid]] = True
        return result
