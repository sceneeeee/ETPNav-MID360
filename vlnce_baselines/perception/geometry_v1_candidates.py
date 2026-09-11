"""Inactive CPU-only Geometry V1 candidate-contract prototype.

This is deliberately *not* imported by an active waypoint provider.  It is a
small, explicit seam between epistemic ray evidence and ETP's existing
angle/distance candidate representation.  It does not manufacture a dense
proposal score and it never treats unknown or range-censored space as free.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from vlnce_baselines.perception.ray_carving_representation import EpistemicState


ETP_ANGLE_BINS = 120
ETP_RADIAL_BINS = 12
ETP_DISTANCE_STEP_M = 0.25
ETP_MAX_CANDIDATES = 5


@dataclass(frozen=True)
class GeometryV1Candidate:
    """A direct candidate before the unchanged ETP feature adapter.

    ``kind`` is deliberately visible for analysis.  It must not be passed to
    the navigation model: the existing provider adapter supplies only the
    unchanged angle/distance/feature contract.
    """
    angle_index: int
    distance_index: int
    kind: str  # ``clearance_ridge`` or ``progress_fallback``
    rank: float

    @property
    def distance_m(self):
        return (self.distance_index + 1) * ETP_DISTANCE_STEP_M


@dataclass(frozen=True)
class PolarEvidence:
    """Certified radial reachability sampled from collision-height evidence."""
    safe: np.ndarray
    clearance_m: np.ndarray
    resolution_m: float


def _circular_distance(a, b, n=ETP_ANGLE_BINS):
    delta = abs(a - b)
    return min(delta, n - delta)


def _collision_height_censor(grid):
    """Project censor evidence only for endpoint rejection, never occupancy."""
    centers = grid.y_min_m + np.arange(grid.ny) * grid.resolution_m
    levels = ((centers >= grid.obstacle_height_min_m) &
              (centers <= grid.obstacle_height_max_m))
    return np.any(grid.range_censored[levels], axis=0)


def _ray_indices(angle_index, distance_m, resolution_m, center):
    """Conservative raster cells along an ETP polar segment, including end."""
    angle = angle_index / float(ETP_ANGLE_BINS) * 2.0 * math.pi
    samples = max(1, int(math.ceil(distance_m / (resolution_m * 0.5))))
    result = []
    for distance in np.linspace(0.0, distance_m, samples + 1):
        row = int(round(-math.cos(angle) * distance / resolution_m)) + center
        col = int(round(math.sin(angle) * distance / resolution_m)) + center
        if not result or result[-1] != (row, col):
            result.append((row, col))
    return result


def polar_evidence_from_grid(grid, safety_invalid=None):
    """Return certified safe ETP bins and physical-hit clearance.

    A bin is safe iff every collision-height raster cell on its path is
    directly observed free, is not occupied/inflated, and its endpoint is not
    range-censored.  Thus UNKNOWN is a hard rejection rather than an implicit
    free-space prior.
    """
    state = grid.collision_column_state()
    invalid = (grid.safety_invalid() if safety_invalid is None else
               np.asarray(safety_invalid, dtype=bool))
    censor = _collision_height_censor(grid)
    if invalid.shape != state.shape or censor.shape != state.shape:
        raise ValueError("safety/censor shape must match collision grid")
    occupied = state == EpistemicState.OCCUPIED
    occupied_rows, _ = np.nonzero(occupied)
    if len(occupied_rows):
        clearance = distance_transform_edt(~occupied) * grid.resolution_m
    else:
        clearance = np.full(state.shape, np.inf, dtype=np.float64)

    safe = np.zeros((ETP_ANGLE_BINS, ETP_RADIAL_BINS), dtype=bool)
    clearance_samples = np.full_like(safe, np.nan, dtype=np.float64)
    center = state.shape[0] // 2
    for angle_index in range(ETP_ANGLE_BINS):
        for distance_index in range(ETP_RADIAL_BINS):
            cells = _ray_indices(
                angle_index,
                (distance_index + 1) * ETP_DISTANCE_STEP_M,
                grid.resolution_m,
                center,
            )
            if any(row < 0 or row >= state.shape[0] or col < 0 or col >= state.shape[1]
                   for row, col in cells):
                continue
            endpoint = cells[-1]
            traversable = all(
                state[row, col] == EpistemicState.OBSERVED_FREE and
                not invalid[row, col]
                for row, col in cells
            )
            if traversable and not censor[endpoint]:
                safe[angle_index, distance_index] = True
                clearance_samples[angle_index, distance_index] = clearance[endpoint]
    return PolarEvidence(
        safe=safe, clearance_m=clearance_samples,
        resolution_m=float(grid.resolution_m),
    )


def _structural_ridges(polar):
    """Strict, finite radial clearance maxima: no plateau or range edge."""
    result = []
    for angle_index in range(ETP_ANGLE_BINS):
        for distance_index in range(1, ETP_RADIAL_BINS - 1):
            if not polar.safe[angle_index, distance_index]:
                continue
            here = polar.clearance_m[angle_index, distance_index]
            before = polar.clearance_m[angle_index, distance_index - 1]
            after = polar.clearance_m[angle_index, distance_index + 1]
            if not (np.isfinite(here) and np.isfinite(before) and np.isfinite(after)):
                continue
            prominence = min(here - before, here - after)
            # A change smaller than one collision-grid cell is rasterization
            # noise, not a stable physical ridge.
            if prominence >= polar.resolution_m:
                result.append(GeometryV1Candidate(
                    angle_index, distance_index, "clearance_ridge", prominence
                ))
    return result


def _bounded_components(flags):
    """Return non-circular runs (start/end inclusive) of true angular bins."""
    flags = np.asarray(flags, dtype=bool)
    if flags.shape != (ETP_ANGLE_BINS,):
        raise ValueError("expected 120 angular flags")
    if not np.any(flags) or np.all(flags):
        return []
    # Rotate at a false cell so runs cannot wrap ambiguously.
    cut = int(np.flatnonzero(~flags)[0])
    rotated = np.roll(flags, -cut - 1)
    components, start = [], None
    for index, flag in enumerate(rotated):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            components.append(((start + cut + 1) % ETP_ANGLE_BINS,
                               (index - 1 + cut + 1) % ETP_ANGLE_BINS,
                               index - start))
            start = None
    if start is not None:
        components.append(((start + cut + 1) % ETP_ANGLE_BINS,
                           (ETP_ANGLE_BINS - 1 + cut + 1) % ETP_ANGLE_BINS,
                           ETP_ANGLE_BINS - start))
    return components


def _progress_fallbacks(polar):
    """Coverage policy for bounded observed-free angular components only.

    A full 360-degree component intentionally returns no arbitrary headings:
    no finite deterministic set can be rotation-equivariant in perfectly open
    rotationally symmetric evidence.  This is policy, not geometric salience.
    """
    terminal = np.full(ETP_ANGLE_BINS, -1, dtype=np.int64)
    for angle_index in range(ETP_ANGLE_BINS):
        good = np.flatnonzero(polar.safe[angle_index])
        if len(good):
            terminal[angle_index] = int(good[-1])
    # A direction exists at its own maximal certified extent.  Components at
    # the global outer ring are what coverage can honestly propose.
    max_terminal = int(np.max(terminal))
    if max_terminal < 0:
        return []
    flags = terminal == max_terminal
    candidates = []
    for start, end, width in _bounded_components(flags):
        # A discrete even-width component has two equally central headings.
        # Keep both rather than break rotational equivalence by index order.
        centers = [(start + (width - 1) // 2) % ETP_ANGLE_BINS]
        if width % 2 == 0:
            centers.append((start + width // 2) % ETP_ANGLE_BINS)
        for angle_index in centers:
            candidates.append(GeometryV1Candidate(
                angle_index, int(terminal[angle_index]), "progress_fallback",
                float(terminal[angle_index] + 1),
            ))
    return candidates


def _select(candidates, maximum_candidates):
    """Rank direct structures without legacy radial NMS.

    Radially distinct ridges at the same heading are allowed.  Near-identical
    angular copies of a *strictly stronger* feature are deduplicated at the
    existing five-bin angular resolution.  An equal-rank orbit is admitted as
    a whole or skipped when it would cross the cap: selecting a coordinate
    prefix from a symmetric orbit would violate rotation consistency.
    """
    by_kind = {"clearance_ridge": 0, "progress_fallback": 1}
    ordered = sorted(candidates, key=lambda c: (by_kind[c.kind], -c.rank,
                                                 c.distance_index, c.angle_index))
    selected = []
    cursor = 0
    while cursor < len(ordered):
        first = ordered[cursor]
        end = cursor + 1
        while (end < len(ordered) and ordered[end].kind == first.kind and
               np.isclose(ordered[end].rank, first.rank, rtol=0.0, atol=1e-12)):
            end += 1
        orbit = []
        for candidate in ordered[cursor:end]:
            duplicate = any(
                old.kind == candidate.kind and
                abs(old.distance_index - candidate.distance_index) <= 1 and
                _circular_distance(old.angle_index, candidate.angle_index) <= 5
                for old in selected
            )
            if not duplicate:
                orbit.append(candidate)
        if len(selected) + len(orbit) <= maximum_candidates:
            selected.extend(orbit)
        cursor = end
    return tuple(selected)


def extract_geometry_v1_candidates(grid, safety_invalid=None,
                                   maximum_candidates=ETP_MAX_CANDIDATES):
    """Extract direct structural candidates, then use policy only if needed."""
    if not 1 <= maximum_candidates <= ETP_MAX_CANDIDATES:
        raise ValueError("maximum candidates must be in [1, 5]")
    polar = polar_evidence_from_grid(grid, safety_invalid)
    structural = _structural_ridges(polar)
    if structural:
        return _select(structural, maximum_candidates), polar
    return _select(_progress_fallbacks(polar), maximum_candidates), polar


def extract_geometry_v1_structural_candidates(grid, safety_invalid=None,
                                              maximum_candidates=ETP_MAX_CANDIDATES):
    """The structural half of V1, intentionally without any progress policy."""
    if not 1 <= maximum_candidates <= ETP_MAX_CANDIDATES:
        raise ValueError("maximum candidates must be in [1, 5]")
    polar = polar_evidence_from_grid(grid, safety_invalid)
    return _select(_structural_ridges(polar), maximum_candidates), polar
