"""Waypoint proposal adapters that preserve the original ETP feature contract."""

from vlnce_baselines.waypoint_provider.base import WaypointProvider
from vlnce_baselines.waypoint_provider.mid360_geometry import (
    MID360GeometryConfig,
    MID360GeometryWaypointProvider,
)
from vlnce_baselines.waypoint_provider.original_etp import (
    OriginalETPWaypointProvider,
)
from vlnce_baselines.waypoint_provider.types import WaypointOutput

__all__ = [
    "MID360GeometryConfig",
    "MID360GeometryWaypointProvider",
    "OriginalETPWaypointProvider",
    "WaypointOutput",
    "WaypointProvider",
]
