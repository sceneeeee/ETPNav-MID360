"""Waypoint proposal adapters that preserve the original ETP feature contract."""

from vlnce_baselines.waypoint_provider.base import WaypointProvider
from vlnce_baselines.waypoint_provider.mid360_geometry import (
    MID360GeometryConfig,
    MID360GeometryWaypointProvider,
)
from vlnce_baselines.waypoint_provider.original_etp import (
    OriginalETPWaypointProvider,
)
from vlnce_baselines.waypoint_provider.mid360_geometry_v1 import (
    MID360GeometryV1Config,
    MID360GeometryV1WaypointProvider,
)
from vlnce_baselines.waypoint_provider.types import WaypointOutput

__all__ = [
    "MID360GeometryConfig",
    "MID360GeometryWaypointProvider",
    "MID360GeometryV1Config",
    "MID360GeometryV1WaypointProvider",
    "OriginalETPWaypointProvider",
    "WaypointOutput",
    "WaypointProvider",
]
