"""Sensor providers for simulation scenarios."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def point_source(
    source_id: int,
) -> dict[str, Any]:
    """Sensor factory: marks a single device as the source."""
    def fn(did: int, *_args: Any) -> dict[str, Any]:
        return {"is_source": did == source_id}
    return fn  # type: ignore[return-value]


def corner_source(cols: int) -> Any:
    """Sensor factory for grid: device 0 is source."""
    def fn(did: int, row: int, col: int) -> dict[str, Any]:
        return {"is_source": did == 0}
    return fn


def crowd_sensors(
    monitor_ids: set[int],
    exit_ids: set[int],
    density_fn: Any = None,
) -> Any:
    """Sensor factory for the crowd monitoring scenario.

    Parameters
    ----------
    density_fn:
        Optional ``(device_id) -> float`` returning local density.
        Defaults to 0.5 for all devices.
    """
    def fn(did: int, *_args: Any) -> dict[str, Any]:
        d = density_fn(did) if density_fn is not None else 0.5
        return {
            "is_monitor": did in monitor_ids,
            "is_exit": did in exit_ids,
            "density": d,
            "emergency_active": False,
        }
    return fn
