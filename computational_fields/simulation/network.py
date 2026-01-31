"""Network topology management.

Manages a collection of devices, computes neighbor relationships based on
Euclidean distance, and supports dynamic topology changes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.device import Device


class Network:
    """A spatial network of devices with distance-based neighbor discovery."""

    def __init__(self, comm_range: float = 1.5) -> None:
        self.devices: dict[int, Device] = {}
        self.comm_range = comm_range
        self._next_id = 0

    def add_device(
        self,
        position: tuple[float, float] | np.ndarray,
        sensors: dict[str, Any] | None = None,
        device_id: int | None = None,
    ) -> Device:
        """Add a device and return it."""
        did = device_id if device_id is not None else self._next_id
        self._next_id = max(self._next_id, did + 1)
        dev = Device(id=did, position=np.asarray(position, dtype=float),
                     sensors=sensors or {})
        self.devices[did] = dev
        return dev

    def remove_device(self, device_id: int) -> None:
        self.devices.pop(device_id, None)

    def update_neighbors(self) -> None:
        """Recompute neighbor lists based on Euclidean distance."""
        ids = list(self.devices.keys())
        for dev in self.devices.values():
            dev.neighbors = []
        for i, id_a in enumerate(ids):
            da = self.devices[id_a]
            for id_b in ids[i + 1 :]:
                db = self.devices[id_b]
                if da.distance_to(db) <= self.comm_range:
                    da.neighbors.append(id_b)
                    db.neighbors.append(id_a)

    def get_distance(self, id_a: int, id_b: int) -> float:
        da = self.devices.get(id_a)
        db = self.devices.get(id_b)
        if da is None or db is None:
            return math.inf
        return da.distance_to(db)

    # ── Factory helpers ──────────────────────────────────────────────

    @classmethod
    def grid(
        cls,
        rows: int,
        cols: int,
        spacing: float = 1.0,
        comm_range: float | None = None,
        sensors_fn: Any = None,
    ) -> Network:
        """Create a regular grid network.

        Parameters
        ----------
        sensors_fn:
            Optional callable ``(device_id, row, col) -> dict`` that
            provides per-device sensor values.
        """
        cr = comm_range if comm_range is not None else spacing * 1.5
        net = cls(comm_range=cr)
        for r in range(rows):
            for c in range(cols):
                did = r * cols + c
                pos = (c * spacing, r * spacing)
                sensors: dict[str, Any] = {}
                if sensors_fn is not None:
                    sensors = sensors_fn(did, r, c)
                net.add_device(pos, sensors, device_id=did)
        net.update_neighbors()
        return net

    @classmethod
    def random(
        cls,
        n: int,
        width: float = 10.0,
        height: float = 10.0,
        comm_range: float = 2.0,
        sensors_fn: Any = None,
        rng: np.random.Generator | None = None,
    ) -> Network:
        """Create a random network with *n* devices in a rectangular area."""
        rng = rng or np.random.default_rng()
        net = cls(comm_range=comm_range)
        for i in range(n):
            pos = (rng.uniform(0, width), rng.uniform(0, height))
            sensors: dict[str, Any] = {}
            if sensors_fn is not None:
                sensors = sensors_fn(i)
            net.add_device(pos, sensors, device_id=i)
        net.update_neighbors()
        return net
