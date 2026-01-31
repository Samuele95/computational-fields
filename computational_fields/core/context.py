"""Execution context for a single device round.

The context tracks the current device, its neighbor exports, a call-path
stack for alignment, and provides helpers for the field calculus primitives.
"""

from __future__ import annotations

import math
from typing import Any

from .device import Device


class Context:
    """Per-round execution context for one device.

    The call stack produces a *call path* string that uniquely identifies each
    primitive invocation site.  Neighbor exports are keyed by the same paths,
    which ensures alignment: ``nbr`` only reads values from neighbors that
    executed the same code path.
    """

    def __init__(
        self,
        device: Device,
        neighbor_devices: dict[int, Device],
        round_count: int = 0,
        delta_time: float = 1.0,
    ) -> None:
        self.device = device
        self.neighbor_devices = neighbor_devices
        # Snapshot of neighbor exports at the start of this round.
        self.neighbor_exports: dict[int, dict[str, Any]] = {
            nid: dict(nd.exports) for nid, nd in neighbor_devices.items()
        }
        self.round_count = round_count
        self.delta_time = delta_time

        # Call-path alignment machinery
        self._call_stack: list[str] = []
        self._slot_counters: list[int] = []  # per-level counters
        self._new_exports: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Call-path helpers
    # ------------------------------------------------------------------

    def _next_slot(self) -> int:
        """Return and increment the counter at the current stack depth."""
        if not self._slot_counters:
            self._slot_counters.append(0)
        idx = self._slot_counters[-1]
        self._slot_counters[-1] += 1
        return idx

    def push(self, tag: str) -> str:
        """Push *tag* onto the call stack and return the full call path."""
        slot = self._next_slot()
        label = f"{tag}@{slot}"
        self._call_stack.append(label)
        self._slot_counters.append(0)
        return self.call_path

    def pop(self) -> None:
        """Pop the most recent call-stack entry."""
        self._call_stack.pop()
        self._slot_counters.pop()

    @property
    def call_path(self) -> str:
        return "/".join(self._call_stack)

    # ------------------------------------------------------------------
    # Export read/write
    # ------------------------------------------------------------------

    def export(self, value: Any) -> None:
        """Write *value* to the export tree at the current call path."""
        self._new_exports[self.call_path] = value

    def read_neighbor_export(self, neighbor_id: int, path: str) -> Any | None:
        """Read a neighbor's export at the given call path, or ``None``."""
        nexp = self.neighbor_exports.get(neighbor_id, {})
        return nexp.get(path)

    def get_exports(self) -> dict[str, Any]:
        """Return the full export tree produced during this round."""
        return dict(self._new_exports)

    # ------------------------------------------------------------------
    # Neighbor geometry
    # ------------------------------------------------------------------

    def nbr_range_to(self, neighbor_id: int) -> float:
        """Euclidean distance from this device to *neighbor_id*."""
        nbr = self.neighbor_devices.get(neighbor_id)
        if nbr is None:
            return math.inf
        return self.device.distance_to(nbr)

    def aligned_neighbors(self, path: str | None = None) -> list[int]:
        """Return neighbor IDs whose exports contain the given path."""
        p = path if path is not None else self.call_path
        return [
            nid
            for nid, nexp in self.neighbor_exports.items()
            if p in nexp
        ]

    # ------------------------------------------------------------------
    # Sensor shortcut
    # ------------------------------------------------------------------

    def sense(self, name: str) -> Any:
        return self.device.sense(name)

    def mid(self) -> int:
        return self.device.id
