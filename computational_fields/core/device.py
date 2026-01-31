"""Device model for aggregate computing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Device:
    """A single device in the computational field network.

    Each device has a unique ID, a 2D position, sensors, persistent state
    for rep/share across rounds, and an export tree that neighbors read.
    """

    id: int
    position: np.ndarray  # shape (2,)
    sensors: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    exports: dict[str, Any] = field(default_factory=dict)
    neighbors: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)

    def distance_to(self, other: Device) -> float:
        return float(np.linalg.norm(self.position - other.position))

    def sense(self, name: str) -> Any:
        return self.sensors.get(name)
