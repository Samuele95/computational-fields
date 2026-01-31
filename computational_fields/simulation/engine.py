"""Simulation engine — orchestrates round execution across the network."""

from __future__ import annotations

from typing import Any, Callable

from ..core.context import Context
from ..core.device import Device
from .network import Network


class SimulationEngine:
    """Synchronous simulation engine for aggregate programs.

    Each :meth:`step` executes the aggregate program on every device,
    collects exports, and makes them available to neighbors in the next
    round.
    """

    def __init__(
        self,
        network: Network,
        program: Callable[[Context], Any],
        delta_time: float = 1.0,
    ) -> None:
        self.network = network
        self.program = program
        self.delta_time = delta_time
        self.round_count = 0
        self.results: dict[int, Any] = {}
        # Per-round history for visualization / analysis
        self.history: list[dict[int, Any]] = []

    def _build_context(self, device: Device) -> Context:
        neighbor_devices = {
            nid: self.network.devices[nid]
            for nid in device.neighbors
            if nid in self.network.devices
        }
        return Context(
            device=device,
            neighbor_devices=neighbor_devices,
            round_count=self.round_count,
            delta_time=self.delta_time,
        )

    def step(self) -> dict[int, Any]:
        """Execute one synchronous round for all devices.

        Returns a dict mapping device IDs to their program outputs.
        """
        round_results: dict[int, Any] = {}
        new_exports: dict[int, dict[str, Any]] = {}

        for dev in self.network.devices.values():
            ctx = self._build_context(dev)
            result = self.program(ctx)
            new_exports[dev.id] = ctx.get_exports()
            round_results[dev.id] = result

        # Commit exports after all devices have executed (synchronous).
        for did, exports in new_exports.items():
            self.network.devices[did].exports = exports

        self.round_count += 1
        self.results = round_results
        self.history.append(dict(round_results))
        return round_results

    def run(self, num_rounds: int) -> list[dict[int, Any]]:
        """Run *num_rounds* synchronous rounds. Returns full history."""
        for _ in range(num_rounds):
            self.step()
        return self.history

    def get_field(self, key: str | None = None) -> dict[int, Any]:
        """Extract a named sub-field from the latest results.

        If results are dicts, returns ``{id: result[key]}``.
        If *key* is ``None``, returns the raw results.
        """
        if key is None:
            return dict(self.results)
        return {
            did: (r[key] if isinstance(r, dict) else r)
            for did, r in self.results.items()
        }
