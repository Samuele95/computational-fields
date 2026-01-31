"""Tests for composite patterns (channel, partition, distributed average)."""

from __future__ import annotations

from computational_fields.core.context import Context
from computational_fields.blocks.composites import channel, partition
from computational_fields.simulation.network import Network
from computational_fields.simulation.engine import SimulationEngine


class TestChannel:
    def test_source_and_dest_on_channel(self):
        """Source and destination should be on the channel."""
        def program(ctx: Context) -> bool:
            src = ctx.sense("is_source") or False
            dst = ctx.sense("is_dest") or False
            return channel(ctx, src, dst, width=2.0)

        net = Network.grid(5, 5, spacing=1.0,
                           sensors_fn=lambda did, r, c: {
                               "is_source": did == 0,
                               "is_dest": did == 24,
                           })
        engine = SimulationEngine(net, program)
        engine.run(20)
        assert engine.results[0] is True
        assert engine.results[24] is True

    def test_channel_forms_path(self):
        """Some intermediate devices should be on the channel."""
        def program(ctx: Context) -> bool:
            src = ctx.sense("is_source") or False
            dst = ctx.sense("is_dest") or False
            return channel(ctx, src, dst, width=2.0)

        net = Network.grid(5, 5, spacing=1.0,
                           sensors_fn=lambda did, r, c: {
                               "is_source": did == 0,
                               "is_dest": did == 24,
                           })
        engine = SimulationEngine(net, program)
        engine.run(20)
        on_channel = [did for did, v in engine.results.items() if v]
        assert len(on_channel) >= 3  # at least src, dst, and some path


class TestPartition:
    def test_all_devices_assigned(self):
        """Every device should be assigned to a partition (leader ID)."""
        def program(ctx: Context) -> int:
            return partition(ctx, 3.0)

        net = Network.grid(6, 6, spacing=1.0)
        engine = SimulationEngine(net, program)
        engine.run(25)
        for did, val in engine.results.items():
            assert isinstance(val, int), f"Device {did} has non-int partition: {val}"

    def test_multiple_partitions(self):
        """A large enough network should produce multiple partitions."""
        def program(ctx: Context) -> int:
            return partition(ctx, 3.0)

        net = Network.grid(8, 8, spacing=1.0)
        engine = SimulationEngine(net, program)
        engine.run(30)
        unique = set(engine.results.values())
        assert len(unique) >= 2
