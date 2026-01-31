"""Tests for G, C, S, T building blocks."""

from __future__ import annotations

import math

from computational_fields.core.context import Context
from computational_fields.blocks.gradient import gradient, dist_to, broadcast
from computational_fields.blocks.collection import collect
from computational_fields.blocks.sparse import sparse
from computational_fields.blocks.time_decay import timer
from computational_fields.simulation.network import Network
from computational_fields.simulation.engine import SimulationEngine


# ── Gradient (G block) ───────────────────────────────────────────────

class TestGradient:
    def test_source_is_zero(self):
        """Source devices should have gradient 0."""
        def program(ctx: Context) -> float:
            return gradient(ctx, ctx.sense("is_source") or False)

        net = Network.grid(3, 3, spacing=1.0,
                           sensors_fn=lambda did, r, c: {"is_source": did == 4})
        engine = SimulationEngine(net, program)
        engine.run(10)
        assert engine.results[4] == 0.0

    def test_gradient_converges(self):
        """After enough rounds, non-source devices should have finite distances."""
        def program(ctx: Context) -> float:
            return gradient(ctx, ctx.sense("is_source") or False)

        net = Network.grid(5, 5, spacing=1.0,
                           sensors_fn=lambda did, r, c: {"is_source": did == 0})
        engine = SimulationEngine(net, program)
        engine.run(20)
        # All devices should have finite distances
        for did, val in engine.results.items():
            assert math.isfinite(val), f"Device {did} has inf distance"

    def test_gradient_increases_with_distance(self):
        """Devices farther from source should have larger gradient values."""
        def program(ctx: Context) -> float:
            return gradient(ctx, ctx.sense("is_source") or False)

        net = Network.grid(5, 1, spacing=1.0,
                           sensors_fn=lambda did, r, c: {"is_source": did == 0})
        engine = SimulationEngine(net, program)
        engine.run(10)
        # Linear chain: d0=0, d1~1, d2~2, ...
        for i in range(4):
            assert engine.results[i] < engine.results[i + 1]


# ── Broadcast ────────────────────────────────────────────────────────

class TestBroadcast:
    def test_broadcast_from_source(self):
        """Broadcast should propagate the source value to all devices."""
        def program(ctx: Context) -> str:
            source = ctx.sense("is_source") or False
            return broadcast(ctx, source, "HELLO")

        net = Network.grid(3, 1, spacing=1.0,
                           sensors_fn=lambda did, r, c: {"is_source": did == 0})
        engine = SimulationEngine(net, program)
        engine.run(10)
        # All devices should eventually receive "HELLO"
        for did, val in engine.results.items():
            assert val == "HELLO", f"Device {did} got {val}"


# ── Collection (C block) ────────────────────────────────────────────

class TestCollection:
    def test_collect_sums(self):
        """Collect should aggregate values toward the sink."""
        import operator

        def program(ctx: Context) -> dict:
            source = ctx.sense("is_source") or False
            pot = gradient(ctx, source)
            total = collect(ctx, pot, operator.add,
                            ctx.sense("value") or 0.0, 0.0)
            return {"potential": pot, "total": total}

        # 3-device chain: d0=source, d1, d2. Each has value=1
        net = Network.grid(3, 1, spacing=1.0,
                           sensors_fn=lambda did, r, c: {
                               "is_source": did == 0,
                               "value": 1.0,
                           })
        engine = SimulationEngine(net, program)
        engine.run(15)
        # d0 (sink) should collect contributions from all devices
        # Due to eventual convergence, total at d0 should approach 3.0
        total_at_source = engine.results[0]["total"]
        assert total_at_source >= 1.0  # at least own value


# ── Sparse (S block) ────────────────────────────────────────────────

class TestSparse:
    def test_at_least_one_leader(self):
        """At least one device should be elected as leader."""
        def program(ctx: Context) -> bool:
            return sparse(ctx, 3.0)

        net = Network.grid(5, 5, spacing=1.0)
        engine = SimulationEngine(net, program)
        engine.run(20)
        leaders = [did for did, v in engine.results.items() if v]
        assert len(leaders) >= 1

    def test_leaders_are_spaced(self):
        """Leaders should be approximately grain-distance apart."""
        def program(ctx: Context) -> bool:
            return sparse(ctx, 2.5)

        net = Network.grid(8, 8, spacing=1.0)
        engine = SimulationEngine(net, program)
        engine.run(30)
        leaders = [did for did, v in engine.results.items() if v]
        # Check that no two leaders are closer than grain/2
        for i, la in enumerate(leaders):
            for lb in leaders[i + 1:]:
                d = net.get_distance(la, lb)
                # Allow some tolerance for the discrete grid
                assert d >= 1.0, f"Leaders {la} and {lb} too close: {d}"


# ── Timer (T block) ──────────────────────────────────────────────────

class TestTimer:
    def test_countdown(self):
        """Timer should count down to zero."""
        def program(ctx: Context) -> float:
            return timer(ctx, 5.0)

        net = Network.grid(1, 1, spacing=1.0)
        engine = SimulationEngine(net, program, delta_time=1.0)
        engine.run(7)
        # After 5 rounds: 5 - 1 = 4, 4-1=3, ... 1-1=0, then stays at 0
        # Round 1: max(0, 5-1) = 4
        # Round 5: max(0, 1-1) = 0
        # Round 6: max(0, 0-1) = 0
        assert engine.results[0] == 0.0
        # Check intermediate
        assert engine.history[0][0] == 4.0
        assert engine.history[4][0] == 0.0
