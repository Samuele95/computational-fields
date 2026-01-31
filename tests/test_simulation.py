"""Integration tests for the simulation engine."""

from __future__ import annotations

import math

from computational_fields.core.context import Context
from computational_fields.core.primitives import rep, nbr, share
from computational_fields.blocks.gradient import gradient
from computational_fields.simulation.network import Network
from computational_fields.simulation.engine import SimulationEngine


class TestSimulationEngine:
    def test_basic_round(self):
        """Engine should execute a program on all devices."""
        def program(ctx: Context) -> int:
            return ctx.mid()

        net = Network.grid(3, 3, spacing=1.0)
        engine = SimulationEngine(net, program)
        results = engine.step()
        assert len(results) == 9
        for did in range(9):
            assert results[did] == did

    def test_multi_round_state(self):
        """State should persist across rounds via rep."""
        def program(ctx: Context) -> int:
            return rep(ctx, 0, lambda x: x + 1)

        net = Network.grid(1, 1, spacing=1.0)
        engine = SimulationEngine(net, program)
        engine.run(5)
        # After 5 rounds: 1, 2, 3, 4, 5
        assert engine.results[0] == 5

    def test_exports_propagate(self):
        """Exports from round N should be visible in round N+1."""
        def program(ctx: Context) -> dict:
            own = ctx.mid()
            neighbor_ids = nbr(ctx, own)
            return {"own": own, "neighbors": neighbor_ids}

        net = Network.grid(3, 1, spacing=1.0)
        engine = SimulationEngine(net, program)
        # Round 1: no exports yet
        engine.step()
        # Round 2: should see neighbors
        engine.step()
        r = engine.results[1]  # middle device
        assert 0 in r["neighbors"] or 2 in r["neighbors"]

    def test_gradient_self_stabilization(self):
        """Gradient should converge to stable values."""
        def program(ctx: Context) -> float:
            return gradient(ctx, ctx.sense("is_source") or False)

        net = Network.grid(5, 1, spacing=1.0,
                           sensors_fn=lambda did, r, c: {"is_source": did == 0})
        engine = SimulationEngine(net, program)
        engine.run(20)
        prev = dict(engine.results)
        engine.run(5)
        # Values should be stable (not changing)
        for did in prev:
            assert abs(prev[did] - engine.results[did]) < 1e-9

    def test_history_recorded(self):
        """Engine should record results for each round."""
        def program(ctx: Context) -> int:
            return rep(ctx, 0, lambda x: x + 1)

        net = Network.grid(1, 1, spacing=1.0)
        engine = SimulationEngine(net, program)
        history = engine.run(3)
        assert len(history) == 3
        assert history[0][0] == 1
        assert history[1][0] == 2
        assert history[2][0] == 3

    def test_delta_time(self):
        """Engine should pass delta_time to context."""
        recorded_dt = []

        def program(ctx: Context) -> float:
            recorded_dt.append(ctx.delta_time)
            return 0.0

        net = Network.grid(1, 1, spacing=1.0)
        engine = SimulationEngine(net, program, delta_time=0.5)
        engine.step()
        assert recorded_dt[0] == 0.5
