"""Tests for field calculus primitives."""

from __future__ import annotations

import math

import numpy as np

from computational_fields.core.device import Device
from computational_fields.core.context import Context
from computational_fields.core.primitives import (
    rep, nbr, share, branch, foldhood, mux, mid, sense,
    min_hood, min_hood_plus, nbr_range,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_device(did: int = 0, pos: tuple = (0, 0), sensors: dict | None = None) -> Device:
    return Device(id=did, position=np.array(pos, dtype=float),
                  sensors=sensors or {})


def _make_ctx(
    device: Device,
    neighbor_devices: dict[int, Device] | None = None,
) -> Context:
    return Context(
        device=device,
        neighbor_devices=neighbor_devices or {},
        round_count=0,
        delta_time=1.0,
    )


def _two_device_setup() -> tuple[Device, Device, Context, Context]:
    """Two devices at distance 1.0 that are neighbors."""
    d0 = _make_device(0, (0, 0))
    d1 = _make_device(1, (1, 0))
    d0.neighbors = [1]
    d1.neighbors = [0]
    ctx0 = _make_ctx(d0, {1: d1})
    ctx1 = _make_ctx(d1, {0: d0})
    return d0, d1, ctx0, ctx1


# ── rep ──────────────────────────────────────────────────────────────

class TestRep:
    def test_first_round_returns_init(self):
        dev = _make_device()
        ctx = _make_ctx(dev)
        result = rep(ctx, 0, lambda x: x + 1)
        # First round: f(init) = 0 + 1 = 1
        assert result == 1

    def test_accumulates_across_rounds(self):
        dev = _make_device()
        # Round 1
        ctx = _make_ctx(dev)
        r1 = rep(ctx, 0, lambda x: x + 1)
        assert r1 == 1
        # Round 2: previous state persisted in dev.state
        ctx2 = _make_ctx(dev)
        r2 = rep(ctx2, 0, lambda x: x + 1)
        assert r2 == 2

    def test_exports_value(self):
        dev = _make_device()
        ctx = _make_ctx(dev)
        rep(ctx, 10, lambda x: x * 2)
        exports = ctx.get_exports()
        assert len(exports) > 0
        assert 20 in exports.values()


# ── nbr ──────────────────────────────────────────────────────────────

class TestNbr:
    def test_no_neighbors_returns_empty(self):
        dev = _make_device()
        ctx = _make_ctx(dev)
        result = nbr(ctx, 42)
        assert result == {}

    def test_reads_neighbor_exports(self):
        d0, d1, ctx0, ctx1 = _two_device_setup()
        # d1 runs first and exports via nbr
        val1 = nbr(ctx1, 100)
        d1.exports = ctx1.get_exports()
        # Now d0 should see d1's export
        ctx0_round2 = Context(d0, {1: d1})
        val0 = nbr(ctx0_round2, 200)
        assert 1 in val0
        assert val0[1] == 100


# ── share ────────────────────────────────────────────────────────────

class TestShare:
    def test_first_round_uses_init(self):
        dev = _make_device()
        ctx = _make_ctx(dev)
        result = share(ctx, 5.0, lambda prev, nbrs: prev + 1)
        # prev = 5.0 (init), nbrs = {}
        assert result == 6.0

    def test_shares_with_neighbors(self):
        d0, d1, ctx0, ctx1 = _two_device_setup()
        # d1 runs and exports
        share(ctx1, 10.0, lambda prev, nbrs: prev)
        d1.exports = ctx1.get_exports()
        # d0 reads d1's shared value
        ctx0_r2 = Context(d0, {1: d1})
        result = share(ctx0_r2, 0.0,
                       lambda prev, nbrs: sum(nbrs.values()) if nbrs else prev)
        assert result == 10.0


# ── branch ───────────────────────────────────────────────────────────

class TestBranch:
    def test_true_branch(self):
        dev = _make_device()
        ctx = _make_ctx(dev)
        result = branch(ctx, True, lambda: "yes", lambda: "no")
        assert result == "yes"

    def test_false_branch(self):
        dev = _make_device()
        ctx = _make_ctx(dev)
        result = branch(ctx, False, lambda: "yes", lambda: "no")
        assert result == "no"

    def test_alignment_isolation(self):
        """Devices in different branches should not see each other's exports."""
        d0, d1, ctx0, ctx1 = _two_device_setup()
        # d0 takes true branch
        branch(ctx0, True, lambda: nbr(ctx0, "A"), lambda: nbr(ctx0, "B"))
        d0.exports = ctx0.get_exports()
        # d1 takes false branch
        ctx1_r2 = Context(d1, {0: d0})
        result = branch(ctx1_r2, False,
                        lambda: nbr(ctx1_r2, "A"),
                        lambda: nbr(ctx1_r2, "B"))
        # d1 should NOT see d0's value since they're in different branches
        assert 0 not in result


# ── foldhood ─────────────────────────────────────────────────────────

class TestFoldhood:
    def test_no_neighbors(self):
        dev = _make_device()
        ctx = _make_ctx(dev)
        result = foldhood(ctx, 0.0, lambda a, b: a + b, lambda: 5.0)
        assert result == 0.0  # no neighbors to fold

    def test_folds_neighbor_values(self):
        d0, d1, ctx0, ctx1 = _two_device_setup()
        # d1 exports via foldhood
        foldhood(ctx1, 0.0, lambda a, b: a + b, lambda: 10.0)
        d1.exports = ctx1.get_exports()
        # d0 folds over d1's value
        ctx0_r2 = Context(d0, {1: d1})
        result = foldhood(ctx0_r2, 0.0, lambda a, b: a + b, lambda: 5.0)
        assert result == 10.0  # only d1's exported value


# ── mux ──────────────────────────────────────────────────────────────

class TestMux:
    def test_true(self):
        dev = _make_device()
        ctx = _make_ctx(dev)
        assert mux(ctx, True, "yes", "no") == "yes"

    def test_false(self):
        dev = _make_device()
        ctx = _make_ctx(dev)
        assert mux(ctx, False, "yes", "no") == "no"


# ── mid / sense ──────────────────────────────────────────────────────

class TestMidSense:
    def test_mid(self):
        dev = _make_device(42)
        ctx = _make_ctx(dev)
        assert mid(ctx) == 42

    def test_sense(self):
        dev = _make_device(sensors={"temp": 25.0})
        ctx = _make_ctx(dev)
        assert sense(ctx, "temp") == 25.0
        assert sense(ctx, "missing") is None


# ── min_hood_plus ────────────────────────────────────────────────────

class TestMinHoodPlus:
    def test_isolated_device(self):
        dev = _make_device()
        ctx = _make_ctx(dev)
        result = min_hood_plus(ctx, lambda: 7.0)
        assert result == 7.0  # own value only

    def test_with_neighbor(self):
        d0, d1, ctx0, ctx1 = _two_device_setup()
        # d1 exports 3.0
        min_hood_plus(ctx1, lambda: 3.0)
        d1.exports = ctx1.get_exports()
        # d0 has own=7.0 and neighbor=3.0
        ctx0_r2 = Context(d0, {1: d1})
        result = min_hood_plus(ctx0_r2, lambda: 7.0)
        assert result == 3.0
