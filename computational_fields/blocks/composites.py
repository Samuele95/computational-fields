"""Composite aggregate computing patterns built from G, C, S, T blocks.

These patterns demonstrate the composability of self-stabilizing building
blocks — each composite inherits the self-stabilization guarantees of its
constituents.
"""

from __future__ import annotations

import operator

from ..core.context import Context
from ..core.primitives import mid, mux
from .gradient import gradient, broadcast, dist_to
from .collection import collect
from .sparse import sparse


def channel(
    ctx: Context,
    source: bool,
    destination: bool,
    width: float,
) -> bool:
    """Form a logical channel between *source* and *destination* regions.

    Returns ``True`` for devices that lie on the shortest path (within
    *width* tolerance) between the two regions.
    """
    to_src = gradient(ctx, source)
    to_dst = gradient(ctx, destination)
    path_dist = to_src + to_dst
    optimal = broadcast(ctx, source, gradient(ctx, destination))
    return path_dist <= optimal + width


def partition(ctx: Context, grain: float) -> int:
    """Partition the network into regions of approximate diameter *grain*.

    Returns the leader device ID for each device's region.
    """
    is_leader = sparse(ctx, grain)
    _potential = gradient(ctx, is_leader)
    return broadcast(ctx, is_leader, mid(ctx))


def distributed_average(ctx: Context, value: float) -> float:
    """Compute a regional average of *value* across partitioned regions.

    Uses S (sparse election) to pick region leaders, G (gradient) for
    routing, C (collection) for aggregation, and G (broadcast) to
    distribute the result back.
    """
    is_leader = sparse(ctx, 100.0)
    potential = gradient(ctx, is_leader)
    total = collect(ctx, potential, operator.add, value, 0.0)
    count = collect(ctx, potential, operator.add, 1.0, 0.0)
    avg = mux(ctx, is_leader, total / max(count, 1.0), 0.0)
    return broadcast(ctx, is_leader, avg)
