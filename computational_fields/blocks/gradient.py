"""G block: gradient (distance estimation) and broadcast.

The gradient is the foundational building block of aggregate computing.
It creates a self-stabilizing distance field from source devices, and
broadcast propagates values outward along that field.
"""

from __future__ import annotations

import math
from typing import Any, TypeVar

from ..core.context import Context
from ..core.primitives import share, mux

T = TypeVar("T")


def gradient(ctx: Context, source: bool) -> float:
    """Self-stabilizing distance estimation from *source* devices.

    Implements the G block using ``share``.  Source devices output ``0.0``;
    others take the minimum ``(neighbor_distance + range_to_neighbor)``
    across aligned neighbors.
    """
    def body(prev: float, nbrs: dict[int, float]) -> float:
        if source:
            return 0.0
        if not nbrs:
            return math.inf
        return min(nbrs[n] + ctx.nbr_range_to(n) for n in nbrs)

    return share(ctx, math.inf, body)


def dist_to(ctx: Context, source: bool) -> float:
    """Alias for :func:`gradient`."""
    return gradient(ctx, source)


def broadcast(ctx: Context, source: bool, value: T) -> T:
    """Broadcast *value* outward from *source* devices along the gradient.

    Each device picks the neighbor closest to a source and relays that
    neighbor's value.  Returns the value originating from the nearest
    source.
    """
    def body(
        prev: tuple[float, T],
        nbrs: dict[int, tuple[float, T]],
    ) -> tuple[float, T]:
        if source:
            return (0.0, value)
        if not nbrs:
            return (math.inf, prev[1])
        best_nid = min(nbrs, key=lambda n: nbrs[n][0] + ctx.nbr_range_to(n))
        best_dist = nbrs[best_nid][0] + ctx.nbr_range_to(best_nid)
        best_val = nbrs[best_nid][1]
        return (best_dist, best_val)

    return share(ctx, (math.inf, value), body)[1]
