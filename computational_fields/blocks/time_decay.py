"""T block: time-based decay and temporal dynamics."""

from __future__ import annotations

from ..core.context import Context
from ..core.primitives import rep


def timer(ctx: Context, duration: float) -> float:
    """Countdown timer that decrements by ``delta_time`` each round.

    Returns the remaining time, clamped to ``0.0``.
    """
    dt = ctx.delta_time
    return rep(ctx, duration, lambda remaining: max(0.0, remaining - dt))


def decay(ctx: Context, value: float, rate: float = 0.9) -> float:
    """Exponential decay: multiplies previous value by *rate* each round."""
    return rep(ctx, value, lambda prev: prev * rate)
