"""Field calculus primitives.

Implements the five core constructs of field calculus plus common derived
operators.  Every function receives an explicit :class:`Context`.
"""

from __future__ import annotations

import math
from typing import Any, Callable, TypeVar

from .context import Context

T = TypeVar("T")


# ── Core primitives ──────────────────────────────────────────────────

def rep(ctx: Context, init: T, f: Callable[[T], T]) -> T:
    """State evolution across rounds.

    On the first round (no prior state), returns *init*.
    On subsequent rounds, applies *f* to the previous value.
    """
    path = ctx.push("rep")
    try:
        prev = ctx.device.state.get(path, init)
        result = f(prev)
        ctx.device.state[path] = result
        ctx.export(result)
        return result
    finally:
        ctx.pop()


def nbr(ctx: Context, value: T) -> dict[int, T]:
    """Neighbor observation.

    Exports *value* (this device's contribution) and returns a dict mapping
    aligned neighbor IDs to their exported values at the same call path.
    """
    path = ctx.push("nbr")
    try:
        ctx.export(value)
        result: dict[int, T] = {}
        for nid in ctx.aligned_neighbors(path):
            nval = ctx.read_neighbor_export(nid, path)
            if nval is not None:
                result[nid] = nval
        return result
    finally:
        ctx.pop()


def share(ctx: Context, init: T, f: Callable[[T, dict[int, T]], T]) -> T:
    """Combined state evolution and neighbor observation.

    *f(previous_own_value, neighbor_values)* returns the new value.
    """
    path = ctx.push("share")
    try:
        prev = ctx.device.state.get(path, init)
        nbrs: dict[int, T] = {}
        for nid in ctx.aligned_neighbors(path):
            nval = ctx.read_neighbor_export(nid, path)
            if nval is not None:
                nbrs[nid] = nval
        result = f(prev, nbrs)
        ctx.device.state[path] = result
        ctx.export(result)
        return result
    finally:
        ctx.pop()


def branch(ctx: Context, cond: bool, then_fn: Callable[[], T], else_fn: Callable[[], T]) -> T:
    """Domain restriction.

    Only the matching branch executes.  Neighbors in the other branch
    are invisible (different call path).
    """
    tag = "branch_T" if cond else "branch_F"
    path = ctx.push(tag)
    try:
        result = then_fn() if cond else else_fn()
        ctx.export(result)
        return result
    finally:
        ctx.pop()


def foldhood(ctx: Context, init: T, acc: Callable[[T, T], T], nbr_expr: Callable[[], T]) -> T:
    """General fold over neighbor values.

    Evaluates *nbr_expr* (which should call ``nbr`` internally or return a
    value to be broadcast), then folds neighbor values with *acc*.

    For simplicity, *nbr_expr* is called once and its result is treated as
    the device's own contribution that is exported.  Neighbor contributions
    come from aligned exports at the same call path.
    """
    path = ctx.push("fold")
    try:
        own = nbr_expr()
        ctx.export(own)
        result = init
        for nid in ctx.aligned_neighbors(path):
            nval = ctx.read_neighbor_export(nid, path)
            if nval is not None:
                result = acc(result, nval)
        return result
    finally:
        ctx.pop()


# ── Derived operators ────────────────────────────────────────────────

def mux(ctx: Context, cond: bool, then_val: T, else_val: T) -> T:
    """Multiplexer — both values are evaluated; no domain restriction."""
    return then_val if cond else else_val


def mid(ctx: Context) -> int:
    """Current device ID."""
    return ctx.mid()


def sense(ctx: Context, name: str) -> Any:
    """Read a sensor value."""
    return ctx.sense(name)


def nbr_range(ctx: Context) -> dict[int, float]:
    """Return a dict mapping each neighbor to its distance."""
    path = ctx.push("nbr_range")
    try:
        own_val = 0.0
        ctx.export(own_val)
        result: dict[int, float] = {}
        for nid in ctx.device.neighbors:
            result[nid] = ctx.nbr_range_to(nid)
        return result
    finally:
        ctx.pop()


def min_hood(ctx: Context, nbr_expr: Callable[[], float]) -> float:
    """Minimum of neighbor values (excluding self)."""
    return foldhood(ctx, math.inf, min, nbr_expr)


def max_hood(ctx: Context, nbr_expr: Callable[[], float]) -> float:
    """Maximum of neighbor values (excluding self)."""
    return foldhood(ctx, -math.inf, max, nbr_expr)


def sum_hood(ctx: Context, nbr_expr: Callable[[], float]) -> float:
    """Sum of neighbor values (excluding self)."""
    return foldhood(ctx, 0.0, lambda a, b: a + b, nbr_expr)


def min_hood_plus(ctx: Context, expr: Callable[[], float]) -> float:
    """Minimum of neighbor values *including self*.

    Evaluates *expr* once, exports it, collects aligned neighbor values,
    folds with ``min``, and also considers the device's own value.
    """
    path = ctx.push("mhp")
    try:
        own = expr()
        ctx.export(own)
        result = own  # include self
        for nid in ctx.aligned_neighbors(path):
            nval = ctx.read_neighbor_export(nid, path)
            if nval is not None:
                result = min(result, nval)
        return result
    finally:
        ctx.pop()
