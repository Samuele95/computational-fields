"""C block: collection / aggregation toward low-potential regions.

Data flows *inward* along a potential field (typically a gradient from
collection points).  Each device accumulates its own local value plus
contributions from higher-potential neighbors.
"""

from __future__ import annotations

import functools
from typing import Callable, TypeVar

from ..core.context import Context
from ..core.primitives import share, nbr

T = TypeVar("T")


def collect(
    ctx: Context,
    potential: float,
    acc: Callable[[T, T], T],
    local: T,
    null: T,
) -> T:
    """Collect values toward low-potential regions.

    Parameters
    ----------
    potential:
        This device's potential value (e.g. distance from a sink).
    acc:
        Associative accumulator (e.g. ``operator.add``).
    local:
        This device's contribution to the aggregation.
    null:
        Identity element for *acc* (e.g. ``0.0`` for addition).
    """
    # Export the current potential so neighbors can read it.
    potential_field = nbr(ctx, potential)

    def body(prev: T, nbrs: dict[int, T]) -> T:
        # Accept contributions only from higher-potential neighbors
        # (data flows downhill toward sinks).
        contributions = [
            nbrs[n]
            for n in nbrs
            if potential_field.get(n, -1) > potential
        ]
        folded = functools.reduce(acc, contributions, null)
        return acc(local, folded)

    return share(ctx, local, body)
