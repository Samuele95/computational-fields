"""S block: sparse leader election.

Selects a set of approximately uniformly-spaced leaders.  The *grain*
parameter controls the minimum distance between leaders.  Symmetry is
broken using device IDs: among competing candidates, the one with the
lowest ID wins.
"""

from __future__ import annotations

import math

from ..core.context import Context
from ..core.primitives import share, rep, nbr


def sparse(ctx: Context, grain: float) -> bool:
    """Elect uniformly-spaced leaders separated by at least *grain*.

    Uses a two-field approach:
    1. Each device computes a gradient-like distance to the nearest
       *potential leader* (the device with the minimum ID within range).
    2. A device becomes a leader if it has the minimum ID within grain
       distance of itself.

    Returns ``True`` for leaders.
    """
    my_id = ctx.device.id

    def body(
        prev: tuple[float, int],
        nbrs: dict[int, tuple[float, int]],
    ) -> tuple[float, int]:
        """State is (distance_to_leader, leader_id)."""
        # Collect candidates: each neighbor's (distance + range, leader_id)
        candidates: list[tuple[float, int]] = []
        for nid, (ndist, nlid) in nbrs.items():
            d = ndist + ctx.nbr_range_to(nid)
            candidates.append((d, nlid))

        if not candidates:
            # Isolated device: become leader
            return (0.0, my_id)

        # Find the nearest leader among neighbors
        best_dist, best_lid = min(candidates, key=lambda x: (x[0], x[1]))

        if best_dist > grain:
            # No leader within grain distance → become leader
            return (0.0, my_id)

        # If this device is the leader being tracked, stay as leader
        if best_lid == my_id:
            return (0.0, my_id)

        # There's a different leader within grain distance — follow it
        # But if I have a lower ID than that leader and was already a
        # leader, I should remain leader (symmetry breaking)
        if my_id < best_lid and prev[0] == 0.0:
            return (0.0, my_id)

        return (best_dist, best_lid)

    result = share(ctx, (0.0, my_id), body)
    return result[0] == 0.0 and result[1] == my_id
