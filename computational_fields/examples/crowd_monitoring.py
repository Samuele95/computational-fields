"""Crowd monitoring case study from the paper.

Demonstrates all building blocks working together:
- G: gradient to monitoring stations and exits
- C: collect density readings to monitors
- S: sparse leader election for emergency lanes
- T: (implicit time smoothing via rep)
- Broadcast: disseminate alert levels back to devices

The network simulates a venue with monitoring stations, exits,
and spatially-varying crowd density.
"""

from __future__ import annotations

import math
import operator

import numpy as np
import matplotlib.pyplot as plt

from ..core.context import Context
from ..core.primitives import rep, mux, mid, sense
from ..blocks.gradient import gradient, dist_to, broadcast
from ..blocks.collection import collect
from ..blocks.sparse import sparse
from ..simulation.network import Network
from ..simulation.engine import SimulationEngine
from ..visualization.renderer import FieldRenderer


MAX_DENSITY = 1.0


def crowd_program(ctx: Context) -> dict:
    """Aggregate program for crowd monitoring.

    Each device outputs:
    - density: smoothed local density
    - alert: broadcast alert level (NORMAL / WARNING / CRITICAL)
    - exit_dist: distance to nearest exit
    - lane_id: emergency lane partition ID
    """
    is_monitor = sense(ctx, "is_monitor") or False
    is_exit = sense(ctx, "is_exit") or False
    local_density = sense(ctx, "density") or 0.0

    # Smooth density with exponential moving average (rep = T-like)
    density = rep(ctx, local_density,
                  lambda prev: 0.7 * prev + 0.3 * local_density)

    # Gradient to monitoring stations
    potential = dist_to(ctx, is_monitor)

    # Collect density readings toward monitors (C block)
    total_density = collect(ctx, potential, operator.add, density, 0.0)
    device_count = collect(ctx, potential, operator.add, 1.0, 0.0)
    avg_density = mux(ctx, is_monitor,
                      total_density / max(device_count, 1.0),
                      0.0)

    # Determine alert level at monitors
    ratio = avg_density / MAX_DENSITY
    if ratio >= 0.9:
        alert = "CRITICAL"
    elif ratio >= 0.7:
        alert = "WARNING"
    else:
        alert = "NORMAL"

    # Broadcast alert from monitors to all devices (G block)
    alert_broadcast = broadcast(ctx, is_monitor, alert)

    # Navigation: distance to nearest exit (G block)
    exit_dist = dist_to(ctx, is_exit)

    # Emergency lane formation (S block)
    is_leader = sparse(ctx, 3.0)
    lane_id = broadcast(ctx, is_leader, mid(ctx))

    return {
        "density": density,
        "alert": alert_broadcast,
        "exit_dist": exit_dist,
        "lane_id": lane_id,
    }


def _density_fn(center: tuple[float, float], radius: float, peak: float):
    """Return a density function with a Gaussian hotspot."""
    def fn(did: int) -> float:
        # Will be set dynamically per-device based on position
        return peak
    return fn


def main() -> None:
    rows, cols = 12, 12
    spacing = 1.0

    # Place monitors at center-ish positions, exits at corners
    monitor_ids = {
        5 * cols + 5,
        5 * cols + 6,
        6 * cols + 5,
        6 * cols + 6,
    }
    exit_ids = {0, cols - 1, (rows - 1) * cols, rows * cols - 1}

    # Density hotspot near center
    center = np.array([5.5, 5.5])
    rng = np.random.default_rng(42)

    def sensors(did: int, row: int, col: int) -> dict:
        pos = np.array([col * spacing, row * spacing])
        dist = float(np.linalg.norm(pos - center))
        # Gaussian density peak at center
        density = 0.95 * math.exp(-(dist ** 2) / 8.0) + rng.uniform(0, 0.05)
        density = min(density, 1.0)
        return {
            "is_monitor": did in monitor_ids,
            "is_exit": did in exit_ids,
            "density": density,
            "emergency_active": False,
        }

    net = Network.grid(rows, cols, spacing=spacing, sensors_fn=sensors)
    engine = SimulationEngine(net, crowd_program)
    engine.run(30)

    renderer = FieldRenderer(engine)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # 1. Density heatmap
    density_field = {did: r["density"] for did, r in engine.results.items()}
    renderer.render_scalar_field(
        density_field, title="Crowd Density",
        cmap="YlOrRd", vmin=0, vmax=1.0, ax=axes[0, 0],
    )

    # 2. Alert levels
    alert_field = {did: r["alert"] for did, r in engine.results.items()}
    renderer.render_categorical_field(
        alert_field, title="Alert Levels",
        color_map={"NORMAL": "green", "WARNING": "orange", "CRITICAL": "red"},
        ax=axes[0, 1],
    )

    # 3. Distance to nearest exit
    exit_field = {did: r["exit_dist"] for did, r in engine.results.items()}
    renderer.render_scalar_field(
        exit_field, title="Distance to Exit",
        cmap="Blues_r", ax=axes[1, 0],
    )

    # 4. Emergency lane partitions
    lane_field = {did: r["lane_id"] for did, r in engine.results.items()}
    renderer.render_scalar_field(
        lane_field, title="Emergency Lanes (Leader IDs)",
        cmap="tab20", ax=axes[1, 1],
    )

    fig.suptitle("Crowd Monitoring Case Study — Round 30", fontsize=14)
    plt.tight_layout()
    plt.savefig("crowd_monitoring.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
