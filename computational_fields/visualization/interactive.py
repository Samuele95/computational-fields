"""Interactive Streamlit UI for Computational Fields simulation.

Run with:
    streamlit run computational_fields/visualization/interactive.py
"""

from __future__ import annotations

import json
import math
import operator
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ── Imports from the framework ──────────────────────────────────────────
import sys, os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from computational_fields.core.context import Context
from computational_fields.core.primitives import rep, mux, mid, sense
from computational_fields.blocks.gradient import gradient, dist_to, broadcast
from computational_fields.blocks.collection import collect
from computational_fields.blocks.sparse import sparse
from computational_fields.blocks.composites import channel, partition
from computational_fields.simulation.network import Network
from computational_fields.simulation.engine import SimulationEngine

# ═══════════════════════════════════════════════════════════════════════
#  Dark plotly theme
# ═══════════════════════════════════════════════════════════════════════

_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="#fafafa"),
    margin=dict(l=20, r=20, t=50, b=20),
    height=600,
    transition=dict(duration=300, easing="cubic-in-out"),
)


# ═══════════════════════════════════════════════════════════════════════
#  Demo program definitions
# ═══════════════════════════════════════════════════════════════════════


def _gradient_program(ctx: Context) -> float:
    source = ctx.sense("is_source") or False
    return gradient(ctx, source)


def _channel_program(ctx: Context) -> dict:
    source = ctx.sense("is_source") or False
    dest = ctx.sense("is_dest") or False
    on_channel = channel(ctx, source, dest, width=1.5)
    return {"on_channel": on_channel}


def _sparse_program(ctx: Context) -> dict:
    grain = ctx.sense("grain") or 3.0
    is_leader = sparse(ctx, grain)
    leader_id = broadcast(ctx, is_leader, mid(ctx))
    return {"is_leader": is_leader, "leader_id": leader_id}


def _crowd_program(ctx: Context) -> dict:
    is_monitor = sense(ctx, "is_monitor") or False
    is_exit = sense(ctx, "is_exit") or False
    local_density = sense(ctx, "density") or 0.0

    density = rep(
        ctx, local_density, lambda prev: 0.7 * prev + 0.3 * local_density
    )

    potential = dist_to(ctx, is_monitor)
    total_density = collect(ctx, potential, operator.add, density, 0.0)
    device_count = collect(ctx, potential, operator.add, 1.0, 0.0)
    avg_density = mux(
        ctx, is_monitor, total_density / max(device_count, 1.0), 0.0
    )

    ratio = avg_density / 1.0
    if ratio >= 0.9:
        alert = "CRITICAL"
    elif ratio >= 0.7:
        alert = "WARNING"
    else:
        alert = "NORMAL"

    alert_broadcast = broadcast(ctx, is_monitor, alert)
    exit_dist = dist_to(ctx, is_exit)

    return {
        "density": density,
        "alert": alert_broadcast,
        "exit_dist": exit_dist,
    }


def _wave_program(ctx: Context) -> float:
    """Gradient with pulsing source — source toggles on/off."""
    is_source_device = ctx.sense("is_source") or False
    pulse_period = ctx.sense("pulse_period") or 6
    # Source is active during even half-periods
    source_active = (ctx.round_count // pulse_period) % 2 == 0
    source = is_source_device and source_active
    return gradient(ctx, source)


def _selfheal_program(ctx: Context) -> float:
    source = ctx.sense("is_source") or False
    return gradient(ctx, source)


# ═══════════════════════════════════════════════════════════════════════
#  Demo configurations
# ═══════════════════════════════════════════════════════════════════════

DEMOS: dict[str, dict[str, Any]] = {
    "Gradient": {
        "program": _gradient_program,
        "field_type": "scalar",
        "field_key": None,
        "colorscale": "Plasma",
        "description": (
            "Each device computes its shortest-path distance to one or more "
            "source devices.  The scalar field converges over rounds as "
            "distance information propagates hop-by-hop through the network."
        ),
        "theory": (
            "### Computational Field Theory\n\n"
            "A **computational field** is a function $\\varphi : D \\times T \\to V$ "
            "that maps every device $\\delta \\in D$ at every time step "
            "$t \\in T$ to a value $v \\in V$.  The Gradient is the most "
            "fundamental building block (**G block**) in the aggregate "
            "computing framework.\n\n"
            "#### The G Block\n\n"
            "Formally, the gradient computes the **shortest-path distance "
            "field** from a set of source devices $S \\subseteq D$:\n\n"
            "$$G(\\delta) = \\begin{cases} 0 & \\text{if } \\delta \\in S \\\\ "
            "\\min_{\\eta \\in N(\\delta)}\\{G(\\eta) + w(\\delta, \\eta)\\} & "
            "\\text{otherwise} \\end{cases}$$\n\n"
            "where $N(\\delta)$ is the set of neighbors of $\\delta$ and "
            "$w(\\delta, \\eta)$ is the link cost (Euclidean distance "
            "between positions).\n\n"
            "#### Self-Stabilization\n\n"
            "The key theoretical property is **self-stabilization**: "
            "starting from *any* arbitrary initial state, the field "
            "converges to the correct fixed point within a finite number "
            "of communication rounds (bounded by the network diameter).  "
            "This is achieved through the **share** primitive, which "
            "combines:\n\n"
            "1. **Local state** maintained via `rep` (state across rounds)\n"
            "2. **Neighbor communication** via `nbr` (exchange of values "
            "with all neighbors within communication range)\n\n"
            "The convergence speed is $O(\\mathrm{diam}(G))$ rounds where "
            "$\\mathrm{diam}(G)$ is the hop-diameter of the network graph.\n\n"
            "#### Alignment & Call-Path Semantics\n\n"
            "Each invocation of `share` is uniquely identified by its "
            "**call-path** (position in the program call stack).  This "
            "ensures that when a device reads its neighbor's exports, it "
            "reads values from the *same* `share` invocation — the formal "
            "notion of **alignment** that makes compositional building "
            "blocks possible."
        ),
        "practical": (
            "### What You See\n\n"
            "- **Colors**: each node is colored by its distance value "
            "(dark = close to source, bright = far away).  The colorbar "
            "on the right maps colors to numeric distances.\n"
            "- **Edges**: thin lines connect devices within communication "
            "range, showing the network topology.\n"
            "- **Hover**: mouse over a node to see its device ID, "
            "position, and exact distance value.\n\n"
            "### How to Explore\n\n"
            "1. **Step** through rounds one at a time and watch the "
            "distance wavefront expand from the source.\n"
            "2. Use the **history slider** to scrub back and forth and "
            "see the propagation frame by frame.\n"
            "3. **Auto-Play** animates the simulation — adjust the speed "
            "slider in the sidebar.\n"
            "4. Try changing the **Topology** to Random to see how the "
            "gradient adapts to irregular networks.\n"
            "5. Use the **Device Role Editor** below the chart to move "
            "the source to a different device, then reset.\n"
            "6. Watch the **Convergence Chart** at the bottom — it "
            "shows the maximum per-round change dropping to zero as "
            "the field stabilizes."
        ),
        "roles": ["Source"],
    },
    "Channel": {
        "program": _channel_program,
        "field_type": "categorical",
        "field_key": "on_channel",
        "colorscale": None,
        "description": (
            "A logical communication channel forms between a source and "
            "a destination.  Devices lying on the shortest-path corridor "
            "are highlighted as channel members."
        ),
        "theory": (
            "### Channel Formation Theory\n\n"
            "The **channel** is a composite building block that "
            "demonstrates the power of functional composition in "
            "aggregate computing.  It creates a *tube-shaped* region "
            "of devices that connects two distinguished points in the "
            "network.\n\n"
            "#### Construction\n\n"
            "A channel between source $s$ and destination $d$ with width "
            "$w$ is defined as:\n\n"
            "$$\\mathrm{channel}(\\delta) = "
            "\\big[G_s(\\delta) + G_d(\\delta) \\leq "
            "G_s(d) + w\\big]$$\n\n"
            "where $G_s$ is the gradient from $s$ and $G_d$ is the "
            "gradient from $d$.  The expression $G_s(\\delta) + G_d(\\delta)$ "
            "is the total path length through $\\delta$; devices where this "
            "is close to the direct distance $G_s(d)$ lie near the "
            "shortest path.\n\n"
            "#### Compositional Design\n\n"
            "Notice how the channel is built from **three independent "
            "gradient computations** (two G blocks + one broadcast) "
            "composed together.  This is the hallmark of aggregate "
            "computing: complex spatial patterns emerge from simple, "
            "reusable building blocks.  Each block self-stabilizes "
            "independently, and their composition self-stabilizes "
            "as well — a property guaranteed by the field calculus "
            "semantics.\n\n"
            "#### The Width Parameter\n\n"
            "The $w$ parameter controls the *tolerance* of the "
            "tube.  At $w = 0$, only devices on the exact shortest path "
            "are included.  Larger $w$ creates a wider corridor, which "
            "is useful for fault tolerance: if a device on the channel "
            "fails, nearby devices can take over."
        ),
        "practical": (
            "### What You See\n\n"
            "- **Blue nodes** (CHANNEL): devices on the logical channel "
            "between source and destination.\n"
            "- **Red/gray nodes** (OFF): devices outside the channel "
            "corridor.\n"
            "- The channel typically forms a band of nodes along the "
            "shortest path.\n\n"
            "### How to Explore\n\n"
            "1. Run a few rounds and watch the channel establish itself.\n"
            "2. Use the **Device Role Editor** to move the source or "
            "destination to different positions and see how the "
            "channel reconfigures.\n"
            "3. Try **Random topology** — the channel follows the "
            "shortest-path corridor even through irregular networks.\n"
            "4. Observe the convergence chart: once the two underlying "
            "gradients stabilize, the channel membership becomes fixed.\n"
            "5. Increase grid size to see longer, more complex channels."
        ),
        "roles": ["Source", "Dest"],
    },
    "Sparse Leaders": {
        "program": _sparse_program,
        "field_type": "leader",
        "field_key": "leader_id",
        "colorscale": "Turbo",
        "description": (
            "Uniformly-spaced leaders are elected across the network.  "
            "Each device follows its nearest leader.  Leaders are shown "
            "as stars, and Voronoi-like regions are colored by leader ID."
        ),
        "theory": (
            "### Sparse Leader Election Theory\n\n"
            "The **S block** (sparse) solves the distributed leader "
            "election problem: select a subset of devices as *leaders* "
            "such that leaders are approximately uniformly spaced with "
            "a configurable minimum inter-leader distance (the *grain* "
            "parameter).\n\n"
            "#### Algorithm\n\n"
            "Each device maintains a tuple $(d, \\ell)$ via `share`, "
            "where $d$ is the distance to the nearest leader and $\\ell$ "
            "is that leader's ID:\n\n"
            "1. Exchange $(d, \\ell)$ with all neighbors.  Each "
            "neighbor's distance is increased by the link cost: "
            "$(d_{\\eta} + w(\\delta, \\eta), \\ell_{\\eta})$.\n"
            "2. Select the candidate with the **smallest total "
            "distance** (ties broken by smallest leader ID).\n"
            "3. If the best distance exceeds the grain $g$, no leader "
            "is close enough — the device **elects itself** as a new "
            "leader: $(0, \\delta)$.\n"
            "4. **Symmetry breaking**: if two nearby devices both try "
            "to become leaders, the one with the lower ID wins.\n\n"
            "#### Theoretical Properties\n\n"
            "- **Spacing guarantee**: after convergence, every point in "
            "the network is within distance $g$ of at least one leader.\n"
            "- **Self-stabilization**: if a leader is removed, nearby "
            "devices detect the gap (distances grow beyond $g$) and "
            "elect a replacement.\n"
            "- **Voronoi partitioning**: the `broadcast` of each "
            "leader's ID creates a natural Voronoi decomposition of "
            "the network — the foundation for spatial partitioning in "
            "aggregate computing.\n\n"
            "#### Relation to the Partition Block\n\n"
            "The S block is used internally by the **partition** "
            "composite, which divides the network into regions, each "
            "managed by a leader.  This enables independent sub-programs "
            "to run within each region — a form of distributed spatial "
            "scoping."
        ),
        "practical": (
            "### What You See\n\n"
            "- **Star markers**: elected leaders.\n"
            "- **Colored regions**: each device is colored by the ID of "
            "its nearest leader, forming a Voronoi-like tessellation.\n"
            "- **Colorscale**: maps leader IDs to distinct colors.\n\n"
            "### How to Explore\n\n"
            "1. Adjust the **Grain** slider in the sidebar to control "
            "inter-leader spacing.  Smaller grain = more leaders, "
            "larger grain = fewer leaders.\n"
            "2. Use **Auto-Play** to watch the election converge — "
            "initially many devices claim leadership, then the field "
            "settles as lower-ID devices win.\n"
            "3. Try **Random topology** to see how leaders adapt to "
            "non-uniform connectivity.\n"
            "4. Increase the grid size and lower the grain to see "
            "many distinct Voronoi cells.\n"
            "5. The convergence chart shows how quickly the leader "
            "assignment stabilizes across the network."
        ),
        "roles": [],
    },
    "Crowd Monitoring": {
        "program": _crowd_program,
        "field_type": "multi",
        "field_key": None,
        "colorscale": None,
        "description": (
            "A full-featured case study combining all building blocks: "
            "crowd density estimation, alert broadcast from monitoring "
            "stations, and exit-distance navigation fields."
        ),
        "theory": (
            "### Crowd Safety Monitoring — Case Study\n\n"
            "This demo implements the complete crowd safety scenario "
            "from the Computational Fields literature, combining all "
            "four building blocks (G, C, S, T) into a single program.\n\n"
            "#### Architecture\n\n"
            "The system has three layers that run simultaneously on "
            "every device:\n\n"
            "**1. Density Estimation (T block — time decay)**\n\n"
            "Each device has a local density sensor.  Raw readings are "
            "smoothed over time using `rep` with exponential averaging:\n\n"
            "$$\\rho_t(\\delta) = \\alpha \\cdot \\rho_{t-1}(\\delta) + "
            "(1-\\alpha) \\cdot \\rho_{\\text{raw}}(\\delta)$$\n\n"
            "This filters out transient noise from the density sensor.\n\n"
            "**2. Density Collection (G + C blocks)**\n\n"
            "A **gradient potential field** $P$ is computed from the "
            "monitoring stations using the G block.  The smoothed "
            "densities are then **collected** along this potential toward "
            "the monitors using the C block.  At each monitor, the "
            "total collected density is divided by the count of "
            "contributing devices to compute the *average regional "
            "density*.\n\n"
            "$$\\bar\\rho(m) = \\frac{\\sum_{\\delta \\in "
            "\\mathrm{region}(m)} \\rho(\\delta)}"
            "{|\\mathrm{region}(m)|}$$\n\n"
            "**3. Alert Broadcast & Exit Navigation (G blocks)**\n\n"
            "The monitor converts the average density to an alert level "
            "(NORMAL / WARNING / CRITICAL) using configurable thresholds.  "
            "This alert is **broadcast** back to all devices in the "
            "region via a gradient-based broadcast (G block).  "
            "Simultaneously, a separate gradient field computes "
            "**exit distances**, giving each device a navigation "
            "direction toward the nearest exit.\n\n"
            "#### Self-Organization\n\n"
            "The entire system is self-organizing: no central "
            "coordinator is needed.  If a monitor fails, its region is "
            "absorbed by neighboring monitors.  If an exit is blocked, "
            "navigation automatically reroutes.  This emerges naturally "
            "from the self-stabilizing properties of the G and C blocks."
        ),
        "practical": (
            "### What You See\n\n"
            "The visualization cycles through three sub-fields using "
            "the tabs at the top of the chart:\n\n"
            "- **Density**: heatmap of smoothed crowd density at each "
            "device (Gaussian distribution centered on the grid center, "
            "simulating a crowd gathering).\n"
            "- **Alert**: categorical map showing NORMAL (green), "
            "WARNING (orange), and CRITICAL (red) zones broadcast from "
            "monitoring stations.\n"
            "- **Exit Distance**: scalar field showing distance to the "
            "nearest exit (corner devices are exits by default).\n\n"
            "### How to Explore\n\n"
            "1. Run several rounds and observe how alerts propagate "
            "outward from monitors.\n"
            "2. Use the **Device Role Editor** to move monitors or "
            "exits — the entire system self-reconfigures.\n"
            "3. Remove all exits except one: exit distances increase "
            "dramatically, reflecting reduced escape capacity.\n"
            "4. Try a large grid (15x15) to see multiple distinct alert "
            "zones and richer density gradients.\n"
            "5. This demo shows the practical motivation for aggregate "
            "computing: writing a single program that self-organizes "
            "across hundreds of devices."
        ),
        "roles": ["Monitor", "Exit"],
    },
    "Wave Propagation": {
        "program": _wave_program,
        "field_type": "scalar",
        "field_key": None,
        "colorscale": "Viridis",
        "description": (
            "The source pulses on and off, creating expanding and "
            "collapsing distance wavefronts that ripple through the "
            "network — a dynamic visualization of self-stabilization."
        ),
        "theory": (
            "### Wave Propagation Theory\n\n"
            "This demo visualizes a fundamental consequence of "
            "**self-stabilization**: the system continuously adapts to "
            "changing inputs.  When the environment changes, the field "
            "does not need to be restarted — it evolves toward the new "
            "correct state.\n\n"
            "#### Mechanism\n\n"
            "The program uses the standard gradient algorithm (G block), "
            "but the source device's status is toggled by a periodic "
            "clock:\n\n"
            "$$\\mathrm{source}(\\delta, t) = "
            "\\delta \\in S \\land "
            "\\lfloor t / T_p \\rfloor \\bmod 2 = 0$$\n\n"
            "where $T_p$ is the pulse period.  During 'on' phases, the "
            "gradient field expands from the source (distances decrease). "
            "During 'off' phases, the source stops anchoring at zero, "
            "and distances rise toward infinity across the network.\n\n"
            "#### What This Reveals\n\n"
            "- **Information propagation speed**: the wavefront expands "
            "at one hop per round (bounded by the communication model).  "
            "This is the *speed of light* for aggregate computing.\n"
            "- **Rising values problem**: when the source turns off, "
            "values increase slowly because each device only sees local "
            "neighbors.  This asymmetry (fast convergence down, slow "
            "convergence up) is a well-known challenge in distance-based "
            "self-stabilizing algorithms.  Advanced algorithms like CRF "
            "(Constraint and Restoring Force) address this.\n"
            "- **Transient dynamics**: the field goes through "
            "intermediate, incorrect states between pulses.  The "
            "convergence chart shows these transients as periodic "
            "spikes.\n\n"
            "#### Connection to Physical Waves\n\n"
            "The expanding wavefront is conceptually similar to "
            "electromagnetic or acoustic wave propagation: information "
            "radiates outward from the source at a bounded speed, "
            "creating concentric rings of equal distance.  However, "
            "unlike physical waves, these are *computational* wavefronts "
            "— they carry distance information, not energy."
        ),
        "practical": (
            "### What You See\n\n"
            "- **Expanding rings**: when the source is active, low-value "
            "(dark) rings expand outward as distance information "
            "propagates.\n"
            "- **Dissolving field**: when the source deactivates, all "
            "values slowly rise (colors brighten) as the field loses "
            "its anchor.\n"
            "- **Periodic pattern**: with Auto-Play, you see a rhythmic "
            "pulse-and-fade cycle.\n\n"
            "### How to Explore\n\n"
            "1. **Auto-Play is recommended** for this demo — set a "
            "moderate speed (300-500ms) and watch the waves.\n"
            "2. Adjust the **Pulse Period** slider: shorter periods "
            "create rapid flicker, longer periods allow full expansion "
            "before the next pulse.\n"
            "3. Increase the grid to 15x15 or larger to see the "
            "wavefront clearly.\n"
            "4. Compare **Grid** vs **Random** topology — waves are "
            "smoother on grids and more irregular on random networks.\n"
            "5. The convergence chart shows periodic spikes corresponding "
            "to each pulse transition — the field never fully stabilizes "
            "because the input keeps changing."
        ),
        "roles": ["Source"],
    },
    "Self-Healing": {
        "program": _selfheal_program,
        "field_type": "scalar",
        "field_key": None,
        "colorscale": "Plasma",
        "description": (
            "Standard gradient field, but you can remove devices "
            "mid-simulation to observe how the field self-heals: "
            "distances recompute through alternative paths."
        ),
        "theory": (
            "### Self-Healing & Resilience Theory\n\n"
            "**Self-stabilization** is the defining theoretical property "
            "of aggregate computing.  A self-stabilizing system "
            "guarantees that:\n\n"
            "> Starting from **any** arbitrary state (including states "
            "caused by failures), the system converges to the correct "
            "output within a bounded number of rounds.\n\n"
            "This is a stronger guarantee than fault tolerance: the "
            "system does not just *survive* failures, it *automatically "
            "recovers* without any external intervention.\n\n"
            "#### Why It Works\n\n"
            "The gradient algorithm uses only **local information**: "
            "each device computes `min(neighbor_distance + link_cost)`.  "
            "When a device is removed:\n\n"
            "1. Its neighbors detect its absence (no more exports "
            "from that device).\n"
            "2. Neighbors that relied on the removed device for their "
            "shortest path now compute a higher distance via remaining "
            "neighbors.\n"
            "3. This higher distance propagates outward, and eventually "
            "all affected devices find new shortest paths through the "
            "remaining network.\n\n"
            "#### Formal Guarantee\n\n"
            "If the network remains connected after the failure, the "
            "field converges to the correct shortest-path distances "
            "within $O(\\mathrm{diam}(G'))$ rounds, where $G'$ is the "
            "post-failure network graph.\n\n"
            "#### Practical Implications\n\n"
            "This property makes aggregate computing suitable for "
            "**unreliable environments** such as:\n\n"
            "- Wireless sensor networks (devices may run out of battery)\n"
            "- Robot swarms (units may leave or join dynamically)\n"
            "- IoT deployments (intermittent connectivity)\n"
            "- Smart city infrastructure (hardware failures)\n\n"
            "No reprogramming, restart, or reconfiguration is needed — "
            "the field *heals itself*."
        ),
        "practical": (
            "### What You See\n\n"
            "- Initially, a standard gradient field from the source "
            "(device 0 at the bottom-left corner).\n"
            "- After removing devices, the gap appears and nearby "
            "distances temporarily spike as paths are rerouted.\n"
            "- Within a few rounds, the field stabilizes to new correct "
            "distances through alternative paths.\n\n"
            "### How to Explore\n\n"
            "1. First, **run 20+ rounds** to let the gradient fully "
            "stabilize.\n"
            "2. Use the **Remove Region** controls (below the chart) to "
            "cut out a rectangular block of devices.\n"
            "3. Continue stepping or use Auto-Play — watch the field "
            "recompute around the gap.\n"
            "4. Try removing devices that lie on the *shortest path* "
            "from the source: the distances beyond the gap increase, "
            "then stabilize via detour paths.\n"
            "5. Remove a large block to partition the network: devices "
            "in the disconnected component will show infinite distance "
            "(very large values), demonstrating that the algorithm "
            "correctly identifies unreachable devices.\n"
            "6. The **convergence chart** shows a spike when devices are "
            "removed, followed by a rapid drop back to zero as the "
            "field heals."
        ),
        "roles": ["Source"],
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  Network builders
# ═══════════════════════════════════════════════════════════════════════


def _build_network(
    demo: str,
    rows: int,
    cols: int,
    spacing: float,
    comm_range: float,
    grain: float,
    pulse_period: int,
    topology: str,
    n_random: int,
    width: float,
    height: float,
    role_overrides: dict[int, str] | None = None,
) -> Network:
    total = rows * cols

    # ── Base sensor factories ───────────────────────────────────────
    def _gradient_sensors(did: int, row: int, col: int) -> dict:
        return {"is_source": did == 0}

    def _channel_sensors(did: int, row: int, col: int) -> dict:
        last = total - 1
        return {"is_source": did == 0, "is_dest": did == last}

    def _sparse_sensors(did: int, row: int, col: int) -> dict:
        return {"grain": grain}

    def _wave_sensors(did: int, row: int, col: int) -> dict:
        return {"is_source": did == 0, "pulse_period": pulse_period}

    def _selfheal_sensors(did: int, row: int, col: int) -> dict:
        return {"is_source": did == 0}

    def _crowd_sensors(did: int, row: int, col: int) -> dict:
        center = np.array(
            [(cols - 1) * spacing / 2, (rows - 1) * spacing / 2]
        )
        rng = np.random.default_rng(42 + did)
        monitor_r, monitor_c = rows // 2, cols // 2
        monitor_ids = set()
        for dr in range(2):
            for dc in range(2):
                mid_ = (monitor_r + dr) * cols + (monitor_c + dc)
                if mid_ < total:
                    monitor_ids.add(mid_)
        exit_ids = {0, cols - 1, (rows - 1) * cols, total - 1}

        pos = np.array([col * spacing, row * spacing])
        dist = float(np.linalg.norm(pos - center))
        density = 0.95 * math.exp(-(dist**2) / 8.0) + rng.uniform(0, 0.05)
        density = min(density, 1.0)
        return {
            "is_monitor": did in monitor_ids,
            "is_exit": did in exit_ids,
            "density": density,
        }

    sensor_map = {
        "Gradient": _gradient_sensors,
        "Channel": _channel_sensors,
        "Sparse Leaders": _sparse_sensors,
        "Crowd Monitoring": _crowd_sensors,
        "Wave Propagation": _wave_sensors,
        "Self-Healing": _selfheal_sensors,
    }
    base_fn = sensor_map.get(demo, _gradient_sensors)

    # Wrap to apply role overrides
    def sensors_fn(did: int, row: int, col: int) -> dict:
        s = base_fn(did, row, col)
        if role_overrides and did in role_overrides:
            role = role_overrides[did]
            # Clear role-related keys first
            for k in ["is_source", "is_dest", "is_monitor", "is_exit"]:
                s[k] = False
            if role == "Source":
                s["is_source"] = True
            elif role == "Dest":
                s["is_dest"] = True
            elif role == "Monitor":
                s["is_monitor"] = True
            elif role == "Exit":
                s["is_exit"] = True
        return s

    # For random networks, use a wrapper that ignores row/col
    if topology == "Random":
        def random_sensor(did: int) -> dict:
            return sensors_fn(did, 0, 0)

        return Network.random(
            n_random,
            width=width,
            height=height,
            comm_range=comm_range,
            sensors_fn=random_sensor,
            rng=np.random.default_rng(7),
        )

    return Network.grid(
        rows, cols, spacing=spacing, comm_range=comm_range, sensors_fn=sensors_fn
    )


# ═══════════════════════════════════════════════════════════════════════
#  Plotly rendering helpers
# ═══════════════════════════════════════════════════════════════════════


def _get_positions(engine: SimulationEngine):
    ids = sorted(engine.network.devices.keys())
    xs = [float(engine.network.devices[i].position[0]) for i in ids]
    ys = [float(engine.network.devices[i].position[1]) for i in ids]
    return xs, ys, ids


def _edge_traces(engine: SimulationEngine) -> go.Scatter:
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    seen: set[tuple[int, int]] = set()
    for dev in engine.network.devices.values():
        for nid in dev.neighbors:
            if nid not in engine.network.devices:
                continue
            key = (min(dev.id, nid), max(dev.id, nid))
            if key in seen:
                continue
            seen.add(key)
            nb = engine.network.devices[nid]
            edge_x += [float(dev.position[0]), float(nb.position[0]), None]
            edge_y += [float(dev.position[1]), float(nb.position[1]), None]
    return go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.7, color="rgba(100,100,120,0.4)"),
        hoverinfo="none",
        showlegend=False,
    )


def _scalar_figure(
    engine: SimulationEngine,
    field: dict[int, float],
    title: str,
    colorscale: str = "Plasma",
    show_edges: bool = True,
) -> go.Figure:
    xs, ys, ids = _get_positions(engine)
    values = [float(field.get(i, 0)) for i in ids]

    fig = go.Figure()
    if show_edges:
        fig.add_trace(_edge_traces(engine))

    hover_text = [
        f"<b>Device {i}</b><br>"
        f"Position: ({x:.1f}, {y:.1f})<br>"
        f"Value: <b>{v:.3f}</b>"
        for i, x, y, v in zip(ids, xs, ys, values)
    ]

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=14,
                color=values,
                colorscale=colorscale,
                showscale=True,
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
                colorbar=dict(title="Value", thickness=15),
            ),
            text=hover_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(scaleanchor="y", constrain="domain", showgrid=False),
        yaxis=dict(constrain="domain", showgrid=False),
        **_LAYOUT_DEFAULTS,
    )
    return fig


def _categorical_figure(
    engine: SimulationEngine,
    field: dict[int, Any],
    title: str,
    color_map: dict[str, str] | None = None,
    show_edges: bool = True,
) -> go.Figure:
    xs, ys, ids = _get_positions(engine)
    categories = [str(field.get(i, "?")) for i in ids]
    unique_cats = sorted(set(categories))

    if color_map is None:
        palette = [
            "#636efa", "#ef553b", "#00cc96", "#ab63fa",
            "#ffa15a", "#19d3f3", "#ff6692", "#b6e880",
        ]
        color_map = {
            cat: palette[i % len(palette)] for i, cat in enumerate(unique_cats)
        }

    fig = go.Figure()
    if show_edges:
        fig.add_trace(_edge_traces(engine))

    for cat in unique_cats:
        mask = [i for i, c in enumerate(categories) if c == cat]
        fig.add_trace(
            go.Scatter(
                x=[xs[i] for i in mask],
                y=[ys[i] for i in mask],
                mode="markers",
                marker=dict(
                    size=14,
                    color=color_map.get(cat, "#999"),
                    line=dict(width=1, color="rgba(255,255,255,0.3)"),
                ),
                name=cat,
                text=[
                    f"<b>Device {ids[i]}</b><br>"
                    f"Position: ({xs[i]:.1f}, {ys[i]:.1f})<br>"
                    f"Category: <b>{cat}</b>"
                    for i in mask
                ],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(scaleanchor="y", constrain="domain", showgrid=False),
        yaxis=dict(constrain="domain", showgrid=False),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        **_LAYOUT_DEFAULTS,
    )
    return fig


def _leader_figure(
    engine: SimulationEngine,
    field: dict[int, Any],
    results: dict[int, Any],
    title: str,
    colorscale: str = "Turbo",
    show_edges: bool = True,
) -> go.Figure:
    xs, ys, ids = _get_positions(engine)
    leader_ids = [float(field.get(i, -1)) for i in ids]

    is_leader_field = {}
    for did, r in results.items():
        if isinstance(r, dict):
            is_leader_field[did] = r.get("is_leader", False)
        else:
            is_leader_field[did] = False

    fig = go.Figure()
    if show_edges:
        fig.add_trace(_edge_traces(engine))

    follower_mask = [
        i for i, did in enumerate(ids) if not is_leader_field.get(did, False)
    ]
    leader_mask = [
        i for i, did in enumerate(ids) if is_leader_field.get(did, False)
    ]

    fig.add_trace(
        go.Scatter(
            x=[xs[i] for i in follower_mask],
            y=[ys[i] for i in follower_mask],
            mode="markers",
            marker=dict(
                size=12,
                color=[leader_ids[i] for i in follower_mask],
                colorscale=colorscale,
                showscale=True,
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
                colorbar=dict(title="Leader ID", thickness=15),
            ),
            text=[
                f"<b>Device {ids[i]}</b><br>"
                f"Position: ({xs[i]:.1f}, {ys[i]:.1f})<br>"
                f"Leader: <b>{int(leader_ids[i])}</b>"
                for i in follower_mask
            ],
            hoverinfo="text",
            name="Followers",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[xs[i] for i in leader_mask],
            y=[ys[i] for i in leader_mask],
            mode="markers",
            marker=dict(
                size=20,
                symbol="star",
                color=[leader_ids[i] for i in leader_mask],
                colorscale=colorscale,
                line=dict(width=2, color="#fafafa"),
            ),
            text=[
                f"<b>Device {ids[i]} (LEADER)</b><br>"
                f"Position: ({xs[i]:.1f}, {ys[i]:.1f})"
                for i in leader_mask
            ],
            hoverinfo="text",
            name="Leaders",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(scaleanchor="y", constrain="domain", showgrid=False),
        yaxis=dict(constrain="domain", showgrid=False),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Convergence chart
# ═══════════════════════════════════════════════════════════════════════


def _convergence_chart(engine: SimulationEngine, field_key: str | None, field_type: str) -> go.Figure | None:
    history = engine.history
    if len(history) < 2:
        return None

    deltas: list[float] = []
    for r in range(1, len(history)):
        prev_round = history[r - 1]
        curr_round = history[r]
        common_ids = set(prev_round.keys()) & set(curr_round.keys())
        if not common_ids:
            deltas.append(0.0)
            continue

        if field_type in ("scalar",):
            def _extract(results, fk):
                if fk is None:
                    return {k: float(v) if v != float("inf") else 0.0
                            for k, v in results.items()}
                return {k: float(v[fk]) if isinstance(v, dict) and v.get(fk) is not None
                        else 0.0 for k, v in results.items()}

            pf = _extract(prev_round, field_key)
            cf = _extract(curr_round, field_key)
            max_delta = max(abs(cf.get(d, 0) - pf.get(d, 0)) for d in common_ids)
            deltas.append(max_delta)
        elif field_type == "categorical":
            def _cat(results, fk):
                if fk is None:
                    return results
                return {k: v[fk] if isinstance(v, dict) else v
                        for k, v in results.items()}
            pf = _cat(prev_round, field_key)
            cf = _cat(curr_round, field_key)
            changed = sum(1 for d in common_ids if str(pf.get(d)) != str(cf.get(d)))
            deltas.append(float(changed))
        elif field_type == "leader":
            pf = {k: v.get("leader_id", -1) if isinstance(v, dict) else v
                  for k, v in prev_round.items()}
            cf = {k: v.get("leader_id", -1) if isinstance(v, dict) else v
                  for k, v in curr_round.items()}
            changed = sum(1 for d in common_ids if pf.get(d) != cf.get(d))
            deltas.append(float(changed))
        elif field_type == "multi":
            # Use density sub-field for convergence
            pf = {k: float(v.get("density", 0)) if isinstance(v, dict) else 0.0
                  for k, v in prev_round.items()}
            cf = {k: float(v.get("density", 0)) if isinstance(v, dict) else 0.0
                  for k, v in curr_round.items()}
            max_delta = max(abs(cf.get(d, 0) - pf.get(d, 0)) for d in common_ids)
            deltas.append(max_delta)
        else:
            deltas.append(0.0)

    rounds = list(range(2, len(history) + 1))

    y_label = ("Max |change|" if field_type in ("scalar", "multi")
               else "Devices changed")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=deltas,
            mode="lines+markers",
            marker=dict(size=4, color="#00cc96"),
            line=dict(width=2, color="#00cc96"),
            hovertemplate="Round %{x}<br>" + y_label + ": %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Convergence (self-stabilization)", font=dict(size=14)),
        xaxis=dict(title="Round"),
        yaxis=dict(title=y_label),
        height=250,
        **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k != "height"},
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Animated HTML rendering (browser-side Plotly.react)
# ═══════════════════════════════════════════════════════════════════════


def _fixed_axis_ranges(engine: SimulationEngine, padding: float = 0.5):
    """Compute fixed x/y axis ranges from device positions."""
    xs, ys, _ = _get_positions(engine)
    if not xs:
        return None, None
    x_range = [min(xs) - padding, max(xs) + padding]
    y_range = [min(ys) - padding, max(ys) + padding]
    return x_range, y_range


def _frame_to_json(
    engine: SimulationEngine,
    demo: dict,
    demo_name: str,
    results: dict[int, Any],
    round_num: int,
    show_edges: bool,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
) -> dict | None:
    """Build a serialised Plotly figure dict for one animation frame.

    When *x_range* / *y_range* are supplied the axes are locked so
    that scale never jumps between frames or across batches.
    """
    field_type = demo["field_type"]

    if field_type == "scalar":
        fk = demo["field_key"]
        if fk is None:
            field = {k: float(v) if v != float("inf") else 0.0
                     for k, v in results.items()}
        else:
            field = {k: float(v[fk]) if isinstance(v, dict) else float(v)
                     for k, v in results.items()}
            field = {k: 0.0 if v == float("inf") else v
                     for k, v in field.items()}
        fig = _scalar_figure(
            engine, field,
            title=f"{demo_name} \u2014 Round {round_num}",
            colorscale=demo["colorscale"],
            show_edges=show_edges,
        )
    elif field_type == "categorical":
        raw = {k: v[demo["field_key"]] if isinstance(v, dict) else v
               for k, v in results.items()}
        cat_field = {did: "CHANNEL" if v else "OFF"
                     for did, v in raw.items()}
        fig = _categorical_figure(
            engine, cat_field,
            title=f"{demo_name} \u2014 Round {round_num}",
            color_map={"CHANNEL": "#ef553b", "OFF": "#636efa"},
            show_edges=show_edges,
        )
    elif field_type == "leader":
        field = {k: float(v.get("leader_id", -1)) if isinstance(v, dict) else float(v)
                 for k, v in results.items()}
        fig = _leader_figure(
            engine, field, results,
            title=f"{demo_name} \u2014 Round {round_num}",
            colorscale=demo["colorscale"],
            show_edges=show_edges,
        )
    elif field_type == "multi":
        # Use density sub-field for animation
        field = {k: float(v.get("density", 0)) if isinstance(v, dict) else 0.0
                 for k, v in results.items()}
        fig = _scalar_figure(
            engine, field,
            title=f"Crowd Density \u2014 Round {round_num}",
            colorscale="YlOrRd",
            show_edges=show_edges,
        )
    else:
        return None

    # Lock axis ranges so there is no scale jump between batches
    if x_range is not None:
        fig.update_layout(xaxis_range=x_range)
    if y_range is not None:
        fig.update_layout(yaxis_range=y_range)

    return json.loads(fig.to_json())


def _build_animation_html(
    frames_json: list[dict],
    play_speed_ms: int,
    height: int = 600,
) -> str:
    """Self-contained HTML with Plotly.js animation via Plotly.react().

    The browser handles the animation loop — no Streamlit reruns needed
    while the batch plays.
    """
    data_json = json.dumps(frames_json)
    transition_ms = int(min(play_speed_ms * 0.7, 300))

    return f"""<!DOCTYPE html>
<html><head>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  html, body {{ margin:0; padding:0; background:#0e1117; overflow:hidden; }}
  #chart {{
    width:100%; height:{height}px;
    animation: fadeIn 0.35s ease-in;
  }}
  @keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
  }}
</style>
</head><body>
<div id="chart"></div>
<script>
(function() {{
  var frames = {data_json};
  var layout = JSON.parse(JSON.stringify(frames[0].layout));
  layout.transition = {{duration:{transition_ms}, easing:'cubic-in-out'}};

  Plotly.newPlot('chart', frames[0].data, layout,
                 {{responsive:true, displayModeBar:false}});

  if (frames.length < 2) return;

  var idx = 0;
  function step() {{
    idx++;
    if (idx >= frames.length) return;   // stop at end (no loop)
    var f = frames[idx];
    var lyt = JSON.parse(JSON.stringify(f.layout));
    lyt.transition = {{duration:{transition_ms}, easing:'cubic-in-out'}};
    Plotly.react('chart', f.data, lyt);
  }}
  setInterval(step, {play_speed_ms});
}})();
</script>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════
#  Field rendering dispatcher
# ═══════════════════════════════════════════════════════════════════════


def _render_field(
    engine: SimulationEngine,
    demo: dict,
    demo_name: str,
    results: dict[int, Any],
    round_num: int,
    show_edges: bool,
):
    """Render the appropriate field visualization."""
    field_type = demo["field_type"]

    if field_type == "scalar":
        fk = demo["field_key"]
        if fk is None:
            field = {k: float(v) if v != float("inf") else 0.0
                     for k, v in results.items()}
        else:
            field = {k: float(v[fk]) if isinstance(v, dict) else float(v)
                     for k, v in results.items()}
            field = {k: 0.0 if v == float("inf") else v for k, v in field.items()}

        vals = list(field.values())
        fig = _scalar_figure(
            engine, field,
            title=f"{demo_name} \u2014 Round {round_num}",
            colorscale=demo["colorscale"],
            show_edges=show_edges,
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_scalar")
        c1, c2, c3 = st.columns(3)
        c1.metric("Min", f"{min(vals):.2f}")
        c2.metric("Max", f"{max(vals):.2f}")
        c3.metric("Mean", f"{np.mean(vals):.2f}")

    elif field_type == "categorical":
        raw = {k: v[demo["field_key"]] if isinstance(v, dict) else v
               for k, v in results.items()}
        cat_field = {did: "CHANNEL" if v else "OFF" for did, v in raw.items()}
        fig = _categorical_figure(
            engine, cat_field,
            title=f"{demo_name} \u2014 Round {round_num}",
            color_map={"CHANNEL": "#ef553b", "OFF": "#636efa"},
            show_edges=show_edges,
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_categorical")
        on_count = sum(1 for v in cat_field.values() if v == "CHANNEL")
        st.metric("Devices on channel", on_count)

    elif field_type == "leader":
        field = {k: float(v.get("leader_id", -1)) if isinstance(v, dict) else float(v)
                 for k, v in results.items()}
        fig = _leader_figure(
            engine, field, results,
            title=f"{demo_name} \u2014 Round {round_num}",
            colorscale=demo["colorscale"],
            show_edges=show_edges,
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_leader")
        is_leader = {
            did: r.get("is_leader", False)
            for did, r in results.items()
            if isinstance(r, dict)
        }
        n_leaders = sum(1 for v in is_leader.values() if v)
        unique_partitions = len(set(int(v) for v in field.values()))
        c1, c2 = st.columns(2)
        c1.metric("Leaders", n_leaders)
        c2.metric("Partitions", unique_partitions)

    elif field_type == "multi":
        tab_density, tab_alert, tab_exit = st.tabs(
            ["Density", "Alert Levels", "Exit Distance"]
        )

        with tab_density:
            field = {k: float(v.get("density", 0)) if isinstance(v, dict) else 0.0
                     for k, v in results.items()}
            vals = list(field.values())
            fig = _scalar_figure(
                engine, field,
                title=f"Crowd Density \u2014 Round {round_num}",
                colorscale="YlOrRd",
                show_edges=show_edges,
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_density")
            c1, c2, c3 = st.columns(3)
            c1.metric("Min", f"{min(vals):.3f}")
            c2.metric("Max", f"{max(vals):.3f}")
            c3.metric("Mean", f"{np.mean(vals):.3f}")

        with tab_alert:
            alert_field = {k: v.get("alert", "?") if isinstance(v, dict) else "?"
                           for k, v in results.items()}
            fig = _categorical_figure(
                engine, alert_field,
                title=f"Alert Levels \u2014 Round {round_num}",
                color_map={
                    "NORMAL": "#2ca02c",
                    "WARNING": "#ff7f0e",
                    "CRITICAL": "#d62728",
                },
                show_edges=show_edges,
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_alert")
            for level in ["NORMAL", "WARNING", "CRITICAL"]:
                count = sum(1 for v in alert_field.values() if v == level)
                if count > 0:
                    st.metric(level, count)

        with tab_exit:
            field = {k: float(v.get("exit_dist", 0)) if isinstance(v, dict) else 0.0
                     for k, v in results.items()}
            field = {k: 0.0 if v == float("inf") else v for k, v in field.items()}
            fig = _scalar_figure(
                engine, field,
                title=f"Distance to Exit \u2014 Round {round_num}",
                colorscale="Blues_r",
                show_edges=show_edges,
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_exit")


# ═══════════════════════════════════════════════════════════════════════
#  Device role editor
# ═══════════════════════════════════════════════════════════════════════


def _role_editor(engine: SimulationEngine, demo: dict, demo_name: str) -> dict[int, str] | None:
    """Show a device role editor. Returns role overrides or None."""
    roles = demo.get("roles", [])
    if not roles:
        return None

    with st.expander("Configure Device Roles", expanded=False):
        st.caption(
            "Edit the **Role** column to assign roles to devices. "
            "Click **Apply & Reset** to restart the simulation with new roles."
        )

        devices = engine.network.devices
        ids = sorted(devices.keys())

        # Build current role data
        rows_data = []
        for did in ids:
            dev = devices[did]
            pos = f"({dev.position[0]:.1f}, {dev.position[1]:.1f})"
            # Determine current role
            role = "None"
            if dev.sensors.get("is_source"):
                role = "Source"
            elif dev.sensors.get("is_dest"):
                role = "Dest"
            elif dev.sensors.get("is_monitor"):
                role = "Monitor"
            elif dev.sensors.get("is_exit"):
                role = "Exit"
            rows_data.append({"Device": did, "Position": pos, "Role": role})

        df = pd.DataFrame(rows_data)
        role_options = ["None"] + roles

        edited = st.data_editor(
            df,
            column_config={
                "Device": st.column_config.NumberColumn(disabled=True),
                "Position": st.column_config.TextColumn(disabled=True),
                "Role": st.column_config.SelectboxColumn(
                    options=role_options, required=True
                ),
            },
            use_container_width=True,
            hide_index=True,
            key=f"role_editor_{demo_name}",
        )

        if st.button("Apply & Reset Simulation", type="primary"):
            overrides = {}
            for _, row in edited.iterrows():
                if row["Role"] != "None":
                    overrides[int(row["Device"])] = row["Role"]
            return overrides

    return None


# ═══════════════════════════════════════════════════════════════════════
#  Self-healing controls
# ═══════════════════════════════════════════════════════════════════════


def _selfheal_controls(engine: SimulationEngine, rows: int, cols: int):
    """Show device removal controls for self-healing demo."""
    with st.expander("Remove Devices (Self-Healing)", expanded=False):
        st.caption(
            "Select a rectangular region of devices to remove, then click "
            "**Remove** and continue stepping to see self-healing."
        )
        c1, c2 = st.columns(2)
        with c1:
            r_start = st.number_input("Row start", 0, rows - 1, rows // 3)
            r_end = st.number_input("Row end", 0, rows - 1, 2 * rows // 3)
        with c2:
            c_start = st.number_input("Col start", 0, cols - 1, cols // 3)
            c_end = st.number_input("Col end", 0, cols - 1, 2 * cols // 3)

        if st.button("Remove Region", type="secondary"):
            removed = 0
            for r in range(int(r_start), int(r_end) + 1):
                for c in range(int(c_start), int(c_end) + 1):
                    did = r * cols + c
                    if did in engine.network.devices:
                        engine.network.remove_device(did)
                        removed += 1
            engine.network.update_neighbors()
            st.success(f"Removed {removed} devices. Continue stepping to see self-healing.")
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════
#  Main app
# ═══════════════════════════════════════════════════════════════════════


def _simulation_panel(
    engine: SimulationEngine,
    demo: dict[str, Any],
    demo_name: str,
    play_speed: int,
    show_edges: bool,
    topology: str,
    rows: int,
    cols: int,
    spacing: float,
    comm_range: float,
    grain: float,
    pulse_period: int,
    n_random: int,
    width: float,
    height: float,
    run_n: int,
) -> None:
    """Simulation panel (called inside a ``@st.fragment``).

    **Auto-play** batch-computes steps and renders a self-contained
    Plotly.js animation in the browser — no Streamlit reruns while
    the batch plays.  ``run_every`` on the enclosing fragment triggers
    the next batch automatically.

    **Manual mode** (Step / Run N) uses standard ``st.plotly_chart``
    for full interactivity.
    """
    # ── Consume pending actions from outer-zone buttons ───────────
    if st.session_state.pop("pending_step", False):
        engine.step()
    if st.session_state.pop("pending_run_n", False):
        engine.run(st.session_state.get("run_n_count", 10))

    _is_playing = st.session_state.get("auto_playing", False)

    # ══════════════════════════════════════════════════════════════
    #  AUTO-PLAY: batch compute + browser-side animation
    # ══════════════════════════════════════════════════════════════
    if _is_playing:
        batch = max(5, min(30, int(8000 / play_speed)))
        engine.run(batch)

        # Fixed axis ranges prevent scale jumps between batches
        x_range, y_range = _fixed_axis_ranges(engine)

        start_round = engine.round_count - batch + 1
        frames: list[dict] = []

        # Overlap: prepend last frame of previous batch so the
        # visual transition starts from where the last batch ended
        prev_frame = st.session_state.get("_last_anim_frame")
        if prev_frame is not None:
            frames.append(prev_frame)

        for i in range(batch):
            r = engine.history[start_round - 1 + i]
            fj = _frame_to_json(engine, demo, demo_name, r,
                                start_round + i, show_edges,
                                x_range=x_range, y_range=y_range)
            if fj is not None:
                frames.append(fj)

        # Store last frame for next batch's overlap
        if frames:
            st.session_state["_last_anim_frame"] = frames[-1]

        st.caption(
            f"**Rounds {start_round}\u2013{engine.round_count}** "
            f"(playing \u2014 {batch} frames)"
        )

        if frames:
            html = _build_animation_html(frames, play_speed)
            components.html(html, height=650)

        # Metrics for the last frame in the batch
        last_results = engine.history[-1]
        _show_summary_metrics(demo, last_results)

        # No sleep / st.rerun() — ``run_every`` triggers next batch
        return

    # ══════════════════════════════════════════════════════════════
    #  MANUAL MODE: standard interactive rendering
    # ══════════════════════════════════════════════════════════════
    st.caption(f"**Round {engine.round_count}**")

    # ── History slider ───────────────────────────────────────────
    if engine.round_count > 0:
        if engine.round_count > 1:
            display_round = st.slider(
                "View round",
                1, engine.round_count, engine.round_count,
                key="history_slider",
            )
        else:
            display_round = 1

        results = engine.history[display_round - 1]
    else:
        results = None
        display_round = 0

    # ── Self-healing controls ────────────────────────────────────
    if demo_name == "Self-Healing" and topology == "Grid":
        _selfheal_controls(engine, rows, cols)

    # ── Device role editor ───────────────────────────────────────
    role_result = _role_editor(engine, demo, demo_name)
    if role_result is not None:
        net = _build_network(
            demo_name, rows, cols, spacing, comm_range, grain,
            pulse_period, topology, n_random, width, height,
            role_overrides=role_result,
        )
        new_engine = SimulationEngine(net, demo["program"])
        st.session_state.engine = new_engine
        st.session_state.auto_playing = False
        st.rerun()

    # ── Render field visualization ───────────────────────────────
    if results is None:
        st.info("Press **Step** or **Auto-Play** to start the simulation.")
        return

    _render_field(engine, demo, demo_name, results, display_round, show_edges)

    # ── Deep description panels ──────────────────────────────────
    desc_left, desc_right = st.columns(2)
    with desc_left:
        with st.expander("Theory & Concepts", expanded=False):
            st.markdown(demo.get("theory", ""), unsafe_allow_html=True)
    with desc_right:
        with st.expander("Practical Guide", expanded=False):
            st.markdown(demo.get("practical", ""), unsafe_allow_html=True)

    # ── Convergence chart ────────────────────────────────────────
    if engine.round_count >= 2:
        conv_fig = _convergence_chart(
            engine, demo["field_key"], demo["field_type"]
        )
        if conv_fig is not None:
            st.plotly_chart(conv_fig, use_container_width=True,
                            key="chart_convergence")


def _show_summary_metrics(demo: dict, results: dict[int, Any]) -> None:
    """Show compact metrics for the last frame (used during auto-play)."""
    ft = demo["field_type"]
    fk = demo["field_key"]

    if ft == "scalar":
        if fk is None:
            vals = [float(v) if v != float("inf") else 0.0
                    for v in results.values()]
        else:
            vals = [float(v[fk]) if isinstance(v, dict) else float(v)
                    for v in results.values()]
            vals = [0.0 if v == float("inf") else v for v in vals]
        c1, c2, c3 = st.columns(3)
        c1.metric("Min", f"{min(vals):.2f}")
        c2.metric("Max", f"{max(vals):.2f}")
        c3.metric("Mean", f"{np.mean(vals):.2f}")

    elif ft == "categorical":
        raw = {k: v[fk] if isinstance(v, dict) else v
               for k, v in results.items()}
        on = sum(1 for v in raw.values() if v)
        st.metric("Devices on channel", on)

    elif ft == "leader":
        is_leader = {did: r.get("is_leader", False)
                     for did, r in results.items()
                     if isinstance(r, dict)}
        st.metric("Leaders", sum(1 for v in is_leader.values() if v))

    elif ft == "multi":
        vals = [float(v.get("density", 0)) if isinstance(v, dict) else 0.0
                for v in results.values()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Min Density", f"{min(vals):.3f}")
        c2.metric("Max Density", f"{max(vals):.3f}")
        c3.metric("Mean Density", f"{np.mean(vals):.3f}")


def main() -> None:
    st.set_page_config(
        page_title="Computational Fields Simulator",
        page_icon="\u26a1",
        layout="wide",
    )

    st.markdown(
        "<h1 style='margin-bottom:0'>\u26a1 Computational Fields Simulator</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Interactive simulation of aggregate computing building blocks")

    # ── Sidebar ─────────────────────────────────────────────────────
    st.sidebar.header("Demo")
    demo_name = st.sidebar.selectbox("Scenario", list(DEMOS.keys()))
    demo = DEMOS[demo_name]
    st.sidebar.caption(demo["description"])

    st.sidebar.header("Topology")
    topology = st.sidebar.radio("Type", ["Grid", "Random"], horizontal=True)

    if topology == "Grid":
        rows = st.sidebar.slider("Rows", 3, 20, 10)
        cols = st.sidebar.slider("Columns", 3, 20, 10)
        spacing = st.sidebar.slider("Spacing", 0.5, 3.0, 1.0, 0.1)
        n_random, width, height = 50, 10.0, 10.0
    else:
        rows, cols, spacing = 10, 10, 1.0
        n_random = st.sidebar.slider("Devices", 10, 200, 50)
        width = st.sidebar.slider("Area width", 3.0, 20.0, 10.0, 0.5)
        height = st.sidebar.slider("Area height", 3.0, 20.0, 10.0, 0.5)

    comm_range = st.sidebar.slider("Comm Range", 0.5, 5.0, 1.5, 0.1)

    # Demo-specific params
    grain = 3.0
    pulse_period = 6
    if demo_name == "Sparse Leaders":
        grain = st.sidebar.slider("Grain (leader spacing)", 1.0, 8.0, 3.0, 0.5)
    if demo_name == "Wave Propagation":
        pulse_period = st.sidebar.slider("Pulse period (rounds)", 2, 20, 6)

    st.sidebar.header("Playback")
    run_n = st.sidebar.slider("Rounds to run", 1, 100, 10)
    play_speed = st.sidebar.slider("Auto-play speed (ms)", 100, 1500, 400, 50)

    st.sidebar.header("Display")
    show_edges = st.sidebar.checkbox("Show edges", True)

    # ── Session state init ──────────────────────────────────────────
    config_key = (
        demo_name, rows, cols, spacing, comm_range, grain,
        pulse_period, topology, n_random, width, height,
    )

    if "config_key" not in st.session_state or st.session_state.config_key != config_key:
        net = _build_network(
            demo_name, rows, cols, spacing, comm_range, grain,
            pulse_period, topology, n_random, width, height,
        )
        engine = SimulationEngine(net, demo["program"])
        st.session_state.engine = engine
        st.session_state.config_key = config_key
        st.session_state.auto_playing = False

    engine: SimulationEngine = st.session_state.engine

    # ── Control bar ─────────────────────────────────────────────────
    # Buttons set session-state flags; the fragment consumes them.
    # This prevents stepping from happening outside the fragment.
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("\u25b6 Step", use_container_width=True):
            st.session_state.pending_step = True
            st.session_state.auto_playing = False
    with col2:
        if st.button(f"\u23e9 Run {run_n}", use_container_width=True):
            st.session_state.pending_run_n = True
            st.session_state.run_n_count = run_n
            st.session_state.auto_playing = False
    with col3:
        play_label = "\u23f8 Pause" if st.session_state.get("auto_playing") else "\u23ef Auto-Play"
        if st.button(play_label, use_container_width=True):
            new_state = not st.session_state.get("auto_playing", False)
            st.session_state.auto_playing = new_state
            if not new_state:
                st.session_state.pop("_last_anim_frame", None)
    with col4:
        if st.button("\u21ba Reset", use_container_width=True):
            net = _build_network(
                demo_name, rows, cols, spacing, comm_range, grain,
                pulse_period, topology, n_random, width, height,
            )
            engine = SimulationEngine(net, demo["program"])
            st.session_state.engine = engine
            st.session_state.auto_playing = False
            st.session_state.pop("_last_anim_frame", None)

    # ── Simulation panel ─────────────────────────────────────────────
    # During auto-play the fragment auto-reruns every `_interval`
    # seconds to batch-compute a new set of frames.  Each batch
    # plays smoothly in the browser via Plotly.react().
    _playing = st.session_state.get("auto_playing", False)
    if _playing:
        _batch = max(5, min(30, int(8000 / play_speed)))
        _interval = _batch * play_speed / 1000.0 + 0.5
    else:
        _interval = None

    @st.fragment(run_every=_interval)
    def _panel():
        _simulation_panel(
            engine, demo, demo_name, play_speed, show_edges,
            topology, rows, cols, spacing, comm_range, grain,
            pulse_period, n_random, width, height, run_n,
        )

    _panel()


if __name__ == "__main__":
    main()
