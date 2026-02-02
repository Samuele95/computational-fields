"""Interactive Dash UI for Computational Fields simulation.

Run with:
    python computational_fields/visualization/dash_app.py

Opens at http://127.0.0.1:8050
"""

from __future__ import annotations

import math
import operator
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash
from dash import dcc, html, ctx, dash_table, Input, Output, State, no_update, callback

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
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#0a0a0f",
    font=dict(family="Inter, -apple-system, sans-serif", color="#e8eaed"),
    margin=dict(l=20, r=20, t=50, b=20),
    height=600,
    uirevision="stable",
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
    """Gradient with pulsing source -- source toggles on/off."""
    is_source_device = ctx.sense("is_source") or False
    pulse_period = ctx.sense("pulse_period") or 6
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
            "reads values from the *same* `share` invocation -- the formal "
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
            "3. **Auto-Play** animates the simulation -- adjust the speed "
            "slider in the sidebar.\n"
            "4. Try changing the **Topology** to Random to see how the "
            "gradient adapts to irregular networks.\n"
            "5. Use the **Device Role Editor** below the chart to move "
            "the source to a different device, then reset.\n"
            "6. Watch the **Convergence Chart** at the bottom -- it "
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
            "as well -- a property guaranteed by the field calculus "
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
            "3. Try **Random topology** -- the channel follows the "
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
            "is close enough -- the device **elects itself** as a new "
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
            "the network -- the foundation for spatial partitioning in "
            "aggregate computing.\n\n"
            "#### Relation to the Partition Block\n\n"
            "The S block is used internally by the **partition** "
            "composite, which divides the network into regions, each "
            "managed by a leader.  This enables independent sub-programs "
            "to run within each region -- a form of distributed spatial "
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
            "2. Use **Auto-Play** to watch the election converge -- "
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
            "### Crowd Safety Monitoring -- Case Study\n\n"
            "This demo implements the complete crowd safety scenario "
            "from the Computational Fields literature, combining all "
            "four building blocks (G, C, S, T) into a single program.\n\n"
            "#### Architecture\n\n"
            "The system has three layers that run simultaneously on "
            "every device:\n\n"
            "**1. Density Estimation (T block -- time decay)**\n\n"
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
            "exits -- the entire system self-reconfigures.\n"
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
            "network -- a dynamic visualization of self-stabilization."
        ),
        "theory": (
            "### Wave Propagation Theory\n\n"
            "This demo visualizes a fundamental consequence of "
            "**self-stabilization**: the system continuously adapts to "
            "changing inputs.  When the environment changes, the field "
            "does not need to be restarted -- it evolves toward the new "
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
            "-- they carry distance information, not energy."
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
            "1. **Auto-Play is recommended** for this demo -- set a "
            "moderate speed (300-500ms) and watch the waves.\n"
            "2. Adjust the **Pulse Period** slider: shorter periods "
            "create rapid flicker, longer periods allow full expansion "
            "before the next pulse.\n"
            "3. Increase the grid to 15x15 or larger to see the "
            "wavefront clearly.\n"
            "4. Compare **Grid** vs **Random** topology -- waves are "
            "smoother on grids and more irregular on random networks.\n"
            "5. The convergence chart shows periodic spikes corresponding "
            "to each pulse transition -- the field never fully stabilizes "
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
            "No reprogramming, restart, or reconfiguration is needed -- "
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
            "3. Continue stepping or use Auto-Play -- watch the field "
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

    def sensors_fn(did: int, row: int, col: int) -> dict:
        s = base_fn(did, row, col)
        if role_overrides and did in role_overrides:
            role = role_overrides[did]
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
            marker=dict(size=4, color="#7c5cfc"),
            line=dict(width=2, color="#7c5cfc"),
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


def _fixed_axis_ranges(engine: SimulationEngine, padding: float = 0.5):
    """Compute fixed x/y axis ranges from device positions."""
    xs, ys, _ = _get_positions(engine)
    if not xs:
        return None, None
    x_range = [min(xs) - padding, max(xs) + padding]
    y_range = [min(ys) - padding, max(ys) + padding]
    return x_range, y_range


# ═══════════════════════════════════════════════════════════════════════
#  Summary metrics (pure-Python, returns HTML)
# ═══════════════════════════════════════════════════════════════════════


def _summary_metrics_html(demo: dict, results: dict[int, Any]) -> str:
    """Return a small HTML snippet with key metrics."""
    ft = demo["field_type"]
    fk = demo["field_key"]

    def _metric(label: str, value: str) -> str:
        return (
            f'<span style="display:inline-block;margin-right:24px;">'
            f'<span style="color:#999;font-size:0.85em;">{label}</span><br>'
            f'<span style="font-size:1.3em;font-weight:bold;color:#fafafa;">{value}</span>'
            f'</span>'
        )

    if ft == "scalar":
        if fk is None:
            vals = [float(v) if v != float("inf") else 0.0 for v in results.values()]
        else:
            vals = [float(v[fk]) if isinstance(v, dict) else float(v) for v in results.values()]
            vals = [0.0 if v == float("inf") else v for v in vals]
        if not vals:
            return ""
        return _metric("Min", f"{min(vals):.2f}") + _metric("Max", f"{max(vals):.2f}") + _metric("Mean", f"{np.mean(vals):.2f}")

    elif ft == "categorical":
        raw = {k: v[fk] if isinstance(v, dict) else v for k, v in results.items()}
        on = sum(1 for v in raw.values() if v)
        return _metric("Devices on channel", str(on))

    elif ft == "leader":
        is_leader = {did: r.get("is_leader", False) for did, r in results.items() if isinstance(r, dict)}
        return _metric("Leaders", str(sum(1 for v in is_leader.values() if v)))

    elif ft == "multi":
        vals = [float(v.get("density", 0)) if isinstance(v, dict) else 0.0 for v in results.values()]
        if not vals:
            return ""
        return _metric("Min Density", f"{min(vals):.3f}") + _metric("Max Density", f"{max(vals):.3f}") + _metric("Mean Density", f"{np.mean(vals):.3f}")

    return ""


# ═══════════════════════════════════════════════════════════════════════
#  Figure builder dispatcher
# ═══════════════════════════════════════════════════════════════════════


def _build_figure(
    engine: SimulationEngine,
    demo_name: str,
    results: dict[int, Any],
    show_edges: bool = True,
    crowd_tab: str = "density",
) -> go.Figure:
    """Build the appropriate figure for the current demo and results."""
    demo = DEMOS[demo_name]
    ft = demo["field_type"]
    fk = demo["field_key"]
    colorscale = demo["colorscale"]
    rnd = engine.round_count

    if ft == "scalar":
        if fk is None:
            field = {k: float(v) for k, v in results.items()}
        else:
            field = {k: float(v[fk]) if isinstance(v, dict) else float(v)
                     for k, v in results.items()}
        fig = _scalar_figure(engine, field, f"{demo_name} -- Round {rnd}",
                             colorscale=colorscale, show_edges=show_edges)

    elif ft == "categorical":
        field = {k: v[fk] if isinstance(v, dict) else v for k, v in results.items()}
        color_map = {"True": "#636efa", "False": "#ef553b"}
        fig = _categorical_figure(engine, field, f"{demo_name} -- Round {rnd}",
                                  color_map=color_map, show_edges=show_edges)

    elif ft == "leader":
        field = {k: v.get(fk, -1) if isinstance(v, dict) else v
                 for k, v in results.items()}
        fig = _leader_figure(engine, field, results, f"{demo_name} -- Round {rnd}",
                             colorscale=colorscale, show_edges=show_edges)

    elif ft == "multi":
        # Crowd monitoring with sub-field selection
        if crowd_tab == "alert":
            field = {k: v.get("alert", "?") if isinstance(v, dict) else "?"
                     for k, v in results.items()}
            color_map = {"NORMAL": "#00cc96", "WARNING": "#ffa15a", "CRITICAL": "#ef553b", "?": "#666"}
            fig = _categorical_figure(engine, field, f"Crowd Alert -- Round {rnd}",
                                      color_map=color_map, show_edges=show_edges)
        elif crowd_tab == "exit_dist":
            field = {k: float(v.get("exit_dist", 0)) if isinstance(v, dict) else 0.0
                     for k, v in results.items()}
            fig = _scalar_figure(engine, field, f"Exit Distance -- Round {rnd}",
                                 colorscale="Cividis", show_edges=show_edges)
        else:  # density
            field = {k: float(v.get("density", 0)) if isinstance(v, dict) else 0.0
                     for k, v in results.items()}
            fig = _scalar_figure(engine, field, f"Crowd Density -- Round {rnd}",
                                 colorscale="Hot", show_edges=show_edges)
    else:
        fig = go.Figure()

    # Apply fixed axis ranges for stability
    x_range, y_range = _fixed_axis_ranges(engine)
    if x_range:
        fig.update_xaxes(range=x_range)
    if y_range:
        fig.update_yaxes(range=y_range)

    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Server-side state (single user)
# ═══════════════════════════════════════════════════════════════════════

_engine: SimulationEngine | None = None
_engine_key: tuple | None = None
_role_overrides: dict[int, str] = {}


def _get_or_create_engine(
    demo_name: str,
    rows: int,
    cols: int,
    spacing: float,
    comm_range: float,
    grain: float,
    pulse_period: int,
    topology: str,
) -> SimulationEngine:
    global _engine, _engine_key, _role_overrides
    key = (demo_name, rows, cols, spacing, comm_range, grain, pulse_period, topology)
    if _engine_key != key or _engine is None:
        n_random = rows * cols
        width = (cols - 1) * spacing
        height = (rows - 1) * spacing
        net = _build_network(
            demo_name, rows, cols, spacing, comm_range, grain, pulse_period,
            topology, n_random, max(width, 1.0), max(height, 1.0),
            role_overrides=_role_overrides if _role_overrides else None,
        )
        _engine = SimulationEngine(net, DEMOS[demo_name]["program"])
        _engine_key = key
    return _engine


# ═══════════════════════════════════════════════════════════════════════
#  Dash app + dark theme
# ═══════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    title="Computational Fields Simulator",
    suppress_callback_exceptions=True,
)

app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0a0a0f;
            --bg-surface: rgba(15, 15, 25, 0.8);
            --bg-elevated: rgba(25, 25, 45, 0.6);
            --bg-hover: rgba(40, 40, 70, 0.5);
            --glass: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
            --glass-border-hover: rgba(255, 255, 255, 0.15);
            --text-primary: #e8eaed;
            --text-secondary: #9aa0a6;
            --text-muted: #5f6368;
            --accent: #7c5cfc;
            --accent-glow: rgba(124, 92, 252, 0.3);
            --accent-hover: #9b7dff;
            --accent-green: #34d399;
            --accent-green-glow: rgba(52, 211, 153, 0.25);
            --accent-red: #f87171;
            --accent-red-glow: rgba(248, 113, 113, 0.25);
            --accent-blue: #60a5fa;
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 20px;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 16px rgba(0,0,0,0.4);
            --shadow-lg: 0 8px 32px rgba(0,0,0,0.5);
            --transition: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg-base); color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6; -webkit-font-smoothing: antialiased;
        }
        /* Subtle gradient bg */
        body::before {
            content: ''; position: fixed; inset: 0; z-index: -1;
            background:
                radial-gradient(ellipse 80% 50% at 20% 40%, rgba(124,92,252,0.08) 0%, transparent 60%),
                radial-gradient(ellipse 60% 40% at 80% 80%, rgba(52,211,153,0.05) 0%, transparent 50%);
        }
        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

        /* ── Sidebar ── */
        .sidebar {
            position: fixed; top: 0; left: 0; bottom: 0; width: 320px;
            background: var(--bg-surface);
            backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
            border-right: 1px solid var(--glass-border);
            padding: 24px 20px; overflow-y: auto;
            box-sizing: border-box;
        }
        .sidebar h2 {
            font-size: 1.35em; font-weight: 700; margin-bottom: 20px;
            background: linear-gradient(135deg, #7c5cfc, #60a5fa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.02em;
        }
        .sidebar label {
            display: block; margin: 10px 0 4px 0;
            color: var(--text-secondary); font-size: 0.8em; font-weight: 500;
            letter-spacing: 0.01em;
        }
        .sidebar-section {
            margin-top: 20px; padding-top: 16px;
            border-top: 1px solid var(--glass-border);
        }
        .sidebar-section-header {
            font-size: 0.65em; color: var(--text-muted); text-transform: uppercase;
            letter-spacing: 0.12em; font-weight: 600; margin-bottom: 8px;
        }

        /* ── Main area ── */
        .main-area { margin-left: 320px; padding: 24px 32px; }

        /* ── Buttons ── */
        .control-bar { display: flex; gap: 10px; margin: 16px 0; flex-wrap: wrap; }
        .control-bar button {
            padding: 10px 20px;
            border: 1px solid var(--glass-border); border-radius: var(--radius-md);
            background: var(--bg-elevated);
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            color: var(--text-secondary); cursor: pointer;
            font-family: 'Inter', sans-serif; font-size: 0.85em; font-weight: 500;
            min-width: 88px;
            transition: all var(--transition);
        }
        .control-bar button:hover {
            background: var(--bg-hover); border-color: var(--glass-border-hover);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
            color: var(--text-primary);
        }
        .control-bar button:active { transform: translateY(0); box-shadow: var(--shadow-sm); }
        .control-bar button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
        .control-bar button.primary {
            background: linear-gradient(135deg, #059669, #34d399);
            border-color: rgba(52,211,153,0.3); color: #fff; font-weight: 600;
        }
        .control-bar button.primary:hover {
            box-shadow: 0 4px 20px var(--accent-green-glow), var(--shadow-md);
            border-color: rgba(52,211,153,0.5);
        }
        .control-bar button.danger {
            background: linear-gradient(135deg, #dc2626, #f87171);
            border-color: rgba(248,113,113,0.3); color: #fff; font-weight: 600;
        }
        .control-bar button.danger:hover {
            box-shadow: 0 4px 20px var(--accent-red-glow), var(--shadow-md);
            border-color: rgba(248,113,113,0.5);
        }

        /* ── Round badge ── */
        .round-badge {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 16px; margin: 10px 0;
            background: var(--bg-elevated);
            border: 1px solid var(--glass-border); border-radius: 999px;
            color: var(--accent-blue); font-weight: 600; font-size: 0.85em;
            backdrop-filter: blur(12px);
            box-shadow: var(--shadow-sm);
        }

        /* ── Metrics card ── */
        .metrics-bar { margin: 12px 0; min-height: 48px; }
        .metrics-card {
            background: var(--bg-elevated);
            border: 1px solid var(--glass-border); border-radius: var(--radius-lg);
            padding: 16px 20px;
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            box-shadow: var(--shadow-md);
        }

        /* ── Details / Summary ── */
        details { margin: 16px 0; }
        details summary {
            cursor: pointer; color: var(--accent); font-weight: 600;
            padding: 10px 16px; list-style: none;
            border: 1px solid var(--glass-border); border-radius: var(--radius-md);
            background: var(--bg-elevated);
            backdrop-filter: blur(12px);
            transition: all var(--transition);
        }
        details summary::-webkit-details-marker { display: none; }
        details summary::before {
            content: "\25B6"; display: inline-block; margin-right: 10px;
            font-size: 0.7em; transition: transform var(--transition);
            opacity: 0.6;
        }
        details[open] summary::before { transform: rotate(90deg); }
        details[open] summary { border-color: var(--glass-border-hover); background: var(--bg-hover); }
        details summary:hover {
            background: var(--bg-hover); border-color: var(--glass-border-hover);
            box-shadow: var(--shadow-sm);
        }
        details .panel-content {
            padding: 16px 20px; margin-top: 8px;
            background: var(--bg-elevated);
            border: 1px solid var(--glass-border); border-radius: var(--radius-md);
            backdrop-filter: blur(12px);
        }

        /* ── Section cards ── */
        .section-card {
            background: var(--bg-elevated);
            border: 1px solid var(--glass-border); border-radius: var(--radius-lg);
            padding: 20px 24px; margin: 20px 0;
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            box-shadow: var(--shadow-md);
            transition: border-color var(--transition);
        }
        .section-card:hover { border-color: var(--glass-border-hover); }
        .section-card-header {
            font-size: 1.1em; font-weight: 600; margin: 0 0 4px 0;
            letter-spacing: -0.01em;
        }
        .section-card-subheader {
            font-size: 0.85em; color: var(--text-muted); margin: 0 0 16px 0;
        }

        /* ── Self-heal controls ── */
        .selfheal-controls {
            display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 12px 0;
        }
        .selfheal-controls input {
            width: 64px; padding: 8px; font-family: 'Inter', sans-serif;
            background: var(--bg-base); border: 1px solid var(--glass-border);
            border-radius: var(--radius-sm); color: var(--text-secondary);
            text-align: center; font-size: 0.85em;
            transition: all var(--transition);
        }
        .selfheal-controls input:hover { border-color: var(--glass-border-hover); }
        .selfheal-controls input:focus {
            border-color: var(--accent); outline: none;
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .selfheal-controls label {
            color: var(--text-muted); font-size: 0.75em; font-weight: 500; margin-right: 2px;
        }

        /* ── Description ── */
        #demo-description {
            color: var(--text-secondary); font-size: 0.9em;
            padding: 12px 16px; margin-bottom: 12px;
            background: var(--bg-elevated);
            border: 1px solid var(--glass-border); border-radius: var(--radius-md);
            border-left: 3px solid var(--accent);
        }

        /* ── Dash component overrides ── */
        .Select-control {
            background: var(--bg-base) !important;
            border-color: var(--glass-border) !important;
            border-radius: var(--radius-sm) !important;
            transition: all var(--transition) !important;
        }
        .Select-control:hover { border-color: var(--glass-border-hover) !important; }
        .Select.is-focused .Select-control { border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-glow) !important; }
        .Select-value-label, .Select-placeholder { color: var(--text-secondary) !important; }
        .Select-menu-outer {
            background: rgba(15,15,25,0.95) !important;
            border-color: var(--glass-border) !important;
            border-radius: var(--radius-sm) !important;
            backdrop-filter: blur(16px) !important;
            box-shadow: var(--shadow-lg) !important;
        }
        .Select-option { color: var(--text-secondary) !important; transition: all var(--transition) !important; }
        .Select-option.is-focused { background: var(--bg-hover) !important; color: var(--text-primary) !important; }
        .rc-slider-track { background: linear-gradient(90deg, #059669, #34d399) !important; height: 4px !important; }
        .rc-slider-handle {
            border-color: var(--accent-green) !important; background: var(--bg-base) !important;
            width: 16px !important; height: 16px !important; margin-top: -6px !important;
            transition: all var(--transition) !important;
            box-shadow: 0 0 0 2px rgba(52,211,153,0.15) !important;
        }
        .rc-slider-handle:hover {
            transform: scale(1.2) !important;
            box-shadow: 0 0 0 4px var(--accent-green-glow) !important;
        }
        .rc-slider-rail { background: rgba(255,255,255,0.06) !important; height: 4px !important; border-radius: 2px !important; }
        .rc-slider-dot { background: rgba(255,255,255,0.1) !important; border-color: rgba(255,255,255,0.1) !important; }
        .rc-slider-mark-text { color: var(--text-muted) !important; font-size: 0.8em !important; }
        /* Table */
        .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
            background: var(--bg-base) !important; color: var(--text-secondary) !important;
            border-color: var(--glass-border) !important;
            transition: background var(--transition) !important;
            font-size: 0.85em !important;
        }
        .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
            background: var(--bg-elevated) !important; color: var(--text-primary) !important;
            border-color: var(--glass-border) !important; font-weight: 600;
            font-size: 0.8em !important; text-transform: uppercase; letter-spacing: 0.05em;
        }
        .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner tr:hover td {
            background: var(--bg-hover) !important;
        }
        /* Checklist / Radio */
        .sidebar input[type="checkbox"], .sidebar input[type="radio"] { accent-color: var(--accent); }
        /* Auto-play active button */
        .control-bar button.btn-active {
            background: linear-gradient(135deg, #7c5cfc, #60a5fa);
            border-color: rgba(124,92,252,0.4); color: #fff; font-weight: 600;
            animation: pulse-glow 2s ease-in-out infinite;
        }
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 8px var(--accent-glow); }
            50% { box-shadow: 0 0 20px var(--accent-glow); }
        }
        /* Welcome page */
        .welcome-page {
            min-height: 100vh; display: flex; flex-direction: column;
            align-items: center; justify-content: center; padding: 40px 24px;
            text-align: center;
        }
        .welcome-page h1 {
            font-size: 3em; font-weight: 700; margin-bottom: 8px;
            background: linear-gradient(135deg, #7c5cfc, #60a5fa, #34d399);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; letter-spacing: -0.03em;
        }
        .welcome-page .subtitle {
            font-size: 1.15em; color: var(--text-secondary); max-width: 600px;
            margin: 0 auto 40px auto; line-height: 1.7;
        }
        .demo-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px; max-width: 900px; width: 100%; margin-bottom: 40px;
        }
        .demo-card {
            background: var(--bg-elevated); border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg); padding: 20px;
            backdrop-filter: blur(12px); text-align: left;
            transition: all var(--transition);
        }
        .demo-card:hover {
            border-color: var(--glass-border-hover);
            transform: translateY(-3px); box-shadow: var(--shadow-lg);
        }
        .demo-card h3 {
            font-size: 1em; font-weight: 600; color: var(--text-primary);
            margin: 0 0 6px 0;
        }
        .demo-card p {
            font-size: 0.8em; color: var(--text-muted); margin: 0;
            line-height: 1.5; display: -webkit-box; -webkit-line-clamp: 3;
            -webkit-box-orient: vertical; overflow: hidden;
        }
        .welcome-btn {
            display: inline-block; padding: 14px 36px;
            background: linear-gradient(135deg, #7c5cfc, #60a5fa);
            border: none; border-radius: var(--radius-md);
            color: #fff; font-family: 'Inter', sans-serif;
            font-size: 1em; font-weight: 600; cursor: pointer;
            transition: all var(--transition); text-decoration: none;
        }
        .welcome-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px var(--accent-glow);
        }
        .welcome-nav {
            display: flex; gap: 16px; margin-top: 16px;
        }
        .welcome-nav a {
            color: var(--text-muted); text-decoration: none;
            font-size: 0.85em; transition: color var(--transition);
        }
        .welcome-nav a:hover { color: var(--accent); }
        /* About page */
        .about-page {
            min-height: 100vh; display: flex; flex-direction: column;
            align-items: center; justify-content: center; padding: 40px 24px;
            max-width: 700px; margin: 0 auto;
        }
        .about-page h1 {
            font-size: 2.2em; font-weight: 700; margin-bottom: 16px;
            background: linear-gradient(135deg, #7c5cfc, #60a5fa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .about-page p, .about-page li {
            color: var(--text-secondary); line-height: 1.8; font-size: 0.95em;
        }
        .about-page a {
            color: var(--accent); text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: border-color var(--transition);
        }
        .about-page a:hover { border-bottom-color: var(--accent); }
        .about-page .back-link {
            margin-top: 32px; color: var(--text-muted); font-size: 0.85em;
        }
        /* Responsive */
        @media (max-width: 900px) {
            .sidebar { width: 260px; padding: 16px 14px; }
            .main-area { margin-left: 260px; }
        }
        @media (max-width: 640px) {
            .sidebar {
                position: relative; width: 100%; height: auto;
                border-right: none; border-bottom: 1px solid var(--glass-border);
            }
            .main-area { margin-left: 0; }
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════
#  Page layouts
# ═══════════════════════════════════════════════════════════════════════

_demo_names = list(DEMOS.keys())

_DEMO_ICONS = {
    "Gradient": "🌡",
    "Channel": "🔀",
    "Sparse Leaders": "👑",
    "Crowd Monitoring": "👥",
    "Wave Propagation": "🌊",
    "Self-Healing": "🩹",
}


def _welcome_layout():
    cards = []
    for name, cfg in DEMOS.items():
        cards.append(
            html.A(
                html.Div([
                    html.Div(_DEMO_ICONS.get(name, ""), style={"fontSize": "1.6em", "marginBottom": "6px"}),
                    html.H3(name),
                    html.P(cfg["description"][:120] + "…" if len(cfg["description"]) > 120 else cfg["description"]),
                ], className="demo-card"),
                href=f"/simulator?demo={name.replace(' ', '+')}",
                style={"textDecoration": "none"},
            )
        )
    return html.Div([
        html.H1("Computational Fields"),
        html.P(
            "Interactive simulator for aggregate computing and self-organising programs",
            className="subtitle",
        ),
        html.Div(cards, className="demo-grid"),
        html.A("Launch Simulator", href="/simulator", className="welcome-btn"),
        html.Div([
            html.A("About this project", href="/about"),
        ], className="welcome-nav"),
    ], className="welcome-page")


def _about_layout():
    return html.Div([
        html.H1("About"),
        html.P([
            "This interactive simulator demonstrates ",
            html.Strong("Computational Fields"),
            " and ",
            html.Strong("Aggregate Computing"),
            " — a macro-programming paradigm for self-organising systems.",
        ]),
        html.P([
            "The project demonstrates the core building blocks (G, C, S) "
            "and higher-level patterns (channel, crowd safety, self-healing) "
            "that emerge when programs are expressed as distributed field computations."
        ]),
        html.H2("Links", style={"marginTop": "24px", "fontSize": "1.4em"}),
        html.Ul([
            html.Li(html.A(
                "GitHub Repository",
                href="https://github.com/Samuele95/computational-fields",
                target="_blank",
            )),
            html.Li(html.A(
                "Documentation (GitHub Pages)",
                href="https://samuele95.github.io/computational-fields/",
                target="_blank",
            )),
        ]),
        html.H2("Tech Stack", style={"marginTop": "24px", "fontSize": "1.4em"}),
        html.Ul([
            html.Li("Python 3.10+ with NumPy"),
            html.Li("Dash / Plotly for interactive visualization"),
            html.Li("Custom aggregate computing runtime"),
        ]),
        html.Div(
            html.A("← Back to Home", href="/"),
            className="back-link",
        ),
    ], className="about-page")


def _simulator_layout():
    return html.Div([
        # ── Sidebar ──────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.A("←", href="/", style={
                    "color": "var(--text-muted)", "textDecoration": "none",
                    "fontSize": "1.2em", "marginRight": "8px",
                }),
                html.H2("Computational Fields", style={"display": "inline", "verticalAlign": "middle"}),
            ], style={"marginBottom": "12px"}),

            html.Label("Demo"),
            dcc.Dropdown(
                id="demo-selector",
                options=[{"label": n, "value": n} for n in _demo_names],
                value="Gradient",
                clearable=False,
                style={"color": "#e8eaed"},
            ),

            # ── Network section ──
            html.Div(className="sidebar-section", children=[
                html.Div("Network", className="sidebar-section-header"),
                html.Label("Topology"),
                dcc.RadioItems(
                    id="topology-type",
                    options=[{"label": "Grid", "value": "Grid"}, {"label": "Random", "value": "Random"}],
                    value="Grid",
                    inline=True,
                    style={"color": "#9aa0a6"},
                ),
                html.Label("Rows"), dcc.Slider(id="rows-slider", min=2, max=20, step=1, value=8, marks={2: "2", 10: "10", 20: "20"}),
                html.Label("Columns"), dcc.Slider(id="cols-slider", min=2, max=20, step=1, value=8, marks={2: "2", 10: "10", 20: "20"}),
                html.Label("Spacing"), dcc.Slider(id="spacing-slider", min=0.5, max=3.0, step=0.1, value=1.0, marks={0.5: "0.5", 1.0: "1.0", 2.0: "2.0", 3.0: "3.0"}),
                html.Label("Comm Range"), dcc.Slider(id="comm-range-slider", min=1.0, max=5.0, step=0.1, value=1.5, marks={1.0: "1.0", 2.0: "2.0", 3.0: "3.0", 5.0: "5.0"}),
            ]),

            # ── Demo Parameters section ──
            html.Div(className="sidebar-section", children=[
                html.Div("Demo Parameters", className="sidebar-section-header"),
                html.Div(id="grain-container", children=[
                    html.Label("Grain (S block)"),
                    dcc.Slider(id="grain-slider", min=1.0, max=8.0, step=0.5, value=3.0, marks={1: "1", 3: "3", 5: "5", 8: "8"}),
                ]),
                html.Div(id="pulse-container", children=[
                    html.Label("Pulse Period"),
                    dcc.Slider(id="pulse-slider", min=2, max=20, step=1, value=6, marks={2: "2", 6: "6", 10: "10", 20: "20"}),
                ]),
            ]),

            # ── Simulation section ──
            html.Div(className="sidebar-section", children=[
                html.Div("Simulation", className="sidebar-section-header"),
                html.Label("Run N Rounds"),
                dcc.Slider(id="run-n-slider", min=1, max=50, step=1, value=10, marks={1: "1", 10: "10", 25: "25", 50: "50"}),
                html.Label("Auto-Play Speed (ms)"),
                dcc.Slider(id="speed-slider", min=100, max=2000, step=100, value=500, marks={100: "100", 500: "500", 1000: "1s", 2000: "2s"}),
            ]),

            # ── Display section ──
            html.Div(className="sidebar-section", children=[
                html.Div("Display", className="sidebar-section-header"),
                dcc.Checklist(
                    id="show-edges",
                    options=[{"label": " Show edges", "value": "yes"}],
                    value=["yes"],
                    style={"color": "#9aa0a6"},
                ),
            ]),
        ], className="sidebar"),

        # ── Main area ────────────────────────────────────────────────
        html.Div([
            html.P(id="demo-description"),

            # Control bar
            html.Div([
                html.Button("Step", id="btn-step", className="primary", n_clicks=0),
                html.Button("Run N", id="btn-run-n", n_clicks=0),
                html.Button("Auto-Play", id="btn-auto-play", n_clicks=0),
                html.Button("Reset", id="btn-reset", className="danger", n_clicks=0),
            ], className="control-bar"),

            # Round display + history slider
            html.Div(id="round-display", children="Round: 0", className="round-badge"),
            dcc.Slider(id="history-slider", min=0, max=0, step=1, value=0,
                       marks={0: "0"}, tooltip={"placement": "bottom"}),

            # Crowd sub-field tabs (visible only for Crowd Monitoring)
            html.Div(id="crowd-tabs-container", children=[
                dcc.RadioItems(
                    id="crowd-tab",
                    options=[
                        {"label": "Density", "value": "density"},
                        {"label": "Alert", "value": "alert"},
                        {"label": "Exit Distance", "value": "exit_dist"},
                    ],
                    value="density",
                    inline=True,
                    style={"margin": "8px 0", "color": "#9aa0a6"},
                ),
            ]),

            # Main graph -- uses Plotly.react() natively for flicker-free updates
            dcc.Graph(id="main-graph", config={"displayModeBar": True, "scrollZoom": True}),

            # Metrics
            html.Div(id="metrics", className="metrics-bar"),

            # Self-heal controls
            html.Div(id="selfheal-container", children=[
                html.Div(className="section-card", children=[
                    html.Div("Remove Region", className="section-card-header", style={"color": "#f87171"}),
                    html.Div([
                        html.Label("Row start"), dcc.Input(id="sh-row-start", type="number", value=2, min=0, max=20),
                        html.Label("Row end"), dcc.Input(id="sh-row-end", type="number", value=4, min=0, max=20),
                        html.Label("Col start"), dcc.Input(id="sh-col-start", type="number", value=2, min=0, max=20),
                        html.Label("Col end"), dcc.Input(id="sh-col-end", type="number", value=4, min=0, max=20),
                        html.Button("Remove", id="btn-remove-region", className="danger", n_clicks=0),
                    ], className="selfheal-controls"),
                ]),
            ]),

            # Role editor
            html.Div(id="role-editor-container", children=[
                html.Div(className="section-card", children=[
                    html.Div("Device Role Editor", className="section-card-header"),
                    html.Div("Edit roles and click Apply to rebuild.", className="section-card-subheader"),
                    dash_table.DataTable(
                        id="role-table",
                        columns=[
                            {"name": "Device", "id": "device", "editable": False},
                            {"name": "Role", "id": "role", "editable": True, "presentation": "dropdown"},
                        ],
                        data=[],
                        dropdown={},
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "center", "padding": "4px 8px", "fontSize": "0.85em"},
                        page_size=15,
                    ),
                    html.Button("Apply Roles", id="btn-apply-roles", n_clicks=0, style={"marginTop": "8px"}),
                ]),
            ]),

            # Theory / Practical panels
            html.Details([
                html.Summary("Theory"),
                html.Div(id="theory-content", className="panel-content"),
            ], style={"marginTop": "16px"}),

            html.Details([
                html.Summary("Practical Guide"),
                html.Div(id="practical-content", className="panel-content"),
            ]),

            # Convergence chart
            dcc.Graph(id="convergence-graph", config={"displayModeBar": False}),

            # Hidden components
            dcc.Interval(id="auto-play-interval", interval=500, disabled=True),
            dcc.Store(id="auto-play-active", data=False),
        ], className="main-area"),
    ])


# ═══════════════════════════════════════════════════════════════════════
#  App layout (URL routing)
# ═══════════════════════════════════════════════════════════════════════

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content"),
])


# ═══════════════════════════════════════════════════════════════════════
#  Callbacks
# ═══════════════════════════════════════════════════════════════════════

# ── CB0: URL routing ─────────────────────────────────────────────────

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    if pathname == "/simulator" or (pathname and pathname.startswith("/simulator")):
        return _simulator_layout()
    elif pathname == "/about":
        return _about_layout()
    return _welcome_layout()


# ── CB8: Conditional visibility ──────────────────────────────────────

@app.callback(
    Output("grain-container", "style"),
    Output("pulse-container", "style"),
    Output("selfheal-container", "style"),
    Output("crowd-tabs-container", "style"),
    Input("demo-selector", "value"),
)
def update_visibility(demo_name):
    hide = {"display": "none"}
    show = {}
    grain_vis = show if demo_name == "Sparse Leaders" else hide
    pulse_vis = show if demo_name == "Wave Propagation" else hide
    selfheal_vis = show if demo_name == "Self-Healing" else hide
    crowd_vis = show if demo_name == "Crowd Monitoring" else hide
    return grain_vis, pulse_vis, selfheal_vis, crowd_vis


# ── CB1: Config change -> rebuild engine + initial figure ────────────

@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Output("convergence-graph", "figure", allow_duplicate=True),
    Output("round-display", "children", allow_duplicate=True),
    Output("history-slider", "max", allow_duplicate=True),
    Output("history-slider", "value", allow_duplicate=True),
    Output("history-slider", "marks", allow_duplicate=True),
    Output("metrics", "children", allow_duplicate=True),
    Output("demo-description", "children"),
    Output("theory-content", "children"),
    Output("practical-content", "children"),
    Output("role-table", "data", allow_duplicate=True),
    Output("role-table", "dropdown", allow_duplicate=True),
    Input("demo-selector", "value"),
    Input("topology-type", "value"),
    Input("rows-slider", "value"),
    Input("cols-slider", "value"),
    Input("spacing-slider", "value"),
    Input("comm-range-slider", "value"),
    Input("grain-slider", "value"),
    Input("pulse-slider", "value"),
    prevent_initial_call="initial_duplicate",
)
def config_change(demo_name, topology, rows, cols, spacing, comm_range, grain, pulse_period):
    global _role_overrides
    _role_overrides = {}

    engine = _get_or_create_engine(demo_name, rows, cols, spacing, comm_range, grain, pulse_period, topology)
    demo = DEMOS[demo_name]

    # Initial figure (no simulation run yet)
    results = engine.results if engine.results else {did: 0.0 for did in engine.network.devices}
    fig = _build_figure(engine, demo_name, results, show_edges=True)
    conv_fig = _convergence_chart(engine, demo["field_key"], demo["field_type"])
    if conv_fig is None:
        conv_fig = go.Figure()
        conv_fig.update_layout(height=250, **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k != "height"})

    # Role table
    roles = demo.get("roles", [])
    all_roles = ["(none)"] + roles
    devices = sorted(engine.network.devices.keys())
    role_data = []
    for did in devices:
        dev = engine.network.devices[did]
        current = "(none)"
        if roles:
            for r in roles:
                key_map = {"Source": "is_source", "Dest": "is_dest", "Monitor": "is_monitor", "Exit": "is_exit"}
                if dev.sensors.get(key_map.get(r, ""), False):
                    current = r
                    break
        role_data.append({"device": did, "role": current})
    role_dropdown = {"role": {"options": [{"label": r, "value": r} for r in all_roles]}}

    # Theory and practical content
    theory_md = dcc.Markdown(demo.get("theory", ""), mathjax=True)
    practical_md = dcc.Markdown(demo.get("practical", ""), mathjax=True)

    h_len = len(engine.history)
    marks = _history_marks(h_len)

    return (
        fig, conv_fig,
        f"Round: {engine.round_count}",
        max(h_len - 1, 0), max(h_len - 1, 0), marks,
        dash.html.Div(dash.dcc.Markdown(_summary_metrics_html(demo, results), dangerously_allow_html=True), className="metrics-card"),
        demo["description"],
        theory_md, practical_md,
        role_data, role_dropdown,
    )


def _history_marks(h_len: int) -> dict:
    if h_len <= 1:
        return {0: "0"}
    marks = {0: "0"}
    step = max(1, (h_len - 1) // 5)
    for i in range(step, h_len, step):
        marks[i] = str(i + 1)
    marks[h_len - 1] = str(h_len)
    return marks


# ── CB2: Simulation advance ─────────────────────────────────────────

@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Output("convergence-graph", "figure", allow_duplicate=True),
    Output("round-display", "children", allow_duplicate=True),
    Output("history-slider", "max", allow_duplicate=True),
    Output("history-slider", "value", allow_duplicate=True),
    Output("history-slider", "marks", allow_duplicate=True),
    Output("metrics", "children", allow_duplicate=True),
    Input("btn-step", "n_clicks"),
    Input("btn-run-n", "n_clicks"),
    Input("auto-play-interval", "n_intervals"),
    Input("history-slider", "value"),
    State("run-n-slider", "value"),
    State("show-edges", "value"),
    State("demo-selector", "value"),
    State("crowd-tab", "value"),
    prevent_initial_call=True,
)
def simulation_advance(step_clicks, run_clicks, interval_ticks, hist_val,
                       run_n, show_edges_val, demo_name, crowd_tab):
    if _engine is None:
        return (no_update,) * 7

    triggered = ctx.triggered_id
    show_edges = "yes" in (show_edges_val or [])
    demo = DEMOS[demo_name]

    if triggered == "btn-step":
        _engine.step()
    elif triggered == "btn-run-n":
        _engine.run(run_n or 10)
    elif triggered == "auto-play-interval":
        _engine.step()
    elif triggered == "history-slider":
        # Replay from history -- no new computation
        h_len = len(_engine.history)
        if h_len > 0 and 0 <= hist_val < h_len:
            results = _engine.history[hist_val]
            fig = _build_figure(_engine, demo_name, results, show_edges=show_edges, crowd_tab=crowd_tab or "density")
            conv_fig = _convergence_chart(_engine, demo["field_key"], demo["field_type"]) or go.Figure()
            if isinstance(conv_fig, go.Figure) and not conv_fig.data:
                conv_fig.update_layout(height=250, **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k != "height"})
            marks = _history_marks(h_len)
            return (
                fig, conv_fig,
                f"Round: {hist_val + 1} (of {h_len})",
                max(h_len - 1, 0), hist_val, marks,
                dash.html.Div(dash.dcc.Markdown(_summary_metrics_html(demo, results), dangerously_allow_html=True), className="metrics-card"),
            )
        return (no_update,) * 7

    # After step/run, show latest results
    results = _engine.results or {}
    fig = _build_figure(_engine, demo_name, results, show_edges=show_edges, crowd_tab=crowd_tab or "density")
    conv_fig = _convergence_chart(_engine, demo["field_key"], demo["field_type"]) or go.Figure()
    if isinstance(conv_fig, go.Figure) and not conv_fig.data:
        conv_fig.update_layout(height=250, **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k != "height"})
    h_len = len(_engine.history)
    marks = _history_marks(h_len)
    return (
        fig, conv_fig,
        f"Round: {_engine.round_count}",
        max(h_len - 1, 0), max(h_len - 1, 0), marks,
        dash.html.Div(dash.dcc.Markdown(_summary_metrics_html(demo, results), dangerously_allow_html=True), className="metrics-card"),
    )


# ── CB3: Auto-play toggle ───────────────────────────────────────────

@app.callback(
    Output("auto-play-interval", "disabled"),
    Output("btn-auto-play", "children"),
    Output("auto-play-active", "data"),
    Output("btn-auto-play", "className"),
    Input("btn-auto-play", "n_clicks"),
    State("auto-play-active", "data"),
    prevent_initial_call=True,
)
def toggle_autoplay(n_clicks, is_active):
    new_active = not is_active
    return (not new_active), ("Pause" if new_active else "Auto-Play"), new_active, ("btn-active" if new_active else "")


# ── CB4: Speed change ───────────────────────────────────────────────

@app.callback(
    Output("auto-play-interval", "interval"),
    Input("speed-slider", "value"),
)
def update_speed(speed_ms):
    return speed_ms or 500


# ── CB5: Reset ───────────────────────────────────────────────────────

@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Output("convergence-graph", "figure", allow_duplicate=True),
    Output("round-display", "children", allow_duplicate=True),
    Output("history-slider", "max", allow_duplicate=True),
    Output("history-slider", "value", allow_duplicate=True),
    Output("history-slider", "marks", allow_duplicate=True),
    Output("metrics", "children", allow_duplicate=True),
    Output("auto-play-interval", "disabled", allow_duplicate=True),
    Output("btn-auto-play", "children", allow_duplicate=True),
    Output("auto-play-active", "data", allow_duplicate=True),
    Output("btn-auto-play", "className", allow_duplicate=True),
    Input("btn-reset", "n_clicks"),
    State("demo-selector", "value"),
    State("topology-type", "value"),
    State("rows-slider", "value"),
    State("cols-slider", "value"),
    State("spacing-slider", "value"),
    State("comm-range-slider", "value"),
    State("grain-slider", "value"),
    State("pulse-slider", "value"),
    prevent_initial_call=True,
)
def reset_simulation(n_clicks, demo_name, topology, rows, cols, spacing, comm_range, grain, pulse_period):
    global _engine, _engine_key
    _engine = None
    _engine_key = None

    engine = _get_or_create_engine(demo_name, rows, cols, spacing, comm_range, grain, pulse_period, topology)
    demo = DEMOS[demo_name]
    results = {did: 0.0 for did in engine.network.devices}
    fig = _build_figure(engine, demo_name, results, show_edges=True)
    conv_fig = go.Figure()
    conv_fig.update_layout(height=250, **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k != "height"})

    return (
        fig, conv_fig,
        "Round: 0", 0, 0, {0: "0"},
        "",
        True, "Auto-Play", False, "",
    )


# ── CB6: Role editor apply ──────────────────────────────────────────

@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Output("convergence-graph", "figure", allow_duplicate=True),
    Output("round-display", "children", allow_duplicate=True),
    Output("history-slider", "max", allow_duplicate=True),
    Output("history-slider", "value", allow_duplicate=True),
    Output("history-slider", "marks", allow_duplicate=True),
    Output("metrics", "children", allow_duplicate=True),
    Input("btn-apply-roles", "n_clicks"),
    State("role-table", "data"),
    State("demo-selector", "value"),
    State("topology-type", "value"),
    State("rows-slider", "value"),
    State("cols-slider", "value"),
    State("spacing-slider", "value"),
    State("comm-range-slider", "value"),
    State("grain-slider", "value"),
    State("pulse-slider", "value"),
    prevent_initial_call=True,
)
def apply_roles(n_clicks, table_data, demo_name, topology, rows, cols, spacing, comm_range, grain, pulse_period):
    global _engine, _engine_key, _role_overrides

    _role_overrides = {}
    if table_data:
        for row in table_data:
            role = row.get("role", "(none)")
            if role and role != "(none)":
                _role_overrides[int(row["device"])] = role

    _engine = None
    _engine_key = None
    engine = _get_or_create_engine(demo_name, rows, cols, spacing, comm_range, grain, pulse_period, topology)
    demo = DEMOS[demo_name]
    results = {did: 0.0 for did in engine.network.devices}
    fig = _build_figure(engine, demo_name, results, show_edges=True)
    conv_fig = go.Figure()
    conv_fig.update_layout(height=250, **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k != "height"})

    return (
        fig, conv_fig,
        "Round: 0", 0, 0, {0: "0"},
        "",
    )


# ── CB7: Self-heal remove region ────────────────────────────────────

@app.callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Output("convergence-graph", "figure", allow_duplicate=True),
    Output("round-display", "children", allow_duplicate=True),
    Output("history-slider", "max", allow_duplicate=True),
    Output("history-slider", "value", allow_duplicate=True),
    Output("history-slider", "marks", allow_duplicate=True),
    Output("metrics", "children", allow_duplicate=True),
    Output("role-table", "data", allow_duplicate=True),
    Output("role-table", "dropdown", allow_duplicate=True),
    Input("btn-remove-region", "n_clicks"),
    State("sh-row-start", "value"),
    State("sh-row-end", "value"),
    State("sh-col-start", "value"),
    State("sh-col-end", "value"),
    State("show-edges", "value"),
    State("demo-selector", "value"),
    State("cols-slider", "value"),
    State("spacing-slider", "value"),
    prevent_initial_call=True,
)
def remove_region(n_clicks, row_start, row_end, col_start, col_end,
                  show_edges_val, demo_name, cols, spacing):
    if _engine is None:
        return (no_update,) * 9

    show_edges = "yes" in (show_edges_val or [])
    demo = DEMOS[demo_name]

    # Find devices in the grid region and remove them
    to_remove = []
    for did, dev in list(_engine.network.devices.items()):
        col_idx = round(dev.position[0] / spacing) if spacing else 0
        row_idx = round(dev.position[1] / spacing) if spacing else 0
        if row_start <= row_idx <= row_end and col_start <= col_idx <= col_end:
            to_remove.append(did)

    for did in to_remove:
        _engine.network.remove_device(did)
    _engine.network.update_neighbors()

    results = _engine.results or {}
    fig = _build_figure(_engine, demo_name, results, show_edges=show_edges)
    conv_fig = _convergence_chart(_engine, demo["field_key"], demo["field_type"]) or go.Figure()
    if isinstance(conv_fig, go.Figure) and not conv_fig.data:
        conv_fig.update_layout(height=250, **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k != "height"})

    h_len = len(_engine.history)
    marks = _history_marks(h_len)

    # Update role table
    roles = demo.get("roles", [])
    all_roles = ["(none)"] + roles
    devices = sorted(_engine.network.devices.keys())
    role_data = [{"device": did, "role": "(none)"} for did in devices]
    role_dropdown = {"role": {"options": [{"label": r, "value": r} for r in all_roles]}}

    return (
        fig, conv_fig,
        f"Round: {_engine.round_count}",
        max(h_len - 1, 0), max(h_len - 1, 0), marks,
        dash.html.Div(dash.dcc.Markdown(_summary_metrics_html(demo, results), dangerously_allow_html=True), className="metrics-card"),
        role_data, role_dropdown,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", debug=False, port=port)
