# Gradient Demo

The **Gradient** demo showcases the most fundamental building block in aggregate computing: the **G block**, which computes a self-stabilising distance field from source devices.

## What It Shows

- Shortest-path distance propagation across a network
- Self-stabilisation: the field converges from arbitrary initial states
- Convergence speed proportional to network diameter

## Building Blocks Used

| Block | Role |
|-------|------|
| **G** (gradient) | Computes minimum-hop distance to source |

## How It Works

One device is designated as the **source** (device 0 by default). The gradient block runs on every device:

```python
def gradient_program(ctx):
    is_source = ctx.sense("is_source")
    return gradient(ctx, is_source)
```

Each round, every device:

1. **Sources** output `0.0`
2. **Non-sources** collect distance estimates from neighbours, add the physical distance to each neighbour, and keep the minimum

Formally:

$$
G(\delta, \text{src}) = \begin{cases}
0 & \text{if src} \\
\min_{n \in N} \left( G_n + d(n, \delta) \right) & \text{otherwise}
\end{cases}
$$

where $N$ is the set of aligned neighbours and $d(n, \delta)$ is the Euclidean distance.

## Convergence

The field stabilises in $O(\text{diam})$ rounds, where **diam** is the network diameter (longest shortest path). On an 8×8 grid with spacing 1.0 and `comm_range=1.5`, convergence typically takes 10–12 rounds.

The **convergence chart** in the simulator tracks the maximum change across all devices per round. When this value drops to zero, the field has stabilised.

## Network Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| Topology | Grid | Regular lattice for predictable propagation |
| Rows / Cols | 8 × 8 | 64 devices |
| Spacing | 1.0 | Unit distance between adjacent devices |
| Comm Range | 1.5 | Each device sees its 4 direct neighbours (and diagonals at distance √2 ≈ 1.41) |

## Interpretation

The colour map shows distance values:

- **Purple / dark** — low distance (near source)
- **Yellow / bright** — high distance (far from source)

The resulting field forms concentric "rings" around the source, similar to a topographic map. This gradient field is the foundation for many higher-level patterns: channels, data collection, broadcasting, and leader election all build on top of it.

## Self-Healing

Try removing devices (click "Remove Random Device") after the field has converged. The gradient automatically recomputes through alternative paths — this is the **self-stabilisation** property in action.

## Try It

Select **Gradient** from the demo dropdown in the [simulator](/simulator).
