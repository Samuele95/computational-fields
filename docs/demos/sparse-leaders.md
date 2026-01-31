# Sparse Leaders Demo

The **Sparse Leaders** demo shows distributed leader election using the **S block**, which selects uniformly-spaced leaders across the network.

## What It Shows

- Symmetry-breaking leader election
- Voronoi-like partitioning of the network
- The `sparse` building block
- Self-stabilising region formation

## Building Blocks Used

| Block | Role |
|-------|------|
| **S** (sparse) | Elects leaders spaced at least `grain` apart |
| **G** (gradient, internal) | Used internally by S to compute inter-leader distances |

## How It Works

The sparse block elects leaders so that no two leaders are closer than `grain` distance apart:

```python
def sparse_program(ctx):
    grain = ctx.sense("grain")  # e.g. 3.0
    is_leader = sparse(ctx, grain)
    return is_leader
```

The algorithm uses a competition mechanism:

1. Each device starts as a **candidate leader**
2. Compute a gradient from all current leaders
3. If a device's distance to its nearest leader is less than `grain` and it is not the leader itself, it **yields**
4. Ties are broken by device ID (lower ID wins)

After convergence, the leaders form an approximately uniform spatial distribution, and every non-leader device is associated with its nearest leader — creating a **Voronoi partition**.

$$
S(\delta, g) = \begin{cases}
\text{true} & \text{if } \min_{n \in \text{leaders}} d(\delta, n) \geq g \text{ or } \delta = \text{closest leader} \\
\text{false} & \text{otherwise}
\end{cases}
$$

## Network Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| Topology | Grid | 8×8 regular grid |
| Grain | 3.0 | Minimum inter-leader distance |
| Comm Range | 1.5 | Neighbour connectivity |

## Interpretation

- **Bright nodes** — elected leaders
- **Dark nodes** — non-leaders
- The colour gradient shows region membership (which leader each device belongs to)

Increasing the `grain` parameter produces fewer, more widely-spaced leaders. Decreasing it produces more leaders with smaller regions.

## Applications

The sparse leader pattern is used as a building block for:

- **Partitioning** — dividing a network into manageable regions
- **Distributed averaging** — computing regional statistics
- **Hierarchical coordination** — multi-level decision making
- **Crowd monitoring** — the full case study demo uses S to create monitoring zones

## Try It

Select **Sparse Leaders** from the demo dropdown in the [simulator](/simulator).
