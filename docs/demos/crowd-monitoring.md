# Crowd Monitoring Demo

The **Crowd Monitoring** demo is a full case study that composes all four building blocks (G, C, S, T) into a realistic crowd-safety application.

## What It Shows

- Composition of all building blocks into a complete application
- Density estimation across network regions
- Alert zone detection and propagation
- Exit guidance through gradient fields
- Real-world applicability of aggregate computing

## Building Blocks Used

| Block | Role |
|-------|------|
| **S** (sparse) | Partition the space into monitoring zones |
| **G** (gradient) | Distance to zone leaders; distance to exits |
| **C** (collect) | Aggregate crowd density toward zone leaders |
| **broadcast** (G-derived) | Propagate alerts and density info outward |

## How It Works

The crowd monitoring program runs four coordinated computations:

```python
def crowd_monitoring(ctx):
    # 1. Partition into zones
    leader = partition(ctx, grain=4.0)

    # 2. Estimate local density
    local_density = ctx.sense("density")

    # 3. Collect density at zone leaders
    zone_density = collect(ctx, gradient(ctx, is_leader), sum, local_density, 0)

    # 4. Broadcast alert if density exceeds threshold
    is_alert = zone_density > THRESHOLD
    alert = broadcast(ctx, is_leader, is_alert)

    # 5. Guide toward nearest exit
    exit_distance = gradient(ctx, ctx.sense("is_exit"))

    return {
        "leader": leader,
        "density": zone_density,
        "alert": alert,
        "exit_dist": exit_distance,
    }
```

### Pipeline

```
┌─────────┐     ┌───────────┐     ┌──────────┐     ┌───────────┐
│ S block │────▶│ G + C     │────▶│ threshold │────▶│ broadcast │
│ (zones) │     │ (collect  │     │ check     │     │ (alert)   │
│         │     │  density) │     │           │     │           │
└─────────┘     └───────────┘     └──────────┘     └───────────┘
                                                          │
                                              ┌───────────▼──────┐
                                              │ G (exit gradient) │
                                              └──────────────────┘
```

1. **Zone formation** — S block elects leaders, creating Voronoi-like monitoring zones
2. **Density collection** — C block aggregates crowd density readings toward each zone leader along the gradient field
3. **Alert detection** — leaders compare zone density against a safety threshold
4. **Alert broadcast** — alerts propagate outward from leaders via the broadcast pattern
5. **Exit guidance** — a gradient from exit devices provides escape-route distances

## Visualisation Tabs

The simulator offers multiple views for this demo:

| Tab | Shows |
|-----|-------|
| **Density** | Raw density values across the network |
| **Zones** | Voronoi partition with leader highlighting |
| **Alerts** | Alert status (red = danger zone) |
| **Exit Distance** | Gradient field toward nearest exit |

## Network Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| Topology | Random | Irregular placement simulating real crowds |
| Devices | 50 | Network size |
| Grain | 4.0 | Zone size for monitoring |
| Density Threshold | 5.0 | Alert trigger level |
| Exits | Corner devices | Designated escape points |

## Why This Matters

This demo shows that complex distributed monitoring systems can be built by **composing simple, well-understood building blocks**. Each block is individually self-stabilising, and their composition preserves this property — the entire system recovers automatically from device failures, network changes, and shifting crowd patterns.

This is the core thesis of aggregate computing: macro-level behaviour emerges from the composition of resilient, reusable primitives.

## Try It

Select **Crowd Monitoring** from the demo dropdown in the [simulator](/simulator).
