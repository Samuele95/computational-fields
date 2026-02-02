# Channel Demo

The **Channel** demo demonstrates how three gradient computations compose to form a logical corridor between a source region and a destination region.

## What It Shows

- Composition of multiple building blocks
- Logical path formation between two regions
- The `channel` composite pattern

## Building Blocks Used

| Block | Role |
|-------|------|
| **G** (gradient from source) | Distance field from source region |
| **G** (gradient from destination) | Distance field from destination region |
| **G** (combined) | Shortest source-to-destination path length |

## How It Works

The channel is defined as the set of devices that lie "close enough" to the shortest path between source and destination:

```python
def channel_program(ctx):
    is_src = ctx.sense("is_source")
    is_dst = ctx.sense("is_destination")
    return channel(ctx, is_src, is_dst, width=0.5)
```

The algorithm:

1. Compute `d_s` = gradient from source
2. Compute `d_d` = gradient from destination
3. Compute `d_sd` = the shortest distance between source and destination (broadcast from source along `d_s`)
4. A device is **on the channel** if: $d_s + d_d \leq d_{sd} + w$

where $w$ is the width tolerance.

$$
\text{channel}(\delta) = \bigl(G_{\text{src}}(\delta) + G_{\text{dst}}(\delta)\bigr) \leq \bigl(d_{\text{src} \to \text{dst}} + w\bigr)
$$

## Network Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| Topology | Grid | 8×8 regular grid |
| Source | Device 0 | Top-left corner |
| Destination | Device 63 | Bottom-right corner |
| Width | 0.5 | Corridor tolerance |
| Comm Range | 1.5 | Neighbour connectivity |

## Interpretation

- **Highlighted devices** (bright) — on the channel
- **Dark devices** — outside the channel

The channel forms a band of devices connecting the source to the destination. The `width` parameter controls how wide this band is: a width of 0 gives the exact shortest path, while larger values include more devices.

## Composition in Action

This demo is a clear example of **functional composition** in aggregate computing. The `channel` pattern is built entirely from `gradient` calls — no new low-level primitives are needed. This compositionality is a key design principle of the building-block approach.

## Try It

Select **Channel** from the demo dropdown in the [simulator](/simulator), or try it on the [live deployment](https://huggingface.co/spaces/Sams995/computational-fields).
