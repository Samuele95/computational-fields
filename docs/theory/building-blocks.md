# Building Blocks

The field calculus primitives are powerful but low-level. The **building block** abstraction provides a small set of reusable, self-stabilising operators that cover the vast majority of distributed coordination patterns.

## The Four Blocks

| Block | Name | Purpose | Core Construct |
|-------|------|---------|----------------|
| **G** | Gradient | Spread information outward from sources | `share` |
| **C** | Collect | Aggregate information inward toward sinks | `share` |
| **S** | Sparse | Elect uniformly-spaced leaders | `share` |
| **T** | Time | Temporal evolution and decay | `rep` |

These four blocks, combined with function composition, can express any self-organising coordination pattern.

## G Block -- Gradient

The gradient computes a **shortest-path distance field** from a set of source devices:

$$G(S) = \begin{cases} 0 & \text{if device} \in S \\ \min_{n \in N} \big( G_n + d(n, \text{self}) \big) & \text{otherwise} \end{cases}$$

where $N$ is the set of aligned neighbours and $d(n, \text{self})$ is the Euclidean distance to neighbour $n$.

```python
from computational_fields.blocks.gradient import gradient

distance = gradient(ctx, is_source=ctx.sense("is_source"))
```

**Properties:**

- Converges in $O(\text{diameter})$ rounds
- Self-stabilising: recovers from perturbations automatically
- Foundation for most higher-level patterns

### Derived: `broadcast`

Pushes a value outward from source devices along the gradient:

```python
from computational_fields.blocks.gradient import broadcast

value = broadcast(ctx, source=is_source, value=42)
```

Each device relays the value from its closest-to-source neighbour.

## C Block -- Collect

Aggregates data **inward** along a potential field toward low-potential regions:

$$C(\text{potential}, \oplus, v, \mathbf{0}) = \bigoplus_{n : p_n > p_\text{self}} C_n \oplus v_\text{self}$$

```python
from computational_fields.blocks.collection import collect
import operator

total = collect(ctx, potential=distance, acc=operator.add,
                local=1, null=0)
```

**Parameters:**

- `potential`: the gradient field guiding data flow (data flows downhill)
- `acc`: associative binary accumulator (e.g., `+`, `max`)
- `local`: this device's contribution
- `null`: identity element for the accumulator

## S Block -- Sparse

Elects **uniformly-spaced leaders** separated by at least `grain` distance:

```python
from computational_fields.blocks.sparse import sparse

is_leader = sparse(ctx, grain=3.0)
```

**Algorithm:**

1. Each device tracks (distance_to_nearest_leader, leader_id)
2. If no leader exists within `grain` distance, the device self-elects
3. Ties are broken by device ID (lower wins)
4. Result: a Voronoi-like partition of the network

## T Block -- Time

Temporal operators for decay and periodic behaviour:

```python
from computational_fields.blocks.time_decay import timer, decay

countdown = timer(ctx, duration=10.0)   # decreases by delta_time each round
smoothed = decay(ctx, raw_value, rate=0.9)  # exponential smoothing
```

## Composite Patterns

Building blocks compose naturally into complex behaviours:

### Channel

A logical corridor between source and destination regions:

```python
from computational_fields.blocks.composites import channel

on_channel = channel(ctx, source=is_src, destination=is_dst, width=0.5)
```

Internally computes three gradients:

1. Distance from source
2. Distance from destination
3. Distance along the shortest path

A device is "on the channel" if $d_\text{source} + d_\text{dest} < d_\text{shortest} + \text{width}$.

### Partition

Divides the network into regions of diameter approximately `grain`:

```python
from computational_fields.blocks.composites import partition

region_leader = partition(ctx, grain=4.0)
```

Combines S (elect leaders) + G (distance to leader) + broadcast (propagate leader ID).

### Distributed Average

Computes regional averages using all four blocks:

```python
from computational_fields.blocks.composites import distributed_average

avg_temp = distributed_average(ctx, value=ctx.sense("temperature"))
```
