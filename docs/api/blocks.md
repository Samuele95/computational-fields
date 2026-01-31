# Building Blocks API

## G Block -- Gradient

::: computational_fields.blocks.gradient

### `gradient(ctx, source) -> float`

Computes the shortest-path distance from the nearest source device.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `Context` | Current execution context |
| `source` | `bool` | Whether this device is a source |

**Returns:** `float` -- distance to nearest source (0.0 for sources).

**Convergence:** $O(\text{diameter})$ rounds.

---

### `dist_to(ctx, source) -> float`

Alias for `gradient`.

---

### `broadcast(ctx, source, value) -> T`

Broadcasts a value outward from source devices along the gradient field.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `Context` | Current execution context |
| `source` | `bool` | Whether this device is a source |
| `value` | `T` | Value to broadcast (only meaningful at sources) |

**Returns:** `T` -- value from the nearest source.

---

## C Block -- Collection

::: computational_fields.blocks.collection

### `collect(ctx, potential, acc, local, null) -> T`

Aggregates data inward along a potential field.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `Context` | Current execution context |
| `potential` | `float` | Potential field value (data flows toward lower potential) |
| `acc` | `(T, T) -> T` | Associative binary accumulator |
| `local` | `T` | This device's contribution |
| `null` | `T` | Identity element for `acc` |

**Returns:** `T` -- accumulated value from upstream devices plus own contribution.

---

## S Block -- Sparse

::: computational_fields.blocks.sparse

### `sparse(ctx, grain) -> bool`

Elects uniformly-spaced leaders separated by at least `grain` distance.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `Context` | Current execution context |
| `grain` | `float` | Minimum distance between leaders |

**Returns:** `bool` -- `True` if this device is a leader.

---

## T Block -- Time

::: computational_fields.blocks.time_decay

### `timer(ctx, duration) -> float`

Countdown timer that decreases by `delta_time` each round.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `Context` | Current execution context |
| `duration` | `float` | Initial countdown value |

**Returns:** `float` -- remaining time (clamped to 0.0).

---

### `decay(ctx, value, rate=0.9) -> float`

Exponential smoothing: multiplies previous value by `rate` each round.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `Context` | Current execution context |
| `value` | `float` | Current raw value |
| `rate` | `float` | Decay rate (0-1) |

**Returns:** `float` -- smoothed value.

---

## Composite Patterns

::: computational_fields.blocks.composites

### `channel(ctx, source, destination, width) -> bool`

Logical corridor between source and destination regions.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `Context` | Current execution context |
| `source` | `bool` | Whether this device is in the source region |
| `destination` | `bool` | Whether this device is in the destination region |
| `width` | `float` | Corridor width tolerance |

**Returns:** `bool` -- `True` if device is on the channel.

---

### `partition(ctx, grain) -> int`

Partitions the network into regions of approximate diameter `grain`.

**Returns:** `int` -- the leader device ID for this device's region.

---

### `distributed_average(ctx, value) -> float`

Computes the regional average of a sensor value.

**Returns:** `float` -- average value within the device's region.
