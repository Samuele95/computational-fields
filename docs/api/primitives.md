# Core Primitives

::: computational_fields.core.primitives

The field calculus core provides five fundamental constructs and several derived operators.

## Fundamental Constructs

### `rep(ctx, init, f)`

State evolution across rounds.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `Context` | Current execution context |
| `init` | `T` | Initial value (used on round 0) |
| `f` | `(T) -> T` | Update function applied to previous state |

**Returns:** `T` -- the current state value.

**Example:**

```python
# Counter that increments each round
count = rep(ctx, 0, lambda x: x + 1)
```

---

### `nbr(ctx, value)`

Neighbour observation. Exports the device's own value and returns a dictionary of neighbour values at the same call path.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `Context` | Current execution context |
| `value` | `T` | Value to export to neighbours |

**Returns:** `dict[int, T]` -- mapping from neighbour IDs to their values.

**Example:**

```python
neighbour_values = nbr(ctx, my_value)
# {1: 3.5, 2: 7.1, 5: 0.0}
```

---

### `share(ctx, init, f)`

Combined state evolution and neighbour communication. The most commonly used construct in building blocks.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `Context` | Current execution context |
| `init` | `T` | Initial value |
| `f` | `(T, dict[int, T]) -> T` | Body function receiving previous state and neighbour values |

**Returns:** `T` -- the new state value.

**Example:**

```python
def body(prev, nbrs):
    # Minimum distance from any source
    candidates = [d + ctx.nbr_range_to(n) for n, d in nbrs.items()]
    return min(candidates) if candidates else prev

distance = share(ctx, float('inf'), body)
```

---

### `branch(ctx, cond, then_fn, else_fn)`

Domain restriction. Creates two isolated communication sub-domains.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `Context` | Current execution context |
| `cond` | `bool` | Branch condition |
| `then_fn` | `() -> T` | Function for the `True` branch |
| `else_fn` | `() -> T` | Function for the `False` branch |

**Returns:** `T` -- result of the selected branch.

!!! warning "Communication isolation"
    Devices in different branches cannot see each other's exports, even if they are physical neighbours.

---

### `foldhood(ctx, init, acc, nbr_expr)`

General fold over the neighbourhood.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `Context` | Current execution context |
| `init` | `T` | Identity element for the accumulator |
| `acc` | `(T, T) -> T` | Associative binary accumulator |
| `nbr_expr` | `T` | Expression to evaluate and export |

**Returns:** `T` -- accumulated result over all neighbours.

---

## Derived Operators

### `mux(ctx, cond, then_val, else_val)`

Multiplexer. Unlike `branch`, both values are evaluated (no communication isolation).

### `mid(ctx) -> int`

Returns the current device's ID.

### `sense(ctx, name) -> Any`

Reads a named sensor value from the device.

### `nbr_range(ctx) -> dict[int, float]`

Returns Euclidean distances to all aligned neighbours.

### `min_hood(ctx, nbr_expr) -> T`

Minimum value across the neighbourhood.

### `max_hood(ctx, nbr_expr) -> T`

Maximum value across the neighbourhood.

### `sum_hood(ctx, nbr_expr) -> T`

Sum of values across the neighbourhood.

### `min_hood_plus(ctx, expr) -> T`

Minimum value across the neighbourhood, including the device's own value.
