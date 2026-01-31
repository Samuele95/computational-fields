# Field Calculus

Field calculus is the formal foundation of aggregate computing. It provides a minimal set of constructs that, when composed, can express any self-organising distributed computation.

## Computational Fields

A **computational field** is a distributed data structure that assigns a value to every device in a network at every point in time:

$$\varphi : D \times T \rightarrow V$$

- $D$ is the set of networked devices
- $T = \{0, 1, 2, \ldots\}$ is discrete time (rounds)
- $V$ is the value domain (floats, booleans, tuples, etc.)

Fields are not stored centrally. Each device holds only its own value and communicates with immediate neighbours.

## The Five Core Constructs

Field calculus defines five language constructs from which all distributed behaviours are built.

### 1. `rep` -- State Evolution

```python
rep(ctx, init, update_fn)
```

Maintains per-device state across rounds.

- **Round 0**: returns `init`
- **Round n > 0**: returns `update_fn(previous_value)`

!!! example
    A simple counter:
    ```python
    count = rep(ctx, 0, lambda x: x + 1)
    ```

### 2. `nbr` -- Neighbour Observation

```python
nbr(ctx, local_value)
```

Exports the device's own value and returns a dictionary mapping each neighbour to their value at the same program point.

!!! example
    Collect neighbour distances:
    ```python
    distances = nbr(ctx, 0.0)
    # {neighbour_id: their_value, ...}
    ```

### 3. `share` -- State + Neighbour Communication

```python
share(ctx, init, body_fn)
```

Combines `rep` and `nbr` in a single construct. The `body_fn` receives the previous state and a dictionary of neighbour values, and returns the new state.

!!! example
    Self-stabilising minimum distance:
    ```python
    def body(prev, nbrs):
        candidates = [d + ctx.nbr_range_to(n) for n, d in nbrs.items()]
        return min(candidates) if candidates else prev
    dist = share(ctx, float('inf'), body)
    ```

### 4. `branch` -- Domain Restriction

```python
branch(ctx, condition, then_fn, else_fn)
```

Partitions the network into two independent sub-domains. Devices in different branches **cannot communicate** with each other, even if they are physical neighbours.

!!! warning
    Unlike a simple `if-else`, `branch` creates communication isolation. This is essential for compositional correctness.

### 5. `foldhood` -- Neighbourhood Aggregation

```python
foldhood(ctx, init, accumulator, nbr_expression)
```

General fold (reduce) over the neighbourhood. Evaluates `nbr_expression`, exports the result, collects neighbour values, and folds them with the `accumulator`.

## Call-Path Alignment

The key mechanism enabling compositional communication is **call-path alignment**. Every invocation of a primitive is tagged with its position in the call tree:

```
program
  ├── share[0]     ← path: "share_0"
  │   └── nbr[0]  ← path: "share_0/nbr_0"
  └── nbr[1]      ← path: "nbr_1"
```

When a device reads a neighbour's exported value, it looks up the **same call path**. This ensures that values from different parts of the program never get mixed up, even when programs are composed.

## Execution Model

All devices execute **synchronously** in discrete rounds:

1. Each device runs the entire program, producing a set of exports
2. All exports are collected
3. Exports become available to neighbours in the next round

This lockstep execution model guarantees deterministic behaviour and simplifies reasoning about convergence.

## References

- Viroli, M., Beal, J., Damiani, F., Audrito, G., Casadei, R., & Pianini, D. (2019). *From distributed coordination to field calculus and aggregate computing*. Journal of Logical and Algebraic Methods in Programming, 109.
- Beal, J., Pianini, D., & Viroli, M. (2015). *Aggregate Programming for the Internet of Things*. IEEE Computer, 48(9).
