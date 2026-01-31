# Self-Stabilisation

Self-stabilisation is the defining property of the building blocks in this framework. A self-stabilising system converges to a correct configuration from **any** initial state, without explicit initialisation or failure recovery logic.

## Definition

A distributed algorithm is **self-stabilising** if, starting from an arbitrary state, it reaches a correct (legitimate) state within a finite number of rounds, and remains correct as long as no further perturbations occur.

Formally, for a building block $B$ with intended output field $\varphi^*$:

$$\forall \varphi_0 \in \text{States}: \exists k \in \mathbb{N}: \forall t \geq k: B^t(\varphi_0) = \varphi^*$$

## Why Self-Stabilisation Matters

In large-scale distributed systems:

- Devices join and leave unpredictably
- Communication links fail and recover
- Sensor readings change continuously
- There is no global clock or coordinator

Self-stabilisation means the system **automatically recovers** from any transient fault without manual intervention.

## Convergence of the G Block

The gradient is the foundational self-stabilising block. Consider a network of diameter $d$:

**Round 0**: Source devices output `0.0`; all others output `+inf`.

**Round k** ($k \geq 1$): Each device computes:

$$G_\delta^{(k)} = \min_{n \in N_\delta} \big( G_n^{(k-1)} + \text{dist}(\delta, n) \big)$$

After at most $d$ rounds, every device holds its true shortest-path distance to the nearest source.

!!! note "Convergence bound"
    The gradient converges in exactly $O(\text{diameter})$ rounds. For an $n \times n$ grid, this is $O(n)$.

## Self-Healing Demo

The simulator includes a **Self-Healing** demonstration that shows self-stabilisation in action:

1. A gradient field stabilises on a grid network
2. A rectangular region of devices is removed (simulating failure)
3. The gradient automatically recomputes around the gap
4. Within $O(\text{new\_diameter})$ rounds, the field is correct again

No special recovery code is needed -- self-stabilisation is inherent in the algorithm.

## Compositionality

A key result from the field calculus literature is that **composition of self-stabilising blocks produces self-stabilising programs**. The `channel` composite, for instance, uses three gradient computations. Because each is independently self-stabilising, the channel as a whole self-stabilises.

This compositionality property is what makes the building block approach practical: complex behaviours inherit correctness guarantees from their components.

## References

- Dijkstra, E. W. (1974). *Self-stabilizing systems in spite of distributed control*. Communications of the ACM, 17(11).
- Viroli, M., Audrito, G., Beal, J., Damiani, F., & Pianini, D. (2018). *Engineering resilient collective adaptive systems by self-stabilisation*. ACM Transactions on Modeling and Computer Simulation, 28(2).
