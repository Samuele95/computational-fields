# Computational Fields Simulator

An interactive simulator for **aggregate computing** and **self-organising programs**.

> **Try it live** — the simulator is deployed and running at [huggingface.co/spaces/Sams995/computational-fields](https://huggingface.co/spaces/Sams995/computational-fields). No installation required.

---

## What is Aggregate Computing?

Aggregate computing is a macro-programming paradigm for distributed systems. Instead of programming individual devices, you write a single program that conceptually executes on the entire network simultaneously. The runtime handles communication, alignment, and convergence automatically.

The central abstraction is the **computational field** -- a function mapping every device in a spatial network to a value at every point in time:

$$\varphi : D \times T \rightarrow V$$

where $D$ is the set of devices, $T$ is discrete time (rounds), and $V$ is an arbitrary value domain.

## What Does This Project Provide?

<div class="grid cards" markdown>

-   **Field Calculus Runtime**

    ---

    A Python implementation of the five core constructs: `rep`, `nbr`, `share`, `branch`, `foldhood`

-   **Self-Stabilising Blocks**

    ---

    The G (gradient), C (collect), S (sparse), and T (time) building blocks that compose into complex behaviours

-   **Interactive Simulator**

    ---

    A web-based UI with real-time visualisation of field evolution, built with Dash and Plotly

-   **Six Demonstrations**

    ---

    Ready-to-run demos: gradient, channel, sparse leaders, crowd monitoring, wave propagation, self-healing

</div>

## Quick Example

A self-stabilising gradient field that computes shortest-path distance from source devices:

```python
from computational_fields.core.context import Context
from computational_fields.blocks.gradient import gradient

def my_program(ctx: Context) -> float:
    is_source = ctx.sense("is_source")
    return gradient(ctx, is_source)
```

That's it. Each device runs this same program. Source devices return `0.0`; all others converge to their minimum hop-count distance from any source. The field self-stabilises in $O(\text{diameter})$ rounds.

## Architecture

```
Interactive Simulator  (Dash / Plotly)
        |
  Simulation Engine    (synchronous rounds, history)
        |
  Building Blocks      (G, C, S, T + composites)
        |
  Field Calculus Core  (rep, nbr, share, branch, foldhood)
        |
  Network Layer        (Device, Context, Network, Sensors)
```

## Getting Started

1. [Install the project](getting-started/installation.md)
2. [Run your first simulation](getting-started/first-steps.md)
3. [Explore the theory](theory/field-calculus.md)
4. [Try the demos](demos/overview.md)
