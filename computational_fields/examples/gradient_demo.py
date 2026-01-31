"""Gradient field demo.

Creates a 10x10 grid with device 0 (bottom-left corner) as source.
Runs the gradient (distance estimation) program for 20 rounds and
visualizes the resulting distance field.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from ..core.context import Context
from ..blocks.gradient import gradient
from ..simulation.network import Network
from ..simulation.engine import SimulationEngine
from ..visualization.renderer import FieldRenderer


def gradient_program(ctx: Context) -> float:
    """Each device computes its distance to the source."""
    source = ctx.sense("is_source") or False
    return gradient(ctx, source)


def main() -> None:
    # 10x10 grid, spacing 1.0, comm_range 1.5
    def sensors(did: int, row: int, col: int) -> dict:
        return {"is_source": did == 0}

    net = Network.grid(10, 10, spacing=1.0, sensors_fn=sensors)
    engine = SimulationEngine(net, gradient_program)

    # Run for 20 rounds (distance propagates 1 hop per round)
    engine.run(20)

    renderer = FieldRenderer(engine)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Show rounds 1, 10, 20
    for ax, round_idx in zip(axes, [0, 9, 19]):
        renderer.render_scalar_field(
            engine.history[round_idx],
            title=f"Gradient — Round {round_idx + 1}",
            cmap="plasma",
            vmin=0, vmax=14,
            show_values=False,
            ax=ax,
        )
    plt.tight_layout()
    plt.savefig("gradient_demo.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
