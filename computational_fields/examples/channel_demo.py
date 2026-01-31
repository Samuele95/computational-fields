"""Channel formation demo.

Creates a 10x10 grid and forms a logical channel between device 0
(bottom-left) and device 99 (top-right).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from ..core.context import Context
from ..blocks.composites import channel
from ..simulation.network import Network
from ..simulation.engine import SimulationEngine
from ..visualization.renderer import FieldRenderer


def channel_program(ctx: Context) -> dict:
    source = ctx.sense("is_source") or False
    dest = ctx.sense("is_dest") or False
    on_channel = channel(ctx, source, dest, width=1.5)
    return {"on_channel": on_channel}


def main() -> None:
    def sensors(did: int, row: int, col: int) -> dict:
        return {
            "is_source": did == 0,
            "is_dest": did == 99,
        }

    net = Network.grid(10, 10, spacing=1.0, sensors_fn=sensors)
    engine = SimulationEngine(net, channel_program)
    engine.run(30)

    # Extract boolean channel field as categorical
    cat_field = {
        did: "CHANNEL" if r["on_channel"] else "OFF"
        for did, r in engine.results.items()
    }

    renderer = FieldRenderer(engine)
    renderer.render_categorical_field(
        cat_field,
        title="Channel Formation (Round 30)",
        color_map={"CHANNEL": "tomato", "OFF": "lightblue"},
    )
    plt.savefig("channel_demo.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
