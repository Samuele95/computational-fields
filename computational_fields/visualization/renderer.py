"""Matplotlib-based 2D visualization for computational field networks."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

from ..simulation.engine import SimulationEngine


class FieldRenderer:
    """Renders a snapshot or animation of a computational field network."""

    def __init__(self, engine: SimulationEngine) -> None:
        self.engine = engine

    def _device_positions(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        ids = sorted(self.engine.network.devices.keys())
        xs = np.array([self.engine.network.devices[i].position[0] for i in ids])
        ys = np.array([self.engine.network.devices[i].position[1] for i in ids])
        return xs, ys, ids

    def _edges(self) -> list[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        edges: list[tuple[int, int]] = []
        for dev in self.engine.network.devices.values():
            for nid in dev.neighbors:
                pair = (min(dev.id, nid), max(dev.id, nid))
                if pair not in seen:
                    seen.add(pair)
                    edges.append((dev.id, nid))
        return edges

    def render_scalar_field(
        self,
        field: dict[int, float],
        *,
        title: str = "Computational Field",
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        show_edges: bool = True,
        show_values: bool = False,
        ax: Any = None,
    ) -> Any:
        """Draw devices colored by a scalar field value."""
        if ax is None:
            _fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        xs, ys, ids = self._device_positions()
        vals = np.array([field.get(i, float("nan")) for i in ids])

        # Replace inf with nan for coloring
        finite = np.isfinite(vals)
        plot_vals = np.where(finite, vals, np.nan)

        if show_edges:
            devs = self.engine.network.devices
            for a, b in self._edges():
                ax.plot(
                    [devs[a].position[0], devs[b].position[0]],
                    [devs[a].position[1], devs[b].position[1]],
                    color="lightgray", linewidth=0.5, zorder=1,
                )

        sc = ax.scatter(
            xs, ys, c=plot_vals, cmap=cmap, s=80, edgecolors="black",
            linewidths=0.5, zorder=2, vmin=vmin, vmax=vmax,
        )
        plt.colorbar(sc, ax=ax, shrink=0.8)

        if show_values:
            for i, did in enumerate(ids):
                v = vals[i]
                label = f"{v:.1f}" if np.isfinite(v) else "inf"
                ax.annotate(
                    label, (xs[i], ys[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=6, color="gray",
                )

        ax.set_title(title)
        ax.set_aspect("equal")
        return ax

    def render_categorical_field(
        self,
        field: dict[int, str],
        *,
        title: str = "Categorical Field",
        color_map: dict[str, str] | None = None,
        show_edges: bool = True,
        ax: Any = None,
    ) -> Any:
        """Draw devices colored by categorical labels (e.g. alert levels)."""
        if ax is None:
            _fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        xs, ys, ids = self._device_positions()
        labels = [field.get(i, "UNKNOWN") for i in ids]
        unique = sorted(set(labels))

        if color_map is None:
            tab_colors = list(mcolors.TABLEAU_COLORS.values())
            color_map = {lbl: tab_colors[i % len(tab_colors)] for i, lbl in enumerate(unique)}

        colors = [color_map.get(lbl, "gray") for lbl in labels]

        if show_edges:
            devs = self.engine.network.devices
            for a, b in self._edges():
                ax.plot(
                    [devs[a].position[0], devs[b].position[0]],
                    [devs[a].position[1], devs[b].position[1]],
                    color="lightgray", linewidth=0.5, zorder=1,
                )

        ax.scatter(xs, ys, c=colors, s=80, edgecolors="black",
                   linewidths=0.5, zorder=2)

        # Legend
        for lbl in unique:
            ax.scatter([], [], c=color_map.get(lbl, "gray"), label=lbl, s=60)
        ax.legend(loc="upper right", fontsize=8)

        ax.set_title(title)
        ax.set_aspect("equal")
        return ax

    def animate_scalar_field(
        self,
        num_rounds: int,
        field_extractor: Callable[[dict[int, Any]], dict[int, float]],
        *,
        title: str = "Field Evolution",
        cmap: str = "viridis",
        vmin: float = 0.0,
        vmax: float | None = None,
        interval_ms: int = 200,
        show_edges: bool = True,
    ) -> FuncAnimation:
        """Animate the evolution of a scalar field across rounds.

        Parameters
        ----------
        field_extractor:
            Callable that takes ``engine.results`` and returns a
            ``{device_id: float}`` dict to visualize.
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        xs, ys, ids = self._device_positions()

        if show_edges:
            devs = self.engine.network.devices
            for a, b in self._edges():
                ax.plot(
                    [devs[a].position[0], devs[b].position[0]],
                    [devs[a].position[1], devs[b].position[1]],
                    color="lightgray", linewidth=0.5, zorder=1,
                )

        sc = ax.scatter(xs, ys, c=[0.0] * len(ids), cmap=cmap, s=80,
                        edgecolors="black", linewidths=0.5, zorder=2,
                        vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_aspect("equal")
        title_obj = ax.set_title(f"{title} — Round 0")

        def update(frame: int) -> Any:
            self.engine.step()
            field = field_extractor(self.engine.results)
            vals = np.array([field.get(i, float("nan")) for i in ids])
            vals = np.where(np.isfinite(vals), vals, np.nan)
            sc.set_array(vals)
            title_obj.set_text(f"{title} — Round {frame + 1}")
            return (sc, title_obj)

        anim = FuncAnimation(fig, update, frames=num_rounds,
                             interval=interval_ms, blit=False)
        return anim
