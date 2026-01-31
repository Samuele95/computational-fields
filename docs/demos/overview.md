# Demo Overview

The simulator includes six pre-built demonstrations that showcase the building blocks and their compositions.

## Welcome Page

![Welcome Page](../assets/screenshots/welcome.png)

Each demo card links directly to the simulator with that demo pre-selected.

## Simulator Interface

![Simulator](../assets/screenshots/simulator.png)

The simulator provides:

- **Real-time field visualisation** with interactive Plotly charts
- **Step-by-step execution** or continuous auto-play
- **History scrubbing** to review past rounds
- **Configurable network** topology, size, and communication range
- **Convergence monitoring** via a dedicated chart
- **Theory panels** with mathematical background for each demo

## Available Demos

| Demo | Type | Blocks Used | What It Shows |
|------|------|-------------|---------------|
| [Gradient](gradient.md) | Scalar field | G | Distance propagation, convergence |
| [Channel](channel.md) | Categorical | G + G + G | Path formation, composition |
| [Sparse Leaders](sparse-leaders.md) | Leader map | S | Election, Voronoi partitioning |
| [Crowd Monitoring](crowd-monitoring.md) | Multi-field | G + C + S + broadcast | Full case study |
| Wave Propagation | Scalar field | G + rep | Periodic behaviour |
| Self-Healing | Scalar field | G | Fault tolerance |
