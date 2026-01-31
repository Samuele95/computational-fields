# First Steps

## Running the Simulator

Start the interactive web interface:

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860). You'll see a welcome page with six demo cards. Click any demo or use the **Launch Simulator** button.

## The Simulator Interface

The simulator has three sections:

### Sidebar (left)

- **Demo selector** -- switch between the six demonstrations
- **Network** -- configure topology (grid/random), size, spacing, communication range
- **Demo Parameters** -- settings specific to the current demo
- **Simulation** -- round count and auto-play speed controls
- **Display** -- toggle edge visibility

### Main Area (centre)

- **Control bar** -- Step, Run N, Auto-Play, and Reset buttons
- **Network visualisation** -- interactive Plotly chart showing the field
- **Metrics** -- summary statistics for the current round
- **History slider** -- scrub through past rounds

### Theory & Practical Panels (below chart)

Expandable panels with mathematical background and practical usage guidance for each demo.

## Writing Your First Program

Every aggregate program is a function that takes a `Context` and returns a value:

```python
from computational_fields.core.context import Context
from computational_fields.blocks.gradient import gradient

def distance_from_source(ctx: Context) -> float:
    source = ctx.sense("is_source")
    return gradient(ctx, source)
```

### Running It Programmatically

```python
from computational_fields.simulation.network import Network
from computational_fields.simulation.engine import SimulationEngine

# Create a 5x5 grid
net = Network.grid(5, 5, spacing=1.0, comm_range=1.5,
                   sensors_fn=lambda did, r, c: {"is_source": did == 0})

# Create engine with your program
engine = SimulationEngine(net, distance_from_source)

# Run 15 rounds
results = engine.run(15)

# Inspect final state
for device_id, value in engine.results.items():
    print(f"Device {device_id}: distance = {value:.2f}")
```

### Visualising It

```python
from computational_fields.visualization.renderer import FieldRenderer

renderer = FieldRenderer(engine)
renderer.render_scalar_field(
    engine.results,
    title="Gradient Field (round 15)",
    cmap="plasma",
)
```

## Running the Examples

Three standalone examples are included:

```bash
# Gradient propagation with animated evolution
python -m computational_fields.examples.gradient_demo

# Channel formation between corners
python -m computational_fields.examples.channel_demo

# Full crowd monitoring case study
python -m computational_fields.examples.crowd_monitoring
```

## Running the Tests

```bash
pip install -e ".[dev]"
pytest -v
```

The test suite covers primitives, building blocks, composite patterns, network topology, and the simulation engine.
