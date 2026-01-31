# Simulation API

## Network

::: computational_fields.simulation.network

### `Network(comm_range)`

Creates an empty network with the given communication range.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `devices` | `dict[int, Device]` | All devices in the network |
| `comm_range` | `float` | Maximum communication distance |

### Methods

#### `add_device(position, sensors=None, device_id=None) -> Device`

Add a device at the given `(x, y)` position.

#### `remove_device(device_id) -> None`

Remove a device and update neighbour lists.

#### `update_neighbors() -> None`

Recompute all neighbour lists based on Euclidean distance and `comm_range`.

#### `get_distance(id_a, id_b) -> float`

Euclidean distance between two devices.

### Factory Methods

#### `Network.grid(rows, cols, spacing=1.0, comm_range=1.5, sensors_fn=None) -> Network`

Create a regular grid topology.

```python
net = Network.grid(
    rows=8, cols=8, spacing=1.0, comm_range=1.5,
    sensors_fn=lambda did, row, col: {"is_source": did == 0}
)
```

The `sensors_fn` receives `(device_id, row, col)` and returns a sensor dict.

#### `Network.random(n, width, height, comm_range, sensors_fn=None, rng=None) -> Network`

Create a network with randomly placed devices.

```python
import numpy as np
net = Network.random(
    n=50, width=10, height=10, comm_range=2.5,
    sensors_fn=lambda did: {"is_source": did == 0},
    rng=np.random.default_rng(42)
)
```

---

## SimulationEngine

::: computational_fields.simulation.engine

### `SimulationEngine(network, program, delta_time=1.0)`

Orchestrates synchronous execution of an aggregate program.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `network` | `Network` | The device network |
| `program` | `(Context) -> Any` | The aggregate program |
| `delta_time` | `float` | Time step per round |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `round_count` | `int` | Current round number |
| `results` | `dict[int, Any]` | Latest round results |
| `history` | `list[dict[int, Any]]` | All past results |

### Methods

#### `step() -> dict[int, Any]`

Execute one synchronous round on all devices. Returns the results dict.

#### `run(num_rounds) -> list[dict[int, Any]]`

Execute multiple rounds. Returns the full history.

#### `get_field(key=None) -> dict[int, Any]`

Extract a named sub-field from the latest results. If `key` is `None`, returns results directly.

---

## Device

::: computational_fields.core.device

### `Device`

Dataclass representing a single device in the network.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `int` | Unique identifier |
| `position` | `np.ndarray` | 2D spatial position |
| `sensors` | `dict[str, Any]` | Sensor readings |
| `state` | `dict[str, Any]` | Persistent state across rounds |
| `exports` | `dict[str, Any]` | Values exported to neighbours |
| `neighbors` | `list[int]` | Neighbour device IDs |

---

## Context

::: computational_fields.core.context

### `Context`

Per-round execution context for a single device.

**Key Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `sense(name)` | `Any` | Read a sensor value |
| `mid()` | `int` | Current device ID |
| `nbr_range_to(nid)` | `float` | Distance to neighbour |
| `aligned_neighbors(path)` | `list[int]` | Neighbours that executed this call path |
