# Installation

## Prerequisites

- **Python 3.10** or later
- **pip** or [uv](https://github.com/astral-sh/uv)

## From Source

```bash
git clone https://github.com/Samuele95/computational-fields.git
cd computational-fields
pip install -e .
```

## Dependencies

The project depends on:

| Package | Version | Purpose |
|---------|---------|---------|
| NumPy | >= 1.24 | Array operations, distance calculations |
| Matplotlib | >= 3.7 | Static field rendering |
| Plotly | >= 5.18 | Interactive charts |
| Pandas | >= 2.0 | Data handling |
| Dash | >= 2.15 | Web application framework |

### Optional: Development Tools

```bash
pip install -e ".[dev]"
```

This adds `pytest` for running the test suite.

## Verify Installation

```bash
python -c "from computational_fields.core.primitives import rep, nbr, share; print('OK')"
```

## Docker

```bash
docker build -t computational-fields .
docker run -p 7860:7860 computational-fields
```

Then open [http://localhost:7860](http://localhost:7860).
