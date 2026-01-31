# Deployment

## Local Development

```bash
git clone https://github.com/Samuele95/computational-fields.git
cd computational-fields
pip install -e .
python app.py
```

The simulator will be available at [http://localhost:7860](http://localhost:7860).

## Docker

Build and run with Docker:

```bash
docker build -t computational-fields .
docker run -p 7860:7860 computational-fields
```

### Custom Port

Override the default port via the `PORT` environment variable:

```bash
docker run -p 8080:8080 -e PORT=8080 computational-fields
```

## Hugging Face Spaces

The project is pre-configured for deployment on [Hugging Face Spaces](https://huggingface.co/spaces) using the Docker SDK.

### Steps

1. Create a new Space on Hugging Face with **Docker** SDK
2. Push the repository:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/computational-fields
git push hf main
```

3. The Space will build automatically and serve the app on port 7860

### Configuration

The `README.md` frontmatter (used by HF Spaces) can be customised:

```yaml
---
title: Computational Fields Simulator
emoji: 🌐
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `7860` | HTTP server port |

## Requirements

The application requires Python 3.10+ and the following packages:

- `numpy >= 1.24`
- `matplotlib >= 3.7`
- `plotly >= 5.18`
- `pandas >= 2.0`
- `dash >= 2.15`

All dependencies are listed in `requirements.txt` and installed automatically with `pip install -e .`.
