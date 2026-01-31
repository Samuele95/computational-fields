"""Hugging Face Spaces entry point."""

from computational_fields.visualization.dash_app import app

server = app.server  # expose Flask server for gunicorn fallback

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", debug=False, port=port)
