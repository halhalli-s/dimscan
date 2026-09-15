"""Minimal DimScan app factory and optional development server entrypoint."""

from __future__ import annotations

from typing import Any

from app.config import DimScanConfig
from app.routes import register_routes


def create_app(cfg: DimScanConfig | None = None) -> Any:
    """Create the DimScan Flask app when Flask is available."""
    active_cfg = cfg or DimScanConfig()

    try:
        from flask import Flask
    except ImportError:
        return {
            "service": "dimscan",
            "flask_available": False,
            "message": "Flask is not installed.",
        }

    app = Flask(__name__)
    register_routes(app, active_cfg)
    return app


def main() -> None:
    """Run the development server when Flask is available."""
    app = create_app()
    if hasattr(app, "run"):
        app.run(host="0.0.0.0", port=8000, debug=True)
        return

    print("DimScan server unavailable: Flask is not installed.")


if __name__ == "__main__":
    main()
