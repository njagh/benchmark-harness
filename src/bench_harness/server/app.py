"""Flask app factory for the Benchmark Harness web UI."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from flask import Flask

from bench_harness.server.routes.configs import configs_bp
from bench_harness.server.routes.runs import runs_bp
from bench_harness.server.routes.results import results_bp
from bench_harness.server.routes.compare import compare_bp
from bench_harness.server.routes.models import models_bp
from bench_harness.server.routes.export import export_bp
from bench_harness.server.utils.storage import resolve_storage_config, get_storage_info

logger = logging.getLogger(__name__)


def create_app(
    storage_root: Optional[str] = None,
    allow_unsafe: bool = False,
) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.url_prefix = ""

    # Storage config
    storage_config = resolve_storage_config(storage_root, allow_unsafe)
    storage_config.ensure_namespaces()
    app.storage_config = storage_config
    app.storage_info = get_storage_info(storage_config)

    # Register blueprints
    app.register_blueprint(configs_bp)
    app.register_blueprint(runs_bp)
    app.register_blueprint(results_bp)
    app.register_blueprint(compare_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(export_bp)

    # Root route
    @app.route("/")
    def index():
        return app.send_static_file("index.html")

    # Health check
    @app.route("/api/health")
    def health():
        return {
            "status": "ok",
            "storage_root": str(storage_config.root),
            "results_runs": str(storage_config.results_runs),
        }

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 5000,
    storage_root: Optional[str] = None,
    debug: bool = False,
    allow_unsafe: bool = False,
) -> None:
    """Run the Flask development server."""
    app = create_app(storage_root, allow_unsafe)
    app.run(host=host, port=port, debug=debug, threaded=True)
