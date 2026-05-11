"""Entry point: python -m bench_harness.server"""

from __future__ import annotations

import argparse
import logging
import sys

from bench_harness.server.app import run_server


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="bench-harness-web",
        description="Start the Benchmark Harness web UI",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)",
    )
    parser.add_argument(
        "--storage-root",
        default=None,
        help="Override storage root path",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--allow-unsafe-storage",
        action="store_true",
        help="Allow unsafe storage root paths",
    )

    args = parser.parse_args()

    print(f"Starting Benchmark Harness Web UI")
    print(f"  Host: {args.host}:{args.port}")
    if args.storage_root:
        print(f"  Storage root: {args.storage_root}")
    print(f"  Debug: {args.debug}")
    print()

    run_server(
        host=args.host,
        port=args.port,
        storage_root=args.storage_root,
        debug=args.debug,
        allow_unsafe=args.allow_unsafe_storage,
    )


if __name__ == "__main__":
    main()
