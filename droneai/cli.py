"""Console entry points for pip install and standalone builds."""

from __future__ import annotations

import argparse
import sys

from droneai import __version__
from paths import ensure_config, project_root


def _print_root_hint() -> None:
    root = project_root()
    print(f"Project data directory: {root}")
    print(f"Config file: {root / 'config.yaml'}")


def run_init() -> None:
    config_path = ensure_config()
    _print_root_hint()
    print(f"Ready. Config at: {config_path}")
    print("Next: run `droneai` (webcam) or edit config.yaml for ESP32-CAM.")


def run_app() -> None:
    ensure_config()
    from app import AIDetectionSystem

    try:
        AIDetectionSystem().run()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def run_test(extra_argv: list[str]) -> None:
    from test_integration import main as test_main

    sys.argv = ["droneai-test", *extra_argv]
    sys.exit(test_main())


def run_fetch_video() -> None:
    ensure_config()
    from fetch_video import main as fetch_main

    fetch_main()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="droneai",
        description="Real-time knife and pistol detection from webcam or ESP32-CAM.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Start live detection (default)")
    sub.add_parser("init", help="Create config.yaml in the project data directory")
    sub.add_parser("test", help="Run integration tests (pass --offline for no camera)")
    sub.add_parser("fetch-video", help="Open ESP32-CAM stream viewer")

    if len(sys.argv) == 1:
        run_app()
        return

    args, extra = parser.parse_known_args()
    command = args.command

    if command is None and extra:
        parser.error(f"unknown arguments: {' '.join(extra)}")
    if command is None:
        run_app()
        return

    if command == "init":
        run_init()
    elif command == "test":
        run_test(extra)
    elif command == "fetch-video":
        run_fetch_video()
    else:
        run_app()


if __name__ == "__main__":
    main()
