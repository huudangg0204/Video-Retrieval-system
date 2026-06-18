#!/usr/bin/env python3
"""
start_services.py — Khởi động Elasticsearch + Kibana bằng Docker Compose.

Usage:
    py -3.10 scripts/start_services.py           # Start all
    py -3.10 scripts/start_services.py --stop    # Stop all
    py -3.10 scripts/start_services.py --status  # Show status
    py -3.10 scripts/start_services.py --logs    # Show ES logs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

COMPOSE_FILE = Path(__file__).parent.parent / "docker" / "docker-compose.yml"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def start() -> None:
    print("\n[Starting Elasticsearch + Kibana via Docker Compose]")
    if not COMPOSE_FILE.exists():
        print(f"  ✗ Not found: {COMPOSE_FILE}")
        sys.exit(1)
    result = run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d"],
        text=True,
    )
    if result.returncode == 0:
        print("\n  ✓ Services started.")
        print("  Elasticsearch: http://localhost:9200")
        print("  Kibana:        http://localhost:5601  (if enabled)")
        print("\n  Wait ~15s for ES to be ready, then run:")
        print("  py -3.10 scripts/check_env.py")
    else:
        print("\n  ✗ Failed to start services.")
        sys.exit(1)


def stop() -> None:
    print("\n[Stopping services]")
    run(["docker", "compose", "-f", str(COMPOSE_FILE), "down"])


def status() -> None:
    print("\n[Service Status]")
    run(["docker", "compose", "-f", str(COMPOSE_FILE), "ps"])


def logs() -> None:
    print("\n[Elasticsearch Logs (last 50 lines)]")
    run(["docker", "compose", "-f", str(COMPOSE_FILE), "logs", "--tail=50", "elasticsearch"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIC-26 Docker service manager")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stop",   action="store_true", help="Stop all services")
    group.add_argument("--status", action="store_true", help="Show service status")
    group.add_argument("--logs",   action="store_true", help="Show Elasticsearch logs")
    args = parser.parse_args()

    if args.stop:
        stop()
    elif args.status:
        status()
    elif args.logs:
        logs()
    else:
        start()
