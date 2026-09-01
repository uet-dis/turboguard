#!/usr/bin/env python3
"""UNSW-NB15 — Entry point.

Usage::

    python datasets/unsw/main.py prepare          --data-dir ./data/CIC-UNSW
    python datasets/unsw/main.py train             --run-dir results/unsw/<ts>_prepare
    python datasets/unsw/main.py generate-attacks  --run-dir results/unsw/<ts>_prepare
    python datasets/unsw/main.py eval              --run-dir ... --tg-dir ... --attack-dir ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from turboguard.console import console

COMMANDS = {
    "prepare": "datasets.unsw.prepare",
    "train": "datasets.unsw.train",
    "generate-attacks": "datasets.unsw.generate_attacks",
    "eval": "datasets.unsw.eval",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        console.print("[bold]Usage:[/bold] python datasets/unsw/main.py <command> [args...]")
        console.print(f"  Commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    import importlib

    mod = importlib.import_module(COMMANDS[command])
    mod.main()


if __name__ == "__main__":
    main()
