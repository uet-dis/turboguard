#!/usr/bin/env python3
"""CIC-IDS2017 — Entry point.

Usage::

    python datasets/cic2017/main.py prepare          --data-dir ./data/cic2017
    python datasets/cic2017/main.py train            --run-dir results/cic2017/<ts>_prepare
    python datasets/cic2017/main.py generate-attacks --run-dir results/cic2017/<ts>_prepare
    python datasets/cic2017/main.py eval             --run-dir ... --tg-dir ... --attack-dir ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from turboguard.console import console

COMMANDS = {
    "download": "datasets.cic2017.download",
    "prepare": "datasets.cic2017.prepare",
    "train": "datasets.cic2017.train",
    "generate-attacks": "datasets.cic2017.generate_attacks",
    "eval": "datasets.cic2017.eval",
    "signal-auc": "datasets.cic2017.signal_auc",
    "explain": "datasets.cic2017.explain",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        console.print("[bold]Usage:[/bold] python datasets/cic2017/main.py <command> [args...]")
        console.print(f"  Commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    import importlib

    mod = importlib.import_module(COMMANDS[command])
    mod.main()


if __name__ == "__main__":
    main()
