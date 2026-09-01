#!/usr/bin/env python3
"""KDD — Data preparation.

Loads raw KDD data via the registry handler, splits into Sectors A/B/C,
and saves everything to a timestamped run folder.

Usage::

    python -m datasets.kdd.prepare --data-dir ./data/CIC-KDD
    python -m datasets.kdd.prepare --data-dir ./data/CIC-KDD --scaler standard
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from joblib import dump as joblib_dump
from datasets.registry import get_handler
import datasets  # noqa: F401 — trigger registration

from experiments.prepare_protocol import prepare_locked_artifacts
from experiments.manifest import make_locked_splits

from turboguard.config import SEED
from turboguard.console import console, header, sector_stats, done
from turboguard.persistence import create_run_dir, save_config, save_run_metadata

RESULTS_BASE = Path("results")


def main():
    parser = argparse.ArgumentParser(description="KDD — Prepare data")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--scaler", default="minmax", choices=["minmax", "standard"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    handler = get_handler("kdd")
    header("Prepare Data", dataset="kdd")
    console.print(f"  scaler={args.scaler}")

    _, y, _ = handler.load(
        data_dir=args.data_dir, scaler_type=args.scaler, fit_indices=np.empty(0, dtype=np.int64)
    )
    preview_splits = make_locked_splits(y, args.seed)
    fit_indices = preview_splits["train"][y[preview_splits["train"]] == 0]
    X, y, scaler = handler.load(
        data_dir=args.data_dir, scaler_type=args.scaler, fit_indices=fit_indices
    )
    preprocessing_audit = handler.load_audit()
    preprocessing_audit.update({"scaler_fit_split": "train", "scaler_fit_benign_rows": int(len(fit_indices))})

    base = Path(args.output_dir) if args.output_dir else RESULTS_BASE
    run_dir = create_run_dir(base, "kdd", "prepare")
    prepared = prepare_locked_artifacts(
        run_dir, "kdd", X, y, args.seed, preprocessing_audit
    )
    for name in ("train", "v_sel", "b_fit", "b_cal", "test"):
        ys = prepared["arrays"][f"y_{name}"]
        yb = (ys > 0).astype(int)
        sector_stats(name, len(ys), int((yb == 0).sum()), int((yb == 1).sum()))
    joblib_dump(scaler, run_dir / "models" / "scaler.joblib")
    save_config(
        run_dir,
        {
            "command": "prepare",
            "dataset": "kdd",
            "input_dim": handler.input_dim(),
            "scaler": args.scaler,
            "seed": args.seed,
            "split_manifest": str(run_dir / "split_manifest.json"),
            "group_file": args.group_file or "",
            "preprocessing_audit": preprocessing_audit,
            "n_train": prepared["statistics"]["splits"]["train"]["n"],
            "n_v_sel": prepared["statistics"]["splits"]["v_sel"]["n"],
            "n_b_fit": prepared["statistics"]["splits"]["b_fit"]["n"],
            "n_b_cal": prepared["statistics"]["splits"]["b_cal"]["n"],
            "n_test": prepared["statistics"]["splits"]["test"]["n"],
        },
    )
    save_run_metadata(run_dir, args)

    done("Saved", path=run_dir)


if __name__ == "__main__":
    main()
