#!/usr/bin/env python3
"""UNSW — Train TurboGuard + baseline models.

Usage::

    python -m datasets.unsw.train --run-dir results/unsw/<ts>_prepare
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from joblib import dump as joblib_dump

from turboguard.config import LATENT_DIM, NUM_EMBEDDINGS, SEED
from turboguard.console import header, step, done
from turboguard.core.turboguard import TurboGuard
from turboguard.classifiers.baselines import BaselineDNN, BaselineXGB
from turboguard.device import get_device
from turboguard.persistence import (
    create_run_dir,
    load_config,
    load_sectors,
    save_config,
    save_run_metadata,
)

RESULTS_BASE = Path("results")


def main():
    parser = argparse.ArgumentParser(description="UNSW — Train models")
    parser.add_argument("--run-dir", required=True, help="Path to prepare run")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--vqvae-epochs", type=int, default=20)
    parser.add_argument("--reset-interval", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=NUM_EMBEDDINGS)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--baseline-epochs", type=int, default=30)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-turboguard", action="store_true")
    parser.add_argument("--fpr-budget", type=float, default=1.5)
    parser.add_argument("--dnn-batch-size", type=int, default=2048)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir_src = Path(args.run_dir)
    cfg = load_config(run_dir_src)
    sectors = load_sectors(run_dir_src)
    device = get_device()

    X_A, y_A = sectors["X_A"], sectors["y_A"]
    X_A_t = torch.tensor(X_A, dtype=torch.float32, device=device)

    base = Path(args.output_dir) if args.output_dir else RESULTS_BASE

    if not args.skip_baseline:
        header("Train Baselines", dataset="unsw")
        bl_dir = create_run_dir(base, "unsw", "baseline")

        step("Training XGBoost")
        xgb = BaselineXGB()
        xgb.fit(X_A, y_A)
        joblib_dump(xgb.clf, bl_dir / "models" / "baseline_xgb.joblib")

        step("Training DNN")
        dnn = BaselineDNN(device)
        dnn.fit(X_A_t, y_A, epochs=args.baseline_epochs)
        torch.save(dnn.dnn.state_dict(), bl_dir / "models" / "baseline_dnn.pth")

        save_config(
            bl_dir,
            {
                "command": "train-baseline",
                "source_prepare": str(run_dir_src),
                "dataset": "unsw",
                "input_dim": cfg["input_dim"],
            },
        )
        save_run_metadata(bl_dir, args)
        done("Baselines saved", path=bl_dir)

    if not args.skip_turboguard:
        header("Train TurboGuard", dataset="unsw")
        tg_dir = create_run_dir(base, "unsw", "turboguard")

        tg = TurboGuard(device)
        tg.fit(
            X_A_t,
            y_A,
            input_dim=cfg["input_dim"],
            seed=args.seed,
            latent_dim=args.latent_dim,
            num_embeddings=args.codebook_size,
            vqvae_epochs=args.vqvae_epochs,
            reset_interval=args.reset_interval,
            dnn_batch_size=args.dnn_batch_size,
        )

        step("Joint calibration on B_fit/B_cal/V_sel")
        X_fit = torch.tensor(sectors.get("X_b_fit", sectors["X_A"]), dtype=torch.float32, device=device)
        y_fit = torch.tensor(sectors.get("y_b_fit", sectors["y_A"]), dtype=torch.long, device=device)
        tg.fit_isolation_forest_reference(X_fit[y_fit == 0], seed=args.seed)
        X_B = torch.tensor(sectors.get("X_b_cal", sectors["X_B"]), dtype=torch.float32, device=device)
        y_B = sectors.get("y_b_cal", sectors["y_B"])
        X_sel = torch.tensor(sectors.get("X_v_sel", sectors["X_B"]), dtype=torch.float32, device=device)
        y_sel = sectors.get("y_v_sel", sectors["y_B"])
        cal_result = tg.calibrate_joint(X_B, y_B, X_sel, y_sel, target_fprs=(0.001, 0.005, 0.01))

        # Display calibration trade-off table for explainability.
        from turboguard.console import console, results_table

        sweep_rows = []
        for row in cal_result["sweep"]:
            marker = " ✓" if row["selected"] else ""
            sweep_rows.append(
                {
                    "P": f"P{row['percentile']}",
                    "FPR": f"{row['FPR']:.2f}%",
                    "ADR": f"{row.get('selection_ADR', row['ADR']):.1f}%",
                    "Budget": "✓" if row["within_budget"] else "✗",
                    "": marker,
                }
            )
        results_table(
            f"Calibration Sweep (budget={args.fpr_budget}%)",
            sweep_rows,
            ["P", "FPR", "ADR", "Budget", ""],
        )
        console.print(
            f"  Selected: [bold]P{cal_result['selected_percentile']}[/bold] "
            f"(threshold={cal_result['selected_threshold']:.6f})"
        )

        torch.save(tg.vqvae.state_dict(), tg_dir / "models" / "vqvae.pth")
        with open(tg_dir / "models" / "semantic_map.pkl", "wb") as f:
            pickle.dump(tg.semantic_map, f)
        joblib_dump(tg._iso_forest, tg_dir / "models" / "iso_forest.joblib")
        torch.save(tg._dnn.state_dict(), tg_dir / "models" / "dnn_greyzone.pth")
        joblib_dump(tg._scaler, tg_dir / "models" / "signal_scaler.joblib")
        np.savez(
            tg_dir / "models" / "ctf_refs.npz",
            pca_mean=tg._pca_mean,
            pca_components=tg._pca_components,
            ctf_mean=tg._ctf_mean,
            ctf_inv_cov=tg._ctf_inv_cov,
        )
        with open(tg_dir / "models" / "cc_stats.json", "w") as f:
            json.dump(
                {
                    "global_mu": tg._global_mu,
                    "global_std": tg._global_std,
                    "per_code": {str(k): list(v) for k, v in tg._code_stats.items()},
                },
                f,
                indent=2,
            )
        with open(tg_dir / "models" / "iso_threshold.json", "w") as f:
            json.dump({"threshold": float(tg._iso_threshold)}, f)
        with open(tg_dir / "models" / "dnn_threshold.json", "w") as f:
            json.dump({"threshold": float(tg._dnn_threshold)}, f)

        # Save full calibration sweep for reproducibility and reporting.
        with open(tg_dir / "calibration_sweep.json", "w") as f:
            json.dump(cal_result, f, indent=2)

        save_config(
            tg_dir,
            {
                "command": "train-turboguard",
                "source_prepare": str(run_dir_src),
                "dataset": "unsw",
                "input_dim": cfg["input_dim"],
                "seed": args.seed,
                "latent_dim": args.latent_dim,
                "codebook_size": args.codebook_size,
                "vqvae_epochs": args.vqvae_epochs,
                "reset_interval": args.reset_interval,
                "fpr_budget": args.fpr_budget,
                "dnn_batch_size": args.dnn_batch_size,
                "calibration_percentile": cal_result["selected_percentile"],
                "signal_names": list(tg._signal_names),
            },
        )
        save_run_metadata(tg_dir, args)
        done("TurboGuard saved", path=tg_dir)


if __name__ == "__main__":
    main()
