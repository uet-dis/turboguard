#!/usr/bin/env python3
"""CIC-IDS2018 — Per-Signal AUC and Calibration Analysis for TurboGuard.

Computes:
  1. Per-signal ROC-AUC (direction-agnostic) for each attack scenario.
  2. σ-separation: (mean_adv − mean_ben) / std_ben per signal.
  3. Isolation Forest calibration sweep (FPR vs ADR at percentiles).

All numbers produced here are the empirical basis for the Signal Hierarchy
table and calibration table in the thesis (Chapter 4/5).

Usage::

    python -m datasets.cic2018.signal_auc \\
        --run-dir   results/cic2018/<ts>_prepare \\
        --tg-dir    results/cic2018/<ts>_turboguard \\
        --atk-dir   results/cic2018/<ts>_attacks

Results are saved to results/cic2018/<ts>_signal_auc/signal_auc_report.json.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from joblib import load as joblib_load
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from turboguard.console import console, header, step, done, results_table
from turboguard.core.turboguard import TurboGuard
from turboguard.device import get_device
from turboguard.models.dnn import DNNClassifier
from turboguard.models.vqvae import VQVAE
from turboguard.persistence import (
    create_run_dir,
    load_config,
    load_sectors,
    save_config,
    save_run_metadata,
)

SIGNAL_NAMES = ["CTF", "VMR", "ENT", "RE", "GEO", "CC"]

# Attacks to report in the thesis table (must match filenames in <atk-dir>/attacks/)
THESIS_ATTACKS = [
    ("FGSM_0.003", "X_fgsm_0003"),
    ("FGSM_0.05",  "X_fgsm_0050"),
    ("C&W_0.5",    "X_cw_0500"),
    ("DeepFool",   "X_deepfool"),
]

# Calibration percentiles to sweep
CALIB_PERCENTILES = [0.1, 0.5, 1.0, 2.0, 5.0]


def _load_turboguard(tg_dir: Path, input_dim: int, device):
    """Reconstruct TurboGuard from saved artifacts (mirrors explain.py)."""
    cfg_tg = load_config(tg_dir)
    tg = TurboGuard(device)
    tg._num_embeddings = cfg_tg.get("codebook_size", 1024)
    tg._latent_dim = cfg_tg.get("latent_dim", input_dim)
    tg.vqvae = VQVAE(input_dim, tg._latent_dim, tg._num_embeddings).to(device)
    tg.vqvae.load_state_dict(
        torch.load(tg_dir / "models" / "vqvae.pth", map_location=device, weights_only=True)
    )
    tg.vqvae.eval()
    with open(tg_dir / "models" / "semantic_map.pkl", "rb") as f:
        tg.semantic_map = pickle.load(f)
    tg._iso_forest = joblib_load(tg_dir / "models" / "iso_forest.joblib")
    dnn_input_dim = input_dim * 3 + 6
    tg._dnn = DNNClassifier(input_dim=dnn_input_dim).to(device)
    tg._dnn.load_state_dict(
        torch.load(tg_dir / "models" / "dnn_greyzone.pth", map_location=device, weights_only=True)
    )
    tg._dnn.eval()
    tg._scaler = joblib_load(tg_dir / "models" / "signal_scaler.joblib")
    ctf = np.load(tg_dir / "models" / "ctf_refs.npz")
    tg._pca_mean, tg._pca_components = ctf["pca_mean"], ctf["pca_components"]
    tg._ctf_mean, tg._ctf_inv_cov = ctf["ctf_mean"], ctf["ctf_inv_cov"]
    with open(tg_dir / "models" / "cc_stats.json") as f:
        cc = json.load(f)
    tg._global_mu, tg._global_std = cc["global_mu"], cc["global_std"]
    tg._code_stats = {int(k): tuple(v) for k, v in cc["per_code"].items()}
    thresh_path = tg_dir / "models" / "iso_threshold.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            tg._iso_threshold = json.load(f)["threshold"]
    return tg


def _auc_and_sep(sig_ben: np.ndarray, sig_adv: np.ndarray, j: int):
    """Return (AUC, sigma_separation) for signal index j."""
    s_b = sig_ben[:, j]
    s_a = sig_adv[:, j]
    sep = (s_a.mean() - s_b.mean()) / max(s_b.std(), 1e-10)
    y_true = np.concatenate([np.zeros(len(s_b)), np.ones(len(s_a))])
    scores = np.concatenate([s_b, s_a])
    try:
        auc = roc_auc_score(y_true, scores)
        auc = max(auc, 1.0 - auc)  # direction-agnostic
    except Exception:
        auc = 0.5
    return float(auc), float(sep)


def main():
    parser = argparse.ArgumentParser(description="CIC-IDS2018 — Signal AUC Analysis")
    parser.add_argument("--run-dir", required=True, help="Path to prepare run")
    parser.add_argument("--tg-dir",  required=True, help="Path to turboguard run")
    parser.add_argument("--atk-dir", required=True, help="Path to attacks run")
    parser.add_argument(
        "--n-benign", type=int, default=0,
        help="Benign samples for reference (0 = use all Sector C benign)",
    )
    parser.add_argument(
        "--n-adv", type=int, default=0,
        help="Adversarial samples per attack file (0 = use all)",
    )
    args = parser.parse_args()

    device = get_device()
    tg_dir   = Path(args.tg_dir)
    prep_dir = Path(args.run_dir)
    atk_dir  = Path(args.atk_dir)

    cfg_prep = load_config(prep_dir)
    sectors  = load_sectors(prep_dir)
    input_dim = cfg_prep["input_dim"]

    header("Signal AUC Analysis", dataset="cic2018")
    tg = _load_turboguard(tg_dir, input_dim, device)

    # ── Benign reference signals ──────────────────────────────────────────
    X_C, y_C = sectors["X_C"], sectors["y_C"]
    ben_idx = np.where(y_C == 0)[0]
    n_ben = len(ben_idx) if args.n_benign <= 0 else min(args.n_benign, len(ben_idx))
    step(f"Extracting benign reference signals ({n_ben:,} / {len(ben_idx):,} available)")
    X_ben_t = torch.tensor(X_C[ben_idx[:n_ben]], dtype=torch.float32, device=device)
    sig_ben = tg.extract_signals(X_ben_t)

    # ── Load attack files ────────────────────────────────────────────────
    atk_file_dir = atk_dir / "attacks"
    step("Loading attack files")
    scenarios = []
    for label, stem in THESIS_ATTACKS:
        matches = list(atk_file_dir.glob(f"{stem}*.pt"))
        if not matches:
            matches = [p for p in atk_file_dir.glob("*.pt") if stem.lower() in p.stem.lower()]
        if matches:
            X_adv_full = torch.load(matches[0], weights_only=True).to(device)
            n_adv = len(X_adv_full) if args.n_adv <= 0 else min(args.n_adv, len(X_adv_full))
            X_adv = X_adv_full[:n_adv]
            scenarios.append((label, X_adv))
            console.print(f"    [green]✓[/green] {label}: {n_adv:,} / {len(X_adv_full):,} samples ({matches[0].name})")
        else:
            console.print(f"    [yellow]⚠[/yellow] {label}: not found (skipping)")

    if not scenarios:
        console.print("  [yellow]No matching files for thesis table — loading all available .pt files[/yellow]")
        for pt in sorted(atk_file_dir.glob("*.pt")):
            X_adv_full = torch.load(pt, weights_only=True).to(device)
            n_adv = len(X_adv_full) if args.n_adv <= 0 else min(args.n_adv, len(X_adv_full))
            scenarios.append((pt.stem, X_adv_full[:n_adv]))

    # ── Per-signal AUC table ─────────────────────────────────────────────
    step("Computing per-signal AUC and σ-separation")
    auc_results = {}   # { scenario_label: { signal_name: {auc, sep} } }

    for label, X_adv in scenarios:
        sig_adv = tg.extract_signals(X_adv)
        auc_results[label] = {}
        for j, name in enumerate(SIGNAL_NAMES):
            auc, sep = _auc_and_sep(sig_ben, sig_adv, j)
            auc_results[label][name] = {"auc": round(auc, 4), "sep_sigma": round(sep, 2)}

    # Print AUC table
    console.print("\n[bold green]Per-Signal AUC (direction-agnostic)[/bold green]")
    header_row = f"{'Signal':<7}" + "".join(f"  {lbl:>12}" for lbl, _ in scenarios)
    console.print(header_row)
    console.print("-" * len(header_row))
    for name in SIGNAL_NAMES:
        row = f"{name:<7}"
        for label, _ in scenarios:
            row += f"  {auc_results[label][name]['auc']:>12.3f}"
        console.print(row)

    # Print σ-separation table
    console.print("\n[bold green]σ-Separation: (mean_adv − mean_ben) / std_ben[/bold green]")
    console.print(header_row)
    console.print("-" * len(header_row))
    for name in SIGNAL_NAMES:
        row = f"{name:<7}"
        for label, _ in scenarios:
            row += f"  {auc_results[label][name]['sep_sigma']:>+12.2f}σ"
        console.print(row)

    # ── Calibration sweep ────────────────────────────────────────────────
    step("Running IF calibration sweep (Sector B benign scores)")
    # Use Sector B for calibration (as in the actual training protocol)
    X_B, y_B = sectors["X_B"], sectors["y_B"]
    X_B_t = torch.tensor(X_B, dtype=torch.float32, device=device)
    sig_B  = tg.extract_signals(X_B_t)

    ben_mask_B = y_B == 0
    atk_mask_B = y_B > 0
    scores_ben_B = tg._iso_forest.decision_function(sig_B[ben_mask_B])
    scores_atk_B = tg._iso_forest.decision_function(sig_B[atk_mask_B])

    calib_results = []
    iso_thresh = getattr(tg, "_iso_threshold", None)
    console.print(f"\n  {'Percentile':>11} | {'FPR %':>7} | {'ADR %':>7} | {'Selected':>10}")
    console.print("  " + "-" * 44)
    for pct in CALIB_PERCENTILES:
        thresh = np.percentile(scores_ben_B, pct)
        fpr = float((scores_ben_B < thresh).mean() * 100)
        adr = float((scores_atk_B < thresh).mean() * 100)
        is_selected = (iso_thresh is not None) and abs(thresh - iso_thresh) < 1e-8
        marker = " ← SELECTED" if is_selected else ""
        console.print(f"  {pct:>10.1f}% | {fpr:>6.2f}% | {adr:>6.1f}% |{marker}")
        calib_results.append({
            "percentile": pct, "fpr_pct": round(fpr, 2),
            "adr_pct": round(adr, 1), "selected": bool(is_selected),
        })

    # ── Save report ──────────────────────────────────────────────────────
    out = create_run_dir(Path("results"), "cic2018", "signal_auc")
    report = {
        "signal_auc": auc_results,
        "calibration_sweep": calib_results,
    }
    with open(out / "signal_auc_report.json", "w") as f:
        json.dump(report, f, indent=2)

    save_config(out, {
        "command": "signal-auc",
        "dataset": "cic2018",
        "source_prepare":    str(prep_dir),
        "source_turboguard": str(tg_dir),
        "source_attacks":    str(atk_dir),
        # CLI params (raw as provided)
        "arg_n_benign": args.n_benign,
        "arg_n_adv":    args.n_adv,
        # Resolved values actually used
        "n_benign_used": n_ben,
        "n_adv_used":    int(len(scenarios[0][1])) if scenarios else 0,
        # Scenarios loaded
        "attack_scenarios": [label for label, _ in scenarios],
        "calib_percentiles": CALIB_PERCENTILES,
    })
    save_run_metadata(out, args)
    done("Signal AUC report saved", path=out)


if __name__ == "__main__":
    main()
