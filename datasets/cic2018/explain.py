#!/usr/bin/env python3
"""CIC-IDS2018 — Explainability and Feature Importance Analysis for TurboGuard.

Usage::

    python -m datasets.cic2018.explain \
        --run-dir   results/cic2018/<ts>_prepare \
        --tg-dir    results/cic2018/<ts>_turboguard
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from joblib import load as joblib_load

# Ensure the framework is in the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import shap
except ImportError:
    print("Error: 'shap' library not found. Please run 'pip install shap' first.")
    sys.exit(1)

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

def _load_turboguard(tg_dir: Path, input_dim: int, device):
    """Reconstruct TurboGuard from saved artifacts."""
    cfg_tg = load_config(tg_dir)
    tg = TurboGuard(device)
    tg._num_embeddings = cfg_tg.get("codebook_size", 1024)
    tg._latent_dim = cfg_tg.get("latent_dim", input_dim)
    tg.vqvae = VQVAE(input_dim, tg._latent_dim, tg._num_embeddings).to(device)
    tg.vqvae.load_state_dict(torch.load(tg_dir / "models" / "vqvae.pth", map_location=device, weights_only=True))
    tg.vqvae.eval()
    with open(tg_dir / "models" / "semantic_map.pkl", "rb") as f:
        tg.semantic_map = pickle.load(f)
    tg._iso_forest = joblib_load(tg_dir / "models" / "iso_forest.joblib")
    dnn_input_dim = input_dim * 3 + 6
    tg._dnn = DNNClassifier(input_dim=dnn_input_dim).to(device)
    tg._dnn.load_state_dict(torch.load(tg_dir / "models" / "dnn_greyzone.pth", map_location=device, weights_only=True))
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

def main():
    parser = argparse.ArgumentParser(description="CIC-IDS2018 — Explainability Analysis")
    parser.add_argument("--run-dir", required=True, help="Path to prepare run")
    parser.add_argument("--tg-dir", required=True, help="Path to turboguard run")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of samples for SHAP")
    args = parser.parse_args()

    device = get_device()
    dataset_name = "cic2018"
    tg_dir, prep_dir = Path(args.tg_dir), Path(args.run_dir)
    cfg_prep = load_config(prep_dir)
    sectors = load_sectors(prep_dir)
    input_dim = cfg_prep["input_dim"]

    header("Explainability Analysis", dataset=dataset_name)
    tg = _load_turboguard(tg_dir, input_dim, device)

    X_C, y_C = sectors["X_C"], sectors["y_C"]
    benign_idx, attack_idx = np.where(y_C == 0)[0], np.where(y_C > 0)[0]
    
    if args.n_samples <= 0:
        step(f"Using FULL Sector C ({len(X_C):,} instances)")
        X_sub_raw = X_C
        n_ben = len(benign_idx)
    else:
        step(f"Sampling {args.n_samples:,} instances (Balanced)")
        rng = np.random.RandomState(42)
        n_ben, n_atk = min(len(benign_idx), args.n_samples // 2), min(len(attack_idx), args.n_samples // 2)
        idx_ben = rng.choice(benign_idx, n_ben, replace=False)
        idx_atk = rng.choice(attack_idx, n_atk, replace=False)
        X_sub_raw = X_C[np.concatenate([idx_ben, idx_atk])]

    X_sub_t = torch.tensor(X_sub_raw, dtype=torch.float32, device=device)

    step("Computing SHAP Values (Grey-Zone DNN)")
    signals = tg.extract_signals(X_sub_t)
    signals_norm = tg._scaler.transform(np.log1p(np.abs(signals)) * np.sign(signals))
    raw = X_sub_raw
    recon = tg._reconstruct(X_sub_t)
    dnn_input = np.column_stack([raw, recon, np.abs(raw - recon), signals_norm])
    background = torch.tensor(dnn_input[:n_ben], dtype=torch.float32, device=device)
    test_data = torch.tensor(dnn_input, dtype=torch.float32, device=device)

    tg._dnn.eval()
    explainer = shap.DeepExplainer(tg._dnn, background)
    with console.status("[bold green]Computing SHAP Values (this may take 2-5 minutes)...[/bold green]"):
        shap_v = explainer.shap_values(test_data)
    if isinstance(shap_v, list): sv = np.array(shap_v[1])
    elif isinstance(shap_v, np.ndarray) and shap_v.ndim == 3: sv = shap_v[:, :, 1]
    else: sv = np.array(shap_v)
    mean_abs_shap = np.abs(sv).mean(axis=0)

    raw_imp, recon_imp = mean_abs_shap[:input_dim].sum(), mean_abs_shap[input_dim:2*input_dim].sum()
    res_imp, sig_imp = mean_abs_shap[2*input_dim:3*input_dim].sum(), mean_abs_shap[3*input_dim:].sum()
    total_shap = raw_imp + recon_imp + res_imp + sig_imp

    step("Computing IF Feature Importance")
    if_imp = np.zeros(6)
    for tree in tg._iso_forest.estimators_:
        if len(tree.feature_importances_) == 6: if_imp += tree.feature_importances_
    if_imp /= len(tg._iso_forest.estimators_)
    if_imp /= if_imp.sum()

    console.print(f"\n[bold green]Report for {dataset_name.upper()}[/bold green]\n")
    group_data = [
        {"Group": "6 Codebook Signals (\u03c3)", "SHAP %": f"{sig_imp/total_shap*100:5.1f}%", "Function": "VQ-VAE topology signals"},
        {"Group": "Raw input (x)", "SHAP %": f"{raw_imp/total_shap*100:5.1f}%", "Function": "Cloud NIDS features"},
        {"Group": "Reconstruction (\u1e8d)", "SHAP %": f"{recon_imp/total_shap*100:5.1f}%", "Function": "Manifold boundary"},
        {"Group": "Residual (|x - \u1e8d|)", "SHAP %": f"{res_imp/total_shap*100:5.1f}%", "Function": "Recon error signal"}
    ]
    results_table("Table 1: Feature Group SHAP Importance", sorted(group_data, key=lambda x: -float(x["SHAP %"][:-1])), ["Group", "SHAP %", "Function"])

    sig_data = []
    sig_shap_vals = mean_abs_shap[3*input_dim:]
    for i, name in enumerate(SIGNAL_NAMES):
        sig_data.append({"Signal": name, "Magnitude": f"{sig_shap_vals[i]:.4f}", "% within Signals": f"{sig_shap_vals[i]/sig_imp*100:5.1f}%"})
    results_table("Table 2: Individual Signal SHAP Ranking", sorted(sig_data, key=lambda x: -float(x["Magnitude"])), ["Signal", "Magnitude", "% within Signals"])

    if_data = []
    for i, name in enumerate(SIGNAL_NAMES):
        if_data.append({"Signal": name, "IF Importance (%)": f"{if_imp[i]*100:5.1f}%", "Distribution Bar": "\u2588" * int(if_imp[i] * 50)})
    results_table("Table 3: Isolation Forest Importance", sorted(if_data, key=lambda x: -float(x["IF Importance (%)"][:-1])), ["Signal", "IF Importance (%)", "Distribution Bar"])

    out = create_run_dir(Path("results"), dataset_name, "explain")
    report = {"groups": {"signals": sig_imp/total_shap, "raw": raw_imp/total_shap}, "signals_shap": {n: float(sig_shap_vals[i]) for i, n in enumerate(SIGNAL_NAMES)}}
    with open(out / "explainability_report.json", "w") as f: json.dump(report, f, indent=2)
    done("Report saved", path=out)

if __name__ == "__main__":
    main()
