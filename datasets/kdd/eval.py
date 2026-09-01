#!/usr/bin/env python3
"""KDD — Evaluate TurboGuard against generated attacks.

Usage::

    python -m datasets.kdd.eval \\
        --run-dir   results/unsw/<ts>_prepare \\
        --tg-dir    results/unsw/<ts>_turboguard \\
        --attack-dir results/unsw/<ts>_attacks
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from joblib import load as joblib_load

from turboguard.config import LATENT_DIM, NUM_EMBEDDINGS
from turboguard.console import console, header, step, metrics_line, done, results_table
from turboguard.core.turboguard import SIGNAL_NAMES, TurboGuard
from turboguard.device import get_device
from turboguard.metrics import (
    compute_adr,
    compute_edr,
    compute_fpr,
    compute_precision_recall_f1,
)
from turboguard.classifiers.baselines import BaselineXGB, BaselineDNN
from turboguard.models.dnn import DNNClassifier
from turboguard.models.vqvae import VQVAE
from turboguard.persistence import (
    create_run_dir,
    load_config,
    load_eval_pool,
    load_sectors,
    save_config,
    save_run_metadata,
)

RESULTS_BASE = Path("results")


def _load_turboguard(tg_dir: Path, input_dim: int, device):
    """Reconstruct TurboGuard from saved artifacts."""
    cfg_tg = load_config(tg_dir)
    tg = TurboGuard(device)
    tg._signal_names = tuple(cfg_tg.get("signal_names", SIGNAL_NAMES))
    tg._num_embeddings = cfg_tg.get("codebook_size", NUM_EMBEDDINGS)
    tg._latent_dim = cfg_tg.get("latent_dim", LATENT_DIM)

    tg.vqvae = VQVAE(input_dim, tg._latent_dim, tg._num_embeddings).to(device)
    tg.vqvae.load_state_dict(
        torch.load(tg_dir / "models" / "vqvae.pth", map_location=device, weights_only=True)
    )
    tg.vqvae.eval()

    with open(tg_dir / "models" / "semantic_map.pkl", "rb") as f:
        tg.semantic_map = pickle.load(f)

    tg._iso_forest = joblib_load(tg_dir / "models" / "iso_forest.joblib")

    dnn_input_dim = input_dim * 3 + len(tg._signal_names)
    tg._dnn = DNNClassifier(input_dim=dnn_input_dim).to(device)
    tg._dnn.load_state_dict(
        torch.load(tg_dir / "models" / "dnn_greyzone.pth", map_location=device, weights_only=True)
    )
    tg._dnn.eval()

    tg._scaler = joblib_load(tg_dir / "models" / "signal_scaler.joblib")

    ctf = np.load(tg_dir / "models" / "ctf_refs.npz")
    tg._pca_mean = ctf["pca_mean"]
    tg._pca_components = ctf["pca_components"]
    tg._ctf_mean = ctf["ctf_mean"]
    tg._ctf_inv_cov = ctf["ctf_inv_cov"]

    with open(tg_dir / "models" / "cc_stats.json") as f:
        cc = json.load(f)
    tg._global_mu = cc["global_mu"]
    tg._global_std = cc["global_std"]
    tg._code_stats = {int(k): tuple(v) for k, v in cc["per_code"].items()}

    thresh_path = tg_dir / "models" / "iso_threshold.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            tg._iso_threshold = json.load(f)["threshold"]
    dnn_thresh_path = tg_dir / "models" / "dnn_threshold.json"
    if dnn_thresh_path.exists():
        with open(dnn_thresh_path) as f:
            tg._dnn_threshold = json.load(f)["threshold"]

    return tg


def _load_baselines(bl_dir: Path, input_dim: int, device):
    """Reconstruct Baselines from saved artifacts."""
    xgb = BaselineXGB()
    xgb.clf = joblib_load(bl_dir / "models" / "baseline_xgb.joblib")

    dnn = BaselineDNN(device)
    dnn.dnn = DNNClassifier(input_dim=input_dim).to(device)
    dnn.dnn.load_state_dict(
        torch.load(bl_dir / "models" / "baseline_dnn.pth", map_location=device, weights_only=True)
    )
    dnn.dnn.eval()
    return {"BaselineXGB": xgb, "BaselineDNN": dnn}

def main():
    parser = argparse.ArgumentParser(description="KDD — Evaluate")
    parser.add_argument("--run-dir", required=True, help="Path to prepare run")
    parser.add_argument("--tg-dir", required=True, help="Path to turboguard run")
    parser.add_argument("--attack-dir", required=True, help="Path to attacks run")
    parser.add_argument("--baseline-dir", default=None, help="Path to baseline run")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    tg_dir = Path(args.tg_dir)
    baseline_dir = args.baseline_dir
    if not baseline_dir:
        # Try to auto-discover
        baseline_dirs = list(tg_dir.parent.glob("*_baseline"))
        if baseline_dirs:
            baseline_dir = max(baseline_dirs)
        else:
            raise ValueError("No baseline-dir provided and none found automatically.")
    else:
        baseline_dir = Path(baseline_dir)

    cfg_prep = load_config(Path(args.run_dir))
    sectors = load_sectors(Path(args.run_dir))
    cfg_atk = load_config(Path(args.attack_dir))
    pool = load_eval_pool(Path(args.attack_dir))
    device = get_device()

    header("Evaluate Benchmark (Sector C)", dataset="kdd")
    console.print("  [dim]Evaluating Baseline XGB/DNN and TurboGuard pipeline.[/dim]")

    X_C = sectors["X_C"]
    atk_idx = pool["atk_idx"]
    ben_idx = pool["ben_idx"]
    N_HALF = cfg_atk["n_half"]

    X_clean_atk = torch.tensor(X_C[atk_idx[:N_HALF]], dtype=torch.float32, device=device)
    X_benign = torch.tensor(X_C[ben_idx], dtype=torch.float32, device=device)

    step("Loading Models")
    models = _load_baselines(baseline_dir, cfg_prep["input_dim"], device)
    models["TurboGuard(DNN)"] = _load_turboguard(tg_dir, cfg_prep["input_dim"], device)

    results = {}
    results["Clean"] = {}
    metrics_line_parts = []
    
    y_eval_clean = np.concatenate(
        [np.zeros(len(X_benign), dtype=int), np.ones(len(X_clean_atk), dtype=int)]
    )

    for model_name, model in models.items():
        if model_name == "BaselineXGB":
            vpc = model.clf.predict(X_C[atk_idx[:N_HALF]])
            vpb = model.clf.predict(X_C[ben_idx])
            preds = np.concatenate([vpb, vpc])
        else:
            vpc = model.predict(X_clean_atk)
            vpb = model.predict(X_benign)
            preds = np.concatenate([vpb, vpc])
        
        adr_clean = compute_adr(vpc, np.ones(len(vpc), dtype=int))
        fpr = compute_fpr(vpb, np.zeros(len(vpb), dtype=int))
        _, _, f1 = compute_precision_recall_f1(preds, y_eval_clean)
        
        results["Clean"][model_name] = {"EDR": float('nan'), "clean_ADR": adr_clean, "ADR": adr_clean, "FPR": fpr, "F1": f1}
        metrics_line_parts.append(f"[bold]{model_name}:[/bold] FPR={fpr:.1f}% ADR={adr_clean:.1f}%")
    
    console.print("  " + "  │  ".join(metrics_line_parts))

    attack_files = sorted((Path(args.attack_dir) / "attacks").glob("*.pt"))
    table_rows = []
    for af in attack_files:
        name = af.stem
        X_adv = torch.load(af, map_location=device, weights_only=True)
        if not isinstance(X_adv, torch.Tensor):
            continue

        X_eval_t = torch.cat([X_benign, X_adv])
        X_eval_np = np.concatenate([X_C[ben_idx], X_adv.cpu().numpy()])

        y_eval = np.concatenate(
            [np.zeros(len(X_benign), dtype=int), np.ones(len(X_adv), dtype=int)]
        )
        is_evasion = np.concatenate(
            [np.zeros(len(X_benign), dtype=bool), np.ones(len(X_adv), dtype=bool)]
        )

        results[name] = {}
        for model_name, model in models.items():
            if model_name == "BaselineXGB":
                preds = model.clf.predict(X_eval_np)
            else:
                preds = model.predict(X_eval_t)

            edr = compute_edr(preds, is_evasion)
            adr = compute_adr(preds, y_eval)
            fpr = compute_fpr(preds, y_eval)
            _, _, f1 = compute_precision_recall_f1(preds, y_eval)

            results[name][model_name] = {"EDR": edr, "ADR": adr, "FPR": fpr, "F1": f1}

    from rich.table import Table
    table = Table(title="Cross-Model Benchmark Results", show_lines=False, border_style="dim")
    table.add_column("Attack", justify="left")
    table.add_column("Model", justify="left", min_width=15)
    table.add_column("EDR", justify="right")
    table.add_column("ADR", justify="right")
    table.add_column("FPR", justify="right")
    table.add_column("F1", justify="right")

    for idx, name in enumerate(["Clean"] + sorted([k for k in results.keys() if k.startswith("X_")])):
        if idx > 0:
            table.add_section()
        
        attack_printed = False
        for m_name in ["BaselineXGB", "BaselineDNN", "TurboGuard(DNN)"]:
            r = results[name][m_name]
            disp_atk = name if not attack_printed else ""
            attack_printed = True
            
            edr_str = "-" if name == "Clean" else f"{r['EDR']:.1f}%"
            
            # Highlight TurboGuard
            fmt = "[bold green]{}[/]" if m_name == "TurboGuard(DNN)" else "{}"
            table.add_row(
                disp_atk,
                fmt.format(m_name),
                fmt.format(edr_str),
                fmt.format(f"{r['ADR']:.1f}%"),
                fmt.format(f"{r['FPR']:.1f}%"),
                fmt.format(f"{r['F1']:.1f}%")
            )
            
    console.print(table)

    base = Path(args.output_dir) if args.output_dir else RESULTS_BASE
    out = create_run_dir(base, "kdd", "eval")

    with open(out / "report.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = ["Benchmark Evaluation — KDD", "=" * 60]
    for m in ["BaselineXGB", "BaselineDNN", "TurboGuard(DNN)"]:
        lines.append(f"{m}: FPR: {results['Clean'][m]['FPR']:.2f}% | Clean ADR: {results['Clean'][m]['ADR']:.1f}%")
    lines.append("")
    lines.append(f"{'Attack':<20s} {'Model':<15s} {'EDR':>6s} {'ADR':>6s} {'FPR':>6s} {'F1':>6s}")
    lines.append("-" * 65)
    
    for name in ["Clean"] + sorted([k for k in results.keys() if k.startswith("X_")]):
        for m_name in ["BaselineXGB", "BaselineDNN", "TurboGuard(DNN)"]:
            r = results[name][m_name]
            edr_str = "-" if name == "Clean" else f"{r['EDR']:5.1f}%"
            lines.append(f"{name:<20s} {m_name:<15s} {edr_str:>6s} {r['ADR']:5.1f}% {r['FPR']:5.1f}% {r['F1']:5.1f}%")
        lines.append("-" * 65)
        
    with open(out / "report.txt", "w") as f:
        f.write("\n".join(lines))

    save_config(
        out,
        {
            "command": "eval_benchmark",
            "source_prepare": args.run_dir,
            "source_turboguard": str(tg_dir),
            "source_baseline": str(baseline_dir),
            "source_attacks": args.attack_dir,
            "dataset": "kdd",
        },
    )
    save_run_metadata(out, args)
    done("Report saved", path=out)


if __name__ == "__main__":
    main()
