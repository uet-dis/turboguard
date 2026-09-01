#!/usr/bin/env python3
"""CIC2018 — Generate adversarial attacks.

Usage::

    python -m datasets.cic2018.generate_attacks --run-dir results/unsw/<ts>_prepare
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from turboguard.attacks.fgsm import fgsm_linf
from turboguard.attacks.pgd import pgd_linf, train_surrogate
from turboguard.attacks.cw import cw_l2
from turboguard.attacks.deepfool import deepfool
from turboguard.config import SEED
from turboguard.console import console, header, step, substep, done
from turboguard.device import get_device
from turboguard.models.dnn import DNNClassifier
from turboguard.persistence import (
    create_run_dir,
    load_config,
    load_sectors,
    save_config,
    save_eval_pool,
    save_run_metadata,
)

RESULTS_BASE = Path("results")


def _log(name, X_clean, X_adv):
    l2 = torch.norm(X_adv - X_clean, p=2, dim=1)
    linf = (X_adv - X_clean).abs().max(dim=1).values
    substep(f"{name}: L2_mean={l2.mean():.4f}, Linf_max={linf.max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="CIC2018 — Generate attacks")
    parser.add_argument("--run-dir", required=True, help="Path to prepare run")
    parser.add_argument(
        "--baseline-dir", default=None, help="Path to baseline run (for WB attacks)"
    )
    parser.add_argument("--attacks", default="fgsm,pgd,cw,deepfool")
    parser.add_argument("--epsilons", default="0.003,0.05,0.1,0.5")
    parser.add_argument("--n-attack", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir_src = Path(args.run_dir)
    cfg = load_config(run_dir_src)
    sectors = load_sectors(run_dir_src)
    device = get_device()
    seed = args.seed

    header("Generate Attacks", dataset="cic2018")

    X_C, y_C = sectors["X_C"], sectors["y_C"]
    atk_idx = np.where(y_C > 0)[0]
    ben_idx = np.where(y_C == 0)[0]
    rng = np.random.RandomState(seed + 1)
    rng.shuffle(atk_idx)
    rng.shuffle(ben_idx)

    N_ATK = (min(args.n_attack, len(atk_idx)) // 2) * 2
    N_BEN = min(N_ATK * 2, len(ben_idx))
    N_HALF = N_ATK // 2

    console.print(
        f"  Pool: [bold]{N_ATK}[/bold] attacks "
        f"({N_HALF} clean + {N_HALF} evasion), "
        f"[bold]{N_BEN}[/bold] benign"
    )

    X_A_t = torch.tensor(sectors["X_A"], dtype=torch.float32, device=device)
    X_C_t = torch.tensor(X_C, dtype=torch.float32, device=device)
    X_atk = X_C_t[atk_idx[N_HALF:N_ATK]]
    y_mod = torch.ones(X_atk.shape[0], device=device, dtype=torch.long)

    step("Training surrogate MLP")
    y_A_bin = torch.tensor((sectors["y_A"] > 0).astype(int), dtype=torch.long, device=device)
    surrogate = train_surrogate(X_A_t, y_A_bin, device, seed=seed)

    base = Path(args.output_dir) if args.output_dir else RESULTS_BASE
    out = create_run_dir(base, "cic2018", "attacks")
    (out / "attacks").mkdir(exist_ok=True)

    epsilons = [float(e) for e in args.epsilons.split(",")]
    attack_types = [a.strip().lower() for a in args.attacks.split(",")]

    saved = []

    def _gen(model, prefix, atk_type):
        for eps in epsilons:
            tag = f"{eps:.3f}".replace(".", "")
            name = f"X_{prefix}{atk_type}_{tag}"
            label = f"{'WB' if prefix == 'wb_' else 'Transfer'} {atk_type.upper()} (eps={eps})"
            step(label)
            if atk_type == "fgsm":
                X_adv = fgsm_linf(model, X_atk, y_mod, eps=eps)
            elif atk_type == "pgd":
                X_adv = pgd_linf(model, X_atk, y_mod, eps=eps, eps_step=max(eps / 10, 0.001))
            elif atk_type == "cw":
                X_adv = cw_l2(model, X_atk, y_mod, eps=eps, c=10.0, max_iter=100)
            _log(name, X_atk, X_adv)
            torch.save(X_adv.cpu(), out / "attacks" / f"{name}.pt")
            saved.append(name)

    def _gen_deepfool(model, prefix):
        name = f"X_{prefix}deepfool"
        label = f"{'WB' if prefix == 'wb_' else 'Transfer'} DeepFool"
        step(label)
        X_adv = deepfool(model, X_atk, y_mod, max_iter=50)
        _log(name, X_atk, X_adv)
        torch.save(X_adv.cpu(), out / "attacks" / f"{name}.pt")
        saved.append(name)

    for atk in attack_types:
        if atk == "deepfool":
            _gen_deepfool(surrogate, "")
        else:
            _gen(surrogate, "", atk)

    if args.baseline_dir:
        dnn_path = Path(args.baseline_dir) / "models" / "baseline_dnn.pth"
        if dnn_path.exists():
            wb_dnn = DNNClassifier(input_dim=cfg["input_dim"]).to(device)
            wb_dnn.load_state_dict(torch.load(dnn_path, map_location=device, weights_only=True))
            wb_dnn.eval()
            for atk in attack_types:
                if atk == "deepfool":
                    _gen_deepfool(wb_dnn, "wb_")
                else:
                    _gen(wb_dnn, "wb_", atk)

    torch.save(surrogate.state_dict(), out / "models" / "surrogate.pth")
    save_eval_pool(out, atk_idx=atk_idx[:N_ATK], ben_idx=ben_idx[:N_BEN])
    save_config(
        out,
        {
            "command": "generate-attacks",
            "source_prepare": str(run_dir_src),
            "source_baseline": args.baseline_dir or "",
            "dataset": "cic2018",
            "input_dim": cfg["input_dim"],
            "seed": seed,
            "n_atk": N_ATK,
            "n_ben": N_BEN,
            "n_half": N_HALF,
            "epsilons": epsilons,
            "attack_types": attack_types,
            "attack_files": saved,
        },
    )
    save_run_metadata(out, args)
    done(f"Saved {len(saved)} attack files", path=out)


if __name__ == "__main__":
    main()
