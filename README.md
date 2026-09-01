# TurboGuard

VQ-VAE-based adversarial defense filter for Network Intrusion Detection Systems (NIDS).

TurboGuard is a **model-agnostic, attack-agnostic** defense that detects adversarial perturbations applied to network traffic samples. It uses a Vector Quantized Variational Autoencoder (VQ-VAE) trained exclusively on benign traffic to extract six complementary anomaly signals from the codebook topology, then applies a two-stage detection pipeline.

## Architecture

```
Input x
  → VQ-VAE (with dead code replacement) → Reconstruct x̂
  → Extract 6 signals: [CTF, VMR, ENT, RE, GEO, CC]
  → IsolationForest: hard-drop obvious outliers
  → DNN on [x, x̂, |x−x̂|, signals]: classify grey-zone
  → Binary output: benign (0) / adversarial (1)
```

### Anomaly Signals

| Signal | Name | Description |
|--------|------|-------------|
| **CTF** | Codebook Topology Fingerprint | Mahalanobis distance in the PCA subspace of codebook distance profiles |
| **VMR** | Voronoi Margin Ratio | Ratio of nearest-code to second-nearest-code distance |
| **ENT** | Codebook Distance Entropy | Softmax entropy over all code distances |
| **RE** | Reconstruction Error | Per-sample MSE between input and VQ-VAE reconstruction |
| **GEO** | Geometric Score | Density-weighted distance to semantic map centroids (~65% feature importance) |
| **CC** | Code-Conditional z-score | Reconstruction error normalised by per-code statistics |

## Project Structure

```
FRAMEWORK_SOURCE_CODE/
├── pyproject.toml               # Package definition (uv/pip installable)
├── README.md
│
├── src/turboguard/              # Pure library — importable, no CLI
│   ├── config.py                # Default hyperparameters
│   ├── device.py                # CUDA/CPU auto-detection
│   ├── metrics.py               # ADR, EDR, FPR, F1
│   ├── persistence.py           # Timestamped run directories
│   ├── models/
│   │   ├── vqvae.py             # VQ-VAE with EMA codebook + dead code replacement
│   │   └── dnn.py               # Binary DNN classifier
│   ├── core/
│   │   ├── turboguard.py        # Main TurboGuard filter (fit/predict/calibrate)
│   │   └── geometry.py          # Semantic map + GEO signal
│   ├── attacks/
│   │   ├── fgsm.py              # FGSM L-inf
│   │   ├── pgd.py               # PGD L-inf + Surrogate MLP
│   │   ├── cw.py                # C&W L2 (batched)
│   │   └── deepfool.py          # DeepFool L2 (batched)
│   └── classifiers/
│       └── baselines.py         # Baseline XGBoost + DNN
│
└── datasets/                    # Per-dataset implementations
    ├── registry.py              # DatasetHandler ABC + @register_dataset
    ├── unsw/                    # UNSW-NB15
    │   ├── handler.py           # Data loading (76 features)
    │   ├── main.py              # Entry point: prepare|train|generate-attacks|eval
    │   ├── prepare.py
    │   ├── train.py
    │   ├── generate_attacks.py
    │   └── eval.py
    ├── kdd/                     # NSL-KDD (122 features)
    ├── iec104/                  # IEC-104 SCADA (76 features)
    └── cic2018/                 # CSE-CIC-IDS2018 (62 features)
        ├── download.py          # Download raw CSVs from AWS S3
        ├── preprocess.py        # Clean + merge → parquet
        ├── handler.py
        ├── main.py
        └── ...
```

## Installation

### With uv (recommended)

```bash
cd FRAMEWORK_SOURCE_CODE
uv sync
```

### With pip

```bash
cd FRAMEWORK_SOURCE_CODE
pip install -e .
```

### CUDA PyTorch

The `pyproject.toml` is pre-configured for CUDA 12.4. To change CUDA version, edit the index URL:

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"   # ← change to cu118, cu121, etc.
```

## Usage

Each dataset has its own entry point. TurboGuard is a library — datasets implement their own pipeline scripts.

### UNSW-NB15

```bash
# 1. Prepare: load data, split into Sectors A/B/C
python datasets/unsw/main.py prepare --data-dir ./data/CIC-UNSW

# 2. Train: baselines + TurboGuard + calibrate threshold
python datasets/unsw/main.py train --run-dir results/unsw/<timestamp>_prepare

# 3. Generate adversarial attacks from Sector C
python datasets/unsw/main.py generate-attacks \
    --run-dir results/unsw/<timestamp>_prepare \
    --baseline-dir results/unsw/<timestamp>_baseline

# 4. Evaluate TurboGuard against attacks
python datasets/unsw/main.py eval \
    --run-dir results/unsw/<timestamp>_prepare \
    --tg-dir results/unsw/<timestamp>_turboguard \
    --attack-dir results/unsw/<timestamp>_attacks
```

### CIC-IDS2018

CIC-IDS2018 requires downloading and preprocessing the raw dataset first:

```bash
# Download 10 CSV files from AWS S3 (~6.4 GB)
python -m datasets.cic2018.download --output-dir ./data/cic2018

# Preprocess into clean parquet
python -m datasets.cic2018.preprocess \
    --raw-csv ./data/cic2018/CICIDS2018_full_raw.csv \
    --output ./data/cic2018/cic2018_clean.parquet

# Then run the standard pipeline
python datasets/cic2018/main.py prepare --data-dir ./data/cic2018
python datasets/cic2018/main.py train --run-dir results/cic2018/<timestamp>_prepare

# 3. Generate adversarial attacks
python datasets/cic2018/main.py generate-attacks \
    --run-dir results/cic2018/<timestamp>_prepare \
    --baseline-dir results/cic2018/<timestamp>_baseline

# 4. Evaluate TurboGuard against attacks
python datasets/cic2018/main.py eval \
    --run-dir results/cic2018/<timestamp>_prepare \
    --tg-dir results/cic2018/<timestamp>_turboguard \
    --attack-dir results/cic2018/<timestamp>_attacks

# 5. Signal AUC Analysis (ROC-AUC & σ-separation table for the thesis)
python -m datasets.cic2018.signal_auc \
    --run-dir results/cic2018/<timestamp>_prepare \
    --tg-dir results/cic2018/<timestamp>_turboguard \
    --atk-dir results/cic2018/<timestamp>_attacks

# 6. Explainability (SHAP Group & Individual Signal Rankings)
python -m datasets.cic2018.explain \
    --run-dir results/cic2018/<timestamp>_prepare \
    --tg-dir results/cic2018/<timestamp>_turboguard
```

### CIC-IDS2017

CIC-IDS2017 requires downloading and preprocessing the raw dataset first:

```bash
# 1. Download and extract raw daily CSVs from official UNB mirror (using active Token cookie from your browser)
uv run python datasets/cic2017/main.py download --output-dir ./data/cic2017 --token <active_token_cookie>

# 2. Preprocess raw CSVs into clean parquet
uv run python datasets/cic2017/preprocess.py \
    --raw-dir ./data/cic2017/raw_csvs \
    --output ./data/cic2017/cic2017_clean.parquet

# 3. Prepare: Split into Sector A (Train), B (Calibrate), and C (Eval)
uv run python datasets/cic2017/main.py prepare --data-dir ./data/cic2017

# 4. Train: Baselines + TurboGuard + calibrate threshold
uv run python datasets/cic2017/main.py train --run-dir results/cic2017/<timestamp>_prepare

# 5. Generate adversarial attacks from Sector C
uv run python datasets/cic2017/main.py generate-attacks \
    --run-dir results/cic2017/<timestamp>_prepare \
    --baseline-dir results/cic2017/<timestamp>_baseline

# 6. Evaluate TurboGuard against attacks
uv run python datasets/cic2017/main.py eval \
    --run-dir results/cic2017/<timestamp>_prepare \
    --tg-dir results/cic2017/<timestamp>_turboguard \
    --attack-dir results/cic2017/<timestamp>_attacks

# 7. Signal AUC Analysis (ROC-AUC & σ-separation table)
uv run python datasets/cic2017/main.py signal-auc \
    --run-dir results/cic2017/<timestamp>_prepare \
    --tg-dir results/cic2017/<timestamp>_turboguard \
    --atk-dir results/cic2017/<timestamp>_attacks

# 8. Explainability (SHAP Importance Rankings)
uv run python datasets/cic2017/main.py explain \
    --run-dir results/cic2017/<timestamp>_prepare \
    --tg-dir results/cic2017/<timestamp>_turboguard
```

## Pipeline Design

### Data Splitting (No-Cheat Protocol)

Training data is split into three sectors to prevent information leakage:

| Sector | Purpose | Size |
|--------|---------|------|
| **A** | Train VQ-VAE, baselines, TurboGuard | 60% |
| **B** | Calibrate IF threshold (FPR budget) | 20% |
| **C** | Final evaluation + attack generation | 20% |

### Timestamped Run Directories

Every pipeline step saves its output to a timestamped directory:

```
results/unsw/
├── 1743879600_prepare/
│   ├── config.json          # Hyperparameters + provenance links
│   ├── sectors.npz          # Sector A/B/C data splits
│   └── models/scaler.joblib
├── 1743879660_baseline/
│   ├── config.json
│   └── models/
│       ├── baseline_xgb.joblib
│       └── baseline_dnn.pth
├── 1743879720_turboguard/
│   ├── config.json
│   └── models/
│       ├── vqvae.pth
│       ├── semantic_map.pkl
│       ├── iso_forest.joblib
│       ├── iso_threshold.json
│       └── ...
└── 1743879900_eval/
    ├── config.json
    ├── report.json
    └── report.txt
```

Each `config.json` links back to its source run for full provenance.

## Adding a New Dataset

1. Create `datasets/<name>/handler.py`:

```python
from datasets.registry import DatasetHandler, register_dataset

@register_dataset
class MyHandler(DatasetHandler):
    def name(self) -> str:
        return "my_dataset"

    def input_dim(self) -> int:
        return 42

    def load(self, data_dir, scaler_type="minmax"):
        # Load data, fit scaler on benign ONLY, return (X, y, scaler)
        ...
```

2. Create `datasets/<name>/__init__.py`:
```python
from datasets.my_dataset.handler import MyHandler
```

3. Copy and adapt any of the existing `prepare.py`, `train.py`, `generate_attacks.py`, `eval.py` from UNSW.

4. Create `datasets/<name>/main.py` as the entry point.

## References

- van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
- Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec", IEEE/ACM TASLP, 2021
- Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018
- Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks", IEEE S&P 2017
- Moosavi-Dezfooli et al., "DeepFool", CVPR 2016
