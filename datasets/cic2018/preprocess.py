#!/usr/bin/env python3
"""CSE-CIC-IDS2018 — Preprocess raw CSVs into a clean parquet file.

Reads the merged CSV (from ``download.py``), applies cleaning (feature
selection, dtype coercion, NaN/inf/negative removal, deduplication), and
saves a single clean parquet file ready for the handler.

Usage::

    python -m datasets.cic2018.preprocess --raw-csv ./data/CICIDS2018_full_raw.csv
    python -m datasets.cic2018.preprocess --raw-dir ./data/raw_csvs
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# 80-column majority schema (9/10 files).
MAJORITY_SCHEMA_COLS = [
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",
    "Fwd Pkt Len Std",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean",
    "Bwd Pkt Len Std",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Tot",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Len",
    "Bwd Header Len",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Min",
    "Pkt Len Max",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "CWE Flag Count",
    "ECE Flag Cnt",
    "Down/Up Ratio",
    "Pkt Size Avg",
    "Fwd Seg Size Avg",
    "Bwd Seg Size Avg",
    "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Pkts",
    "Subflow Bwd Byts",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Label",
]

# 62 features kept after dropping Protocol, Timestamp, constant columns.
FEATURES_62 = [
    "Dst Port",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",
    "Fwd Pkt Len Std",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean",
    "Bwd Pkt Len Std",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Tot",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Header Len",
    "Bwd Header Len",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Min",
    "Pkt Len Max",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "ECE Flag Cnt",
    "Down/Up Ratio",
    "Pkt Size Avg",
    "Fwd Seg Size Avg",
    "Bwd Seg Size Avg",
    "Subflow Fwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Pkts",
    "Subflow Bwd Byts",
    "Init Fwd Win Byts",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

KEEP_COLS = FEATURES_62 + ["Label"]


def process_from_csvs(raw_dir: str, output_path: str) -> None:
    """Reads multiple raw CSVs and processes into a single parquet.

    Args:
        raw_dir: Directory containing the 10 raw CICFlowMeter CSVs.
        output_path: Output path for the clean parquet file.
    """
    csv_files = sorted(
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.lower().endswith(".csv")
    )
    if not csv_files:
        print(f"ERROR: No CSV files found in {raw_dir}")
        sys.exit(1)

    print(f"[1/5] Found {len(csv_files)} CSV files")
    t0 = time.time()
    dfs = []
    for i, fp in enumerate(csv_files):
        fname = os.path.basename(fp)
        print(f"  [{i + 1}/{len(csv_files)}] {fname}...", end=" ", flush=True)
        df = pd.read_csv(
            fp,
            usecols=MAJORITY_SCHEMA_COLS,
            low_memory=False,
            skipinitialspace=True,
        )
        print(f"{len(df):,} rows")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    print(f"  Total: {len(df):,} rows ({time.time() - t0:.1f}s)")

    _clean_and_save(df, output_path)


def process_from_merged_csv(csv_path: str, output_path: str) -> None:
    """Reads a single merged CSV and processes into parquet.

    Args:
        csv_path: Path to the merged CSV from ``download.py``.
        output_path: Output path for the clean parquet file.
    """
    print(f"[1/5] Reading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False, skipinitialspace=True)
    print(f"  {len(df):,} rows")

    _clean_and_save(df, output_path)


def _clean_and_save(df: pd.DataFrame, output_path: str) -> None:
    """Applies cleaning pipeline and saves to parquet.

    Steps:
        1. Strip labels, drop header-as-data rows.
        2. Select 62 features + Label.
        3. Coerce to numeric, drop NaN/inf/negative rows.
        4. Deduplicate.
        5. Save as parquet.

    Args:
        df: Raw DataFrame with at least the 80 standard columns.
        output_path: Output parquet path.
    """
    print(f"[2/5] Feature selection → {len(FEATURES_62)} features")
    df["Label"] = df["Label"].astype(str).str.strip()
    # Drop rows where the header was accidentally read as data.
    header_mask = df["Label"] == "Label"
    if header_mask.any():
        print(f"  Dropped {header_mask.sum():,} header-as-data rows")
        df = df[~header_mask]
    df = df[KEEP_COLS]

    print("[3/5] Dtype coercion + cleaning")
    before = len(df)
    for col in FEATURES_62:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES_62)
    print(f"  Dropped {before - len(df):,} rows with NaN/inf")

    # Drop rows with negative values (matches original pipeline).
    neg_mask = (df[FEATURES_62] < 0).any(axis=1)
    if neg_mask.any():
        df = df[~neg_mask]
        print(f"  Dropped {neg_mask.sum():,} rows with negative values")

    print("[4/5] Deduplication")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=FEATURES_62)
    print(f"  Dropped {before_dedup - len(df):,} duplicates → {len(df):,} rows")

    print("[5/5] Label distribution:")
    counts = df["Label"].value_counts()
    for lbl, cnt in counts.items():
        print(f"  {lbl:35s} {cnt:>10,}")

    df[FEATURES_62] = df[FEATURES_62].astype(np.float32)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ Saved to {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="CIC-IDS2018 — Preprocess raw CSVs to parquet")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--raw-dir", help="Directory containing 10 raw CICFlowMeter CSVs")
    group.add_argument("--raw-csv", help="Single merged CSV from download.py")
    parser.add_argument(
        "--output",
        default="cic2018_clean.parquet",
        help="Output parquet path (default: cic2018_clean.parquet)",
    )
    args = parser.parse_args()

    if args.raw_dir:
        process_from_csvs(args.raw_dir, args.output)
    else:
        process_from_merged_csv(args.raw_csv, args.output)


if __name__ == "__main__":
    main()
