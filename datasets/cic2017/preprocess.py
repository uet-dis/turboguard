#!/usr/bin/env python3
"""CIC-IDS2017 — Preprocess raw CSVs into a clean parquet file.

Reads a directory of CSVs or a single CSV, standardises and maps the 2017-style
columns to the standard 62 features, handles NaNs/infs/negatives, and
saves a single clean parquet file ready for the handler.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# The 62 standard features we want to produce.
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

# Normalized mapping dictionary (lowercased & stripped keys)
RAW_COL_MAPPING = {
    "destination port": "Dst Port",
    "dst port": "Dst Port",
    "flow duration": "Flow Duration",
    "total fwd packets": "Tot Fwd Pkts",
    "tot fwd pkts": "Tot Fwd Pkts",
    "total backward packets": "Tot Bwd Pkts",
    "tot bwd pkts": "Tot Bwd Pkts",
    "total length of fwd packets": "TotLen Fwd Pkts",
    "totlen fwd pkts": "TotLen Fwd Pkts",
    "total length of bwd packets": "TotLen Bwd Pkts",
    "totlen bwd pkts": "TotLen Bwd Pkts",
    "fwd packet length max": "Fwd Pkt Len Max",
    "fwd pkt len max": "Fwd Pkt Len Max",
    "fwd packet length min": "Fwd Pkt Len Min",
    "fwd pkt len min": "Fwd Pkt Len Min",
    "fwd packet length mean": "Fwd Pkt Len Mean",
    "fwd pkt len mean": "Fwd Pkt Len Mean",
    "fwd packet length std": "Fwd Pkt Len Std",
    "fwd pkt len std": "Fwd Pkt Len Std",
    "bwd packet length max": "Bwd Pkt Len Max",
    "bwd pkt len max": "Bwd Pkt Len Max",
    "bwd packet length min": "Bwd Pkt Len Min",
    "bwd pkt len min": "Bwd Pkt Len Min",
    "bwd packet length mean": "Bwd Pkt Len Mean",
    "bwd pkt len mean": "Bwd Pkt Len Mean",
    "bwd packet length std": "Bwd Pkt Len Std",
    "bwd pkt len std": "Bwd Pkt Len Std",
    "flow bytes/s": "Flow Byts/s",
    "flow byts/s": "Flow Byts/s",
    "flow packets/s": "Flow Pkts/s",
    "flow pkts/s": "Flow Pkts/s",
    "flow iat mean": "Flow IAT Mean",
    "flow iat std": "Flow IAT Std",
    "flow iat max": "Flow IAT Max",
    "flow iat min": "Flow IAT Min",
    "fwd iat total": "Fwd IAT Tot",
    "fwd iat tot": "Fwd IAT Tot",
    "fwd iat mean": "Fwd IAT Mean",
    "fwd iat std": "Fwd IAT Std",
    "fwd iat max": "Fwd IAT Max",
    "fwd iat min": "Fwd IAT Min",
    "bwd iat total": "Bwd IAT Tot",
    "bwd iat tot": "Bwd IAT Tot",
    "bwd iat mean": "Bwd IAT Mean",
    "bwd iat std": "Bwd IAT Std",
    "bwd iat max": "Bwd IAT Max",
    "bwd iat min": "Bwd IAT Min",
    "fwd header length": "Fwd Header Len",
    "fwd header len": "Fwd Header Len",
    "bwd header length": "Bwd Header Len",
    "bwd header len": "Bwd Header Len",
    "fwd packets/s": "Fwd Pkts/s",
    "fwd pkts/s": "Fwd Pkts/s",
    "bwd packets/s": "Bwd Pkts/s",
    "bwd pkts/s": "Bwd Pkts/s",
    "min packet length": "Pkt Len Min",
    "pkt len min": "Pkt Len Min",
    "max packet length": "Pkt Len Max",
    "pkt len max": "Pkt Len Max",
    "packet length mean": "Pkt Len Mean",
    "pkt len mean": "Pkt Len Mean",
    "packet length std": "Pkt Len Std",
    "pkt len std": "Pkt Len Std",
    "packet length variance": "Pkt Len Var",
    "pkt len var": "Pkt Len Var",
    "rst flag count": "RST Flag Cnt",
    "rst flag cnt": "RST Flag Cnt",
    "psh flag count": "PSH Flag Cnt",
    "psh flag cnt": "PSH Flag Cnt",
    "ack flag count": "ACK Flag Cnt",
    "ack flag cnt": "ACK Flag Cnt",
    "ece flag count": "ECE Flag Cnt",
    "ece flag cnt": "ECE Flag Cnt",
    "down/up ratio": "Down/Up Ratio",
    "average packet size": "Pkt Size Avg",
    "pkt size avg": "Pkt Size Avg",
    "avg fwd segment size": "Fwd Seg Size Avg",
    "fwd seg size avg": "Fwd Seg Size Avg",
    "avg bwd segment size": "Bwd Seg Size Avg",
    "bwd seg size avg": "Bwd Seg Size Avg",
    "subflow fwd packets": "Subflow Fwd Pkts",
    "subflow fwd pkts": "Subflow Fwd Pkts",
    "subflow fwd bytes": "Subflow Fwd Byts",
    "subflow fwd byts": "Subflow Fwd Byts",
    "subflow bwd packets": "Subflow Bwd Pkts",
    "subflow bwd pkts": "Subflow Bwd Pkts",
    "subflow bwd bytes": "Subflow Bwd Byts",
    "subflow bwd byts": "Subflow Bwd Byts",
    "init_win_bytes_forward": "Init Fwd Win Byts",
    "init fwd win byts": "Init Fwd Win Byts",
    "act_data_pkt_fwd": "Fwd Act Data Pkts",
    "fwd act data pkts": "Fwd Act Data Pkts",
    "min_seg_size_forward": "Fwd Seg Size Min",
    "fwd seg size min": "Fwd Seg Size Min",
    "active mean": "Active Mean",
    "active std": "Active Std",
    "active max": "Active Max",
    "active min": "Active Min",
    "idle mean": "Idle Mean",
    "idle std": "Idle Std",
    "idle max": "Idle Max",
    "idle min": "Idle Min",
    "label": "Label",
}


def align_and_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardises the column names of the DataFrame and filters out noise.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned and standardized DataFrame.
    """
    # 1. Clean column names
    col_map = {}
    for exact in df.columns:
        norm = exact.strip().lower()
        if norm in RAW_COL_MAPPING:
            col_map[exact] = RAW_COL_MAPPING[norm]

    # Handle duplicate columns like 'Fwd Header Length.1' in some variants of 2017
    # by matching exact.strip().lower().split('.')[0]
    for exact in df.columns:
        if exact in col_map:
            continue
        norm_split = exact.strip().lower().split(".")[0]
        if norm_split in RAW_COL_MAPPING:
            col_map[exact] = RAW_COL_MAPPING[norm_split]

    df = df.rename(columns=col_map)
    # Deduplicate columns if any mapped to the same target name
    df = df.loc[:, ~df.columns.duplicated()]

    # Verify we have all required columns
    missing = [c for c in FEATURES_62 + ["Label"] if c not in df.columns]
    if missing:
        print(f"WARNING: The following expected columns were missing: {missing}")
        # Add missing columns as 0 to avoid crashes
        for c in missing:
            if c == "Label":
                df[c] = "Benign"
            else:
                df[c] = 0.0

    df = df[FEATURES_62 + ["Label"]].copy()

    # 2. Clean Labels
    df["Label"] = df["Label"].astype(str).str.strip()
    # Normalize Benign label
    df.loc[df["Label"].str.lower() == "benign", "Label"] = "Benign"

    # Drop header rows accidentally read as data
    header_mask = df["Label"].str.lower() == "label"
    if header_mask.any():
        df = df[~header_mask]

    # 3. Numeric coercion
    for col in FEATURES_62:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace infs with NaNs and drop
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES_62)

    # 4. Drop rows with negative values (noise/artifacts)
    neg_mask = (df[FEATURES_62] < 0).any(axis=1)
    if neg_mask.any():
        df = df[~neg_mask]

    return df


def process_from_dir(raw_dir: str, output_path: str) -> None:
    """Processes all raw CSV files in a directory.

    Args:
        raw_dir: Path to directory containing raw CSVs.
        output_path: Path to write the parquet file.
    """
    csv_files = sorted(
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith(".csv") and not f.startswith(".")
    )
    if not csv_files:
        print(f"ERROR: No CSV files found in {raw_dir}")
        sys.exit(1)

    print(f"[1/5] Found {len(csv_files)} CSV files in {raw_dir}")
    t0 = time.time()
    dfs = []
    for i, fp in enumerate(csv_files):
        fname = os.path.basename(fp)
        print(f"  [{i + 1}/{len(csv_files)}] Reading {fname}...", end=" ", flush=True)
        try:
            # Read first row to inspect columns case-insensitively
            head = pd.read_csv(fp, nrows=0, skipinitialspace=True)
            cols_to_use = []
            for exact in head.columns:
                norm = exact.strip().lower().split(".")[0]
                if norm in RAW_COL_MAPPING:
                    cols_to_use.append(exact)

            df_chunk = pd.read_csv(
                fp,
                usecols=cols_to_use,
                low_memory=False,
                skipinitialspace=True,
            )
            print(f"{len(df_chunk):,} rows")
            dfs.append(df_chunk)
        except Exception as e:
            print(f"ERROR: Failed to read {fp}: {e}")

    if not dfs:
        print("ERROR: No data loaded.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    print(f"  Total read: {len(df):,} rows ({time.time() - t0:.1f}s)")

    _clean_and_save(df, output_path)


def process_from_single_csv(csv_path: str, output_path: str) -> None:
    """Processes a single raw CSV file.

    Args:
        csv_path: Path to the single CSV file.
        output_path: Path to write the parquet file.
    """
    print(f"[1/5] Reading single CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False, skipinitialspace=True)
    print(f"  Read {len(df):,} rows")
    _clean_and_save(df, output_path)


def _clean_and_save(df: pd.DataFrame, output_path: str) -> None:
    """Applies column mapping, filters out noise, deduplicates, and saves."""
    print(f"[2/5] Cleaning and standardising columns → {len(FEATURES_62)} features")
    before_len = len(df)
    df = align_and_clean_df(df)
    print(f"  Kept {len(df):,} / {before_len:,} rows after schema alignment + cleaning")

    print("[3/5] Deduplication")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=FEATURES_62)
    print(f"  Dropped {before_dedup - len(df):,} duplicate rows → {len(df):,} rows")

    print("[4/5] Label distribution:")
    counts = df["Label"].value_counts()
    for lbl, cnt in counts.items():
        print(f"  {lbl:35s} {cnt:>10,}")

    print("[5/5] Saving to parquet")
    df[FEATURES_62] = df[FEATURES_62].astype(np.float32)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ Successfully saved clean parquet to {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="CIC-IDS2017 — Preprocess raw CSVs to parquet")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--raw-dir", help="Directory containing raw CIC-IDS2017 CSV files")
    group.add_argument("--raw-csv", help="Single raw/merged CIC-IDS2017 CSV file")
    parser.add_argument(
        "--output",
        default="./data/cic2017/cic2017_clean.parquet",
        help="Output parquet path (default: ./data/cic2017/cic2017_clean.parquet)",
    )
    args = parser.parse_args()

    if args.raw_dir:
        process_from_dir(args.raw_dir, args.output)
    else:
        process_from_single_csv(args.raw_csv, args.output)


if __name__ == "__main__":
    main()
