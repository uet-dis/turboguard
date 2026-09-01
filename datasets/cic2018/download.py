#!/usr/bin/env python3
"""CSE-CIC-IDS2018 — Download raw CSVs from AWS S3 and merge.

Downloads the 10 CICFlowMeter CSV files from the public S3 bucket
``cse-cic-ids2018``, aligns schemas (Tuesday has 4 extra columns),
and merges them into a single CSV file.

Prerequisites:
    pip install boto3 tqdm

Usage::

    python -m datasets.cic2018.download --output-dir ./data/cic2018_raw
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Standard 80 columns shared by 9/10 files (Tuesday has 4 extras we drop).
STANDARDIZED_COLUMNS = [
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

BUCKET = "cse-cic-ids2018"
PREFIX = "Processed Traffic Data for ML Algorithms/"


def download_from_s3(output_dir: str) -> list[str]:
    """Downloads all CSV files from the CIC-IDS2018 S3 bucket.

    Skips files that already exist locally.

    Args:
        output_dir: Directory to save downloaded CSVs.

    Returns:
        List of local file paths for the downloaded CSVs.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    os.makedirs(output_dir, exist_ok=True)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)

    csv_files = []
    total_bytes = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".csv"):
                csv_files.append((key, obj["Size"]))
                total_bytes += obj["Size"]

    print(f"Found {len(csv_files)} CSV files ({total_bytes / 1e9:.2f} GB)")

    downloaded = []
    with tqdm(total=total_bytes, unit="B", unit_scale=True) as pbar:
        for key, size in csv_files:
            fname = key.split("/")[-1]
            local_path = os.path.join(output_dir, fname)
            downloaded.append(local_path)

            if os.path.exists(local_path):
                pbar.update(size)
                continue

            pbar.set_description(fname)
            tmp = local_path + ".part"
            s3_dl = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            with open(tmp, "wb") as f:
                s3_dl.download_fileobj(
                    BUCKET,
                    key,
                    f,
                    Callback=lambda n: pbar.update(n),
                )
            os.rename(tmp, local_path)

    return downloaded


def merge_csvs(csv_paths: list[str], output_csv: str) -> None:
    """Merges multiple CIC CSVs into a single file with aligned schema.

    Tuesday's CSV has 4 extra columns (Flow ID, Src IP, etc.) that are
    dropped to align with the 80-column majority schema.

    Args:
        csv_paths: List of downloaded CSV file paths.
        output_csv: Output path for the merged CSV.
    """
    if os.path.exists(output_csv):
        print(f"Merged file already exists: {output_csv}")
        return

    print(f"Merging {len(csv_paths)} files → {output_csv}")
    write_header = True

    for i, fp in enumerate(csv_paths):
        fname = os.path.basename(fp)
        print(f"  [{i + 1}/{len(csv_paths)}] {fname}")

        try:
            head = pd.read_csv(fp, nrows=0)
        except Exception as e:
            print(f"    Skipped ({e})")
            continue

        # Match columns case-insensitively to handle whitespace variations.
        col_map = {}
        for exact in head.columns:
            for std in STANDARDIZED_COLUMNS:
                if std.lower() == exact.strip().lower():
                    col_map[exact] = std
        cols_to_read = list(col_map.keys())

        for chunk in pd.read_csv(fp, usecols=cols_to_read, chunksize=1_000_000, low_memory=False):
            chunk.rename(columns=col_map, inplace=True)
            chunk.to_csv(output_csv, mode="a", header=write_header, index=False)
            write_header = False

    print(f"✓ Merged to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="CIC-IDS2018 — Download from S3 and merge")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save raw CSVs and merged output",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip S3 download (use existing CSVs in output-dir)",
    )
    args = parser.parse_args()

    csv_dir = os.path.join(args.output_dir, "raw_csvs")
    merged_csv = os.path.join(args.output_dir, "CICIDS2018_full_raw.csv")

    if args.skip_download:
        paths = sorted(str(p) for p in Path(csv_dir).glob("*.csv"))
    else:
        paths = download_from_s3(csv_dir)

    merge_csvs(paths, merged_csv)


if __name__ == "__main__":
    main()
