#!/usr/bin/env python3
"""CIC-IDS2017 — Download and extract raw CSV files.

Downloads the MachineLearningCSV.zip file from the official UNB mirror,
extracts it, and saves the individual day CSVs into a raw directory.

Usage::

    python -m datasets.cic2017.download --output-dir ./data/cic2017 --token ovhtjdc5v7jdk6t6eoea46vqe9
"""

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from zipfile import ZipFile

from tqdm import tqdm

DOWNLOAD_URL = "https://cicresearch.ca/CICDataset/CIC-IDS-2017/download.php?file=CIC-IDS-2017%2FCSVs%2FMachineLearningCSV.zip"
DEFAULT_TOKEN = "ovhtjdc5v7jdk6t6eoea46vqe9"


def download_file(url: str, output_path: Path, token: str):
    """Downloads a file from a URL using custom headers (User-Agent, Cookies) with a progress bar.

    Args:
        url: The direct download URL.
        output_path: Path to write the downloaded file.
        token: Session token cookie value.
    """
    print(f"Downloading {url} to {output_path}...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9,vi-VN;q=0.8,vi;q=0.7",
        "Connection": "keep-alive",
        "Referer": "https://cicresearch.ca/CICDataset/CIC-IDS-2017/browse.php?p=CIC-IDS-2017%2FCSVs",
        "Cookie": f"Token={token}",
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response:
            # Check response size if available
            tsize = response.getheader("Content-Length")
            tsize = int(tsize) if tsize else None

            # Verify that we are not receiving an HTML error/forbidden page
            content_type = response.getheader("Content-Type", "")
            if "text/html" in content_type and (tsize is not None and tsize < 500 * 1024):
                # Small HTML responses are usually 403 / 404 or login pages
                body = response.read(2048).decode("utf-8", errors="ignore")
                if "forbidden" in body.lower() or "denied" in body.lower() or "403" in body:
                    raise ValueError(
                        "Server returned 403 Forbidden. Your session token has likely expired. "
                        "Please copy a fresh Token cookie from your browser and pass it using the --token option."
                    )

            with open(output_path, "wb") as out_file:
                with tqdm(
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1,
                    desc="Download",
                    total=tsize,
                ) as t:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        t.update(len(chunk))
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nERROR during download: {e}")
        print(
            "\n[Tip] If you received a 403 Forbidden, your session Token has expired. "
            "Please log in via your browser, copy your active 'Token' cookie value, "
            "and run this script using:\n"
            "  uv run python datasets/cic2017/main.py download --token <fresh_token_here>"
        )
        sys.exit(1)


def extract_zip(zip_path: Path, extract_dir: Path):
    """Extracts a ZIP archive and flattens/cleans the output structure.

    Args:
        zip_path: Path to the ZIP file.
        extract_dir: Path to extract the files to.
    """
    print(f"Extracting {zip_path} to {extract_dir}...")
    temp_dir = extract_dir / "temp_extract"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with ZipFile(zip_path, "r") as zip_ref:
            # Get list of files to show extraction progress
            members = zip_ref.namelist()
            for member in tqdm(members, desc="Extraction", unit="file"):
                zip_ref.extract(member, temp_dir)
    except Exception as e:
        print(f"ERROR: Extraction failed because the downloaded file is corrupted or incomplete: {e}")
        sys.exit(1)

    print("Cleaning and structuring extracted files...")
    # Find all CSV files recursively and move them directly to extract_dir
    csv_count = 0
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(".csv") and not file.startswith("."):
                src_path = Path(root) / file
                dest_path = extract_dir / file
                shutil.move(src_path, dest_path)
                csv_count += 1

    # Remove temporary extraction directory
    shutil.rmtree(temp_dir)
    print(f"✓ Successfully extracted {csv_count} CSV files directly to {extract_dir}")


def main():
    parser = argparse.ArgumentParser(description="CIC-IDS2017 — Download and Extract")
    parser.add_argument(
        "--output-dir",
        default="./data/cic2017",
        help="Directory to save raw CSVs (default: ./data/cic2017)",
    )
    parser.add_argument(
        "--token",
        default=DEFAULT_TOKEN,
        help=f"Active session Token cookie value (default: {DEFAULT_TOKEN})",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Do not delete the downloaded ZIP archive after extraction",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    csv_dir = output_dir / "raw_csvs"
    zip_path = output_dir / "MachineLearningCSV.zip"

    # Create target directories
    os.makedirs(csv_dir, exist_ok=True)

    # 1. Download
    if not zip_path.exists() and not any(csv_dir.glob("*.csv")):
        download_file(DOWNLOAD_URL, zip_path, args.token)
    elif zip_path.exists():
        print(f"ZIP archive already exists at {zip_path}, skipping download.")
    else:
        print(f"CSV files already exist in {csv_dir}, skipping download.")

    # 2. Extract
    if zip_path.exists():
        try:
            extract_zip(zip_path, csv_dir)
            # Remove ZIP file unless requested to keep
            if not args.keep_zip:
                os.remove(zip_path)
                print(f"Removed ZIP archive {zip_path}")
        except Exception as e:
            print(f"ERROR during extraction: {e}")
            sys.exit(1)

    print(f"\n✓ CIC-IDS2017 dataset is ready at {csv_dir}")


if __name__ == "__main__":
    main()
