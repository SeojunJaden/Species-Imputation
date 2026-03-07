#!/usr/bin/env python3
"""
Keep only rows with valid latitude/longitude so the combine script can extract env for all of them.

Usage (from project root):
  python cleaning-pipeline/filter-valid-coords.py [input.csv] [output.csv]

Defaults: data/400k-obs-clean.csv -> data/400k-obs-clean.csv (overwrites with filtered version)
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "400k-obs-clean.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "400k-obs-clean.csv"


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if not input_path.exists():
        print(f"ERROR: Not found: {input_path}")
        return 1

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    n_before = len(df)

    df = df.dropna(subset=["latitude", "longitude"])
    n_after = len(df)
    n_dropped = n_before - n_after

    print(f"  Rows before:     {n_before:,}")
    print(f"  Rows dropped:   {n_dropped:,} (missing lat/lon)")
    print(f"  Rows kept:      {n_after:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
