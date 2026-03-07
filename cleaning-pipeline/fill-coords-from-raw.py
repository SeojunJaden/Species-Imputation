#!/usr/bin/env python3
"""
Fill missing latitude/longitude in the clean CSV from the raw observation files
(second-200k-obs.csv, and optionally first-200k-obs.csv) by matching on observation id.

Usage (from project root):
  python cleaning-pipeline/fill-coords-from-raw.py

Reads: data/400k-obs-clean.csv, data/second-200k-obs.csv, data/first-200k-obs.csv (if present)
Writes: data/400k-obs-clean.csv (overwrites with filled version)
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

CLEAN_PATH = DATA_DIR / "400k-obs-clean.csv"
SECOND_RAW_PATH = DATA_DIR / "second-200k-obs.csv"
FIRST_RAW_PATH = DATA_DIR / "first-200k-obs.csv"


def main():
    if not CLEAN_PATH.exists():
        print(f"ERROR: Not found: {CLEAN_PATH}")
        return 1
    if not SECOND_RAW_PATH.exists():
        print(f"ERROR: Not found: {SECOND_RAW_PATH}")
        return 1

    print("Loading clean CSV...")
    clean = pd.read_csv(CLEAN_PATH, low_memory=False)
    n_nan_before = (clean["latitude"].isna() | clean["longitude"].isna()).sum()
    print(f"  Rows with missing lat/lon: {n_nan_before:,}")

    # Build id -> lat, lon from raw files (second first, then first)
    raw_dfs = []
    for path in [SECOND_RAW_PATH, FIRST_RAW_PATH]:
        if not path.exists():
            continue
        print(f"Loading {path.name}...")
        raw = pd.read_csv(path, usecols=["id", "latitude", "longitude"], low_memory=False)
        raw = raw.dropna(subset=["latitude", "longitude"]).drop_duplicates(subset=["id"], keep="first")
        raw_dfs.append(raw)
    coords = pd.concat(raw_dfs, ignore_index=True).drop_duplicates(subset=["id"], keep="first")
    coords = coords.rename(columns={"latitude": "lat_fill", "longitude": "lon_fill"})
    print(f"  Unique ids with coords: {len(coords):,}")

    # Left join and fill where clean has NaN
    clean = clean.merge(coords, on="id", how="left", suffixes=("", "_from_raw"))
    mask = clean["latitude"].isna() | clean["longitude"].isna()
    filled = mask & clean["lat_fill"].notna()
    clean.loc[filled, "latitude"] = clean.loc[filled, "lat_fill"]
    clean.loc[filled, "longitude"] = clean.loc[filled, "lon_fill"]
    clean = clean.drop(columns=["lat_fill", "lon_fill"])

    n_nan_after = (clean["latitude"].isna() | clean["longitude"].isna()).sum()
    print(f"\nFilled {filled.sum():,} rows from raw files")
    print(f"  Rows still missing lat/lon: {n_nan_after:,}")

    clean.to_csv(CLEAN_PATH, index=False)
    print(f"Wrote {CLEAN_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
