#!/usr/bin/env python3
"""
Fill missing taxon_id in 400k-obs-with-env.csv from raw observation files by matching
on (latitude, longitude). Use when with-env has rows with valid coords but NaN taxon_id.

Usage (from project root):
  python cleaning-pipeline/fill-taxon-id-from-raw.py

Reads: data/400k-obs-with-env.csv, data/second-200k-obs.csv, data/first-200k-obs.csv (if present)
Writes: data/400k-obs-with-env.csv (overwrites with filled taxon_id)
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

WITH_ENV_PATH = DATA_DIR / "400k-obs-with-env.csv"
SECOND_RAW_PATH = DATA_DIR / "second-200k-obs.csv"
FIRST_RAW_PATH = DATA_DIR / "first-200k-obs.csv"


def main():
    if not WITH_ENV_PATH.exists():
        print(f"ERROR: Not found: {WITH_ENV_PATH}")
        return 1
    if not SECOND_RAW_PATH.exists():
        print(f"ERROR: Not found: {SECOND_RAW_PATH}")
        return 1

    print("Loading with-env CSV...")
    env = pd.read_csv(WITH_ENV_PATH, low_memory=False)
    n_nan_before = env["taxon_id"].isna().sum()
    print(f"  Rows with missing taxon_id: {n_nan_before:,}")

    # Build (lat, lon) -> taxon_id from raw (first occurrence per point)
    raw_dfs = []
    for path in [SECOND_RAW_PATH, FIRST_RAW_PATH]:
        if not path.exists():
            continue
        print(f"Loading {path.name}...")
        raw = pd.read_csv(
            path, usecols=["latitude", "longitude", "taxon_id"], low_memory=False
        )
        raw = raw.dropna(subset=["latitude", "longitude", "taxon_id"])
        raw_dfs.append(raw)
    raw_combined = pd.concat(raw_dfs, ignore_index=True)
    # One taxon_id per (lat, lon) - keep first
    latlon_to_taxon = raw_combined.drop_duplicates(
        subset=["latitude", "longitude"], keep="first"
    )[["latitude", "longitude", "taxon_id"]]
    latlon_to_taxon = latlon_to_taxon.rename(columns={"taxon_id": "taxon_id_fill"})
    print(f"  Unique (lat, lon) with taxon_id: {len(latlon_to_taxon):,}")

    # Merge and fill
    env = env.merge(
        latlon_to_taxon, on=["latitude", "longitude"], how="left", suffixes=("", "_y")
    )
    mask = env["taxon_id"].isna() & env["taxon_id_fill"].notna()
    env.loc[mask, "taxon_id"] = env.loc[mask, "taxon_id_fill"]
    env = env.drop(columns=["taxon_id_fill"], errors="ignore")

    n_nan_after = env["taxon_id"].isna().sum()
    filled = n_nan_before - n_nan_after
    print(f"\nFilled {filled:,} taxon_id from raw (by lat/lon)")
    print(f"  Rows still missing taxon_id: {n_nan_after:,}")

    env.to_csv(WITH_ENV_PATH, index=False)
    print(f"Wrote {WITH_ENV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
