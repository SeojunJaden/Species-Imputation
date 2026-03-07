#!/usr/bin/env python3
"""
Fill missing taxon_name in 400k-obs-with-env.csv from raw observation files.
Uses taxon_id -> name from raw (scientific_name, else common_name).

Usage (from project root):
  python cleaning-pipeline/fill-taxon-name-from-raw.py

Reads: data/400k-obs-with-env.csv, data/second-200k-obs.csv, data/first-200k-obs.csv (if present)
Writes: data/400k-obs-with-env.csv (overwrites with filled taxon_name)
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
    missing = env["taxon_name"].isna() | (env["taxon_name"].astype(str).str.strip() == "")
    n_missing_before = missing.sum()
    print(f"  Rows with missing/empty taxon_name: {n_missing_before:,}")

    # Build taxon_id -> name from raw (scientific_name preferred, else common_name)
    raw_dfs = []
    for path in [SECOND_RAW_PATH, FIRST_RAW_PATH]:
        if not path.exists():
            continue
        print(f"Loading {path.name}...")
        raw = pd.read_csv(
            path,
            usecols=["taxon_id", "scientific_name", "common_name"],
            low_memory=False,
        )
        raw = raw.dropna(subset=["taxon_id"])
        raw["name"] = raw["scientific_name"].fillna(raw["common_name"]).astype(str)
        raw = raw[raw["name"].str.strip() != ""]
        raw_dfs.append(raw)
    raw_combined = pd.concat(raw_dfs, ignore_index=True)
    id_to_name = raw_combined.drop_duplicates(subset=["taxon_id"], keep="first").set_index("taxon_id")["name"]
    print(f"  Unique taxon_id -> name: {len(id_to_name):,}")

    # Fill missing taxon_name
    fill_mask = missing & env["taxon_id"].notna() & env["taxon_id"].isin(id_to_name.index)
    env.loc[fill_mask, "taxon_name"] = env.loc[fill_mask, "taxon_id"].map(id_to_name)

    still_missing = env["taxon_name"].isna() | (env["taxon_name"].astype(str).str.strip() == "")
    n_still_missing = still_missing.sum()
    filled = n_missing_before - n_still_missing
    print(f"\nFilled {filled:,} taxon_name from raw (by taxon_id)")
    print(f"  Rows still missing taxon_name: {n_still_missing:,}")

    env.to_csv(WITH_ENV_PATH, index=False)
    print(f"Wrote {WITH_ENV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
