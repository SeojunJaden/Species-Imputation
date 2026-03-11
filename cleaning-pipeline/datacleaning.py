#!/usr/bin/env python3
"""
Run the data cleaning pipeline on an iNaturalist observations CSV.

This file: Drops unused columns, removes rows with missing required fields, unduplicates
to one observation per species per 5-minute window, and writes the cleaned CSV.

Usage: python run_datacleaning.py [input_csv] [output_csv]
Defaults (CHANGE THESE!!!!!): data/observations-685862.csv -> data/observations_685862_Filtered.csv
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
"""CHANGE THESE!!!"""
DEFAULT_INPUT = PROJECT_ROOT / "data" / "observations-685862.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "observations_685862_Filtered.csv"

COLUMNS_TO_DROP = [
    "url", "image_url", "sound_url", "tag_list", "description",
    "captive_cultivated", "oauth_application_id", "private_place_guess",
    "private_latitude", "private_longitude", "public_positional_accuracy",
    "geoprivacy", "taxon_geoprivacy", "coordinates_obscured",
    "positioning_method", "positioning_device", "species_guess",
    "created_at", "updated_at", "coordinates",
]

REQUIRED_COLUMNS = ["latitude", "longitude", "taxon_id", "scientific_name"]

# Keep one observation per species per 5-minute window.
def filter_observations(group):
    kept_rows = []
    last_kept_time = None
    for _, row in group.iterrows():
        if last_kept_time is None or (row["time_observed_at"] - last_kept_time) >= pd.Timedelta(minutes=5):
            kept_rows.append(row)
            last_kept_time = row["time_observed_at"]
    return pd.DataFrame(kept_rows)


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df)}")

    # Drop columns that we don't need
    to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=to_drop)

    # Ensure required columns exist
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"Error: Input CSV is missing required columns: {missing_cols}")
        sys.exit(1)

    # Drop rows missing lat/lon, taxon_id, or scientific_name (required columns)
    df = df.dropna(subset=REQUIRED_COLUMNS)

    # Parse time and sort to unduplicate species observations within 5-minute windows
    df["time_observed_at"] = pd.to_datetime(df["time_observed_at"])
    df = df.sort_values(by=["scientific_name", "time_observed_at"])

    # Unduplicate: one observation per species per 5 min
    print("Applying 5-minute unduplication per species...")
    df_clean = df.groupby("scientific_name", group_keys=False).apply(filter_observations, include_groups=False)
    print(f"  Rows after unduplication: {len(df_clean)}")

    # Write to clean csv
    df_clean.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(df_clean)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
