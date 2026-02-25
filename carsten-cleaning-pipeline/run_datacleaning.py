#!/usr/bin/env python3
"""
Run the datacleaning pipeline (from datacleaning.ipynb) on an iNaturalist observations CSV.
Usage: python run_datacleaning.py [input_csv] [output_csv]
Defaults: full_sd_obs.csv/observations-685862.csv -> CleanedData/observations_685862_Filtered.csv
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from global_land_mask import globe
    HAS_LAND_MASK = True
except ImportError:
    HAS_LAND_MASK = False

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "full_sd_obs.csv" / "observations-685862.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "CleanedData" / "observations_685862_Filtered.csv"

COLUMNS_TO_DROP = [
    "url", "image_url", "sound_url", "tag_list", "description",
    "captive_cultivated", "oauth_application_id", "private_place_guess",
    "private_latitude", "private_longitude", "public_positional_accuracy",
    "geoprivacy", "taxon_geoprivacy", "coordinates_obscured",
    "positioning_method", "positioning_device", "species_guess",
    "created_at", "updated_at",
]


def filter_observations(group):
    """Keep one observation per species per 5-minute window."""
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

    # Drop columns that exist
    to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=to_drop)

    # Parse time and sort
    df["time_observed_at"] = pd.to_datetime(df["time_observed_at"])
    df = df.sort_values(by=["scientific_name", "time_observed_at"])

    # Deduplicate: one observation per species per 5 min
    print("Applying 5-minute deduplication per species...")
    df_clean = df.groupby("scientific_name", group_keys=False).apply(filter_observations, include_groups=False)
    print(f"  Rows after dedup: {len(df_clean)}")

    # Add land mask if available (lat, lon order for globe.is_land)
    if HAS_LAND_MASK:
        print("Computing land mask...")
        coords = list(zip(df_clean["latitude"], df_clean["longitude"]))
        df_clean = df_clean.assign(
            is_land=pd.Series([globe.is_land(lat, lon) for lat, lon in coords], index=df_clean.index)
        )
    else:
        print("Skipping land mask (global-land-mask not installed).")

    out_cols = [c for c in df_clean.columns if c != "coordinates"]
    df_out = df_clean[out_cols]

    df_out.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(df_out)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
