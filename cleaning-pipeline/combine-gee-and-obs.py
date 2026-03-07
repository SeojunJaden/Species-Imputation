#!/usr/bin/env python3
"""
This script combines environmental data from GeoTIFF files with observation data from CSV files.

Usage:
  python combine-gee-and-obs.py <input.csv> [output.csv]

  If output is omitted, writes to data/{input_basename}_with_env_data.csv
  Example: python combine-gee-and-obs.py data/first-200k-obs-cleaned.csv data/first-200k-obs-clean-with-env.csv
"""

import os
import sys
import pandas as pd
import rasterio
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

import warnings

warnings.filterwarnings('ignore')

# Paths relative to script location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

TIFF_DIR = os.path.join(PROJECT_ROOT, "google-earth-data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")

# All 18 GEE rasters (must match exports from get-gee-rasters.py)
TIFF_FILES = {
    'elevation.tif': 'elevation',
    'slope.tif': 'slope',
    'aspect.tif': 'aspect',
    'ndvi.tif': 'ndvi',
    'landcover.tif': 'landcover',
    'canopy_cover.tif': 'canopy_cover',
    'impervious.tif': 'impervious',
    'soil_clay.tif': 'soil_clay',
    'soil_sand.tif': 'soil_sand',
    'soil_ph.tif': 'soil_ph',
    'bathymetry.tif': 'bathymetry',
    'temperature_mean.tif': 'temperature_mean',
    'precipitation_annual.tif': 'precipitation_annual',
    'temperature_seasonality.tif': 'temperature_seasonality',
    'ndvi_seasonality.tif': 'ndvi_seasonality',
    'topographic_wetness.tif': 'topographic_wetness',
    'distance_to_coast.tif': 'distance_to_coast',
    'distance_to_water.tif': 'distance_to_water',
}


def extract_pixel_values_batch(raster_path, coordinates):
    """
    Sample a GeoTIFF at many (lon, lat) points in one go.

    Returns a list of values (or None where out-of-bounds / nodata).
    """
    try:
        with rasterio.open(raster_path) as src:
            values = list(src.sample(coordinates))
            result = []
            nodata = src.nodata

            for val_array in values:
                if len(val_array) == 0:
                    result.append(None)
                    continue
                value = float(val_array[0])
                if nodata is not None and (np.isnan(value) or value == nodata):
                    result.append(None)
                else:
                    result.append(value)

            return result

    except Exception:
        return [None] * len(coordinates)


def process_csv_file(csv_path, tiff_dir, output_path):
    """
    Read observations CSV, sample env rasters at each (lat, lon), write enriched CSV to data/.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(csv_path)}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} observations")

    required = ['latitude', 'longitude', 'taxon_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        return None

    if 'scientific_name' in df.columns:
        df = df.assign(taxon_name=df['scientific_name'].astype(str))
    elif 'common_name' in df.columns:
        df = df.assign(taxon_name=df['common_name'].astype(str))
    else:
        print(f"  ERROR: Need scientific_name or common_name for taxon_name.")
        return None

    n_before = len(df)
    df = df.dropna(subset=['latitude', 'longitude'])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped:,} rows with missing lat/lon (cannot extract env without coordinates)")
    print(f"  {len(df):,} observations with valid coordinates")

    if len(df) == 0:
        print(f"  ERROR: No valid coordinates found!")
        return None

    print(f"\nExtracting environmental data from {len(TIFF_FILES)} TIFF files...")
    coordinates = list(zip(df['longitude'], df['latitude']))

    for tiff_filename, column_name in tqdm(TIFF_FILES.items(), desc="  Processing TIFFs"):
        tiff_path = os.path.join(tiff_dir, tiff_filename)
        if not os.path.exists(tiff_path):
            print(f"  WARNING: {tiff_filename} not found, skipping...")
            df[column_name] = None
            continue
        values = extract_pixel_values_batch(tiff_path, coordinates)
        df[column_name] = values

    env_columns = list(TIFF_FILES.values())
    out = df[['taxon_name', 'latitude', 'longitude', 'taxon_id'] + env_columns].copy()

    all_ok = out[env_columns].notna().all(axis=1).sum()
    print(f"\n  Observations with all env data: {all_ok:,}")
    print(f"\nPer-variable completeness:")
    for col in env_columns:
        complete = out[col].notna().sum()
        pct = (complete / len(out)) * 100
        print(f"  {col:15s}: {complete:6,} ({pct:5.1f}%)")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python combine-gee-and-obs.py <input.csv> [output.csv]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Resolve paths relative to project root so same file is used regardless of cwd
    if not os.path.isabs(input_path):
        input_path = os.path.join(PROJECT_ROOT, input_path)
    if output_path is not None and not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    if not os.path.exists(TIFF_DIR):
        print(f"ERROR: TIFF directory not found: {TIFF_DIR}")
        sys.exit(1)

    if output_path is None:
        base = os.path.basename(input_path).replace('.csv', '')
        output_path = os.path.join(OUTPUT_DIR, f"{base}_with_env_data.csv")

    process_csv_file(input_path, TIFF_DIR, output_path)
    print(f"\n{'='*60}\nDone.")


if __name__ == "__main__":
    main()
