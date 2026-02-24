#!/usr/bin/env python3
"""
Species Distribution Modeling - Environmental Data Extraction

This script extracts environmental data from GeoTIFF files at species observation
coordinates and combines them with observation data from CSV files.
"""

import os
import glob
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform as rasterio_transform
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# Configuration
TIFF_DIR = "GEE Data/drive-download-20260217T221411Z-1-001"
CSV_DIR = "CleanedData"
OUTPUT_DIR = "ProcessedData"

# Required columns to keep from CSV files
REQUIRED_COLUMNS = [
    'latitude', 'longitude', 'time_observed_at', 
    'scientific_name', 'common_name', 'iconic_taxon_name'
]

# Expected TIFF files and their corresponding column names
TIFF_FILES = {
    'elevation.tif': 'elevation',
    'slope.tif': 'slope',
    'aspect.tif': 'aspect',
    'ndvi.tif': 'ndvi',
    'landcover.tif': 'landcover',
    'impervious.tif': 'impervious',
    'bathymetry.tif': 'bathymetry',
    'soil_sand.tif': 'soil_sand',
    'soil_ph.tif': 'soil_ph',
    'soil_clay.tif': 'soil_clay'
}


def extract_pixel_values_batch(raster_path, coordinates):
    """
    Extract pixel values from a GeoTIFF for multiple coordinates at once (vectorized).
    
    Parameters:
    -----------
    raster_path : str
        Path to the GeoTIFF file
    coordinates : list of tuples
        List of (lon, lat) coordinate tuples
    
    Returns:
    --------
    list
        List of pixel values (or None for invalid/out-of-bounds coordinates)
    """
    try:
        with rasterio.open(raster_path) as src:
            # Use sample() to extract values for all points at once
            # This is much faster than opening the file for each point
            values = list(src.sample(coordinates))
            
            # Process the results
            result = []
            nodata = src.nodata
            
            for val_array in values:
                if len(val_array) > 0:
                    value = float(val_array[0])
                    # Check for nodata
                    if nodata is not None and (np.isnan(value) or value == nodata):
                        result.append(None)
                    else:
                        result.append(value)
                else:
                    result.append(None)
            
            return result
    
    except Exception as e:
        # If batch extraction fails, return None for all points
        return [None] * len(coordinates)


def load_raster_metadata(tiff_path):
    """
    Load raster metadata for validation.
    
    Parameters:
    -----------
    tiff_path : str
        Path to the GeoTIFF file
    
    Returns:
    --------
    dict or None
        Dictionary with bounds and nodata info, or None if file can't be opened
    """
    try:
        with rasterio.open(tiff_path) as src:
            return {
                'bounds': src.bounds,
                'nodata': src.nodata,
                'crs': src.crs,
                'shape': src.shape
            }
    except Exception:
        return None


def process_csv_file(csv_path, tiff_dir, output_dir):
    """
    Process a single CSV file by extracting environmental data from TIFFs.
    
    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file
    tiff_dir : str
        Directory containing TIFF files
    output_dir : str
        Directory to save processed CSV
    
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with environmental data
    """
    # Extract reserve name from filename
    reserve_name = os.path.basename(csv_path).replace('_Filtered.csv', '')
    
    print(f"\n{'='*60}")
    print(f"Processing: {reserve_name}")
    print(f"{'='*60}")
    
    # Read CSV file
    print(f"Reading CSV file...")
    df = pd.read_csv(csv_path)
    original_count = len(df)
    print(f"  Loaded {original_count:,} observations")
    
    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"  WARNING: Missing columns: {missing_cols}")
        # Keep only available required columns
        available_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
        df = df[available_cols + ['latitude', 'longitude']]
    else:
        df = df[REQUIRED_COLUMNS].copy()
    
    # Remove rows with missing coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    valid_count = len(df)
    print(f"  {valid_count:,} observations with valid coordinates")
    
    if valid_count == 0:
        print(f"  ERROR: No valid coordinates found!")
        return None
    
    # Extract environmental data from each TIFF
    print(f"\nExtracting environmental data from {len(TIFF_FILES)} TIFF files...")
    
    # Prepare coordinates as list of tuples for batch processing
    coordinates = list(zip(df['longitude'], df['latitude']))
    
    for tiff_filename, column_name in tqdm(TIFF_FILES.items(), desc="  Processing TIFFs"):
        tiff_path = os.path.join(tiff_dir, tiff_filename)
        
        if not os.path.exists(tiff_path):
            print(f"  WARNING: {tiff_filename} not found, skipping...")
            df[column_name] = None
            continue
        
        # Extract pixel values for all coordinates at once (much faster!)
        values = extract_pixel_values_batch(tiff_path, coordinates)
        df[column_name] = values
    
    # Calculate data completeness statistics
    env_columns = list(TIFF_FILES.values())
    df['env_data_completeness'] = df[env_columns].notna().sum(axis=1) / len(env_columns)
    
    # Print statistics
    print(f"\nData Completeness Statistics:")
    print(f"  Total observations: {len(df):,}")
    print(f"  Observations with all env data: {(df['env_data_completeness'] == 1.0).sum():,}")
    print(f"  Observations with >=50% env data: {(df['env_data_completeness'] >= 0.5).sum():,}")
    print(f"  Observations with <50% env data: {(df['env_data_completeness'] < 0.5).sum():,}")
    
    # Per-variable completeness
    print(f"\nPer-variable completeness:")
    for col in env_columns:
        if col in df.columns:
            complete = df[col].notna().sum()
            pct = (complete / len(df)) * 100
            print(f"  {col:15s}: {complete:6,} ({pct:5.1f}%)")
    
    # Save individual processed CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{reserve_name}_with_env_data.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    # Add reserve name column for later combination
    df['reserve_name'] = reserve_name
    
    return df


def main():
    """Main function to process all CSV files and create combined dataset."""
    
    print("="*60)
    print("Species Distribution Modeling - Environmental Data Extraction")
    print("="*60)
    
    # Validate directories
    if not os.path.exists(TIFF_DIR):
        print(f"ERROR: TIFF directory not found: {TIFF_DIR}")
        return
    
    if not os.path.exists(CSV_DIR):
        print(f"ERROR: CSV directory not found: {CSV_DIR}")
        return
    
    # Check for TIFF files
    print(f"\nChecking TIFF files in {TIFF_DIR}...")
    missing_tiffs = []
    for tiff_filename in TIFF_FILES.keys():
        tiff_path = os.path.join(TIFF_DIR, tiff_filename)
        if os.path.exists(tiff_path):
            print(f"  ✓ {tiff_filename}")
        else:
            print(f"  ✗ {tiff_filename} - NOT FOUND")
            missing_tiffs.append(tiff_filename)
    
    if missing_tiffs:
        print(f"\nWARNING: {len(missing_tiffs)} TIFF file(s) not found!")
    
    # Find all CSV files
    csv_pattern = os.path.join(CSV_DIR, "*_Filtered.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        print(f"\nERROR: No CSV files found matching pattern: {csv_pattern}")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s) to process:")
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")
    
    # Process each CSV file
    all_dataframes = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for csv_path in csv_files:
        reserve_name = os.path.basename(csv_path).replace('_Filtered.csv', '')
        output_path = os.path.join(OUTPUT_DIR, f"{reserve_name}_with_env_data.csv")
        
        # Check if already processed
        if os.path.exists(output_path):
            print(f"\n{'='*60}")
            print(f"Skipping {reserve_name} (already processed)")
            print(f"{'='*60}")
            print(f"Loading existing file: {output_path}")
            try:
                df_existing = pd.read_csv(output_path)
                # Ensure reserve_name column exists
                if 'reserve_name' not in df_existing.columns:
                    df_existing['reserve_name'] = reserve_name
                all_dataframes.append(df_existing)
                print(f"  Loaded {len(df_existing):,} observations")
            except Exception as e:
                print(f"  ERROR loading existing file: {e}")
                print(f"  Will reprocess...")
                df_processed = process_csv_file(csv_path, TIFF_DIR, OUTPUT_DIR)
                if df_processed is not None:
                    all_dataframes.append(df_processed)
        else:
            df_processed = process_csv_file(csv_path, TIFF_DIR, OUTPUT_DIR)
            if df_processed is not None:
                all_dataframes.append(df_processed)
    
    # Combine all reserves
    if all_dataframes:
        print(f"\n{'='*60}")
        print("Combining all reserves...")
        print(f"{'='*60}")
        
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Overall statistics
        print(f"\nCombined Dataset Statistics:")
        print(f"  Total observations: {len(combined_df):,}")
        print(f"  Number of reserves: {combined_df['reserve_name'].nunique()}")
        print(f"  Reserves: {', '.join(sorted(combined_df['reserve_name'].unique()))}")
        
        env_columns = list(TIFF_FILES.values())
        print(f"\nOverall Data Completeness:")
        print(f"  Observations with all env data: {(combined_df['env_data_completeness'] == 1.0).sum():,}")
        print(f"  Observations with >=50% env data: {(combined_df['env_data_completeness'] >= 0.5).sum():,}")
        print(f"  Observations with <50% env data: {(combined_df['env_data_completeness'] < 0.5).sum():,}")
        
        # Save combined CSV
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        combined_output_path = os.path.join(OUTPUT_DIR, "all_reserves_with_env_data.csv")
        combined_df.to_csv(combined_output_path, index=False)
        print(f"\nSaved combined dataset: {combined_output_path}")
        
        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"{'='*60}")
    else:
        print("\nERROR: No data was successfully processed!")


if __name__ == "__main__":
    main()
