#!/usr/bin/env python3
"""
Sample habitat rasters at every v2 partner-reserve observation.

Replaces the (lost) script that produced ProcessedData/*_with_env_data.csv for v1:
nothing in the repo writes that file's schema, so this reproduces it exactly --
same column names, same 10 environmental layers, same join keys -- against the
v2 pull in RefreshedData/2026-09-14/filtered/.

Only PARTNER reserves need habitat values: the model's features all describe where
a species occurs at its partner (METHODOLOGY 2). Study reserves contribute only
presence/absence, so they are not sampled here.

Unlike v1, an observation that falls OUTSIDE the raster footprint is written as
empty, never 0.0, and the per-reserve out-of-bounds count is printed and stored in
the manifest. Silent 0.0 for out-of-bbox points is exactly what made Kendall-Frost
habitat-blind in v1 (METHODOLOGY 6).

Usage:
  python cleaning-pipeline/sample_env_v2.py [Reserve ...]     # default: all partners
  python cleaning-pipeline/sample_env_v2.py --tiff-dir google-earth-data-v2 SweetwaterMarsh

--tiff-dir picks the raster folder (default google-earth-data/). The v2 rasters extend
south to 32.5 N and are used for Sweetwater Marsh only (METHODOLOGY 4).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIFF_DIR = os.path.join(ROOT, "google-earth-data")
V2 = os.path.join(ROOT, "RefreshedData", "2026-09-14")
FILTERED = os.path.join(V2, "filtered")
OUT_DIR = os.path.join(V2, "env")

# The 10 layers v1's ProcessedData/*_with_env_data.csv carries, in its column order.
ENV_LAYERS = [
    ("elevation.tif", "elevation"),
    ("slope.tif", "slope"),
    ("aspect.tif", "aspect"),
    ("ndvi.tif", "ndvi"),
    ("landcover.tif", "landcover"),
    ("impervious.tif", "impervious"),
    ("bathymetry.tif", "bathymetry"),
    ("soil_sand.tif", "soil_sand"),
    ("soil_ph.tif", "soil_ph"),
    ("soil_clay.tif", "soil_clay"),
]

# Partner reserves only (see module docstring).
PARTNERS = ["TorreyPines", "MissionTrails", "BuenaVista", "SweetwaterMarsh"]

# Fill values the v2 rasters (google-earth-data-v2/) use for masked pixels (water)
# without declaring them as nodata -- the files' nodata tag is 0. None of these values
# is in the layer's valid range (soils 0-100 / pH x10, impervious 0-100), and the v1
# rasters never contain them, so masking them is a no-op there.
FILL = {"soil_sand": 255, "soil_ph": 255, "soil_clay": 255, "impervious": 127}

KEY_COLS = ["latitude", "longitude", "time_observed_at",
            "scientific_name", "common_name", "iconic_taxon_name"]


def sample_raster(path, coords, fill=None):
    """Sample one raster at (lon, lat) pairs -> float array, NaN where unusable."""
    with rasterio.open(path) as src:
        left, bottom, right, top = src.bounds
        nodata = src.nodata
        out = np.full(len(coords), np.nan)
        # rasterio.sample returns the edge pixel for out-of-bounds points on some
        # drivers, so screen on the footprint ourselves before sampling.
        inside = np.array([left <= lon <= right and bottom <= lat <= top
                           for lon, lat in coords])
        idx = np.flatnonzero(inside)
        if idx.size:
            vals = np.array([v[0] for v in src.sample([coords[i] for i in idx])],
                            dtype="float64")
            if nodata is not None:
                vals[vals == nodata] = np.nan
            if fill is not None:
                vals[vals == fill] = np.nan
            out[idx] = vals
        return out, inside


def process(reserve):
    src_csv = os.path.join(FILTERED, f"{reserve}_Filtered.csv")
    if not os.path.exists(src_csv):
        raise SystemExit(f"missing {src_csv}")

    df = pd.read_csv(src_csv, low_memory=False)
    df = df.dropna(subset=["latitude", "longitude"])
    coords = list(zip(df["longitude"].to_numpy(), df["latitude"].to_numpy()))
    print(f"\n[*] {reserve}: {len(df):,} observations")

    out = df[KEY_COLS].copy()
    in_footprint = None
    for tif, col in ENV_LAYERS:
        path = os.path.join(TIFF_DIR, tif)
        if not os.path.exists(path):
            raise SystemExit(f"missing raster {path}")
        vals, inside = sample_raster(path, coords, FILL.get(col))
        out[col] = vals
        in_footprint = inside if in_footprint is None else (in_footprint & inside)

    env_cols = [c for _, c in ENV_LAYERS]
    out["env_data_completeness"] = out[env_cols].notna().mean(axis=1).round(4)

    n_outside = int((~in_footprint).sum())
    pct_out = 100.0 * n_outside / max(len(out), 1)
    for col in env_cols:
        n = int(out[col].notna().sum())
        print(f"      {col:12s} {n:7,} ({100.0*n/max(len(out),1):5.1f}%)")
    if n_outside:
        print(f"    !! {n_outside:,} of {len(out):,} observations ({pct_out:.1f}%) fall "
              f"OUTSIDE the raster footprint -- habitat features are unusable for them.")
        if pct_out > 99:
            print(f"    !! {reserve} is effectively habitat-blind. Re-export the rasters "
                  f"over a bbox that covers it before modelling with these values.")

    os.makedirs(OUT_DIR, exist_ok=True)
    dest = os.path.join(OUT_DIR, f"{reserve}_with_env_data.csv")
    out.to_csv(dest, index=False)
    print(f"    -> {dest}")

    return dict(reserve=reserve, observations=len(out),
                outside_raster_footprint=n_outside,
                pct_outside=round(pct_out, 2),
                fully_sampled=int(out["env_data_completeness"].eq(1.0).sum()),
                layers=env_cols, rasters=os.path.relpath(TIFF_DIR, ROOT), source=os.path.relpath(src_csv, ROOT),
                file=os.path.relpath(dest, ROOT))


def main():
    global TIFF_DIR
    args = sys.argv[1:]
    if "--tiff-dir" in args:
        i = args.index("--tiff-dir")
        TIFF_DIR = os.path.join(ROOT, args[i + 1])
        del args[i:i + 2]
    print(f"[*] rasters: {os.path.relpath(TIFF_DIR, ROOT)}")
    targets = args or PARTNERS
    report = [process(r) for r in targets]

    os.makedirs(OUT_DIR, exist_ok=True)
    man_path = os.path.join(OUT_DIR, "env_manifest.json")
    manifest = {}
    if os.path.exists(man_path):
        with open(man_path) as f:
            manifest = json.load(f)
    for r in report:
        manifest[r["reserve"]] = r
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n[+] manifest -> {os.path.relpath(man_path, ROOT)}")


if __name__ == "__main__":
    main()
