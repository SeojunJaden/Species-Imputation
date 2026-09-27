#!/usr/bin/env python3
"""
Re-export the 10 habitat layers the v2 model samples, extended south to 32.5 N,
and download them straight to google-earth-data-v2/ (no Google Drive round trip).

Why: google-earth-data/ stops at 32.7 N, which excludes Sweetwater Marsh
(32.634-32.651 N), Kendall-Frost's v2 partner (METHODOLOGY 4). The layer
definitions are copied from get-gee-rasters.py unchanged; only the extent differs.

The grid is pinned to the old rasters' exact pixel grid (same origin and pixel size)
and simply extended 743 rows further south (1857 -> 2600), so every old pixel has an identical
twin here. `--check` compares the two over their overlap -- if they match, the old
rasters and the new ones are interchangeable north of 32.7 N, and the Scripps /
Elliott / Dawson results already sent to Kellie need no re-run.

The old rasters are left untouched. Only Sweetwater Marsh is sampled from these.

Usage:
  EE_PROJECT=<cloud-project-id> python cleaning-pipeline/download_gee_rasters_v2.py
  python cleaning-pipeline/download_gee_rasters_v2.py --check   # compare with old
"""

import os
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_DIR = os.path.join(ROOT, "google-earth-data")
OUT_DIR = os.path.join(ROOT, "google-earth-data-v2")

# The old rasters' grid (read from google-earth-data/elevation.tif).
PIXEL = 0.00026949458523585647
LEFT, TOP = -117.4001956608814, 33.20038542813134
WIDTH = 1857
HEIGHT = 2600          # old grid was 1857 rows; bottom edge now ~32.4997 N
CRS_TRANSFORM = [PIXEL, 0, LEFT, 0, -PIXEL, TOP]

# Every layer is a public Earth Engine catalog dataset, so any EE-registered Cloud
# project can run this; the project only identifies who is making the requests.
# The v1 rasters came from Carsten's "species-imputation" project; v2 from Kylan's.
PROJECT = os.environ.get("EE_PROJECT", "species-imputation-kylan")


def build_layers(ee):
    """The 10 layers sample_env_v2.py reads, defined exactly as in get-gee-rasters.py."""
    bbox = ee.Geometry.Rectangle([LEFT, TOP - HEIGHT * PIXEL, LEFT + WIDTH * PIXEL, TOP])

    def mask_clouds_landsat8(image):
        qa = image.select('QA_PIXEL')
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        return image.updateMask(cloud_mask)

    def calculate_ndvi(image):
        return image.addBands(image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI'))

    srtm = ee.Image('USGS/SRTMGL1_003')
    nlcd19 = ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2019')
    ndvi = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterDate('2020-01-01', '2024-12-31')
            .filterBounds(bbox)
            .map(mask_clouds_landsat8)
            .map(calculate_ndvi)
            .select('NDVI').median())

    layers = {
        'elevation': srtm,
        'slope': ee.Terrain.slope(srtm),
        'aspect': ee.Terrain.aspect(srtm),
        'ndvi': ndvi,
        'landcover': ee.Image('USGS/NLCD_RELEASES/2021_REL/NLCD/2021').select('landcover'),
        'impervious': nlcd19.select('impervious'),
        'bathymetry': ee.Image('NOAA/NGDC/ETOPO1').select('bedrock'),
        'soil_sand': ee.Image('OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02').select('b0'),
        'soil_ph': ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02').select('b0'),
        'soil_clay': ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02').select('b0'),
    }
    return {name: img.clip(bbox) for name, img in layers.items()}


# Layers too expensive to compute in one interactive request ("User memory limit
# exceeded" -- the 5-year Landsat median). They are fetched in horizontal strips on the
# same grid and stitched; a median is per-pixel, so strips give identical values.
STRIPS = {'ndvi': 6}


def _url(img, name, top, rows):
    return img.getDownloadURL({
        'name': name,
        'format': 'GEO_TIFF',
        'crs': 'EPSG:4326',
        'crs_transform': [PIXEL, 0, LEFT, 0, -PIXEL, top],
        'dimensions': f'{WIDTH}x{rows}',
    })


def _download_strips(img, name, dest, n):
    import tempfile
    import rasterio
    from rasterio.transform import Affine

    edges = np.linspace(0, HEIGHT, n + 1).astype(int)
    parts, profile = [], None
    with tempfile.TemporaryDirectory() as tmp:
        for i, (r0, r1) in enumerate(zip(edges[:-1], edges[1:])):
            part = os.path.join(tmp, f'{name}_{i}.tif')
            urllib.request.urlretrieve(_url(img, name, TOP - r0 * PIXEL, r1 - r0), part)
            with rasterio.open(part) as src:
                parts.append(src.read(1))
                profile = profile or src.profile
    profile.update(height=HEIGHT, width=WIDTH,
                   transform=Affine(PIXEL, 0, LEFT, 0, -PIXEL, TOP))
    with rasterio.open(dest, 'w', **profile) as out:
        out.write(np.vstack(parts), 1)


def download():
    import ee
    ee.Initialize(project=PROJECT)
    os.makedirs(OUT_DIR, exist_ok=True)
    only = [a for a in sys.argv[1:] if not a.startswith('-')]
    for name, img in build_layers(ee).items():
        if only and name not in only:
            continue
        dest = os.path.join(OUT_DIR, f'{name}.tif')
        if name in STRIPS:
            _download_strips(img.toFloat(), name, dest, STRIPS[name])
        else:
            urllib.request.urlretrieve(_url(img, name, TOP, HEIGHT), dest)
        print(f"[+] {name:11s} -> {os.path.relpath(dest, ROOT)} "
              f"({os.path.getsize(dest) / 1e6:.1f} MB)")


def check():
    """Compare new vs old rasters over the old footprint, pixel for pixel."""
    import rasterio
    from rasterio.windows import from_bounds

    for name in sorted(f[:-4] for f in os.listdir(OUT_DIR) if f.endswith('.tif')):
        with rasterio.open(os.path.join(OLD_DIR, f'{name}.tif')) as old, \
             rasterio.open(os.path.join(OUT_DIR, f'{name}.tif')) as new:
            a = old.read(1).astype('float64')
            win = from_bounds(*old.bounds, transform=new.transform).round_offsets().round_lengths()
            b = new.read(1, window=win).astype('float64')
            if a.shape != b.shape:
                print(f"{name:11s} SHAPE MISMATCH old {a.shape} new {b.shape}")
                continue
            both = np.isfinite(a) & np.isfinite(b)
            diff = np.abs(a[both] - b[both])
            exact = 100.0 * (diff == 0).mean()
            print(f"{name:11s} bounds {tuple(round(x, 4) for x in new.bounds)}  "
                  f"identical {exact:6.2f}%  max|diff| {diff.max():.4g}  "
                  f"mean|diff| {diff.mean():.4g}")


if __name__ == '__main__':
    check() if '--check' in sys.argv else download()
