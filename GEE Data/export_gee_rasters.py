"""
Google Earth Engine Script: Export Environmental Rasters for Species Distribution Modeling
San Diego Coastal Reserves Study Area

This script exports 11 environmental layers (10 terrestrial + 1 ocean) for use in
machine learning models predicting species presence/absence in San Diego coastal reserves.

Study Area: San Diego coastal area from Oceanside to Mission Bay
Bounding Box: [-117.4, 32.7, -116.9, 33.2]
Resolution: 30 meters
Format: GeoTIFF
Destination: Google Drive folder 'sd_species_rasters'
"""

import ee

ee.Authenticate()

# Initialize Earth Engine
ee.Initialize(project='species-imputation')

# ============================================================================
# Bounding Box contains all UCSD natural reserves
# ============================================================================
# San Diego coastal reserves bounding box
# North: 33.2° (above Oceanside/Carlsbad)
# South: 32.7° (below Mission Bay)
# East: -116.9° (past Scripps Ranch)
# West: -117.4° (into Pacific Ocean)
bbox = ee.Geometry.Rectangle([-117.4, 32.7, -116.9, 33.2])

print("=" * 70)
print("Google Earth Engine Raster Export for San Diego Coastal Reserves")
print("=" * 70)
print(f"Bounding Box: [-117.4, 32.7, -116.9, 33.2]")
print(f"Export folder: 'sd_species_rasters'")
print(f"Resolution: 30 meters")
print(f"CRS: EPSG:4326")
print("=" * 70)
print()

# ============================================================================
# CLOUD MASKING FUNCTION FOR LANDSAT 8
# ============================================================================
def mask_clouds_landsat8(image):
    """
    Mask clouds and cloud shadows from Landsat 8 images using QA_PIXEL band.
    
    Args:
        image: Landsat 8 image with QA_PIXEL band
        
    Returns:
        Image with clouds and cloud shadows masked out
    """
    # Use QA_PIXEL band for cloud masking
    qa = image.select('QA_PIXEL')
    # Bits 3 and 4 are cloud and cloud shadow
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(
                 qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(cloud_mask)


# ============================================================================
# EXPORT FUNCTION
# ============================================================================
def export_raster(image, description, folder='sd_species_rasters', 
                  scale=30, region=None, crs='EPSG:4326', 
                  maxPixels=1e13, fileFormat='GeoTIFF'):
    """
    Create and start an export task for a raster image.
    
    Args:
        image: Earth Engine Image to export
        description: Task description and fileNamePrefix
        folder: Google Drive folder name
        scale: Pixel resolution in meters
        region: Region of interest (Geometry)
        crs: Coordinate reference system
        maxPixels: Maximum pixels allowed in export
        fileFormat: Export file format
        
    Returns:
        Export task object
    """
    try:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            fileNamePrefix=description,
            scale=scale,
            region=region,
            crs=crs,
            maxPixels=maxPixels,
            fileFormat=fileFormat
        )
        task.start()
        print(f"✓ Export started: {description}")
        return task
    except Exception as e:
        print(f"✗ Error exporting {description}: {str(e)}")
        return None


# ============================================================================
# TERRESTRIAL LAYERS
# ============================================================================

print("TERRESTRIAL LAYERS")
print("-" * 70)

# ----------------------------------------------------------------------------
# 1. ELEVATION
# ----------------------------------------------------------------------------
print("\n1. Processing elevation...")
try:
    # USGS SRTM 30m Digital Elevation Model
    # Units: meters above sea level
    elevation = ee.Image('USGS/SRTMGL1_003').clip(bbox)
    export_raster(elevation, 'elevation', region=bbox)
except Exception as e:
    print(f"✗ Error loading elevation dataset: {str(e)}")

# ----------------------------------------------------------------------------
# 2. SLOPE
# ----------------------------------------------------------------------------
print("\n2. Processing slope...")
try:
    # Calculate slope in degrees from elevation
    # Units: degrees (0-90)
    elevation_for_slope = ee.Image('USGS/SRTMGL1_003')
    slope = ee.Terrain.slope(elevation_for_slope).clip(bbox)
    export_raster(slope, 'slope', region=bbox)
except Exception as e:
    print(f"✗ Error calculating slope: {str(e)}")

# ----------------------------------------------------------------------------
# 3. ASPECT
# ----------------------------------------------------------------------------
print("\n3. Processing aspect...")
try:
    # Calculate aspect (direction slope faces) from elevation
    # Units: degrees (0=North, 90=East, 180=South, 270=West)
    elevation_for_aspect = ee.Image('USGS/SRTMGL1_003')
    aspect = ee.Terrain.aspect(elevation_for_aspect).clip(bbox)
    export_raster(aspect, 'aspect', region=bbox)
except Exception as e:
    print(f"✗ Error calculating aspect: {str(e)}")

# ----------------------------------------------------------------------------
# 4. LANDCOVER
# ----------------------------------------------------------------------------
print("\n4. Processing landcover...")
try:
    # NLCD 2021 land cover classification
    # Categorical values: 11=Open Water, 21=Developed Open Space, 
    # 41=Deciduous Forest, 42=Evergreen Forest, 43=Mixed Forest, etc.
    landcover = ee.Image('USGS/NLCD_RELEASES/2021_REL/NLCD/2021').select('landcover').clip(bbox)
    export_raster(landcover, 'landcover', region=bbox)
except Exception as e:
    print(f"✗ Error loading landcover dataset: {str(e)}")

# ----------------------------------------------------------------------------
# 5. NDVI
# ----------------------------------------------------------------------------
print("\n5. Processing NDVI...")
try:
    # Landsat 8 Collection 2 Level 2 Surface Reflectance
    # Calculate annual median NDVI (vegetation greenness index)
    # Date range: 2020-01-01 to 2024-12-31
    # Units: -1 to 1 (higher = more/healthier vegetation)
    
    landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterDate('2020-01-01', '2024-12-31') \
        .filterBounds(bbox) \
        .map(mask_clouds_landsat8)
    
    # Calculate NDVI: (NIR - Red) / (NIR + Red)
    # Landsat 8: SR_B5 = NIR, SR_B4 = Red
    def calculate_ndvi(image):
        ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        return image.addBands(ndvi)
    
    landsat8_ndvi = landsat8.map(calculate_ndvi)
    
    # Calculate median NDVI across all images
    ndvi_median = landsat8_ndvi.select('NDVI').median().clip(bbox)
    export_raster(ndvi_median, 'ndvi', region=bbox)
except Exception as e:
    print(f"✗ Error processing NDVI: {str(e)}")

# ----------------------------------------------------------------------------
# 6. CANOPY COVER
# ----------------------------------------------------------------------------
print("\n6. Processing canopy_cover...")
try:
    # NLCD 2019 percentage of tree canopy cover
    # Units: 0-100%
    canopy_cover = ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2019') \
        .select('percent_tree_cover') \
        .clip(bbox)
    export_raster(canopy_cover, 'canopy_cover', region=bbox)
except Exception as e:
    print(f"✗ Error loading canopy_cover dataset: {str(e)}")

# ----------------------------------------------------------------------------
# 7. IMPERVIOUS SURFACE
# ----------------------------------------------------------------------------
print("\n7. Processing impervious...")
try:
    # NLCD 2019 percentage of impervious surface (pavement, buildings)
    # Units: 0-100%
    impervious = ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2019') \
        .select('impervious') \
        .clip(bbox)
    export_raster(impervious, 'impervious', region=bbox)
except Exception as e:
    print(f"✗ Error loading impervious dataset: {str(e)}")

# ----------------------------------------------------------------------------
# 8. SOIL CLAY CONTENT
# ----------------------------------------------------------------------------
print("\n8. Processing soil_clay...")
try:
    # OpenLandMap soil clay content at 0-5cm depth
    # Units: g/kg (multiply by 0.1 for percentage)
    soil_clay = ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02').select('b0').clip(bbox)
    export_raster(soil_clay, 'soil_clay', region=bbox)
except Exception as e:
    print(f"✗ Error loading soil_clay dataset: {str(e)}")

# ----------------------------------------------------------------------------
# 9. SOIL SAND CONTENT
# ----------------------------------------------------------------------------
print("\n9. Processing soil_sand...")
try:
    # OpenLandMap soil sand content at 0-5cm depth
    # Units: g/kg (multiply by 0.1 for percentage)
    soil_sand = ee.Image('OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02').select('b0').clip(bbox)
    export_raster(soil_sand, 'soil_sand', region=bbox)
except Exception as e:
    print(f"✗ Error loading soil_sand dataset: {str(e)}")

# ----------------------------------------------------------------------------
# 10. SOIL PH
# ----------------------------------------------------------------------------
print("\n10. Processing soil_ph...")
try:
    # OpenLandMap soil pH in H2O at 0-5cm depth
    # Units: pH × 10 (divide by 10 for actual pH)
    soil_ph = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02').select('b0').clip(bbox)
    export_raster(soil_ph, 'soil_ph', region=bbox)
except Exception as e:
    print(f"✗ Error loading soil_ph dataset: {str(e)}")

# ============================================================================
# OCEAN LAYER
# ============================================================================

print("\n" + "=" * 70)
print("OCEAN LAYER")
print("-" * 70)

# ----------------------------------------------------------------------------
# 11. BATHYMETRY
# ----------------------------------------------------------------------------
print("\n11. Processing bathymetry...")
try:
    # NOAA ETOPO1 global bathymetry/topography
    # Negative values = below sea level (ocean depth)
    # Positive values = elevation above sea level
    # Units: meters
    bathymetry = ee.Image('NOAA/NGDC/ETOPO1').select('bedrock').clip(bbox)
    export_raster(bathymetry, 'bathymetry', region=bbox)
except Exception as e:
    print(f"✗ Error loading bathymetry dataset: {str(e)}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("EXPORT SUMMARY")
print("=" * 70)
print("All export tasks have been submitted to Google Earth Engine.")
print("Exports will appear in Google Drive folder: 'sd_species_rasters'")
print()
print("Total layers exported: 11")
print("  - Terrestrial: 10 layers")
print("  - Ocean: 1 layer")
print()
print("Expected export time: 30-60 minutes")
print("File size per layer: ~10-50 MB")
print()
print("To check export status:")
print("  1. Visit: https://code.earthengine.google.com/tasks")
print("  2. Or use: ee.batch.Task.list()")
print("=" * 70)

