"""
This Google Earth Engine Script exports environmental rasters 

This script exports 17 environmental layers (16 terrestrial + 1 ocean)

Area: San Diego coastal area from Oceanside to Mission Bay, containing all 4 ucsd natural reserves
Bounding Box: [-117.4, 32.7, -116.9, 33.2]
Collects data every 30 meters
Exports files as .tif
Destination: Google Drive folder 'sd_species_rasters' (and are in google-earth-data folder)
"""

import ee

ee.Authenticate()
ee.Initialize(project='species-imputation')

# San Diego coastal area bbox
bbox = ee.Geometry.Rectangle([-117.4, 32.7, -116.9, 33.2])

print()
print("Google Earth Engine Raster Export for San Diego Coastal Reserves")
print()
print(f"Bounding Box: [-117.4, 32.7, -116.9, 33.2]")
print(f"Export folder: 'sd_species_rasters'")
print(f"Resolution: 30 meters")
print()


def mask_clouds_landsat8(image):
    """Mask clouds and cloud shadows using QA_PIXEL band."""
    qa = image.select('QA_PIXEL')
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(
                 qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(cloud_mask)


def calculate_ndvi(image):
    """NDVI from Landsat 8: (NIR - Red) / (NIR + Red), SR_B5=NIR, SR_B4=Red."""
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    return image.addBands(ndvi)


def export_raster(image, description, folder='sd_species_rasters',
                  scale=30, region=None, crs='EPSG:4326',
                  maxPixels=1e13, fileFormat='GeoTIFF'):
    """Start a GEE export task to Google Drive."""
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


print("TERRESTRIAL LAYERS")
print()

# 1. Elevation
print("\n1. Processing elevation...")
try:
    elevation = ee.Image('USGS/SRTMGL1_003').clip(bbox)
    export_raster(elevation, 'elevation', region=bbox)
except Exception as e:
    print(f"✗ Error loading elevation dataset: {str(e)}")

# 2. Slope
print("\n2. Processing slope...")
try:
    elevation_for_slope = ee.Image('USGS/SRTMGL1_003')
    slope = ee.Terrain.slope(elevation_for_slope).clip(bbox)
    export_raster(slope, 'slope', region=bbox)
except Exception as e:
    print(f"✗ Error calculating slope: {str(e)}")

# 3. Aspect
print("\n3. Processing aspect...")
try:
    elevation_for_aspect = ee.Image('USGS/SRTMGL1_003')
    aspect = ee.Terrain.aspect(elevation_for_aspect).clip(bbox)
    export_raster(aspect, 'aspect', region=bbox)
except Exception as e:
    print(f"✗ Error calculating aspect: {str(e)}")

# 4. Landcover
print("\n4. Processing landcover...")
try:
    landcover = ee.Image('USGS/NLCD_RELEASES/2021_REL/NLCD/2021').select('landcover').clip(bbox)
    export_raster(landcover, 'landcover', region=bbox)
except Exception as e:
    print(f"✗ Error loading landcover dataset: {str(e)}")

# 5. NDVI
print("\n5. Processing NDVI...")
try:
    landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterDate('2020-01-01', '2024-12-31') \
        .filterBounds(bbox) \
        .map(mask_clouds_landsat8)
    landsat8_ndvi = landsat8.map(calculate_ndvi)
    ndvi_median = landsat8_ndvi.select('NDVI').median().clip(bbox)
    export_raster(ndvi_median, 'ndvi', region=bbox)
except Exception as e:
    print(f"✗ Error processing NDVI: {str(e)}")

# 6. Canopy cover
print("\n6. Processing canopy_cover...")
try:
    canopy_cover = ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2019') \
        .select('percent_tree_cover') \
        .clip(bbox)
    export_raster(canopy_cover, 'canopy_cover', region=bbox)
except Exception as e:
    print(f"✗ Error loading canopy_cover dataset: {str(e)}")

# 7. Impervious
print("\n7. Processing impervious...")
try:
    impervious = ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2019') \
        .select('impervious') \
        .clip(bbox)
    export_raster(impervious, 'impervious', region=bbox)
except Exception as e:
    print(f"✗ Error loading impervious dataset: {str(e)}")

# 8. Soil clay
print("\n8. Processing soil_clay...")
try:
    soil_clay = ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02').select('b0').clip(bbox)
    export_raster(soil_clay, 'soil_clay', region=bbox)
except Exception as e:
    print(f"✗ Error loading soil_clay dataset: {str(e)}")

# 9. Soil sand
print("\n9. Processing soil_sand...")
try:
    soil_sand = ee.Image('OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02').select('b0').clip(bbox)
    export_raster(soil_sand, 'soil_sand', region=bbox)
except Exception as e:
    print(f"✗ Error loading soil_sand dataset: {str(e)}")

# 10. Soil pH
print("\n10. Processing soil_ph...")
try:
    soil_ph = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02').select('b0').clip(bbox)
    export_raster(soil_ph, 'soil_ph', region=bbox)
except Exception as e:
    print(f"✗ Error loading soil_ph dataset: {str(e)}")

print()
print("OCEAN LAYER")
print()

# 11. Bathymetry
print("\n11. Processing bathymetry...")
try:
    bathymetry = ee.Image('NOAA/NGDC/ETOPO1').select('bedrock').clip(bbox)
    export_raster(bathymetry, 'bathymetry', region=bbox)
except Exception as e:
    print(f"✗ Error loading bathymetry dataset: {str(e)}")

print()
print("ADDITIONAL TERRESTRIAL LAYERS")
print()

# 12. Temperature mean
print("\n12. Processing temperature_mean...")
try:
    terraclimate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE') \
        .filterDate('2015-01-01', '2024-12-31') \
        .filterBounds(bbox) \
        .select('tmmn')
    temperature_mean = terraclimate.mean().divide(10).clip(bbox)
    export_raster(temperature_mean, 'temperature_mean', region=bbox)
except Exception as e:
    print(f"✗ Error processing temperature_mean: {str(e)}")

# 13. Precipitation annual
print("\n13. Processing precipitation_annual...")
try:
    terraclimate_pr = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE') \
        .filterDate('2015-01-01', '2024-12-31') \
        .filterBounds(bbox) \
        .select('pr')
    precipitation_annual = terraclimate_pr.sum().toDouble().clip(bbox)
    export_raster(precipitation_annual, 'precipitation_annual', region=bbox)
except Exception as e:
    print(f"✗ Error processing precipitation_annual: {str(e)}")

# 14. Temperature seasonality
print("\n14. Processing temperature_seasonality...")
try:
    terraclimate_tmmn = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE') \
        .filterDate('2015-01-01', '2024-12-31') \
        .filterBounds(bbox) \
        .select('tmmn')
    temperature_seasonality = terraclimate_tmmn.reduce(ee.Reducer.stdDev()).divide(10).clip(bbox)
    export_raster(temperature_seasonality, 'temperature_seasonality', region=bbox)
except Exception as e:
    print(f"✗ Error processing temperature_seasonality: {str(e)}")

# 15. NDVI seasonality
print("\n15. Processing ndvi_seasonality...")
try:
    landsat8_seasonality = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterDate('2020-01-01', '2024-12-31') \
        .filterBounds(bbox) \
        .map(mask_clouds_landsat8) \
        .map(calculate_ndvi)
    ndvi_max = landsat8_seasonality.select('NDVI').max()
    ndvi_min = landsat8_seasonality.select('NDVI').min()
    ndvi_seasonality = ndvi_max.subtract(ndvi_min).clip(bbox)
    export_raster(ndvi_seasonality, 'ndvi_seasonality', region=bbox)
except Exception as e:
    print(f"✗ Error processing ndvi_seasonality: {str(e)}")

# 16. Distance to coast
print("\n16. Processing distance_to_coast...")
try:
    bathymetry_coast = ee.Image('NOAA/NGDC/ETOPO1').select('bedrock')
    coast_mask = bathymetry_coast.gte(-2).And(bathymetry_coast.lte(2)).selfMask()
    scale_m = 30
    distance_pixels = coast_mask.fastDistanceTransform(neighborhood=256, metric='squared_euclidean').sqrt()
    distance_to_coast = distance_pixels.multiply(scale_m).clip(bbox)
    export_raster(distance_to_coast, 'distance_to_coast', region=bbox)
except Exception as e:
    print(f"✗ Error processing distance_to_coast: {str(e)}")

# 17. Distance to water
print("\n17. Processing distance_to_water...")
try:
    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('max_extent')
    water_mask = gsw.gt(0).selfMask()
    scale_m = 30
    distance_pixels_water = water_mask.fastDistanceTransform(neighborhood=256, metric='squared_euclidean').sqrt()
    distance_to_water = distance_pixels_water.multiply(scale_m).clip(bbox)
    export_raster(distance_to_water, 'distance_to_water', region=bbox)
except Exception as e:
    print(f"✗ Error processing distance_to_water: {str(e)}")

print()
print("EXPORT SUMMARY")
print()
print("All export tasks have been submitted to Google Earth Engine.")
print("Exports will appear in Google Drive folder: 'sd_species_rasters'")
print()
print("Total layers exported: 17")
print("  - Terrestrial: 16 layers")
print("  - Ocean: 1 layer")
print()
print("To check export status:")
print("  1. Visit: https://code.earthengine.google.com/tasks")
print("  2. Or use: ee.batch.Task.list()")
print()
