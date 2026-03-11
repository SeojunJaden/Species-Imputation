# Cleaning Pipeline

datacleaning.py removes potentailly invalid species observations.  
get-gee-rasters.py extracts environmental variables from GeoTIFFs at observation coordinates.  
combine-gee-and-obs.py adds google earth data to each species observation, and outputs the CSVs to `data/`.  
