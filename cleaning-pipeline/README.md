# Cleaning Pipeline

Extract environmental variables from GeoTIFFs at observation coordinates. Writes enriched CSVs to `data/`.

## Usage

```bash
python combine-gee-and-obs.py <input.csv> [output.csv]
```

If output is omitted, writes to `data/{input_basename}_with_env_data.csv`.

Example:
```bash
python combine-gee-and-obs.py data/first-200k-obs-cleaned.csv data/first-200k-obs-clean-with-env.csv
```

## Paths

- **Input:** Any CSV with latitude, longitude, taxon_id, and scientific_name or common_name
- **Output:** `data/` (project root)
- **Rasters:** `GEE Data/drive-download-20260217T221411Z-1-001/` (project root)
