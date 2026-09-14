#!/usr/bin/env python3
"""
Refresh the per-reserve iNaturalist observation pulls through the public iNat API
(no account / login needed), replacing the manual website export behind
AllSpeciesRawData.zip (exported Jan 27 - Feb 7 2026).

Two steps:
  python cleaning-pipeline/pull_inat_api.py pull  [Reserve ...]   # API -> raw CSVs
  python cleaning-pipeline/pull_inat_api.py clean [Reserve ...]   # raw -> *_Filtered.csv

How each reserve is scoped:
  - TARGET reserves: every verifiable observation in a bounding box around Kellie
    Uyeda's unofficial-extent polygon (ReserveExtents.zip) plus a small buffer.
    The exact polygon filter is applied later by the modeling step, as before.
  - DONOR ("partner") reserves: the same iNaturalist place the original export used,
    identified by matching sampled observation IDs from the old files to place IDs.

Filter matches the original export: verifiable=true (research + needs_id grades;
casual/captive excluded). Raw CSVs keep the original export's column names.
"""

import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "CleanedData"))
from run_unofficial_extent_analysis import dedup_5min, parse_polygons  # noqa: E402

PULL_DATE = "2026-09-14"
OUT = os.path.join(ROOT, "RefreshedData", PULL_DATE)
RAW_DIR = os.path.join(OUT, "raw")
FILTERED_DIR = os.path.join(OUT, "filtered")
API = "https://api.inaturalist.org/v1/observations"
HEADERS = {"User-Agent": "species-imputation-research (UCSD student project)"}
BBOX_BUFFER_DEG = 0.002  # ~200 m around the polygon's bounding box

# name -> scope. Targets use Kellie's KML; donors use the original iNat place.
RESERVES = {
    "Scripps":          dict(role="target", raw="ScrippsData.csv",
                             kml="Scripps Coastal Reserve Extent.kml", polygon="unofficial upland extent"),
    "ElliottChaparral": dict(role="target", raw="ElliottChaparralData.csv",
                             kml="Elliott extent.kml", polygon="Unofficial extent"),
    "LosMonos":         dict(role="target", raw="LosMonosCanyonData.csv",
                             kml="Dawson Extent.kml", polygon="unofficial extent"),
    "MissionBay":       dict(role="target", raw="MissionBayData.csv",
                             kml="Kendall Frost Extent.kml", polygon="Unofficial extent"),
    "TorreyPines":      dict(role="donor", raw="TorreyPinesData.csv", place_id=4605),
    "MissionTrails":    dict(role="donor", raw="MissionTrailsData.csv", place_id=81859),
    "TijuanaRiver":     dict(role="donor", raw="TijuanaRiverData.csv", place_id=152893),
    # Buena Vista has no matching iNat place; the original export was a bounding box.
    # This box reproduces it: 4,277 verifiable obs created before the export vs 4,279 in the old file.
    "BuenaVista":       dict(role="donor", raw="BuenaVistaData.csv",
                             bbox=dict(swlat=33.1514, swlng=-117.2492, nelat=33.1606, nelng=-117.2416)),
}

RAW_COLUMNS = [
    "id", "uuid", "observed_on_string", "observed_on", "time_observed_at", "time_zone",
    "user_id", "user_login", "user_name", "created_at", "updated_at", "quality_grade",
    "license", "url", "image_url", "sound_url", "tag_list", "description",
    "num_identification_agreements", "num_identification_disagreements",
    "captive_cultivated", "oauth_application_id", "place_guess", "latitude", "longitude",
    "positional_accuracy", "private_place_guess", "private_latitude", "private_longitude",
    "public_positional_accuracy", "geoprivacy", "taxon_geoprivacy", "coordinates_obscured",
    "positioning_method", "positioning_device", "species_guess", "scientific_name",
    "common_name", "iconic_taxon_name", "taxon_id",
]


def api_get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            time.sleep(1.1)  # iNat asks for <= ~1 request/second
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            wait = min(10 * (attempt + 1), 60)
            print(f"      retry {attempt + 1} in {wait}s ({e})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"API request kept failing: {url}")


def to_utc_str(ts):
    """ISO timestamp with offset -> 'YYYY-MM-DD HH:MM:SS UTC' (the website export format)."""
    if not ts:
        return None
    return datetime.fromisoformat(ts).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def to_row(o):
    taxon = o.get("taxon") or {}
    user = o.get("user") or {}
    lat = lon = None
    if o.get("location"):
        lat, lon = (float(v) for v in o["location"].split(","))
    photos = o.get("photos") or []
    sounds = o.get("sounds") or []
    return {
        "id": o["id"],
        "uuid": o.get("uuid"),
        "observed_on_string": o.get("observed_on_string"),
        "observed_on": o.get("observed_on"),
        "time_observed_at": to_utc_str(o.get("time_observed_at")),
        "time_zone": o.get("observed_time_zone"),
        "user_id": user.get("id"),
        "user_login": user.get("login"),
        "user_name": user.get("name"),
        "created_at": to_utc_str(o.get("created_at")),
        "updated_at": to_utc_str(o.get("updated_at")),
        "quality_grade": o.get("quality_grade"),
        "license": (o.get("license_code") or "").upper() or None,
        "url": f"https://www.inaturalist.org/observations/{o['id']}",
        "image_url": photos[0]["url"].replace("square", "medium") if photos else None,
        "sound_url": sounds[0].get("file_url") if sounds else None,
        "tag_list": ", ".join(o.get("tags") or []) or None,
        "description": o.get("description") or None,
        "num_identification_agreements": o.get("num_identification_agreements"),
        "num_identification_disagreements": o.get("num_identification_disagreements"),
        "captive_cultivated": o.get("captive"),
        "oauth_application_id": o.get("oauth_application_id"),
        "place_guess": o.get("place_guess"),
        "latitude": lat,
        "longitude": lon,
        "positional_accuracy": o.get("positional_accuracy"),
        "private_place_guess": None,
        "private_latitude": None,
        "private_longitude": None,
        "public_positional_accuracy": o.get("public_positional_accuracy"),
        "geoprivacy": o.get("geoprivacy"),
        "taxon_geoprivacy": o.get("taxon_geoprivacy"),
        "coordinates_obscured": o.get("obscured"),
        "positioning_method": o.get("positioning_method"),
        "positioning_device": o.get("positioning_device"),
        "species_guess": o.get("species_guess"),
        "scientific_name": taxon.get("name"),
        "common_name": taxon.get("preferred_common_name"),
        "iconic_taxon_name": taxon.get("iconic_taxon_name"),
        "taxon_id": taxon.get("id"),
    }


def scope_params(name, cfg):
    if cfg["role"] == "donor":
        return {"place_id": cfg["place_id"]} if "place_id" in cfg else dict(cfg["bbox"])
    kz = zipfile.ZipFile(os.path.join(ROOT, "ReserveExtents.zip"))
    members = {os.path.basename(n): n for n in kz.namelist() if n.endswith(".kml")}
    rings = parse_polygons(kz.read(members[cfg["kml"]]))[cfg["polygon"]]
    lons = [p[0] for r in rings for p in r]
    lats = [p[1] for r in rings for p in r]
    b = BBOX_BUFFER_DEG
    return {"swlat": round(min(lats) - b, 5), "swlng": round(min(lons) - b, 5),
            "nelat": round(max(lats) + b, 5), "nelng": round(max(lons) + b, 5)}


def pull(name, cfg, manifest):
    scope = scope_params(name, cfg)
    base = dict(scope, verifiable="true", locale="en", order_by="id", order="asc", per_page=200)
    expected = api_get(dict(base, per_page=0))["total_results"]
    print(f"[*] {name} ({cfg['role']}) {scope}: {expected} observations expected", flush=True)

    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, cfg["raw"])
    n, id_above = 0, 0
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        w.writeheader()
        while True:
            # id_above paging avoids the API's 10,000-result cap on page-based paging
            results = api_get(dict(base, id_above=id_above))["results"]
            if not results:
                break
            for o in results:
                w.writerow(to_row(o))
            n += len(results)
            id_above = results[-1]["id"]
            if n % 5000 < 200:
                print(f"    {n}/{expected}", flush=True)
    print(f"    -> {path} ({n} rows)", flush=True)
    manifest[name] = dict(role=cfg["role"], scope=scope, filters="verifiable=true",
                          expected_total=expected, rows_written=n, file=os.path.relpath(path, ROOT),
                          pulled_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))


def clean(name, cfg):
    """Same steps as cleaning-pipeline/datacleaning.py: keep required fields, 5-min dedup."""
    raw = pd.read_csv(os.path.join(RAW_DIR, cfg["raw"]), low_memory=False)
    keep = ["id", "uuid", "observed_on", "time_observed_at", "time_zone", "user_id", "user_login",
            "user_name", "quality_grade", "license", "place_guess", "latitude", "longitude",
            "positional_accuracy", "scientific_name", "common_name", "iconic_taxon_name", "taxon_id"]
    df = raw[keep].dropna(subset=["latitude", "longitude", "taxon_id", "scientific_name"])
    out = dedup_5min(df)[keep]
    os.makedirs(FILTERED_DIR, exist_ok=True)
    path = os.path.join(FILTERED_DIR, f"{name}_Filtered.csv")
    out.to_csv(path, index=False)
    print(f"[*] {name}: {len(raw)} raw -> {len(out)} after 5-min dedup -> {path}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("pull", "clean"):
        raise SystemExit(__doc__)
    names = sys.argv[2:] or list(RESERVES)
    if sys.argv[1] == "clean":
        for name in names:
            clean(name, RESERVES[name])
        return
    manifest_path = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    for name in names:
        pull(name, RESERVES[name], manifest)
        os.makedirs(OUT, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
