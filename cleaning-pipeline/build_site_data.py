#!/usr/bin/env python3
"""
Build the website's data: the species recorded at each reserve on iNaturalist.

Reads the Sep 14 2026 API pull (RefreshedData/2026-09-14/raw/), keeps the
observations inside each reserve's unofficial boundary (ReserveExtents.zip), and
writes one JSON file the frontend reads at build time.

What goes on the public site, and what does not:
  - observed species only. Model predictions stay off the site until our UC NRS
    advisor has reviewed them (Kylan, Oct 3); the frontend shows a "pending
    review" panel in their place. `predictions_status` in the output says so;
  - observations whose coordinates iNaturalist obscures (threatened species, or
    an observer's own privacy setting) are dropped -- their true location is
    unknown, and a species list should not help anyone find a sensitive taxon;
  - species rank only. Subspecies, varieties and forms roll up to their species
    (taxonomy.json); records identified only to genus or coarser are left out.

Photos come from iNaturalist: the taxon's default photo when it carries a Creative
Commons licence, otherwise the first CC-licensed photo among the taxon's other
photos. The attribution is stored with it and shown on the site.
They are cached in RefreshedData/2026-09-14/taxon_photos.json.

Usage:
  python cleaning-pipeline/build_site_data.py

Output:
  frontend/data/reserves.json
"""

import json
import math
import os
import sys
import time
import urllib.request
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "CleanedData"))
import run_v2_analysis as r  # noqa: E402  (KML parsing + point-in-polygon)

V2 = os.path.join(ROOT, "RefreshedData", "2026-09-14")
PHOTO_CACHE = os.path.join(V2, "taxon_photos.json")
OUT = os.path.join(ROOT, "frontend", "data", "reserves.json")
DATA_THROUGH = "2026-09-14"
UA = "Species-Imputation/1.0 (UC NRS reserve species explorer; research use)"
CC_LICENSES = {"cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa", "cc-by-nd", "cc-by-nc-nd"}
INFRASPECIFIC = {"subspecies", "variety", "form"}

# Order the site lists groups in: animals first (Kylan, Oct 3).
GROUP_ORDER = ["Aves", "Mammalia", "Reptilia", "Amphibia", "Insecta", "Arachnida",
               "Mollusca", "Actinopterygii", "Animalia", "Plantae", "Fungi",
               "Chromista", "Protozoa", "Unknown"]

RESERVES = [
    dict(slug="scripps", name="Scripps Coastal Reserve", short="Scripps",
         raw="ScrippsData.csv", kml="Scripps Coastal Reserve Extent.kml",
         polygon="unofficial upland extent", color="#4f7d5c",
         setting="La Jolla",
         habitats=["Coastal bluffs", "Coastal sage scrub", "Canyon"],
         blurb="Sea-cliff bluffs and canyon above the Pacific, next to the UC San Diego "
               "campus. This explorer covers the upland part of the reserve, where coastal "
               "sage scrub meets the edge of the continent."),
    dict(slug="elliott", name="Elliott Chaparral Reserve", short="Elliott",
         raw="ElliottChaparralData.csv", kml="Elliott extent.kml",
         polygon="Unofficial extent", color="#b5793a",
         setting="Northern San Diego",
         habitats=["Chaparral", "Coastal sage scrub"],
         blurb="Rolling mesa country of dense chaparral and sage scrub on the north side "
               "of San Diego, one of the inland shrublands that once covered much of the "
               "county."),
    dict(slug="dawson", name="Dawson Los Monos Canyon Reserve", short="Dawson",
         raw="LosMonosCanyonData.csv", kml="Dawson Extent.kml",
         polygon="unofficial extent", color="#7a6a9e",
         setting="Vista",
         habitats=["Riparian woodland", "Coastal sage scrub", "Canyon"],
         blurb="A steep canyon along Agua Hedionda Creek in North County, where a shaded "
               "streamside woodland runs between slopes of sage scrub."),
    dict(slug="kendall-frost", name="Kendall-Frost Mission Bay Marsh Reserve",
         short="Kendall-Frost", raw="MissionBayData.csv", kml="Kendall Frost Extent.kml",
         polygon="Unofficial extent", color="#3f7f8f",
         setting="Mission Bay",
         habitats=["Salt marsh", "Mudflat", "Tidal channels"],
         blurb="A remnant of the salt marsh that once ringed Mission Bay. Tides wash "
               "through pickleweed and cordgrass, and shorebirds work the mudflats."),
]


def polygon_area_ha(ring):
    """Shoelace area in a local equirectangular projection -- fine at reserve scale."""
    lat0 = sum(p[1] for p in ring) / len(ring)
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    pts = [(lon * kx, lat * ky) for lon, lat in ring]
    a = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))
    return abs(a) / 2 / 10_000


def pick_photo(taxon):
    """Default photo if CC-licensed, else the first CC-licensed taxon photo."""
    candidates = [taxon.get("default_photo") or {}] + \
        [tp.get("photo") or {} for tp in taxon.get("taxon_photos") or []]
    for ph in candidates:
        lic = (ph.get("license_code") or "").lower()
        if lic in CC_LICENSES and ph.get("medium_url"):
            return dict(url=ph["medium_url"], attribution=ph.get("attribution"), license=lic)
    return {"none": True}   # checked, nothing usable -- not re-fetched


def fetch_photos(taxon_ids):
    cache = {}
    if os.path.exists(PHOTO_CACHE):
        with open(PHOTO_CACHE) as f:
            # Entries cached as null predate the taxon_photos fallback; re-check them.
            cache = {k: v for k, v in json.load(f).items() if v is not None}
    todo = sorted({int(t) for t in taxon_ids if str(int(t)) not in cache})
    if todo:
        print(f"[*] photos: fetching {len(todo):,} taxa ({-(-len(todo) // 30)} requests)")
    for n in range(0, len(todo), 30):
        chunk = todo[n:n + 30]
        req = urllib.request.Request(
            f"https://api.inaturalist.org/v1/taxa/{','.join(map(str, chunk))}",
            headers={"User-Agent": UA})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    for t in json.load(resp).get("results", []):
                        cache[str(t["id"])] = pick_photo(t)
                # Answered but absent from results: record as no usable photo.
                for i in chunk:
                    cache.setdefault(str(i), {"none": True})
                break
            except Exception as e:  # retry; a failed batch is never cached
                print(f"    retry {attempt + 1} ({e})")
                time.sleep(2 ** attempt)
        time.sleep(1.0)
        if n and n % 600 == 0:
            with open(PHOTO_CACHE, "w") as f:
                json.dump(cache, f)
    with open(PHOTO_CACHE, "w") as f:
        json.dump(cache, f)
    return cache


def to_species(df, taxa):
    """Map each record to its species (taxon id, name); None if coarser than species."""
    sp_id, sp_name = [], []
    for tid in df["taxon_id"]:
        rec = taxa.get(int(tid)) if pd.notna(tid) else None
        if rec and rec.get("rank") in INFRASPECIFIC:
            parent = taxa.get(rec.get("parent_id"))
            if parent and parent.get("rank") == "species":
                sp_id.append(rec["parent_id"]); sp_name.append(parent["name"])
                continue
        if rec and rec.get("rank_level") == 10:
            sp_id.append(int(tid)); sp_name.append(rec["name"])
        else:
            sp_id.append(None); sp_name.append(None)
    return sp_id, sp_name


def main():
    with open(os.path.join(V2, "taxonomy.json")) as f:
        taxa = {int(k): v for k, v in json.load(f).items() if v}
    zf = zipfile.ZipFile(os.path.join(ROOT, "ReserveExtents.zip"))
    members = {os.path.basename(n): n for n in zf.namelist() if n.endswith(".kml")}

    built = []
    for cfg in RESERVES:
        rings = r.parse_polygons(zf.read(members[cfg["kml"]]))[cfg["polygon"]]
        raw = pd.read_csv(os.path.join(V2, "raw", cfg["raw"]), low_memory=False,
                          usecols=["id", "observed_on", "latitude", "longitude", "user_id",
                                   "scientific_name", "common_name", "iconic_taxon_name",
                                   "taxon_id", "coordinates_obscured"])
        raw = raw.dropna(subset=["latitude", "longitude", "taxon_id"])
        inside = r.filter_to_polygon(raw, rings)
        obscured = inside["coordinates_obscured"].astype(str).str.lower().eq("true")
        obs = inside[~obscured].copy()
        obs["species_id"], obs["species"] = to_species(obs, taxa)
        coarse = int(obs["species_id"].isna().sum())
        obs = obs.dropna(subset=["species_id"])
        obs["observed_on"] = pd.to_datetime(obs["observed_on"], errors="coerce")

        species = []
        for sid, g in obs.groupby("species_id"):
            sid = int(sid)
            rec = taxa.get(sid, {})
            months = [0] * 12
            for m in g["observed_on"].dropna().dt.month:
                months[int(m) - 1] += 1
            group = g["iconic_taxon_name"].mode()
            species.append(dict(
                id=sid, name=g["species"].iloc[0],
                common=rec.get("common_name") or (g["common_name"].dropna().mode().iloc[0]
                                                  if g["common_name"].notna().any() else None),
                group=group.iloc[0] if len(group) else "Unknown",
                observations=len(g), months=months,
                lastSeen=g["observed_on"].max().strftime("%Y-%m-%d")
                if g["observed_on"].notna().any() else None))
        print(f"[*] {cfg['short']}: {len(inside):,} records in boundary, {int(obscured.sum())} "
              f"obscured dropped, {coarse:,} coarser than species, {len(species):,} species")
        built.append((cfg, rings, obs, species))

    photos = fetch_photos({s["id"] for _, _, _, sp in built for s in sp})

    out = dict(dataThrough=DATA_THROUGH, predictionsStatus="pending", groupOrder=GROUP_ORDER,
               reserves=[])
    for cfg, rings, obs, species in built:
        for s in species:
            ph = photos.get(str(s["id"]))
            s["photo"] = ph if ph and not ph.get("none") else None
        species.sort(key=lambda s: (-s["observations"], s["name"]))
        groups = {}
        for s in species:
            groups[s["group"]] = groups.get(s["group"], 0) + 1
        ring = max(rings, key=len)
        out["reserves"].append(dict(
            slug=cfg["slug"], name=cfg["name"], shortName=cfg["short"], color=cfg["color"],
            setting=cfg["setting"], habitats=cfg["habitats"], description=cfg["blurb"],
            boundary=[[round(lat, 6), round(lon, 6)] for lon, lat in ring],
            center=[round(sum(p[1] for p in ring) / len(ring), 6),
                    round(sum(p[0] for p in ring) / len(ring), 6)],
            areaHa=round(sum(polygon_area_ha(rg) for rg in rings)),
            stats=dict(observations=len(obs), species=len(species),
                       observers=int(obs["user_id"].nunique()),
                       firstYear=int(obs["observed_on"].dt.year.min())),
            groups=dict(sorted(groups.items(), key=lambda kv: GROUP_ORDER.index(kv[0])
                               if kv[0] in GROUP_ORDER else 99)),
            species=species, predicted=[]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    n_photo = sum(1 for res in out["reserves"] for s in res["species"] if s["photo"])
    n_sp = sum(len(res["species"]) for res in out["reserves"])
    print(f"[+] {os.path.relpath(OUT, ROOT)}  ({os.path.getsize(OUT) / 1e6:.2f} MB; "
          f"{n_photo:,} of {n_sp:,} species entries have a CC-licensed photo)")


if __name__ == "__main__":
    main()
