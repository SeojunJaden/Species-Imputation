#!/usr/bin/env python3
"""
Turn the cleaned v2 predictions into a field list Kellie can actually work from.

Decisions behind this (Kylan, Sep 22):
  - ~25 species per reserve, GROUP-BALANCED rather than straight top-by-probability,
    so the list isn't 20 plants and nothing else;
  - introduced/ornamental species stay in but are FLAGGED, so the judgement of what
    is worth field time stays with Kellie;
  - Scripps / Elliott / Dawson go out now; Kendall-Frost follows after the raster
    re-export.

Our advisor asked (Sep 25) that clearly non-native species be omitted; unknown
origin is fine to keep. Reserves with omit_introduced=True hold introduced species out
of the short list (they stay in the appendix, flagged). Kendall-Frost had it from the
start (Sep 26); since v2.2 (Oct 3) all four reserves do.

`manual_introduced` covers species iNat gives no California status for but that are
clearly non-native (Kylan, Sep 26: Atriplex lindleyi, an Australian saltbush). They are
labelled "introduced (manual)" so the override stays visible.

The most useful column for fieldwork is `nearest_record_m`: how far outside the
reserve boundary the closest known record of that species sits. A species recorded
80 m over the line is a different proposition from one only known 2 km away. It is
computed from the padded-bbox pull, which extends ~200 m beyond each boundary.

"Introduced" is iNaturalist's establishment means for California (place 14), not our
own judgement -- cached in RefreshedData/2026-09-14/establishment.json.

Usage:
  python CleanedData/build_field_list.py [Reserve ...]

Outputs:
  CleanedData/FieldLists_v2.2/<Reserve>_Field_List.csv     (the ~25)
  CleanedData/FieldLists_v2.2/<Reserve>_All_Imputed.csv    (the full appendix)
"""

import json
import math
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED = os.path.join(ROOT, "CleanedData")
V2 = os.path.join(ROOT, "RefreshedData", "2026-09-14")
RUN = "v2.2"                     # audit fixes + out-of-fold scoring (METHODOLOGY 6)
IN_DIR = os.path.join(CLEANED, f"FinalPredictions_{RUN}_Clean")
OUT_DIR = os.path.join(CLEANED, f"FieldLists_{RUN}")
EST_CACHE = os.path.join(V2, "establishment.json")
RESERVE_ZIP = os.path.join(ROOT, "ReserveExtents.zip")

CALIFORNIA_PLACE_ID = 14
TARGET_N = 25
MAX_PER_GROUP = 10
UA = "Species-Imputation/1.0 (UC NRS reserve species imputation; research use)"

RESERVES = {
    # Scripps is UPLAND ONLY -- the marine conservation area is out of scope
    # (METHODOLOGY 3; agreed with our advisor Jul 27 on the condition that the
    # marine portion of SCR is omitted). Torrey Pines contributes plenty of intertidal records, so without
    # this the short list fills up with limpets and a great white shark.
    "Scripps": dict(label="Scripps Coastal Reserve (upland)", partner="TorreyPines",
                    raw="ScrippsData.csv", kml="Scripps Coastal Reserve Extent.kml",
                    polygon="unofficial upland extent",
                    out_of_scope_groups=["Mollusca", "Actinopterygii", "Animalia"],
                    omit_introduced=True),
    "ElliottChaparral": dict(label="Elliott Chaparral Reserve", partner="MissionTrails",
                             raw="ElliottChaparralData.csv", kml="Elliott extent.kml",
                             polygon="Unofficial extent", omit_introduced=True),
    "LosMonos": dict(label="Dawson Los Monos Canyon Reserve", partner="BuenaVista",
                     raw="LosMonosCanyonData.csv", kml="Dawson Extent.kml",
                     polygon="unofficial extent", omit_introduced=True),
    "MissionBay": dict(label="Kendall-Frost Mission Bay Marsh Reserve",
                       partner="SweetwaterMarsh", raw="MissionBayData.csv",
                       kml="Kendall Frost Extent.kml", polygon="Unofficial extent",
                       omit_introduced=True,
                       manual_introduced=["Atriplex lindleyi"]),
}


# ------------------------------------------------------------------ geometry
def _strip(tag):
    return tag.split("}")[-1]


def parse_polygons(kml_bytes):
    root = ET.fromstring(kml_bytes)
    out = {}
    for pm in root.iter():
        if _strip(pm.tag) != "Placemark":
            continue
        name = None
        for ch in pm:
            if _strip(ch.tag) == "name":
                name = (ch.text or "").strip()
        rings = []
        for poly in pm.iter():
            if _strip(poly.tag) != "Polygon":
                continue
            for ob in poly.iter():
                if _strip(ob.tag) != "outerBoundaryIs":
                    continue
                for c in ob.iter():
                    if _strip(c.tag) == "coordinates" and c.text:
                        ring = [tuple(map(float, t.split(",")[:2]))
                                for t in c.text.split() if len(t.split(",")) >= 2]
                        if ring:
                            rings.append(ring)
        if name and rings:
            out[name] = rings
    return out


def project(rings):
    """Equirectangular metres about the polygon centroid -- fine at reserve scale."""
    lats = [p[1] for r in rings for p in r]
    lat0 = sum(lats) / len(lats)
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110540.0
    return lambda lon, lat: (lon * kx, lat * ky)


def point_segment_m(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_boundary_m(lon, lat, rings, proj):
    px, py = proj(lon, lat)
    best = float("inf")
    for ring in rings:
        pts = [proj(*p) for p in ring]
        for i in range(len(pts) - 1):
            best = min(best, point_segment_m(px, py, *pts[i], *pts[i + 1]))
    return best


# ------------------------------------------------------- establishment means
def fetch_establishment(taxon_ids):
    cache = {}
    if os.path.exists(EST_CACHE):
        with open(EST_CACHE) as f:
            cache = json.load(f)
    todo = sorted({int(t) for t in taxon_ids if str(int(t)) not in cache})
    if todo:
        print(f"[*] establishment means: fetching {len(todo):,} taxa "
              f"({-(-len(todo)//30)} requests)")
        for n in range(0, len(todo), 30):
            chunk = todo[n:n + 30]
            url = (f"https://api.inaturalist.org/v1/taxa/{','.join(map(str, chunk))}"
                   f"?preferred_place_id={CALIFORNIA_PLACE_ID}")
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            ok = False
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        for t in json.load(r).get("results", []):
                            em = t.get("establishment_means")
                            if isinstance(em, dict):
                                em = em.get("establishment_means")
                            cache[str(t["id"])] = em
                    ok = True
                    break
                except Exception as e:
                    print(f"    batch retry {attempt + 1} ({e})")
                    time.sleep(2 ** attempt)
            if ok:
                # Only record "no establishment means" for ids the API actually
                # answered for. Caching a failed batch as None would make the gap
                # permanent -- it silently cost Elliott its introduced flags once.
                for i in chunk:
                    cache.setdefault(str(i), None)
            else:
                print(f"    giving up on {len(chunk)} ids -- will retry next run")
            time.sleep(1.0)
        with open(EST_CACHE, "w") as f:
            json.dump(cache, f)
    return cache


# ------------------------------------------------------------------ helpers
def name_matches(series, sp):
    """A species plus anything infraspecific beneath it (merged rows)."""
    return (series == sp) | series.str.startswith(sp + " ", na=False)


def balanced_pick(df, n=TARGET_N, cap=MAX_PER_GROUP):
    """
    Allocate n slots across taxon groups in proportion to each group's share of
    the imputed list (largest remainder), capped so no group dominates, then fill
    each group's slots by probability.
    """
    groups = df["iconic_taxon_name"].fillna("Unknown")
    sizes = groups.value_counts()
    total = sizes.sum()
    raw = {g: n * c / total for g, c in sizes.items()}
    quota = {g: min(int(v), cap, sizes[g]) for g, v in raw.items()}
    # hand out the remaining slots by largest fractional part
    while sum(quota.values()) < min(n, len(df)):
        order = sorted(raw, key=lambda g: (raw[g] - quota[g]), reverse=True)
        progressed = False
        for g in order:
            if quota[g] < min(cap, sizes[g]):
                quota[g] += 1
                progressed = True
                break
        if not progressed:
            break
    picks = []
    for g, k in quota.items():
        if k:
            picks.append(df[groups == g].nlargest(k, "probability_of_presence"))
    out = pd.concat(picks) if picks else df.head(0)
    return out.sort_values("probability_of_presence", ascending=False)


def process(key):
    cfg = RESERVES[key]
    src = os.path.join(IN_DIR, f"{key}_Final_Predictions.csv")
    if not os.path.exists(src):
        print(f"[skip] {key}: no cleaned predictions yet")
        return None

    df = pd.read_csv(src)
    imp = df[df["Prediction_Result"] == "False Positive (Imputed/Missing)"].copy()
    print(f"\n[*] {cfg['label']}: {len(imp):,} imputed species")

    # --- how recently, and how well, the partner knows each species ----------
    partner = pd.read_csv(os.path.join(V2, "filtered", f"{cfg['partner']}_Filtered.csv"),
                          usecols=["scientific_name", "observed_on"], low_memory=False)
    partner["observed_on"] = pd.to_datetime(partner["observed_on"], errors="coerce")
    # Match infraspecific names too: a merged species' partner records may all be
    # filed under a subspecies (Sweetwater's Crotalus oreganus are all C. o. helleri).
    last_seen = partner.groupby("scientific_name")["observed_on"].max()
    imp["partner_last_seen"] = pd.to_datetime(imp["scientific_name"].map(
        lambda s: last_seen[name_matches(last_seen.index.to_series(), s)].max())
    ).dt.strftime("%Y-%m-%d")

    # --- nearest known record outside the boundary ---------------------------
    with zipfile.ZipFile(RESERVE_ZIP) as z:
        member = next(n for n in z.namelist() if os.path.basename(n) == cfg["kml"])
        rings = parse_polygons(z.read(member))[cfg["polygon"]]
    proj = project(rings)

    raw = pd.read_csv(os.path.join(V2, "raw", cfg["raw"]),
                      usecols=["scientific_name", "latitude", "longitude"],
                      low_memory=False).dropna(subset=["latitude", "longitude"])

    nearest = []
    for sp in imp["scientific_name"]:
        hits = raw[name_matches(raw["scientific_name"], sp)]
        if len(hits) == 0:
            nearest.append((None, 0))
            continue
        d = min(distance_to_boundary_m(lo, la, rings, proj)
                for lo, la in zip(hits["longitude"], hits["latitude"]))
        nearest.append((round(d), len(hits)))
    imp["nearest_record_m"] = [n[0] for n in nearest]
    imp["records_just_outside"] = [n[1] for n in nearest]

    # --- introduced flag -----------------------------------------------------
    est = fetch_establishment(imp["taxon_id"].dropna())
    imp["establishment_ca"] = imp["taxon_id"].map(
        lambda t: est.get(str(int(t))) if pd.notna(t) else None)
    manual = imp["scientific_name"].isin(cfg.get("manual_introduced", []))
    imp.loc[manual & imp["establishment_ca"].isna(), "establishment_ca"] = "introduced (manual)"
    imp["likely_planted"] = imp["establishment_ca"].isin(["introduced", "introduced (manual)"])

    # Taxa outside the reserve's scope stay in the appendix, marked, but are
    # kept out of the list Kellie is asked to walk.
    out_groups = cfg.get("out_of_scope_groups", [])
    imp["out_of_scope"] = imp["iconic_taxon_name"].isin(out_groups)

    cols = ["scientific_name", "common_name", "iconic_taxon_name", "rank",
            "probability_of_presence", "backup_obs_count", "partner_last_seen",
            "nearest_record_m", "records_just_outside", "establishment_ca",
            "likely_planted", "out_of_scope", "taxon_id"]
    imp = imp[cols].rename(columns={"backup_obs_count": "partner_records"})
    imp["probability_of_presence"] = imp["probability_of_presence"].round(3)

    os.makedirs(OUT_DIR, exist_ok=True)
    imp.sort_values("probability_of_presence", ascending=False).to_csv(
        os.path.join(OUT_DIR, f"{key}_All_Imputed.csv"), index=False)

    in_scope = imp[~imp["out_of_scope"]]
    if len(out_groups):
        print(f"    {int(imp['out_of_scope'].sum())} out-of-scope ({', '.join(out_groups)}) "
              f"held back from the short list")
    if cfg.get("omit_introduced"):
        n_held = int(in_scope["likely_planted"].sum())
        in_scope = in_scope[~in_scope["likely_planted"]]
        print(f"    {n_held} introduced (iNat, California) held back from the short list")
    short = balanced_pick(in_scope)
    short.to_csv(os.path.join(OUT_DIR, f"{key}_Field_List.csv"), index=False)

    mix = short["iconic_taxon_name"].value_counts().to_dict()
    n_intro = int(short["likely_planted"].sum())
    n_close = int(short["nearest_record_m"].notna().sum())
    print(f"    field list: {len(short)} species  {mix}")
    print(f"    {n_intro} flagged introduced; {n_close} have a record within the padded bbox")
    return dict(key=key, label=cfg["label"], total=len(imp), short=short, mix=mix,
                introduced=n_intro)


def main():
    targets = [a for a in sys.argv[1:] if not a.startswith("--")] or list(RESERVES)
    results = [r for r in (process(k) for k in targets if k in RESERVES) if r]
    if results:
        print(f"\n[+] {os.path.relpath(OUT_DIR, ROOT)}/")
    return results


if __name__ == "__main__":
    main()
