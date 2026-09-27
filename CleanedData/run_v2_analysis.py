#!/usr/bin/env python3
"""
v2 re-run of the sister-reserve species-imputation model.

Successor to run_unofficial_extent_analysis.py. Same model, same unofficial
boundaries; what changes is the data and one pairing:

  - observations come from the Sep 14 2026 iNaturalist API pull
    (RefreshedData/2026-09-14/) instead of the Jan-Feb 2026 website export, so
    they cover the full unofficial boundaries and run to Sep 2026;
  - Kendall-Frost's partner is Sweetwater Marsh, not Tijuana River (Kellie,
    Aug 21 / Sep 14);
  - partner habitat values come from cleaning-pipeline/sample_env_v2.py, which
    marks out-of-raster observations empty instead of 0.0;
  - the pandas 3.0.2 fillna no-op is fixed (see fill_missing below).

A pair whose partner has no usable habitat data is SKIPPED rather than quietly
modelled on zeros -- that silent failure is what made Kendall-Frost
habitat-blind in v1. Pass --allow-habitat-blind to run it anyway.

Usage:
  python CleanedData/run_v2_analysis.py [Reserve ...] [--allow-habitat-blind]

Outputs:
  CleanedData/FinalPredictions_v2/<Reserve>_Final_Predictions.csv
  CleanedData/V2_Comparison.csv / .md      (v2 vs the v1 unofficial-extent run)
"""

import io
import json
import os
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESERVE_ZIP = os.path.join(ROOT, "ReserveExtents.zip")
CLEANED = os.path.join(ROOT, "CleanedData")
V2 = os.path.join(ROOT, "RefreshedData", "2026-09-14")
V2_RAW, V2_FILTERED, V2_ENV = (os.path.join(V2, d) for d in ("raw", "filtered", "env"))
OUT_DIR = os.path.join(CLEANED, "FinalPredictions_v2")
V1_DIR = os.path.join(CLEANED, "FinalPredictions_Unofficial")

# Kellie's hand-drawn Sweetwater boundary is gitignored and lives loose at the
# repo root -- deliberately NOT inside the tracked ReserveExtents.zip.
SWEETWATER_KML = os.path.join(ROOT, "Sweetwater Marsh.kml")

PAIRS = [
    dict(target="Scripps", raw="ScrippsData.csv",
         kml="Scripps Coastal Reserve Extent.kml", unofficial="unofficial upland extent",
         partner="TorreyPines", partner_label="Torrey Pines State Reserve"),
    dict(target="ElliottChaparral", raw="ElliottChaparralData.csv",
         kml="Elliott extent.kml", unofficial="Unofficial extent",
         partner="MissionTrails", partner_label="Mission Trails Regional Park"),
    dict(target="LosMonos", raw="LosMonosCanyonData.csv",
         kml="Dawson Extent.kml", unofficial="unofficial extent",
         partner="BuenaVista", partner_label="Buena Vista Park"),
    dict(target="MissionBay", raw="MissionBayData.csv",  # Kendall-Frost
         kml="Kendall Frost Extent.kml", unofficial="Unofficial extent",
         partner="SweetwaterMarsh", partner_label="Sweetwater Marsh",
         # the partner itself is scoped by a polygon, not by an iNat place
         partner_kml=SWEETWATER_KML, partner_polygon="Sweetwater polygon unofficial"),
]

UNDERREPRESENTED_THRESHOLD = 3
MIN_PARTNER_OBS = 10          # imputed species below this are demoted to noise
ENV_FEATURES = ["avg_elevation", "avg_slope", "avg_ndvi",
                "avg_soil_sand", "avg_soil_ph", "avg_soil_clay"]


# ---------------------------------------------------------------- KML parsing
def _strip(tag):
    return tag.split("}")[-1]


def parse_polygons(kml_bytes):
    """name -> list of outer rings; each ring = list of (lon, lat)."""
    import xml.etree.ElementTree as ET
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
                        ring = []
                        for trip in c.text.split():
                            p = trip.split(",")
                            if len(p) >= 2:
                                ring.append((float(p[0]), float(p[1])))
                        if ring:
                            rings.append(ring)
        if name and rings:
            out[name] = rings
    return out


def point_in_ring(x, y, ring):
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def in_polygon(x, y, rings):
    return any(point_in_ring(x, y, r) for r in rings)


def filter_to_polygon(df, rings):
    mask = [in_polygon(lon, lat, rings)
            for lon, lat in zip(df["longitude"].values, df["latitude"].values)]
    return df[pd.Series(mask, index=df.index)]


# ------------------------------------------------ 5-min dedup (datacleaning.py)
def dedup_5min(df):
    df = df.dropna(subset=["scientific_name", "latitude", "longitude", "time_observed_at"]).copy()
    df["time_observed_at"] = pd.to_datetime(df["time_observed_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["time_observed_at"]).sort_values(["scientific_name", "time_observed_at"])
    keep = []
    for _, g in df.groupby("scientific_name", sort=False):
        last = None
        for idx, t in zip(g.index, g["time_observed_at"]):
            if last is None or (t - last) >= pd.Timedelta(minutes=5):
                keep.append(idx)
                last = t
    return df.loc[keep]


# --------------------------------------------------------------------- helpers
def fill_missing(df, col, value):
    """
    Assign instead of df[col].fillna(value, inplace=True).

    Under pandas 3.0.2 copy-on-write that call mutates a temporary and silently
    does nothing, so v1 never actually labelled missing taxa "Unknown"
    (METHODOLOGY 6).
    """
    df[col] = df[col].fillna(value)


def load_partner(cfg):
    """Partner observations joined to their sampled habitat values."""
    base = pd.read_csv(os.path.join(V2_FILTERED, f"{cfg['partner']}_Filtered.csv"),
                       low_memory=False)
    env_path = os.path.join(V2_ENV, f"{cfg['partner']}_with_env_data.csv")
    if not os.path.exists(env_path):
        raise SystemExit(f"    no habitat file for {cfg['partner']} -- run "
                         f"cleaning-pipeline/sample_env_v2.py first")
    env = pd.read_csv(env_path, low_memory=False)
    merged = pd.merge(base, env,
                      on=["scientific_name", "time_observed_at", "latitude", "longitude"],
                      how="inner")
    for stem in ("common_name", "iconic_taxon_name"):
        if f"{stem}_x" in merged.columns:
            merged = merged.rename(columns={f"{stem}_x": stem})

    # A partner scoped by a bounding box (Sweetwater) still has to be cut to its
    # polygon; one scoped by an iNat place is already its own boundary.
    if cfg.get("partner_kml"):
        with open(cfg["partner_kml"], "rb") as f:
            polys = parse_polygons(f.read())
        rings = polys[cfg["partner_polygon"]]
        merged = filter_to_polygon(merged, rings)
    return merged


def habitat_is_usable(partner_df):
    """True if the partner's habitat values actually carry signal."""
    cols = ["elevation", "slope", "ndvi", "soil_sand", "soil_ph", "soil_clay"]
    present = [c for c in cols if c in partner_df.columns]
    if not present:
        return False, "no habitat columns"
    frac = partner_df[present].notna().mean().mean()
    if frac < 0.5:
        return False, f"only {frac:.1%} of habitat values are populated"
    return True, f"{frac:.1%} of habitat values populated"


# ---------------------------------------------------- features (4Models.py logic)
def build_environmental_features(main_df, partner_df):
    partner = partner_df[partner_df["scientific_name"].notna()].copy()
    partner["observed_on"] = pd.to_datetime(partner["observed_on"], errors="coerce")
    features = partner.groupby("scientific_name").agg(
        backup_obs_count=("id", "count"),
        backup_unique_users=("user_id", "nunique"),
        backup_unique_days=("observed_on", "nunique"),
        avg_elevation=("elevation", "mean"),
        avg_slope=("slope", "mean"),
        avg_ndvi=("ndvi", "mean"),
        avg_soil_sand=("soil_sand", "mean"),
        avg_soil_ph=("soil_ph", "mean"),
        avg_soil_clay=("soil_clay", "mean"),
    ).reset_index()
    features = features.fillna(0)
    features["log_obs_count"] = np.log1p(features["backup_obs_count"])

    taxons = partner[["scientific_name", "iconic_taxon_name", "common_name"]] \
        .dropna(subset=["iconic_taxon_name"]).drop_duplicates("scientific_name")
    features = features.merge(taxons, on="scientific_name", how="left")
    fill_missing(features, "iconic_taxon_name", "Unknown")
    fill_missing(features, "common_name", "Unknown")

    main_counts = main_df["scientific_name"].value_counts().reset_index()
    main_counts.columns = ["scientific_name", "main_obs_count"]
    features = features.merge(main_counts, on="scientific_name", how="left")
    features["main_obs_count"] = features["main_obs_count"].fillna(0)
    features["present_in_main"] = (features["main_obs_count"] > 0).astype(int)
    return features


def run_model(features):
    feature_cols_ml = ["log_obs_count", "backup_unique_users", "backup_unique_days"] + ENV_FEATURES
    X_cat = pd.get_dummies(features["iconic_taxon_name"], prefix="taxon", drop_first=True)
    X = pd.concat([features[feature_cols_ml], X_cat], axis=1)
    y = features["present_in_main"]
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                random_state=42, class_weight="balanced")
    rf.fit(X, y)
    features = features.copy()
    features["probability_of_presence"] = rf.predict_proba(X)[:, 1]
    features["predicted_present"] = (features["probability_of_presence"] >= 0.5).astype(int)

    out = features[["scientific_name", "common_name", "iconic_taxon_name",
                    "present_in_main", "predicted_present", "probability_of_presence",
                    "main_obs_count", "backup_obs_count",
                    "avg_elevation", "avg_ndvi", "avg_soil_clay"]].copy()
    conditions = [
        (out["present_in_main"] == 1) & (out["predicted_present"] == 1) & (out["main_obs_count"] <= UNDERREPRESENTED_THRESHOLD),
        (out["present_in_main"] == 1) & (out["predicted_present"] == 1) & (out["main_obs_count"] > UNDERREPRESENTED_THRESHOLD),
        (out["present_in_main"] == 0) & (out["predicted_present"] == 1),
        (out["present_in_main"] == 1) & (out["predicted_present"] == 0),
        (out["present_in_main"] == 0) & (out["predicted_present"] == 0),
    ]
    choices = ["True Positive (Underrepresented)", "True Positive (Well Documented)",
               "False Positive (Imputed/Missing)", "False Negative", "True Negative"]
    out["Prediction_Result"] = np.select(conditions, choices, default="Unknown")
    noise = (out["Prediction_Result"] == "False Positive (Imputed/Missing)") & \
            (out["backup_obs_count"] < MIN_PARTNER_OBS)
    out.loc[noise, "Prediction_Result"] = "True Negative (Filtered)"
    out.loc[noise, "predicted_present"] = 0
    sort_key = {"False Positive (Imputed/Missing)": 1, "True Positive (Underrepresented)": 2,
                "True Positive (Well Documented)": 3, "False Negative": 4,
                "True Negative": 5, "True Negative (Filtered)": 6}
    out["sort_key"] = out["Prediction_Result"].map(sort_key)
    out = out.sort_values(by=["sort_key", "probability_of_presence"],
                          ascending=[True, False]).drop(columns=["sort_key"])
    return out


def v1_counts(target):
    """Imputed/candidate counts from the v1 unofficial-extent run, for comparison."""
    path = os.path.join(V1_DIR, f"{target}_Final_Predictions.csv")
    if not os.path.exists(path):
        return None, None
    v1 = pd.read_csv(path)
    return len(v1), int((v1["Prediction_Result"] == "False Positive (Imputed/Missing)").sum())


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------- main
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    allow_blind = "--allow-habitat-blind" in sys.argv[1:]
    selected = args or [c["target"] for c in PAIRS]

    os.makedirs(OUT_DIR, exist_ok=True)
    kml_zip = zipfile.ZipFile(RESERVE_ZIP)
    kml_members = {os.path.basename(n): n for n in kml_zip.namelist() if n.endswith(".kml")}

    rows, skipped = [], []
    for cfg in PAIRS:
        if cfg["target"] not in selected:
            continue
        print(f"\n[*] {cfg['target']}  <- partner: {cfg['partner_label']}")

        polys = parse_polygons(kml_zip.read(kml_members[cfg["kml"]]))
        if cfg["unofficial"] not in polys:
            raise SystemExit(f"    polygon '{cfg['unofficial']}' not in {cfg['kml']}; "
                             f"available: {list(polys)}")
        rings = polys[cfg["unofficial"]]

        # Study reserve: raw -> unofficial polygon -> 5-min dedup, the v1 order.
        raw = pd.read_csv(os.path.join(V2_RAW, cfg["raw"]),
                          usecols=["scientific_name", "latitude", "longitude", "time_observed_at"],
                          low_memory=False).dropna(subset=["latitude", "longitude"])
        main_df = dedup_5min(filter_to_polygon(raw, rings))
        print(f"    study    : {len(main_df):6,} obs, {main_df['scientific_name'].nunique():5,} species")

        partner_df = load_partner(cfg)
        print(f"    partner  : {len(partner_df):6,} obs, "
              f"{partner_df['scientific_name'].nunique():5,} species")

        ok, why = habitat_is_usable(partner_df)
        print(f"    habitat  : {why}")
        if not ok and not allow_blind:
            print(f"    SKIPPED -- modelling this pair now would repeat v1's habitat-blind\n"
                  f"               Kendall-Frost. Re-export the rasters to cover the partner,\n"
                  f"               re-run sample_env_v2.py, then run this again.\n"
                  f"               (--allow-habitat-blind overrides.)")
            skipped.append(dict(reserve=cfg["target"], partner=cfg["partner_label"], reason=why))
            continue

        feats = build_environmental_features(main_df, partner_df)
        out = run_model(feats)
        dest = os.path.join(OUT_DIR, f"{cfg['target']}_Final_Predictions.csv")
        out.to_csv(dest, index=False)

        n_imputed = int((out["Prediction_Result"] == "False Positive (Imputed/Missing)").sum())
        v1_pool, v1_imputed = v1_counts(cfg["target"])
        print(f"    -> {os.path.relpath(dest, ROOT)}  ({n_imputed} imputed/missing)")
        rows.append(dict(
            reserve=cfg["target"], partner=cfg["partner_label"],
            study_obs=len(main_df), study_species=main_df["scientific_name"].nunique(),
            candidate_pool=len(feats), imputed_missing=n_imputed,
            v1_candidate_pool=v1_pool if v1_pool is not None else "n/a",
            v1_imputed_missing=v1_imputed if v1_imputed is not None else "n/a",
            habitat_blind=not ok,
        ))

    if not rows:
        print("\n[!] nothing modelled.")
        return

    # Merge into the existing table: a run on one reserve must not drop the others.
    comp = pd.DataFrame(rows)
    comp_path = os.path.join(CLEANED, "V2_Comparison.csv")
    if os.path.exists(comp_path):
        old = pd.read_csv(comp_path)
        comp = pd.concat([old[~old["reserve"].isin(comp["reserve"])], comp])
    order = {c["target"]: i for i, c in enumerate(PAIRS)}
    comp = comp.sort_values("reserve", key=lambda s: s.map(order)).reset_index(drop=True)
    comp.to_csv(comp_path, index=False)
    with open(os.path.join(CLEANED, "V2_Comparison.md"), "w") as f:
        f.write("# v2 run -- refreshed data, unofficial extents\n\n")
        f.write("Observations are from the Sep 14 2026 iNaturalist API pull, cut to each\n")
        f.write("reserve's unofficial boundary and 5-min deduplicated. `candidate_pool` is the\n")
        f.write("partner's species list -- the only species that can be predicted.\n")
        f.write("`imputed_missing` = predicted present at the reserve with zero observations\n")
        f.write("there. v1 columns are the Jan-Feb 2026 website-export run for comparison;\n")
        f.write("Kendall-Frost's v1 partner was Tijuana River, not Sweetwater Marsh.\n\n")
        f.write(md_table(comp) + "\n")
        if skipped:
            f.write("\n## Skipped\n\n")
            for s in skipped:
                f.write(f"- **{s['reserve']}** (partner {s['partner']}): {s['reason']}\n")
            f.write("\n")
    print("\n[+] CleanedData/V2_Comparison.{csv,md}")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
