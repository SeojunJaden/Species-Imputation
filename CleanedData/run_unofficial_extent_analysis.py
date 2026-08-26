#!/usr/bin/env python3
"""
Unofficial-extent re-run of the sister-reserve species-imputation model.

Per Kellie Uyeda's Jun 25 2026 decisions (see planning/2026-06-25-post-kellie-analysis-plan.md):
  - Use the UNOFFICIAL reserve boundaries she sent (ReserveExtents.zip).
  - Scripps = UPLAND only; the marine MPA polygon is excluded.
  - No official-vs-unofficial dual analysis for the paper, but we DO compute
    both here so we can show Kellie the before/after delta as a sanity check.

Only the TARGET reserve's observation set changes (point-in-polygon on the
unofficial boundary). All environmental features + the candidate species pool
come from the unchanged DONOR ("sister") reserve, so no env re-sampling is
needed. Model logic mirrors CleanedData/4Models.py exactly.

Outputs:
  CleanedData/FinalPredictions_Unofficial/<Reserve>_Final_Predictions.csv
  CleanedData/UnofficialExtent_Comparison.csv / .md
"""

import os
import io
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESERVE_ZIP = os.path.join(ROOT, "ReserveExtents.zip")
RAW_ZIP = os.path.join(ROOT, "AllSpeciesRawData.zip")
CLEANED = os.path.join(ROOT, "CleanedData")
PROCESSED = os.path.join(ROOT, "ProcessedData")
OUT_DIR = os.path.join(CLEANED, "FinalPredictions_Unofficial")

# target -> config. Names are EXACT KML placemark names (verified from the zip).
PAIRS = [
    dict(target="Scripps", raw="ScrippsData.csv",
         kml="Scripps Coastal Reserve Extent.kml",
         unofficial="unofficial upland extent", official="Official upland extent",
         donor_base="TorreyPines_Filtered.csv", donor_env="TorreyPines_with_env_data.csv"),
    dict(target="ElliottChaparral", raw="ElliottChaparralData.csv",
         kml="Elliott extent.kml",
         unofficial="Unofficial extent", official="Official extent",
         donor_base="MissionTrails_Filtered.csv", donor_env="MissionTrails_with_env_data.csv"),
    dict(target="LosMonos", raw="LosMonosCanyonData.csv",
         kml="Dawson Extent.kml",
         unofficial="unofficial extent", official="Official extent primary",
         donor_base="BuenaVista_Filtered.csv", donor_env="BuenaVista_with_env_data.csv"),
    dict(target="MissionBay", raw="MissionBayData.csv",  # Kendall-Frost
         kml="Kendall Frost Extent.kml",
         unofficial="Unofficial extent", official="Official extent",
         donor_base="TijuanaRiver_Filtered.csv", donor_env="TijuanaRiver_with_env_data.csv"),
]

UNDERREPRESENTED_THRESHOLD = 3


# ---------------------------------------------------------------- KML parsing
def _strip(tag):
    return tag.split("}")[-1]


def parse_polygons(kml_bytes):
    """name -> list of outer rings; each ring = list of (lon, lat)."""
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
            # outer boundary only (ignore holes)
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


# ----------------------------------------------------- donor loaders (4Models.py)
def load_and_merge(base_csv, env_csv):
    df_base = pd.read_csv(base_csv)
    df_env = pd.read_csv(env_csv)
    merged = pd.merge(df_base, df_env,
                      on=["scientific_name", "time_observed_at", "latitude", "longitude"],
                      how="inner")
    if "common_name_x" in merged.columns:
        merged.rename(columns={"common_name_x": "common_name"}, inplace=True)
    if "iconic_taxon_name_x" in merged.columns:
        merged.rename(columns={"iconic_taxon_name_x": "iconic_taxon_name"}, inplace=True)
    return merged


def build_environmental_features(main_df, backup_df):
    backup_species = backup_df[backup_df["scientific_name"].notna()].copy()
    backup_species["observed_on"] = pd.to_datetime(backup_species["observed_on"], errors="coerce")
    features = backup_species.groupby("scientific_name").agg(
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
    features.fillna(0, inplace=True)
    features["log_obs_count"] = np.log1p(features["backup_obs_count"])
    taxons = backup_species[["scientific_name", "iconic_taxon_name", "common_name"]] \
        .dropna(subset=["iconic_taxon_name"]).drop_duplicates("scientific_name")
    features = features.merge(taxons, on="scientific_name", how="left")
    features["iconic_taxon_name"].fillna("Unknown", inplace=True)
    features["common_name"].fillna("Unknown", inplace=True)
    main_counts = main_df["scientific_name"].value_counts().reset_index()
    main_counts.columns = ["scientific_name", "main_obs_count"]
    features = features.merge(main_counts, on="scientific_name", how="left")
    features["main_obs_count"] = features["main_obs_count"].fillna(0)
    features["present_in_main"] = (features["main_obs_count"] > 0).astype(int)
    return features


def run_model(features):
    feature_cols_ml = ["log_obs_count", "backup_unique_users", "backup_unique_days",
                       "avg_elevation", "avg_slope", "avg_ndvi",
                       "avg_soil_sand", "avg_soil_ph", "avg_soil_clay"]
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
    noise = (out["Prediction_Result"] == "False Positive (Imputed/Missing)") & (out["backup_obs_count"] < 10)
    out.loc[noise, "Prediction_Result"] = "True Negative (Filtered)"
    out.loc[noise, "predicted_present"] = 0
    sort_key = {"False Positive (Imputed/Missing)": 1, "True Positive (Underrepresented)": 2,
                "True Positive (Well Documented)": 3, "False Negative": 4,
                "True Negative": 5, "True Negative (Filtered)": 6}
    out["sort_key"] = out["Prediction_Result"].map(sort_key)
    out = out.sort_values(by=["sort_key", "probability_of_presence"],
                          ascending=[True, False]).drop(columns=["sort_key"])
    return out


# ------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    kml_zip = zipfile.ZipFile(RESERVE_ZIP)
    raw_zip = zipfile.ZipFile(RAW_ZIP)
    kml_members = {os.path.basename(n): n for n in kml_zip.namelist() if n.endswith(".kml")}
    raw_members = {os.path.basename(n): n for n in raw_zip.namelist() if n.endswith(".csv")}

    comparison = []
    for cfg in PAIRS:
        print(f"\n[*] {cfg['target']}  (donor: {cfg['donor_base'].replace('_Filtered.csv','')})")
        polys = parse_polygons(kml_zip.read(kml_members[cfg["kml"]]))
        for key in ("unofficial", "official"):
            if cfg[key] not in polys:
                raise SystemExit(f"    polygon '{cfg[key]}' not found in {cfg['kml']}; "
                                 f"available: {list(polys)}")
        uno_rings, off_rings = polys[cfg["unofficial"]], polys[cfg["official"]]

        raw = pd.read_csv(io.BytesIO(raw_zip.read(raw_members[cfg["raw"]])),
                          usecols=["scientific_name", "latitude", "longitude", "time_observed_at"],
                          low_memory=False)
        raw = raw.dropna(subset=["latitude", "longitude"])

        off_clean = dedup_5min(filter_to_polygon(raw, off_rings))
        uno_clean = dedup_5min(filter_to_polygon(raw, uno_rings))
        off_sp = off_clean["scientific_name"].nunique()
        uno_sp = uno_clean["scientific_name"].nunique()
        print(f"    official : {len(off_clean):5d} obs, {off_sp:4d} species")
        print(f"    unofficial:{len(uno_clean):5d} obs, {uno_sp:4d} species  "
              f"(+{len(uno_clean)-len(off_clean)} obs, +{uno_sp-off_sp} species)")

        # model on the UNOFFICIAL target set (donor unchanged)
        backup_df = load_and_merge(os.path.join(CLEANED, cfg["donor_base"]),
                                   os.path.join(PROCESSED, cfg["donor_env"]))
        feats = build_environmental_features(uno_clean, backup_df)
        out = run_model(feats)
        out_path = os.path.join(OUT_DIR, f"{cfg['target']}_Final_Predictions.csv")
        out.to_csv(out_path, index=False)

        n_imputed = int((out["Prediction_Result"] == "False Positive (Imputed/Missing)").sum())
        print(f"    -> {out_path}  ({n_imputed} imputed/missing species)")
        comparison.append(dict(
            reserve=cfg["target"],
            donor=cfg["donor_base"].replace("_Filtered.csv", ""),
            official_obs=len(off_clean), unofficial_obs=len(uno_clean),
            obs_delta=len(uno_clean) - len(off_clean),
            official_species=off_sp, unofficial_species=uno_sp,
            species_delta=uno_sp - off_sp,
            candidate_pool=len(feats), imputed_missing=n_imputed,
        ))

    comp = pd.DataFrame(comparison)
    comp.to_csv(os.path.join(CLEANED, "UnofficialExtent_Comparison.csv"), index=False)

    def md_table(df):
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |",
                 "| " + " | ".join("---" for _ in cols) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    with open(os.path.join(CLEANED, "UnofficialExtent_Comparison.md"), "w") as f:
        f.write("# Official vs. Unofficial Extent — Comparison\n\n")
        f.write("Observation/species counts are 5-min-deduplicated within each polygon, so the\n")
        f.write("only difference between columns is the boundary. Scripps = upland only (MPA excluded).\n")
        f.write("`imputed_missing` = species the model predicts present in the reserve but with zero\n")
        f.write("observations there (the 'missing species' output), under the unofficial extent.\n\n")
        f.write(md_table(comp))
        f.write("\n")
    print("\n[+] Wrote comparison -> CleanedData/UnofficialExtent_Comparison.{csv,md}")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
