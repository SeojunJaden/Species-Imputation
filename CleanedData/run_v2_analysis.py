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

v2.1 (Sep 30 2026) fixes three problems found by the Sep 29 leakage audit
(METHODOLOGY 6). Outputs go to new *_v2.1 paths so the v2 lists already sent stay
reproducible from the untouched *_v2 folders:

  - partner observations that fall inside the study reserve's polygon are
    dropped. Buena Vista's bbox reaches into Dawson, which put Dawson's own
    records into its partner's features and candidate pool;
  - the "recorded at the reserve" label uses every in-polygon record, not only
    the ones surviving the 5-min dedup, which drops date-only records;
  - the label is rolled up the iNat taxonomy: a species counts as recorded when
    anything at or below it (a subspecies, say) was recorded at the reserve.

v2.2 (Oct 3 2026) changes only how species are scored. Each species' probability
is out-of-fold -- from forests that never saw its own label -- averaged over
REPEATS fold splits and forest seeds. In-sample scoring measured as harmless in
the backtest, but it cannot be defended in a paper. The top of each list is a
plateau of near-equal scores that a single seed reshuffled (METHODOLOGY 6);
averaging over 20 repeats makes it reproducible to ~24 of the top 25.

A pair whose partner has no usable habitat data is SKIPPED rather than quietly
modelled on zeros -- that silent failure is what made Kendall-Frost
habitat-blind in v1. Pass --allow-habitat-blind to run it anyway.

Usage:
  python CleanedData/run_v2_analysis.py [Reserve ...] [--allow-habitat-blind]

Outputs:
  CleanedData/FinalPredictions_v2.2/<Reserve>_Final_Predictions.csv
  CleanedData/V2.2_Comparison.csv / .md    (v2.2 vs the v1 unofficial-extent run)
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
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESERVE_ZIP = os.path.join(ROOT, "ReserveExtents.zip")
CLEANED = os.path.join(ROOT, "CleanedData")
V2 = os.path.join(ROOT, "RefreshedData", "2026-09-14")
V2_RAW, V2_FILTERED, V2_ENV = (os.path.join(V2, d) for d in ("raw", "filtered", "env"))
TAXONOMY = os.path.join(V2, "taxonomy.json")
RUN = "v2.2"
OUT_DIR = os.path.join(CLEANED, f"FinalPredictions_{RUN}")
COMPARISON = os.path.join(CLEANED, f"V{RUN[1:]}_Comparison")
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
EFFORT_FEATURES = ["log_obs_count", "backup_unique_users", "backup_unique_days"]
FOLDS = 5
REPEATS = 20                  # fold splits x forest seeds averaged per species


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


def load_taxonomy():
    """taxon_id -> set of ids at or above it (itself + iNat ancestors)."""
    with open(TAXONOMY) as f:
        raw = json.load(f)
    return {int(k): {int(k), *map(int, v.get("ancestor_ids", []))}
            for k, v in raw.items() if v}


def study_records(study_raw):
    """
    Everything the study reserve's label needs, from all in-polygon records.

    recorded_ids: every taxon recorded at the reserve plus all its ancestors, so a
      candidate counts as recorded if anything at or below it was.
    counts: records per exact name -- timestamped records after the 5-min dedup,
      plus date-only records at one per species per day (the dedup cannot place
      them in time, and v2.0 silently dropped them).
    """
    taxa = load_taxonomy()
    ids = study_raw["taxon_id"].dropna().astype(int).unique()
    recorded_ids, unresolved = set(), 0
    for t in ids:
        if t in taxa:
            recorded_ids |= taxa[t]
        else:
            recorded_ids.add(t)
            unresolved += 1
    timed = dedup_5min(study_raw)
    date_only = study_raw[study_raw["time_observed_at"].isna()] \
        .drop_duplicates(["scientific_name", "observed_on"])
    counts = pd.concat([timed["scientific_name"], date_only["scientific_name"]]).value_counts()
    return dict(recorded_ids=recorded_ids, counts=counts, names=set(study_raw["scientific_name"]),
                n_timed=len(timed), n_date_only=len(date_only), unresolved=unresolved)


def load_partner(cfg, study_rings=None):
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

    # Never let the study reserve's own records into its partner (Buena Vista's
    # bbox reaches into Dawson). They would set the partner's features and put
    # species in the candidate pool that are "recorded" by construction.
    if study_rings is not None:
        inside = filter_to_polygon(merged, study_rings).index
        merged = merged.drop(inside)
        merged.attrs["dropped_in_study"] = len(inside)
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
def build_environmental_features(study, partner_df):
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

    # Candidate taxon_id: the partner's most frequent id for that name.
    tid = partner.dropna(subset=["taxon_id"]).groupby("scientific_name")["taxon_id"] \
        .agg(lambda s: int(s.mode().iloc[0]))
    features["taxon_id"] = features["scientific_name"].map(tid)

    features["main_obs_count"] = features["scientific_name"].map(study["counts"]).fillna(0)
    by_name = features["scientific_name"].isin(study["names"])
    by_taxon = features["taxon_id"].isin(study["recorded_ids"])
    features["present_in_main"] = (by_name | by_taxon).astype(int)
    return features


def new_forest(seed):
    return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                  random_state=seed, class_weight="balanced", n_jobs=-1)


def design_matrix(features, numeric=None):
    """Numeric features plus one-hot taxon group (the full feature set by default)."""
    numeric = EFFORT_FEATURES + ENV_FEATURES if numeric is None else numeric
    X_cat = pd.get_dummies(features["iconic_taxon_name"], prefix="taxon", drop_first=True)
    return pd.concat([features[numeric], X_cat], axis=1)


def score(X, y, repeats=REPEATS):
    """
    Out-of-fold probability of presence, averaged over `repeats` runs.

    Every species is scored only by forests trained without it, so its own label
    never feeds its score. Each repeat uses a fresh fold split and forest seed.
    """
    p = np.zeros(len(y))
    for k in range(repeats):
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=k).split(X, y):
            p[te] += new_forest(k).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    return p / repeats


def run_model(features):
    X = design_matrix(features)
    y = features["present_in_main"]
    features = features.copy()
    features["probability_of_presence"] = score(X, y)
    features["predicted_present"] = (features["probability_of_presence"] >= 0.5).astype(int)

    out = features[["scientific_name", "common_name", "iconic_taxon_name",
                    "present_in_main", "predicted_present", "probability_of_presence",
                    "main_obs_count", "backup_obs_count", "taxon_id",
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

        # Study reserve: raw -> unofficial polygon. The label uses every record;
        # the 5-min dedup only shapes the counts (study_records).
        raw = pd.read_csv(os.path.join(V2_RAW, cfg["raw"]),
                          usecols=["scientific_name", "latitude", "longitude", "time_observed_at",
                                   "observed_on", "taxon_id"],
                          low_memory=False).dropna(subset=["latitude", "longitude", "scientific_name"])
        study = study_records(filter_to_polygon(raw, rings))
        print(f"    study    : {study['n_timed']:6,} obs after dedup + {study['n_date_only']} "
              f"date-only, {len(study['names']):5,} names"
              + (f" ({study['unresolved']} taxa not in taxonomy.json)" if study["unresolved"] else ""))

        partner_df = load_partner(cfg, study_rings=rings)
        dropped_in_study = partner_df.attrs.get("dropped_in_study", 0)
        print(f"    partner  : {len(partner_df):6,} obs, "
              f"{partner_df['scientific_name'].nunique():5,} species "
              f"({dropped_in_study:,} dropped: inside the study polygon)")

        ok, why = habitat_is_usable(partner_df)
        print(f"    habitat  : {why}")
        if not ok and not allow_blind:
            print(f"    SKIPPED -- modelling this pair now would repeat v1's habitat-blind\n"
                  f"               Kendall-Frost. Re-export the rasters to cover the partner,\n"
                  f"               re-run sample_env_v2.py, then run this again.\n"
                  f"               (--allow-habitat-blind overrides.)")
            skipped.append(dict(reserve=cfg["target"], partner=cfg["partner_label"], reason=why))
            continue

        feats = build_environmental_features(study, partner_df)
        out = run_model(feats)
        dest = os.path.join(OUT_DIR, f"{cfg['target']}_Final_Predictions.csv")
        out.to_csv(dest, index=False)

        n_imputed = int((out["Prediction_Result"] == "False Positive (Imputed/Missing)").sum())
        v1_pool, v1_imputed = v1_counts(cfg["target"])
        print(f"    -> {os.path.relpath(dest, ROOT)}  ({n_imputed} imputed/missing)")
        rows.append(dict(
            reserve=cfg["target"], partner=cfg["partner_label"],
            study_obs=study["n_timed"], study_date_only=study["n_date_only"],
            study_names=len(study["names"]),
            partner_obs_dropped_in_study=dropped_in_study,
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
    comp_path = COMPARISON + ".csv"
    if os.path.exists(comp_path):
        old = pd.read_csv(comp_path)
        comp = pd.concat([old[~old["reserve"].isin(comp["reserve"])], comp])
    order = {c["target"]: i for i, c in enumerate(PAIRS)}
    comp = comp.sort_values("reserve", key=lambda s: s.map(order)).reset_index(drop=True)
    comp.to_csv(comp_path, index=False)
    with open(COMPARISON + ".md", "w") as f:
        f.write(f"# {RUN} run -- refreshed data, unofficial extents, audit fixes\n\n")
        f.write("Observations are from the Sep 14 2026 iNaturalist API pull, cut to each\n")
        f.write("reserve's unofficial boundary. `study_obs` is after the 5-min dedup;\n")
        f.write("`study_date_only` counts records with no time of day (one per species per day).\n")
        f.write("`candidate_pool` is the partner's species list, excluding partner records inside\n")
        f.write("the study polygon (`partner_obs_dropped_in_study`) -- the only species that can\n")
        f.write("be predicted. `imputed_missing` = predicted present at the reserve with no record\n")
        f.write("there of that taxon or anything below it. v1 columns are the Jan-Feb 2026 website-export run for comparison;\n")
        f.write("Kendall-Frost's v1 partner was Tijuana River, not Sweetwater Marsh.\n\n")
        f.write(md_table(comp) + "\n")
        if skipped:
            f.write("\n## Skipped\n\n")
            for s in skipped:
                f.write(f"- **{s['reserve']}** (partner {s['partner']}): {s['reason']}\n")
            f.write("\n")
    print(f"\n[+] {os.path.relpath(COMPARISON, ROOT)}.{{csv,md}}")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
