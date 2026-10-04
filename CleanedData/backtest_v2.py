#!/usr/bin/env python3
"""
Temporal backtest of the sister-reserve model: would it have anticipated the
species that were later recorded at each reserve?

The production model is fitted and scored on the same species, so its
probabilities say nothing about predictive skill (METHODOLOGY 6). This script
tests skill the only way the data allows without fieldwork: go back to a cutoff
date T, rebuild everything from what iNaturalist held at T, and check which
"missing" species were then uploaded from inside the reserve during the next
HORIZON months.

  knowledge at T : records (study AND partner) with created_at < T
  outcome        : a record created in [T, T + HORIZON) inside the study
                   polygon, of the candidate taxon or anything below it
  evaluated      : species-rank candidates not recorded at the reserve at T

The pipeline is run_v2_analysis.py v2.1 unchanged (same features, labels, RF,
partner-in-study exclusion); only the data is cut at T.

Scorers compared on the same species:
  rf_insample  -- the production model (fitted and scored on the same rows)
  rf_oof       -- the same RF, each species scored by a model that never saw
                  its label (5-fold)
  partner_obs  -- partner observation count alone ("common at the partner")
  partner_users-- partner observer count alone

Known limits (state in the methods): "uploaded later" stands in for "present"
and favours easy-to-find species; taxonomy and identifications are as of the
Sep 2026 pull, not as of T; NDVI is a 2020-24 median, so partly postdates the
early cutoffs.

Usage:
  python CleanedData/backtest_v2.py [Reserve ...]

Outputs (gitignored -- they name species):
  CleanedData/Backtest_v2.1/backtest_metrics.csv   one row per reserve x cutoff
  CleanedData/Backtest_v2.1/backtest_species.csv   every evaluated species, scores + outcome
  CleanedData/Backtest_v2.1/backtest_summary.md
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_v2_analysis as r  # noqa: E402

warnings.filterwarnings("ignore")

CUTOFFS = ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]
HORIZON_MONTHS = 18            # the last window ends 2026-07-01, inside the pull
TOP_K = 25                     # the size of a field list
OUT_DIR = os.path.join(r.CLEANED, f"Backtest_{r.RUN}")
PARTNER_RAW = {"TorreyPines": "TorreyPinesData.csv", "MissionTrails": "MissionTrailsData.csv",
               "BuenaVista": "BuenaVistaData.csv", "SweetwaterMarsh": "SweetwaterMarshData.csv"}
SPECIES_RANK_LEVEL = 10


def created(path):
    """observation id -> created_at (UTC)."""
    df = pd.read_csv(path, usecols=["id", "created_at"], low_memory=False)
    return pd.to_datetime(df["created_at"], utc=True, errors="coerce").set_axis(df["id"].values)


def feature_matrix(f):
    cols = ["log_obs_count", "backup_unique_users", "backup_unique_days"] + r.ENV_FEATURES
    X = pd.concat([f[cols], pd.get_dummies(f["iconic_taxon_name"], prefix="taxon",
                                           drop_first=True)], axis=1)
    return X, f["present_in_main"]


def new_rf():
    # the production settings (run_v2_analysis.run_model)
    return r.RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                    random_state=42, class_weight="balanced")


def out_of_fold(X, y, folds=5):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, y):
        p[te] = new_rf().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    return p


def at_or_below(ids, taxa):
    """Every taxon at or above any of `ids` -- i.e. the candidates they confirm."""
    out = set()
    for t in ids:
        out |= taxa.get(int(t), {int(t)})
    return out


def precision_at(scores, hits, k):
    top = np.argsort(-scores, kind="stable")[:k]
    return hits[top].mean() if len(top) else np.nan


def run_pair(cfg, kml_zip, kml_members, taxa, rank_level):
    rings = r.parse_polygons(kml_zip.read(kml_members[cfg["kml"]]))[cfg["unofficial"]]
    study_all = pd.read_csv(os.path.join(r.V2_RAW, cfg["raw"]),
                            usecols=["scientific_name", "latitude", "longitude", "time_observed_at",
                                     "observed_on", "taxon_id", "created_at"],
                            low_memory=False).dropna(subset=["latitude", "longitude", "scientific_name"])
    study_all = r.filter_to_polygon(study_all, rings)
    study_all["created_at"] = pd.to_datetime(study_all["created_at"], utc=True, errors="coerce")

    partner_all = r.load_partner(cfg, study_rings=rings)
    partner_all["created_at"] = partner_all["id"].map(
        created(os.path.join(r.V2_RAW, PARTNER_RAW[cfg["partner"]])))

    metrics, species = [], []
    for cut in CUTOFFS:
        T = pd.Timestamp(cut, tz="UTC")
        end = T + pd.DateOffset(months=HORIZON_MONTHS)
        study_T = study_all[study_all["created_at"] < T]
        partner_T = partner_all[partner_all["created_at"] < T]
        if study_T.empty or partner_T["scientific_name"].nunique() < 50:
            print(f"    {cut}: too little data, skipped")
            continue

        f = r.build_environmental_features(r.study_records(study_T), partner_T)
        X, y = feature_matrix(f)
        f["rf_insample"] = new_rf().fit(X, y).predict_proba(X)[:, 1]
        f["rf_oof"] = out_of_fold(X, y)

        window = study_all[(study_all["created_at"] >= T) & (study_all["created_at"] < end)]
        confirmed = at_or_below(window["taxon_id"].dropna().astype(int).unique(), taxa)
        f["recorded_later"] = (f["taxon_id"].isin(confirmed)
                               | f["scientific_name"].isin(set(window["scientific_name"]))).astype(int)

        ev = f[(f["present_in_main"] == 0)
               & (f["taxon_id"].map(rank_level) == SPECIES_RANK_LEVEL)].copy()
        hits = ev["recorded_later"].values
        imputed = (ev["rf_insample"] >= 0.5) & (ev["backup_obs_count"] >= r.MIN_PARTNER_OBS)
        row = dict(reserve=cfg["target"], cutoff=cut, candidates=len(f),
                   evaluated=len(ev), recorded_later=int(hits.sum()),
                   base_rate=hits.mean() if len(ev) else np.nan,
                   imputed=int(imputed.sum()),
                   imputed_hits=int(ev.loc[imputed, "recorded_later"].sum()),
                   imputed_precision=ev.loc[imputed, "recorded_later"].mean() if imputed.any() else np.nan,
                   not_imputed_rate=ev.loc[~imputed, "recorded_later"].mean() if (~imputed).any() else np.nan)
        for s in ("rf_insample", "rf_oof", "log_obs_count", "backup_unique_users"):
            sc = ev[s].values
            row[f"auc_{s}"] = roc_auc_score(hits, sc) if 0 < hits.sum() < len(hits) else np.nan
            row[f"p@{TOP_K}_{s}"] = precision_at(sc, hits, TOP_K)
        metrics.append(row)
        ev["reserve"], ev["cutoff"] = cfg["target"], cut
        species.append(ev[["reserve", "cutoff", "scientific_name", "iconic_taxon_name",
                           "backup_obs_count", "backup_unique_users", "rf_insample", "rf_oof",
                           "recorded_later"]])
        print(f"    {cut}: {len(ev):5,} evaluated, {int(hits.sum()):4} recorded later "
              f"({row['base_rate']:.1%}) | imputed {row['imputed']:4} -> {row['imputed_hits']} hits "
              f"| AUC rf {row['auc_rf_insample']:.3f} oof {row['auc_rf_oof']:.3f} "
              f"count {row['auc_log_obs_count']:.3f} users {row['auc_backup_unique_users']:.3f}")
    return metrics, species


def main():
    import zipfile
    selected = [a for a in sys.argv[1:] if not a.startswith("--")] or [c["target"] for c in r.PAIRS]
    kml_zip = zipfile.ZipFile(r.RESERVE_ZIP)
    kml_members = {os.path.basename(n): n for n in kml_zip.namelist() if n.endswith(".kml")}
    taxa = r.load_taxonomy()
    with open(r.TAXONOMY) as fh:
        rank_level = {int(k): v.get("rank_level") for k, v in json.load(fh).items() if v}

    metrics, species = [], []
    for cfg in r.PAIRS:
        if cfg["target"] not in selected:
            continue
        print(f"\n[*] {cfg['target']}  <- {cfg['partner_label']}")
        m, s = run_pair(cfg, kml_zip, kml_members, taxa, rank_level)
        metrics += m
        species += s

    os.makedirs(OUT_DIR, exist_ok=True)
    met = pd.DataFrame(metrics)
    met.to_csv(os.path.join(OUT_DIR, "backtest_metrics.csv"), index=False)
    pd.concat(species).to_csv(os.path.join(OUT_DIR, "backtest_species.csv"), index=False)

    # Pooled over cutoffs: one line per reserve.
    sp = pd.concat(species)
    pooled = []
    for res, g in sp.groupby("reserve", sort=False):
        imp = (g["rf_insample"] >= 0.5) & (g["backup_obs_count"] >= r.MIN_PARTNER_OBS)
        pooled.append(dict(
            reserve=res, evaluated=len(g), recorded_later=int(g["recorded_later"].sum()),
            base_rate=round(g["recorded_later"].mean(), 3),
            imputed=int(imp.sum()),
            imputed_precision=round(g.loc[imp, "recorded_later"].mean(), 3) if imp.any() else None,
            not_imputed_rate=round(g.loc[~imp, "recorded_later"].mean(), 3),
            **{f"mean_auc_{s}": round(met.loc[met["reserve"] == res, f"auc_{s}"].mean(), 3)
               for s in ("rf_insample", "rf_oof", "log_obs_count", "backup_unique_users")},
            **{f"mean_p@{TOP_K}_{s}": round(met.loc[met["reserve"] == res, f"p@{TOP_K}_{s}"].mean(), 3)
               for s in ("rf_insample", "rf_oof", "log_obs_count")},
        ))
    pooled = pd.DataFrame(pooled)
    with open(os.path.join(OUT_DIR, "backtest_summary.md"), "w") as fh:
        fh.write(f"# Temporal backtest ({r.RUN} pipeline)\n\n")
        fh.write(f"Cutoffs {', '.join(CUTOFFS)}; outcome window {HORIZON_MONTHS} months; "
                 f"species-rank candidates not recorded at the reserve at the cutoff.\n\n")
        fh.write("## Pooled over cutoffs\n\n" + r.md_table(pooled) + "\n\n")
        fh.write("## Per cutoff\n\n" + r.md_table(met.round(3)) + "\n")
    print(f"\n[+] {os.path.relpath(OUT_DIR, r.ROOT)}/")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
