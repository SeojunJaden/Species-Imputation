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

The pipeline is run_v2_analysis.py unchanged (same features, labels, scoring,
partner-in-study exclusion); only the data is cut at T.

Scorers compared on the same species:
  model        -- production scoring: out-of-fold, averaged (run_v2_analysis.score)
  no_habitat   -- the same model without the 6 habitat features (ablation)
  habitat_only -- the same model without the 3 partner-effort features (ablation)
  rf_insample  -- the v2.0/v2.1 scoring (fitted and scored on the same rows)
  log_obs_count       -- partner observation count alone ("common at the partner")
  backup_unique_users -- partner observer count alone

Confidence intervals: paired bootstrap over species within each cutoff
(BOOT resamples), on the mean AUC across cutoffs and on the model's AUC gain
over each comparison scorer.

Known limits (state in the methods): "uploaded later" stands in for "present"
and favours easy-to-find species; taxonomy and identifications are as of the
Sep 2026 pull, not as of T; NDVI is a 2020-24 median, so partly postdates the
early cutoffs.

Usage:
  python CleanedData/backtest_v2.py [Reserve ...]

Outputs (gitignored -- they name species):
  CleanedData/Backtest_v2.2/backtest_metrics.csv   one row per reserve x cutoff
  CleanedData/Backtest_v2.2/backtest_species.csv   every evaluated species, scores + outcome
  CleanedData/Backtest_v2.2/backtest_auc_ci.csv     mean AUC + gains with 95% CIs
  CleanedData/Backtest_v2.2/backtest_summary.md
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_v2_analysis as r  # noqa: E402

warnings.filterwarnings("ignore")

CUTOFFS = ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]
HORIZON_MONTHS = 18            # the last window ends 2026-07-01, inside the pull
TOP_K = 25                     # the size of a field list
BOOT = 1000
BT_REPEATS = 5                 # repeats for the averaged scorers, for run time;
                               # production lists use run_v2_analysis.REPEATS (20)
SCORERS = ["model", "no_habitat", "habitat_only", "rf_insample",
           "log_obs_count", "backup_unique_users"]
OUT_DIR = os.path.join(r.CLEANED, f"Backtest_{r.RUN}")
PARTNER_RAW = {"TorreyPines": "TorreyPinesData.csv", "MissionTrails": "MissionTrailsData.csv",
               "BuenaVista": "BuenaVistaData.csv", "SweetwaterMarsh": "SweetwaterMarshData.csv"}
SPECIES_RANK_LEVEL = 10


def created(path):
    """observation id -> created_at (UTC)."""
    df = pd.read_csv(path, usecols=["id", "created_at"], low_memory=False)
    return pd.to_datetime(df["created_at"], utc=True, errors="coerce").set_axis(df["id"].values)


def all_scores(f):
    """Every model-based scorer, on the same candidates."""
    y = f["present_in_main"]
    X = r.design_matrix(f)
    out = {"model": r.score(X, y, repeats=BT_REPEATS),
           "no_habitat": r.score(r.design_matrix(f, r.EFFORT_FEATURES), y, repeats=BT_REPEATS),
           "habitat_only": r.score(r.design_matrix(f, r.ENV_FEATURES), y, repeats=BT_REPEATS),
           "rf_insample": r.new_forest(42).fit(X, y).predict_proba(X)[:, 1]}
    return out


def boot_auc(groups, rng):
    """
    Paired bootstrap over species within each cutoff. Returns {scorer: (mean, lo, hi)}
    for the mean AUC across cutoffs, and {scorer: (gain, lo, hi)} for model minus it.
    """
    point = {s: np.mean([roc_auc_score(h, sc[s]) for h, sc in groups]) for s in SCORERS}
    draws = {s: [] for s in SCORERS}
    for _ in range(BOOT):
        per = {s: [] for s in SCORERS}
        for h, sc in groups:
            idx = rng.integers(0, len(h), len(h))
            if 0 < h[idx].sum() < len(idx):
                for s in SCORERS:
                    per[s].append(roc_auc_score(h[idx], sc[s][idx]))
        for s in SCORERS:
            draws[s].append(np.mean(per[s]))
    draws = {s: np.array(v) for s, v in draws.items()}
    ci = {s: (point[s], *np.percentile(draws[s], [2.5, 97.5])) for s in SCORERS}
    gain = {s: (point["model"] - point[s],
                *np.percentile(draws["model"] - draws[s], [2.5, 97.5]))
            for s in SCORERS if s != "model"}
    return ci, gain


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
        for name, p in all_scores(f).items():
            f[name] = p

        window = study_all[(study_all["created_at"] >= T) & (study_all["created_at"] < end)]
        confirmed = at_or_below(window["taxon_id"].dropna().astype(int).unique(), taxa)
        f["recorded_later"] = (f["taxon_id"].isin(confirmed)
                               | f["scientific_name"].isin(set(window["scientific_name"]))).astype(int)

        ev = f[(f["present_in_main"] == 0)
               & (f["taxon_id"].map(rank_level) == SPECIES_RANK_LEVEL)].copy()
        hits = ev["recorded_later"].values
        imputed = (ev["model"] >= 0.5) & (ev["backup_obs_count"] >= r.MIN_PARTNER_OBS)
        row = dict(reserve=cfg["target"], cutoff=cut, candidates=len(f),
                   evaluated=len(ev), recorded_later=int(hits.sum()),
                   base_rate=hits.mean() if len(ev) else np.nan,
                   imputed=int(imputed.sum()),
                   imputed_hits=int(ev.loc[imputed, "recorded_later"].sum()),
                   imputed_precision=ev.loc[imputed, "recorded_later"].mean() if imputed.any() else np.nan,
                   not_imputed_rate=ev.loc[~imputed, "recorded_later"].mean() if (~imputed).any() else np.nan)
        for s in SCORERS:
            sc = ev[s].values
            row[f"auc_{s}"] = roc_auc_score(hits, sc) if 0 < hits.sum() < len(hits) else np.nan
            row[f"p@{TOP_K}_{s}"] = precision_at(sc, hits, TOP_K)
        metrics.append(row)
        ev["reserve"], ev["cutoff"] = cfg["target"], cut
        species.append(ev[["reserve", "cutoff", "scientific_name", "iconic_taxon_name",
                           "backup_obs_count", "log_obs_count", "backup_unique_users",
                           "model", "no_habitat", "habitat_only", "rf_insample",
                           "recorded_later"]])
        print(f"    {cut}: {len(ev):5,} evaluated, {int(hits.sum()):4} recorded later "
              f"({row['base_rate']:.1%}) | imputed {row['imputed']:4} -> {row['imputed_hits']} hits "
              f"| AUC model {row['auc_model']:.3f} no-habitat {row['auc_no_habitat']:.3f} "
              f"habitat-only {row['auc_habitat_only']:.3f} count {row['auc_log_obs_count']:.3f}")
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

    # Pooled over cutoffs: one line per reserve, with bootstrap intervals.
    sp = pd.concat(species)
    rng = np.random.default_rng(0)
    pooled, cis = [], []
    for res, g in sp.groupby("reserve", sort=False):
        imp = (g["model"] >= 0.5) & (g["backup_obs_count"] >= r.MIN_PARTNER_OBS)
        groups = [(c["recorded_later"].values, {s: c[s].values for s in SCORERS})
                  for _, c in g.groupby("cutoff") if 0 < c["recorded_later"].sum() < len(c)]
        ci, gain = boot_auc(groups, rng)
        pooled.append(dict(
            reserve=res, evaluated=len(g), recorded_later=int(g["recorded_later"].sum()),
            base_rate=round(g["recorded_later"].mean(), 3),
            imputed=int(imp.sum()),
            imputed_precision=round(g.loc[imp, "recorded_later"].mean(), 3) if imp.any() else None,
            not_imputed_rate=round(g.loc[~imp, "recorded_later"].mean(), 3),
            **{f"p@{TOP_K}_{s}": round(met.loc[met["reserve"] == res, f"p@{TOP_K}_{s}"].mean(), 3)
               for s in ("model", "no_habitat", "log_obs_count")},
        ))
        for s in SCORERS:
            m, lo, hi = ci[s]
            row = dict(reserve=res, scorer=s, mean_auc=round(m, 3),
                       auc_95ci=f"{lo:.3f}-{hi:.3f}")
            if s in gain:
                d, dlo, dhi = gain[s]
                row.update(model_gain=round(d, 3), gain_95ci=f"{dlo:+.3f} to {dhi:+.3f}")
            cis.append(row)
    pooled = pd.DataFrame(pooled)
    cis = pd.DataFrame(cis)
    cis.to_csv(os.path.join(OUT_DIR, "backtest_auc_ci.csv"), index=False)
    with open(os.path.join(OUT_DIR, "backtest_summary.md"), "w") as fh:
        fh.write(f"# Temporal backtest ({r.RUN} pipeline)\n\n")
        fh.write(f"Cutoffs {', '.join(CUTOFFS)}; outcome window {HORIZON_MONTHS} months; "
                 f"species-rank candidates not recorded at the reserve at the cutoff.\n\n")
        fh.write("## Pooled over cutoffs\n\n" + r.md_table(pooled) + "\n\n")
        fh.write(f"## Mean AUC across cutoffs, paired bootstrap ({BOOT} resamples)\n\n"
                 + r.md_table(cis.fillna("")) + "\n\n")
        fh.write("## Per cutoff\n\n" + r.md_table(met.round(3)) + "\n")
    print(f"\n[+] {os.path.relpath(OUT_DIR, r.ROOT)}/")
    print(pooled.to_string(index=False))
    print(cis.fillna("").to_string(index=False))


if __name__ == "__main__":
    main()
