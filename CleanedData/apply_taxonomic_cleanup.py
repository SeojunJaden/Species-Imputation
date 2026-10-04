#!/usr/bin/env python3
"""
Apply Kellie's taxonomic cleanup rules to the v2 predictions.

Her three rules (METHODOLOGY 5, spring 2026), applied PER RESERVE rather than
globally (her Jul 23 clarification):

  1. Remove a genus/family (any rank above species) when a taxon inside it is
     present in that reserve's list -- the coarse entry is redundant.
  2. Merge subspecies / varieties / forms into their parent species.
  3. KEEP coarse-only taxa. If nothing inside a genus is present, the genus
     itself is a legitimate list entry; this is not a "species only" filter.

Ranks and ancestry come from the iNat taxonomy cached by
cleaning-pipeline/fetch_taxonomy.py.

Merging combines rows rather than picking one: observation counts are summed,
probability is the maximum over the merged rows, habitat averages are weighted
by partner observation count, and the prediction category is then RECOMPUTED
from the combined counts with the same thresholds the model used. A species can
therefore change category through merging -- that is intended, since the merged
row describes a different (broader) taxon than either input row.

Usage:
  python CleanedData/apply_taxonomic_cleanup.py [Reserve ...]

Outputs:
  CleanedData/FinalPredictions_v2.1_Clean/<Reserve>_Final_Predictions.csv
  CleanedData/TaxonomicCleanup_Report_v2.1.md
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED = os.path.join(ROOT, "CleanedData")
V2 = os.path.join(ROOT, "RefreshedData", "2026-09-14")
FILTERED = os.path.join(V2, "filtered")
TAXONOMY = os.path.join(V2, "taxonomy.json")
RUN = "v2.1"                     # the audit-fixed run (METHODOLOGY 6, Sep 29)
IN_DIR = os.path.join(CLEANED, f"FinalPredictions_{RUN}")
OUT_DIR = os.path.join(CLEANED, f"FinalPredictions_{RUN}_Clean")
REPORT = os.path.join(CLEANED, f"TaxonomicCleanup_Report_{RUN}.md")

# reserve -> partner, because the candidate pool (and so every name in the
# prediction file) comes from the partner's observations.
PARTNER = {
    "Scripps": "TorreyPines",
    "ElliottChaparral": "MissionTrails",
    "LosMonos": "BuenaVista",
    "MissionBay": "SweetwaterMarsh",
}

INFRASPECIFIC = {"subspecies", "variety", "form"}
SPECIES_RANK_LEVEL = 10          # iNat: species=10, genus=20, family=30, subspecies=5
UNDERREPRESENTED_THRESHOLD = 3
MIN_PARTNER_OBS = 10


def load_taxonomy():
    if not os.path.exists(TAXONOMY):
        raise SystemExit("no taxonomy cache -- run cleaning-pipeline/fetch_taxonomy.py first")
    with open(TAXONOMY) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items() if v}


def name_to_id(partner):
    """Most frequent taxon_id per scientific_name in the partner's observations."""
    df = pd.read_csv(os.path.join(FILTERED, f"{partner}_Filtered.csv"),
                     usecols=["scientific_name", "taxon_id"], low_memory=False)
    df = df.dropna(subset=["scientific_name", "taxon_id"])
    df["taxon_id"] = df["taxon_id"].astype(int)
    counts = df.groupby(["scientific_name", "taxon_id"]).size().reset_index(name="n")
    counts = counts.sort_values(["scientific_name", "n"], ascending=[True, False])
    best = counts.drop_duplicates("scientific_name")
    return dict(zip(best["scientific_name"], best["taxon_id"]))


def categorize(df):
    """Re-apply the model's category rules to (possibly merged) counts."""
    conditions = [
        (df["present_in_main"] == 1) & (df["predicted_present"] == 1) & (df["main_obs_count"] <= UNDERREPRESENTED_THRESHOLD),
        (df["present_in_main"] == 1) & (df["predicted_present"] == 1) & (df["main_obs_count"] > UNDERREPRESENTED_THRESHOLD),
        (df["present_in_main"] == 0) & (df["predicted_present"] == 1),
        (df["present_in_main"] == 1) & (df["predicted_present"] == 0),
        (df["present_in_main"] == 0) & (df["predicted_present"] == 0),
    ]
    choices = ["True Positive (Underrepresented)", "True Positive (Well Documented)",
               "False Positive (Imputed/Missing)", "False Negative", "True Negative"]
    out = np.select(conditions, choices, default="Unknown")
    df = df.copy()
    df["Prediction_Result"] = out
    noise = (df["Prediction_Result"] == "False Positive (Imputed/Missing)") & \
            (df["backup_obs_count"] < MIN_PARTNER_OBS)
    df.loc[noise, "Prediction_Result"] = "True Negative (Filtered)"
    df.loc[noise, "predicted_present"] = 0
    return df


def merge_group(g, taxa):
    """Collapse rows that resolved to the same taxon into one."""
    if len(g) == 1:
        return g.iloc[0]
    row = g.iloc[0].copy()
    backup = g["backup_obs_count"].fillna(0)
    weight = backup if backup.sum() > 0 else pd.Series(1.0, index=g.index)
    for col in ("avg_elevation", "avg_ndvi", "avg_soil_clay"):
        if col in g.columns:
            row[col] = np.average(g[col].fillna(0), weights=weight)
    row["main_obs_count"] = g["main_obs_count"].sum()
    row["backup_obs_count"] = backup.sum()
    row["probability_of_presence"] = g["probability_of_presence"].max()
    # Since v2.1 a row can be "recorded" with a zero exact-name count (the record
    # is filed under a finer taxon), so keep any merged row's label.
    row["present_in_main"] = int(g["present_in_main"].max() > 0 or row["main_obs_count"] > 0)
    row["predicted_present"] = int(g["predicted_present"].max())
    # Prefer a real common name over the "Unknown" placeholder.
    named = g[g["common_name"].notna() & (g["common_name"] != "Unknown")]
    if len(named):
        row["common_name"] = named.iloc[0]["common_name"]
    return row


def process(reserve, taxa, report):
    src = os.path.join(IN_DIR, f"{reserve}_Final_Predictions.csv")
    if not os.path.exists(src):
        print(f"[skip] {reserve}: no v2 predictions ({os.path.relpath(src, ROOT)})")
        return None
    df = pd.read_csv(src)
    n_start = len(df)
    print(f"\n[*] {reserve}: {n_start:,} rows in")

    ids = name_to_id(PARTNER[reserve])
    df["taxon_id"] = df["scientific_name"].map(ids)
    unresolved = df["taxon_id"].isna().sum()
    df["rank"] = df["taxon_id"].map(lambda t: taxa.get(int(t), {}).get("rank") if pd.notna(t) else None)
    df["rank_level"] = df["taxon_id"].map(
        lambda t: taxa.get(int(t), {}).get("rank_level") if pd.notna(t) else None)
    no_rank = int(df["rank"].isna().sum())
    print(f"    {no_rank:,} rows without a resolved rank ({unresolved:,} with no taxon_id) -- kept as-is")

    # --- rule 2: merge infraspecific taxa into their parent species -----------
    def target(row):
        if pd.isna(row["taxon_id"]) or row["rank"] not in INFRASPECIFIC:
            return row["taxon_id"], row["scientific_name"]
        rec = taxa.get(int(row["taxon_id"]), {})
        pid = rec.get("parent_id")
        prec = taxa.get(int(pid)) if pid else None
        if prec and prec.get("rank") == "species":
            return int(pid), prec["name"]
        return row["taxon_id"], row["scientific_name"]

    resolved = df.apply(target, axis=1, result_type="expand")
    df["merge_id"], df["merge_name"] = resolved[0], resolved[1]
    n_infra = int((df["merge_name"] != df["scientific_name"]).sum())

    # Name the grouper something that is NOT a column: pandas drops a grouping
    # key from the result when its name matches one, which would lose merge_name.
    key = df["merge_name"].where(df["merge_name"].notna(),
                                 df["scientific_name"]).rename("_merge_key")
    merged = df.groupby(key, sort=False).apply(lambda g: merge_group(g, taxa))
    merged = merged.reset_index(drop=True)
    merged["scientific_name"] = merged["merge_name"].where(
        merged["merge_name"].notna(), merged["scientific_name"])
    merged["taxon_id"] = merged["merge_id"]
    n_after_merge = len(merged)
    collapsed = n_start - n_after_merge
    print(f"    rule 2: {n_infra:,} infraspecific rows -> {collapsed:,} rows absorbed")

    # ranks must be re-read: a merged row is now its parent species
    merged["rank"] = merged["taxon_id"].map(
        lambda t: taxa.get(int(t), {}).get("rank") if pd.notna(t) else None)
    merged["rank_level"] = merged["taxon_id"].map(
        lambda t: taxa.get(int(t), {}).get("rank_level") if pd.notna(t) else None)

    # --- rule 1 / rule 3: drop coarse taxa that contain something on the list --
    present_ids = set(int(t) for t in merged["taxon_id"].dropna())
    covered = set()
    for t in present_ids:
        for anc in taxa.get(t, {}).get("ancestor_ids", []):
            if anc != t:
                covered.add(int(anc))

    def is_redundant(row):
        if pd.isna(row["taxon_id"]) or pd.isna(row["rank_level"]):
            return False                       # unknown rank: keep, never guess
        if row["rank_level"] <= SPECIES_RANK_LEVEL:
            return False                       # species or finer: never dropped
        return int(row["taxon_id"]) in covered  # rule 1; else rule 3 keeps it

    redundant_mask = merged.apply(is_redundant, axis=1)
    dropped = merged[redundant_mask]
    kept = merged[~redundant_mask].copy()
    coarse_kept = int(((kept["rank_level"] > SPECIES_RANK_LEVEL) & kept["rank_level"].notna()).sum())
    print(f"    rule 1: {len(dropped):,} redundant coarse taxa dropped")
    print(f"    rule 3: {coarse_kept:,} coarse-only taxa kept")

    kept = categorize(kept)
    sort_key = {"False Positive (Imputed/Missing)": 1, "True Positive (Underrepresented)": 2,
                "True Positive (Well Documented)": 3, "False Negative": 4,
                "True Negative": 5, "True Negative (Filtered)": 6}
    kept["sort_key"] = kept["Prediction_Result"].map(sort_key)
    kept = kept.sort_values(["sort_key", "probability_of_presence"], ascending=[True, False])

    cols = ["scientific_name", "common_name", "iconic_taxon_name", "rank", "taxon_id",
            "present_in_main", "predicted_present", "probability_of_presence",
            "main_obs_count", "backup_obs_count",
            "avg_elevation", "avg_ndvi", "avg_soil_clay", "Prediction_Result"]
    kept = kept[[c for c in cols if c in kept.columns]]

    os.makedirs(OUT_DIR, exist_ok=True)
    dest = os.path.join(OUT_DIR, f"{reserve}_Final_Predictions.csv")
    kept.to_csv(dest, index=False)

    imp_before = int((df["Prediction_Result"] == "False Positive (Imputed/Missing)").sum())
    imp_after = int((kept["Prediction_Result"] == "False Positive (Imputed/Missing)").sum())
    print(f"    imputed/missing: {imp_before} -> {imp_after}")
    print(f"    -> {os.path.relpath(dest, ROOT)}  ({len(kept):,} rows)")

    report.append(dict(reserve=reserve, rows_in=n_start, rows_out=len(kept),
                       infraspecific_merged=n_infra, rows_absorbed=collapsed,
                       coarse_dropped=len(dropped), coarse_kept=coarse_kept,
                       unresolved_rank=no_rank,
                       imputed_before=imp_before, imputed_after=imp_after))
    return dropped


def main():
    taxa = load_taxonomy()
    print(f"[i] taxonomy: {len(taxa):,} taxa")
    targets = [a for a in sys.argv[1:] if not a.startswith("--")] or list(PARTNER)

    report, examples = [], {}
    for r in targets:
        dropped = process(r, taxa, report)
        if dropped is not None and len(dropped):
            examples[r] = dropped.nlargest(8, "probability_of_presence")[
                ["scientific_name", "rank", "Prediction_Result"]]

    if not report:
        print("\n[!] nothing processed.")
        return
    rep = pd.DataFrame(report)

    def md_table(df):
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |",
                 "| " + " | ".join("---" for _ in cols) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    with open(REPORT, "w") as f:
        f.write("# Taxonomic cleanup of the v2 predictions\n\n")
        f.write("Kellie's three rules, applied per reserve (her Jul 23 clarification):\n")
        f.write("1. drop a genus/family when something inside it is on that reserve's list;\n")
        f.write("2. merge subspecies/varieties/forms into their parent species;\n")
        f.write("3. keep coarse-only taxa -- this is not a species-only filter.\n\n")
        f.write("Ranks and ancestry from the iNaturalist taxonomy\n")
        f.write("(`RefreshedData/2026-09-14/taxonomy.json`). Merged rows sum observation\n")
        f.write("counts, take the maximum probability, and have their prediction category\n")
        f.write("recomputed, so a taxon can change category through merging.\n\n")
        f.write(md_table(rep) + "\n")
        for r, ex in examples.items():
            f.write(f"\n## {r} -- examples of coarse taxa dropped by rule 1\n\n")
            f.write(md_table(ex) + "\n")
    print(f"\n[+] {os.path.relpath(REPORT, ROOT)}")
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
