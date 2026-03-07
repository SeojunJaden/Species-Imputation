import os
import joblib
import numpy as np
import pandas as pd
from utils import load_data, ENV_COLS, FEATURE_COLS, SPECIES_COL

# we can change this to test different thresholds
PRESENCE_THRESHOLD = 0.6

#load model
model = joblib.load("models/lightgbm_model.pkl")
df = load_data()

taxon_names = df[["taxon_id", "taxon_name"]].drop_duplicates().set_index("taxon_id")["taxon_name"]
all_taxon_ids = df["taxon_id"].unique()

RESERVES = {
    "scripps_coastal":   [-117.26, 32.86, -117.24, 32.88],
    "kendall_frost":     [-117.26, 32.86, -117.24, 32.88],
    "elliott_chaparral": [-117.26, 32.86, -117.24, 32.88],
    "sweeney_granite":   [-117.26, 32.86, -117.24, 32.88],
}

# ── PREDICT ───────────────────────────────────────────────────────────────────
for reserve_name, (W, S, E, N) in RESERVES.items():
    print(f"\n{reserve_name}...")

    reserve_env = df[
        (df["longitude"] >= W) & (df["longitude"] <= E) &
        (df["latitude"]  >= S) & (df["latitude"]  <= N)
    ][ENV_COLS].drop_duplicates()

    if len(reserve_env) == 0:
        print(f"  No points found — check bounding box")
        continue

    rows = []
    for taxon_id in all_taxon_ids:
        pred_df = reserve_env.copy()
        pred_df[SPECIES_COL] = taxon_id
        probs = model.predict_proba(pred_df[FEATURE_COLS])[:, 1]

        if probs.max() >= PRESENCE_THRESHOLD:
            rows.append({
                "taxon_id":         taxon_id,
                "taxon_name":       taxon_names.get(taxon_id, "unknown"),
                "max_probability":  round(probs.max(), 4),
                "mean_probability": round(probs.mean(), 4),
            })

    results = pd.DataFrame(rows).sort_values("max_probability", ascending=False)
    results.to_csv(f"predictions/{reserve_name}_species_list.csv", index=False)
    print(f"  {len(results)} species predicted → predictions/{reserve_name}_species_list.csv")
