import os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "ProcessedData")
MODEL_DIR = os.path.join(BASE_DIR, "model_outputs")
OUT_DIR   = os.path.join(BASE_DIR, "imputed_species")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "elevation", "slope", "aspect", "ndvi",
    "landcover", "impervious", "bathymetry",
    "soil_sand", "soil_ph", "soil_clay",
]

SPP_COL = "scientific_name"
COMMON_COL = "common_name"
TAXON_COL = "iconic_taxon_name"

PROB_THRESHOLD = 0.30 

IMPUTATION_TASKS = [
    {
        "name":         "ElliottChaparral_imputed",
        "display":      "Elliott Chaparral  ←  Mission Trails Regional Park",
        "source_csv":   "MissionTrails_with_env_data.csv",
        "target_csv":   "ElliottChaparral_with_env_data.csv",
        "model_prefix": "Torrey_Pines_vs_Elliott_Chaparral",
        "source_label": "Mission Trails",
        "target_label": "Elliott Chaparral",
    },
    {
        "name":         "KendallFrost_imputed",
        "display":      "Kendall Frost Mission Bay  ←  Tijuana River",
        "source_csv":   "TijuanaRiver_with_env_data.csv",
        "target_csv":   "MissionBay_with_env_data.csv",
        "model_prefix": "Tijuana_Estuary_vs_Tijuana_River",
        "source_label": "Tijuana River",
        "target_label": "Kendall Frost Mission Bay",
    },
    {
        "name":         "LosMonos_imputed",
        "display":      "Dawson Los Monos Canyon Reserve  ←  Buena Vista Park",
        "source_csv":   "BuenaVista_with_env_data.csv",
        "target_csv":   "LosMonos_with_env_data.csv",
        "model_prefix": "BuenaVista_vs_LosMonos",
        "source_label": "Buena Vista Park",
        "target_label": "Dawson Los Monos Canyon Reserve",
    },
]

MODEL_NAMES = ["RF_v1", "RF_v2", "RF_v3", "XGB_v1", "XGB_v2", "XGB_v3"]

def load_models(model_prefix: str) -> dict:
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODEL_DIR, f"{model_prefix}_{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


def safe_mode(x):
    vals = x.dropna()
    if len(vals) == 0:
        return ""
    m = vals.mode()
    return m.iloc[0] if len(m) > 0 else ""


def get_species_env_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile = (df.groupby(SPP_COL)[FEATURE_COLS]
                 .median()
                 .reset_index())

    if COMMON_COL in df.columns:
        common = (df.groupby(SPP_COL)[COMMON_COL]
                    .agg(safe_mode)
                    .reset_index())
        profile = profile.merge(common, on=SPP_COL, how="left")

    if TAXON_COL in df.columns:
        taxon = (df.groupby(SPP_COL)[TAXON_COL]
                   .agg(safe_mode)
                   .reset_index())
        profile = profile.merge(taxon, on=SPP_COL, how="left")

    return profile


def predict_with_all_models(models: dict,
                             profile_df: pd.DataFrame) -> pd.DataFrame:
    X = profile_df[FEATURE_COLS].fillna(0).values.astype(np.float32)

    for model_name, model in models.items():
        probs = model.predict_proba(X)[:, 1]
        profile_df[f"prob_{model_name}"] = probs

    prob_cols = [f"prob_{m}" for m in models]
    profile_df["prob_mean"] = profile_df[prob_cols].mean(axis=1)
    profile_df["prob_std"]  = profile_df[prob_cols].std(axis=1)
    profile_df["prob_min"]  = profile_df[prob_cols].min(axis=1)
    profile_df["prob_max"]  = profile_df[prob_cols].max(axis=1)
    profile_df["model_agreement"] = (
        (profile_df[prob_cols] >= 0.5).sum(axis=1).astype(str)
        + f"/{len(prob_cols)} models agree"
    )

    return profile_df

def run_imputation(task: dict) -> pd.DataFrame:
    print(f"")
    print(f"  {task['display']}")

    source_path = os.path.join(DATA_DIR, task["source_csv"])
    target_path = os.path.join(DATA_DIR, task["target_csv"])

    source_df = pd.read_csv(source_path).dropna(subset=FEATURE_COLS + [SPP_COL])
    target_df = pd.read_csv(target_path).dropna(subset=[SPP_COL])

    source_species = set(source_df[SPP_COL].unique())
    target_species = set(target_df[SPP_COL].unique())
    candidate_species = source_species - target_species
    print(f"  Source species  : {len(source_species):,}")
    print(f"  Target species  : {len(target_species):,}")
    print(f"  Shared species  : {len(source_species & target_species):,}")
    print(f"  Candidate (missing from target): {len(candidate_species):,}")

    if not candidate_species:
        print("  No missing species to impute.")
        return pd.DataFrame()

    candidate_df = source_df[source_df[SPP_COL].isin(candidate_species)].copy()
    profile_df   = get_species_env_profile(candidate_df)
    print(f"  Species with env profiles: {len(profile_df):,}")

    models = load_models(task["model_prefix"])
    if not models:
        print("  [ERROR] No models found. Run predictionmodel.py first.")
        return pd.DataFrame()
    print(f"  Loaded {len(models)} models: {list(models.keys())}")

    result_df = predict_with_all_models(models, profile_df)

    result_df = result_df.sort_values("prob_mean", ascending=False).reset_index(drop=True)
    result_df.index += 1
    result_df.index.name = "rank"

    n_above = (result_df["prob_mean"] >= PROB_THRESHOLD).sum()
    print(f"  Total candidate species: {len(result_df):,}")
    print(f"  Above {PROB_THRESHOLD:.0%} threshold: {n_above:,}")

    result_df["confidence"] = pd.cut(
        result_df["prob_mean"],
        bins=[0, 0.30, 0.50, 0.70, 0.90, 1.01],
        labels=["Low (<30%)", "Moderate (30-50%)", "Good (50-70%)",
                "High (70-90%)", "Very High (>90%)"]
    )

    out_cols = [SPP_COL]
    if COMMON_COL in result_df.columns:
        out_cols.append(COMMON_COL)
    if TAXON_COL in result_df.columns:
        out_cols.append(TAXON_COL)
    out_cols += ["confidence", "prob_mean", "prob_std", "prob_min", "prob_max",
                 "model_agreement",
                 "prob_RF_v1", "prob_RF_v2", "prob_RF_v3",
                 "prob_XGB_v1", "prob_XGB_v2", "prob_XGB_v3"]
    out_cols = [c for c in out_cols if c in result_df.columns]

    result_df = result_df[out_cols].round(4)


    csv_path = os.path.join(OUT_DIR, f"{task['name']}.csv")
    result_df.to_csv(csv_path)

    return result_df

if __name__ == "__main__":
    print(f"Data dir   : {DATA_DIR}")
    print(f"Model dir  : {MODEL_DIR}")
    print(f"Output dir : {OUT_DIR}")

    all_results = {}
    for task in IMPUTATION_TASKS:
        try:
            result = run_imputation(task)
            all_results[task["name"]] = result
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {task['name']}: {e}")
            traceback.print_exc()

    combined_rows = []
    for task_name, df in all_results.items():
        if df is not None and len(df) > 0:
            task_cfg = next(t for t in IMPUTATION_TASKS if t["name"] == task_name)
            df = df.copy()
            df["imputed_into"] = task_cfg["target_label"]
            df["sourced_from"] = task_cfg["source_label"]
            combined_rows.append(df.reset_index())

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined_path = os.path.join(OUT_DIR, "all_reserves_imputed_species.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n\nCombined list saved: {combined_path}")
        print(f"Total imputed species across all reserves: {len(combined):,}")