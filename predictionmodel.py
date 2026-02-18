import os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay
import xgboost as xgb


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "ProcessedData")
OUTPUT_DIR = os.path.join(BASE_DIR, "model_outputs")
IMPUTE_DIR = os.path.join(BASE_DIR, "imputed_species")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMPUTE_DIR, exist_ok=True)

FEATURE_COLS = [
    "elevation", "slope", "aspect", "ndvi",
    "landcover", "impervious", "bathymetry",
    "soil_sand", "soil_ph", "soil_clay",
]
SPP_COL    = "scientific_name"
COMMON_COL = "common_name"
TAXON_COL  = "iconic_taxon_name"
RANDOM_STATE = 42

TASKS = [
    {
        "name":         "ElliottChaparral",
        "display":      "Elliott Chaparral  ←  Mission Trails",
        "target_csv":   "ElliottChaparral_with_env_data.csv",
        "source_csv":   "MissionTrails_with_env_data.csv",
        "target_label": "Elliott Chaparral",
        "source_label": "Mission Trails",
    },
    {
        "name":         "KendallFrost",
        "display":      "Kendall Frost / Mission Bay  ←  Tijuana River",
        "target_csv":   "MissionBay_with_env_data.csv",
        "source_csv":   "TijuanaRiver_with_env_data.csv",
        "target_label": "Kendall Frost / Mission Bay",
        "source_label": "Tijuana River",
    },
    {
        "name":         "LosMonos",
        "display":      "Dawson Los Monos Canyon  ←  Buena Vista Park",
        "target_csv":   "LosMonos_with_env_data.csv",
        "source_csv":   "BuenaVista_with_env_data.csv",
        "target_label": "Dawson Los Monos Canyon",
        "source_label": "Buena Vista Park",
    },
]

MODEL_NAMES = ["RF_v1", "RF_v2", "RF_v3", "XGB_v1", "XGB_v2", "XGB_v3"]



def build_rf(v):
    configs = {
        1: dict(n_estimators=300, max_depth=None, min_samples_leaf=1,
                max_features="sqrt",  class_weight="balanced"),
        2: dict(n_estimators=500, max_depth=15,   min_samples_leaf=3,
                max_features=0.5,     class_weight="balanced_subsample"),
        3: dict(n_estimators=200, max_depth=10,   min_samples_leaf=5,
                max_features="log2",  class_weight="balanced", oob_score=True),
    }
    return RandomForestClassifier(**configs[v], random_state=RANDOM_STATE, n_jobs=-1)


def build_xgb(v, spw=1.0):
    configs = {
        1: dict(n_estimators=300, max_depth=6,  learning_rate=0.10,
                subsample=0.8, colsample_bytree=0.8, gamma=0,
                reg_alpha=0,   reg_lambda=1),
        2: dict(n_estimators=500, max_depth=4,  learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.6, gamma=1,
                reg_alpha=0.1, reg_lambda=1.5),
        3: dict(n_estimators=200, max_depth=8,  learning_rate=0.15,
                subsample=0.9, colsample_bytree=1.0, gamma=0.5,
                reg_alpha=0.5, reg_lambda=2.0),
    }
    return xgb.XGBClassifier(**configs[v], scale_pos_weight=spw,
                              eval_metric="logloss", random_state=RANDOM_STATE,
                              n_jobs=-1, verbosity=0)

def build_habitat_training_data(target_df, all_df, target_label):

    pos = target_df[FEATURE_COLS].copy()
    pos["y"] = 1

    neg = all_df[~all_df["reserve"].isin([target_label])][FEATURE_COLS].copy()
    n_neg = min(len(neg), len(pos) * 3)
    neg = neg.sample(n=n_neg, random_state=RANDOM_STATE)
    neg["y"] = 0

    combined = pd.concat([pos, neg], ignore_index=True)
    combined = combined.dropna(subset=FEATURE_COLS)

    X = combined[FEATURE_COLS].values.astype(np.float32)
    y = combined["y"].values.astype(int)

    print(f"  Training data: {y.sum():,} target-reserve locations (y=1)  |  "
          f"{(y==0).sum():,} other-reserve locations (y=0)")
    return X, y


def train_habitat_models(task, all_df):
    name   = task["name"]
    label  = task["target_label"]
    print(f"\n{'='*60}")
    print(f"  Training habitat model for: {label}")
    print(f"{'='*60}")

    target_df = all_df[all_df["reserve"] == label]
    X, y = build_habitat_training_data(target_df, all_df, label)

    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = {}
    results = {}

    for v in [1, 2, 3]:
        mname = f"RF_v{v}"
        print(f"  Training {mname}...", end=" ", flush=True)
        clf = build_rf(v)
        clf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        print(f"AUC={auc:.4f}")
        models[mname] = clf
        results[mname] = auc
        joblib.dump(clf, os.path.join(OUTPUT_DIR, f"{name}_{mname}.pkl"))

    for v in [1, 2, 3]:
        mname = f"XGB_v{v}"
        print(f"  Training {mname}...", end=" ", flush=True)
        clf = build_xgb(v, spw=spw)
        clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        print(f"AUC={auc:.4f}")
        models[mname] = clf
        results[mname] = auc
        joblib.dump(clf, os.path.join(OUTPUT_DIR, f"{name}_{mname}.pkl"))

    best = max(results, key=results.get)
    print(f"\n  Best model: {best}  AUC={results[best]:.4f}")
    return models, results


def safe_mode(x):
    vals = x.dropna()
    if len(vals) == 0:
        return ""
    m = vals.mode()
    return m.iloc[0] if len(m) > 0 else ""


def get_species_env_profile(df):
    """Median env fingerprint per species + common name + taxon."""
    profile = df.groupby(SPP_COL)[FEATURE_COLS].median().reset_index()
    if COMMON_COL in df.columns:
        common = df.groupby(SPP_COL)[COMMON_COL].agg(safe_mode).reset_index()
        profile = profile.merge(common, on=SPP_COL, how="left")
    if TAXON_COL in df.columns:
        taxon = df.groupby(SPP_COL)[TAXON_COL].agg(safe_mode).reset_index()
        profile = profile.merge(taxon, on=SPP_COL, how="left")
    return profile


def run_imputation(task, models):
    print(f"\n{'='*60}")
    print(f"  IMPUTING: {task['display']}")
    print(f"{'='*60}")

    source_df = pd.read_csv(os.path.join(DATA_DIR, task["source_csv"])).dropna(subset=[SPP_COL])
    target_df = pd.read_csv(os.path.join(DATA_DIR, task["target_csv"])).dropna(subset=[SPP_COL])

    source_species = set(source_df[SPP_COL].unique())
    target_species = set(target_df[SPP_COL].unique())
    candidates     = source_species - target_species

    print(f"  Source species : {len(source_species):,}")
    print(f"  Target species : {len(target_species):,}")
    print(f"  Already shared : {len(source_species & target_species):,}")
    print(f"  Candidates (missing from target): {len(candidates):,}")

    if not candidates:
        print("  No missing species — nothing to impute.")
        return pd.DataFrame()
    cand_df  = source_df[source_df[SPP_COL].isin(candidates)].copy()
    cand_df  = cand_df.dropna(subset=FEATURE_COLS)
    profile  = get_species_env_profile(cand_df)
    print(f"  Candidates with full env profiles: {len(profile):,}")
    X = profile[FEATURE_COLS].fillna(0).values.astype(np.float32)

    for mname, model in models.items():
        profile[f"prob_{mname}"] = model.predict_proba(X)[:, 1]

    prob_cols = [f"prob_{m}" for m in models]
    profile["prob_mean"] = profile[prob_cols].mean(axis=1)
    profile["prob_std"]  = profile[prob_cols].std(axis=1)
    profile["models_agree"] = (profile[prob_cols] >= 0.5).sum(axis=1)

    profile = profile.sort_values("prob_mean", ascending=False).reset_index(drop=True)
    profile.index += 1
    profile.index.name = "rank"
    profile["confidence"] = pd.cut(
        profile["prob_mean"],
        bins=[0, 0.30, 0.50, 0.70, 0.90, 1.01],
        labels=["Low (<30%)", "Moderate (30-50%)", "Good (50-70%)",
                "High (70-90%)", "Very High (>90%)"]
    )

    print(f"\n  Confidence breakdown:")
    print(profile["confidence"].value_counts().sort_index().to_string())


    out_cols = [SPP_COL]
    for c in [COMMON_COL, TAXON_COL]:
        if c in profile.columns:
            out_cols.append(c)
    out_cols += ["confidence", "prob_mean", "prob_std", "models_agree"] + prob_cols
    out_cols = [c for c in out_cols if c in profile.columns]
    profile_filtered = profile[profile["prob_mean"] >= 0.30]
    out_path = os.path.join(IMPUTE_DIR, f"{task['name']}_imputed.csv")
    profile_filtered[out_cols].round(4).to_csv(out_path)
    print(f"  Species above 30% saved to CSV: {len(profile_filtered):,}")
    print(f"\n  Saved: {out_path}")
    plot_imputed(profile, task)

    return profile


def plot_imputed(df, task, top_n=25):
    top = df.head(top_n).iloc[::-1]
    labels = top[SPP_COL].str[:40]
    means  = top["prob_mean"]
    stds   = top["prob_std"]

    taxon_colors = {
        "Plantae":"#4CAF50","Aves":"#2196F3","Mammalia":"#FF9800",
        "Reptilia":"#9C27B0","Amphibia":"#00BCD4","Insecta":"#FFEB3B",
        "Arachnida":"#F44336","Fungi":"#795548","Animalia":"#607D8B",
    }
    colors = [taxon_colors.get(str(t), "#B0BEC5")
              for t in top.get(TAXON_COL, pd.Series([""] * len(top)))]

    fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.35)))
    ax.barh(range(len(top)), means, xerr=stds, color=colors,
            edgecolor="white", linewidth=0.4,
            error_kw=dict(ecolor="gray", capsize=3, linewidth=0.8))
    ax.axvline(0.5, color="red", linestyle="--", lw=1, label="50% threshold")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("P(species belongs in target reserve habitat)")
    ax.set_title(f"Predicted Missing Species\n{task['display']}", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1)

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=t) for t, c in taxon_colors.items()
               if t in top.get(TAXON_COL, pd.Series()).values]
    if patches:
        ax.legend(handles=patches, fontsize=7, loc="lower right", title="Taxon")

    plt.tight_layout()
    plt.savefig(os.path.join(IMPUTE_DIR, f"{task['name']}_top{top_n}.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    print(f"Data dir   : {DATA_DIR}  (exists={os.path.isdir(DATA_DIR)})")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Impute dir : {IMPUTE_DIR}")
    reserve_files = {
        "Elliott Chaparral":          "ElliottChaparral_with_env_data.csv",
        "Mission Trails":             "MissionTrails_with_env_data.csv",
        "Kendall Frost / Mission Bay": "MissionBay_with_env_data.csv",
        "Tijuana River":              "TijuanaRiver_with_env_data.csv",
        "Dawson Los Monos Canyon":    "LosMonos_with_env_data.csv",
        "Buena Vista Park":           "BuenaVista_with_env_data.csv",
    }

    frames = []
    for reserve_label, fname in reserve_files.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  [WARNING] Missing: {fname}")
            continue
        df = pd.read_csv(path).dropna(subset=FEATURE_COLS)
        df["reserve"] = reserve_label
        print(f"  Loaded {len(df):,} rows — {reserve_label}")
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    print(f"\n  Total observations across all reserves: {len(all_df):,}")
    all_results = {}
    for task in TASKS:
        try:
            models, auc_results = train_habitat_models(task, all_df)
            result_df = run_imputation(task, models)
            all_results[task["name"]] = result_df
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {task['name']}: {e}")
            traceback.print_exc()

    rows = []
    for task in TASKS:
        df = all_results.get(task["name"])
        if df is not None and len(df) > 0:
            d = df.reset_index().copy()
            d["imputed_into"] = task["target_label"]
            d["sourced_from"] = task["source_label"]
            rows.append(d)

    if rows:
        combined = pd.concat(rows, ignore_index=True)
        combined.to_csv(os.path.join(IMPUTE_DIR, "all_reserves_imputed.csv"), index=False)
        print(f"\n  Combined CSV: {os.path.join(IMPUTE_DIR, 'all_reserves_imputed.csv')}")
        print(f"  Total imputed species: {len(combined):,}")