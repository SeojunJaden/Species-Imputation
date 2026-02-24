"""
UCSD Natural Reserve System - ML Model Error Test
==================================================
Evaluates 4 reserve-specific Random Forest classifiers using an 80/20
stratified train-test split. Reports per-class and aggregate error metrics
for each reserve model.

Metrics reported per model:
  - Total misclassifications & error rate
  - Accuracy, Balanced Accuracy
  - ROC-AUC, Average Precision (PR-AUC)
  - Matthews Correlation Coefficient (MCC)
  - F1 Weighted, F1 Macro
  - Confusion Matrix (TN, FP, FN, TP)
  - Per-class Precision, Recall, F1
    (class 0 = underrepresented, class 1 = well-represented)

Output:
  model_test_errors.csv  — full numeric results for all 4 reserves
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, f1_score,
    precision_score, recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────

RESERVE_FILES = {
    'Elliott_Chaparral': '/mnt/user-data/uploads/ElliottChaparral_Filtered.csv',
    'Mission_Bay':       '/mnt/user-data/uploads/MissionBay_Filtered.csv',
    'Scripps':           '/mnt/user-data/uploads/Scripps_Filtered.csv',
    'Los_Monos':         '/mnt/user-data/uploads/LosMonos_Filtered.csv',
}

SNAPSHOT_DATE = pd.Timestamp('2026-02-18')

TAXON_ORDER = [
    'Plantae', 'Insecta', 'Aves', 'Mammalia', 'Reptilia', 'Amphibia',
    'Arachnida', 'Fungi', 'Mollusca', 'Actinopterygii', 'Chromista',
    'Animalia', 'Protozoa', 'Other',
]

FEATURE_COLS = [
    'obs_count', 'unique_observers', 'unique_dates', 'research_count',
    'research_ratio', 'days_since_last', 'days_since_first',
    'observation_span_days', 'years_observed', 'lat_std', 'lon_std',
    'spatial_spread', 'taxon_encoded', 'n_other_reserves', 'in_other_reserves',
]

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)

TEST_SIZE   = 0.2
RANDOM_SEED = 42

# ── 1. Load Data ──────────────────────────────────────────────────────────────

print("Loading reserve data...")
raw = {}
for name, path in RESERVE_FILES.items():
    df = pd.read_csv(path, parse_dates=['observed_on'])
    df['reserve'] = name
    raw[name] = df
    print(f"  {name}: {len(df):,} observations")

all_obs = pd.concat(raw.values(), ignore_index=True)

# ── 2. Feature Engineering ────────────────────────────────────────────────────

def engineer_features(df_reserve, reserve_name):
    """Compute per-species features from a single reserve's observations."""
    grp = df_reserve.groupby('scientific_name')

    feats = pd.DataFrame(
        {'scientific_name': list(grp.groups.keys())}
    ).set_index('scientific_name')

    # Observation frequency
    feats['obs_count']        = grp['id'].count()
    feats['unique_observers'] = grp['user_id'].nunique()
    feats['unique_dates']     = grp['observed_on'].nunique()

    # Quality grade
    research_obs = df_reserve[df_reserve['quality_grade'] == 'research']
    feats['research_count']   = (
        research_obs.groupby('scientific_name')['id'].count().reindex(feats.index).fillna(0)
    )
    feats['research_ratio']   = feats['research_count'] / feats['obs_count']

    # Temporal features
    last_seen  = grp['observed_on'].max()
    first_seen = grp['observed_on'].min()
    feats['days_since_last']       = (SNAPSHOT_DATE - last_seen).dt.days
    feats['days_since_first']      = (SNAPSHOT_DATE - first_seen).dt.days
    feats['observation_span_days'] = (last_seen - first_seen).dt.days
    feats['years_observed']        = grp['observed_on'].apply(
        lambda x: x.dt.year.nunique()
    )

    # Spatial spread
    feats['lat_std']         = grp['latitude'].std().fillna(0)
    feats['lon_std']         = grp['longitude'].std().fillna(0)
    feats['spatial_spread']  = feats['lat_std'] + feats['lon_std']

    # Taxonomic group (label-encoded)
    taxon_map = (
        df_reserve.drop_duplicates('scientific_name')
        .set_index('scientific_name')['iconic_taxon_name']
    )
    feats['taxon_group'] = taxon_map.reindex(feats.index).fillna('Other')
    le = LabelEncoder()
    le.fit(TAXON_ORDER)
    feats['taxon_encoded'] = feats['taxon_group'].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else len(TAXON_ORDER) - 1
    )

    # Cross-reserve presence
    species_reserve_counts     = all_obs.groupby('scientific_name')['reserve'].nunique()
    feats['n_other_reserves']  = (
        species_reserve_counts.reindex(feats.index).fillna(1) - 1
    )
    feats['in_other_reserves'] = (feats['n_other_reserves'] > 0).astype(int)

    return feats.reset_index()


# ── 3. Labeling ───────────────────────────────────────────────────────────────

def label_species(feats):
    """
    Label each species as underrepresented (0) or well-represented (1).

    Underrepresented if ANY of:
      - Total observations <= 3
      - Only 1 unique observer
      - Observed in only 1 year AND total observations <= 5
    """
    under = (
        (feats['obs_count'] <= 3) |
        (feats['unique_observers'] == 1) |
        ((feats['years_observed'] == 1) & (feats['obs_count'] <= 5))
    )
    feats['label'] = np.where(under, 0, 1)
    return feats


# ── 4. Train-Test Split & Evaluation ─────────────────────────────────────────

results = []

print("\nRunning train-test evaluation (80/20 stratified split)...")
print("=" * 60)

for name, df in raw.items():
    feats = label_species(engineer_features(df, name))
    X = feats[FEATURE_COLS].fillna(0)
    y = feats['label']

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # Train model
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    # Predict
    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc        = accuracy_score(y_test, y_pred)
    bal_acc    = balanced_accuracy_score(y_test, y_pred)
    roc_auc    = roc_auc_score(y_test, y_proba)
    avg_prec   = average_precision_score(y_test, y_proba)
    mcc        = matthews_corrcoef(y_test, y_pred)
    f1_w       = f1_score(y_test, y_pred, average='weighted')
    f1_macro   = f1_score(y_test, y_pred, average='macro')

    # Per-class (0 = underrepresented, 1 = well-represented)
    prec_under = precision_score(y_test, y_pred, pos_label=0)
    rec_under  = recall_score(y_test, y_pred, pos_label=0)
    f1_under   = f1_score(y_test, y_pred, pos_label=0)
    prec_well  = precision_score(y_test, y_pred, pos_label=1)
    rec_well   = recall_score(y_test, y_pred, pos_label=1)
    f1_well    = f1_score(y_test, y_pred, pos_label=1)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    n_test   = len(y_test)
    n_errors = int((y_test != y_pred).sum())

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n=== {name} ===")
    print(f"Train: {len(X_train)} species | Test: {n_test} species "
          f"({int(y_test.sum())} well-represented, {int((y_test==0).sum())} underrepresented)")
    print(f"Misclassifications : {n_errors} / {n_test}  ({100*n_errors/n_test:.1f}% error rate)")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Balanced Accuracy  : {bal_acc:.4f}")
    print(f"ROC-AUC            : {roc_auc:.4f}")
    print(f"Avg Precision (PR) : {avg_prec:.4f}")
    print(f"MCC                : {mcc:.4f}")
    print(f"F1 Weighted        : {f1_w:.4f}")
    print(f"F1 Macro           : {f1_macro:.4f}")
    print(f"Confusion Matrix   : TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  Underrepresented | Precision={prec_under:.4f}  Recall={rec_under:.4f}  F1={f1_under:.4f}")
    print(f"  Well-represented | Precision={prec_well:.4f}  Recall={rec_well:.4f}  F1={f1_well:.4f}")

    results.append({
        'Reserve':             name,
        'Train_Size':          len(X_train),
        'Test_Size':           n_test,
        'Misclassifications':  n_errors,
        'Error_Rate_%':        round(100 * n_errors / n_test, 2),
        'Accuracy':            round(acc, 4),
        'Balanced_Accuracy':   round(bal_acc, 4),
        'ROC_AUC':             round(roc_auc, 4),
        'Avg_Precision_PR':    round(avg_prec, 4),
        'MCC':                 round(mcc, 4),
        'F1_Weighted':         round(f1_w, 4),
        'F1_Macro':            round(f1_macro, 4),
        'TN':                  int(tn),
        'FP':                  int(fp),
        'FN':                  int(fn),
        'TP':                  int(tp),
        'Underrep_Precision':  round(prec_under, 4),
        'Underrep_Recall':     round(rec_under, 4),
        'Underrep_F1':         round(f1_under, 4),
        'WellRep_Precision':   round(prec_well, 4),
        'WellRep_Recall':      round(rec_well, 4),
        'WellRep_F1':          round(f1_well, 4),
    })

# ── 5. Save Results ───────────────────────────────────────────────────────────

output_path = '/mnt/user-data/outputs/model_test_errors.csv'
pd.DataFrame(results).to_csv(output_path, index=False)
print(f"\n{'='*60}")
print(f"Results saved to: {output_path}")
