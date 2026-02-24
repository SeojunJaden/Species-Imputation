"""
UCSD Natural Reserve System - Species Gap Analysis ML Pipeline
=============================================================
Build 4 reserve-specific Random Forest models to predict missing/underrepresented species.
Each reserve:
  1. Engineers species-level features from iNaturalist observations
  2. Trains a Random Forest classifier (well-represented vs underrepresented)
  3. Scores candidate species (observed elsewhere but absent/sparse locally)
  4. Outputs ranked predictions for domain expert review
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ─────────────────────────────────────────────────────────────

RESERVE_FILES = {
    'Elliott_Chaparral': 'Desktop/ImputatingSpecies/Species-Imputation/CleanedData/ElliottChaparral_Filtered.csv',
    'Mission_Bay':       'Desktop/ImputatingSpecies/Species-Imputation/CleanedData/MissionBay_Filtered.csv',
    'Scripps':           'Desktop/ImputatingSpecies/Species-Imputation/CleanedData/Scripps_Filtered.csv',
    'Los_Monos':         'Desktop/ImputatingSpecies/Species-Imputation/CleanedData/LosMonos_Filtered.csv',
}

raw = {}
for name, path in RESERVE_FILES.items():
    df = pd.read_csv(path, parse_dates=['observed_on'])
    df['reserve'] = name
    raw[name] = df
    print(f"Loaded {name}: {len(df)} observations, {df['scientific_name'].nunique()} species")

all_obs = pd.concat(raw.values(), ignore_index=True)
SNAPSHOT_DATE = pd.Timestamp('2026-02-18')  # today

# ── 2. Feature Engineering (species-level, per reserve) ──────────────────────

TAXON_ORDER = ['Plantae','Insecta','Aves','Mammalia','Reptilia','Amphibia',
               'Arachnida','Fungi','Mollusca','Actinopterygii','Chromista','Animalia','Protozoa','Other']

def engineer_features(df_reserve, reserve_name):
    """Compute per-species features from a single reserve's observations."""
    grp = df_reserve.groupby('scientific_name')
    
    feats = pd.DataFrame()
    feats['scientific_name']   = list(grp.groups.keys())
    feats = feats.set_index('scientific_name')

    # Observation frequency
    feats['obs_count']         = grp['id'].count()
    feats['unique_observers']  = grp['user_id'].nunique()
    feats['unique_dates']      = grp['observed_on'].nunique()

    # Quality
    research_mask = df_reserve['quality_grade'] == 'research'
    feats['research_count']    = df_reserve[research_mask].groupby('scientific_name')['id'].count()
    feats['research_count']    = feats['research_count'].fillna(0)
    feats['research_ratio']    = feats['research_count'] / feats['obs_count']

    # Temporal recency
    last_seen = grp['observed_on'].max()
    feats['days_since_last']   = (SNAPSHOT_DATE - last_seen).dt.days
    first_seen = grp['observed_on'].min()
    feats['days_since_first']  = (SNAPSHOT_DATE - first_seen).dt.days
    feats['observation_span_days'] = (last_seen - first_seen).dt.days

    # Temporal spread (how many distinct years)
    feats['years_observed']    = grp['observed_on'].apply(lambda x: x.dt.year.nunique())

    # Spatial spread
    feats['lat_std']           = grp['latitude'].std().fillna(0)
    feats['lon_std']           = grp['longitude'].std().fillna(0)
    feats['spatial_spread']    = feats['lat_std'] + feats['lon_std']

    # Taxonomic group (encoded)
    taxon_map = df_reserve.drop_duplicates('scientific_name').set_index('scientific_name')['iconic_taxon_name']
    feats['taxon_group']       = taxon_map.reindex(feats.index).fillna('Other')
    le = LabelEncoder()
    le.fit(TAXON_ORDER)
    feats['taxon_encoded']     = feats['taxon_group'].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else len(TAXON_ORDER)-1
    )

    # Cross-reserve presence: how many OTHER reserves have this species
    species_reserve_counts = all_obs.groupby('scientific_name')['reserve'].nunique()
    feats['n_other_reserves']  = species_reserve_counts.reindex(feats.index).fillna(1) - 1

    # Is species in any other reserve?
    feats['in_other_reserves'] = (feats['n_other_reserves'] > 0).astype(int)

    feats = feats.reset_index()
    feats['reserve'] = reserve_name
    return feats


print("\n── Engineering features ──")
reserve_features = {}
for name, df in raw.items():
    feats = engineer_features(df, name)
    reserve_features[name] = feats
    print(f"{name}: {len(feats)} species × {feats.shape[1]} features")

# ── 3. Labeling: well-represented vs underrepresented ────────────────────────

def label_species(feats):
    """
    Underrepresented heuristic:
      - obs_count <= 3  OR
      - unique_observers == 1  OR
      - years_observed == 1 AND obs_count <= 5
    Well-represented: observed 4+ times, by 2+ observers, across 2+ years (or high research ratio).
    """
    under = (
        (feats['obs_count'] <= 3) |
        (feats['unique_observers'] == 1) |
        ((feats['years_observed'] == 1) & (feats['obs_count'] <= 5))
    )
    feats['label'] = np.where(under, 0, 1)  # 0 = underrepresented, 1 = well-represented
    return feats

print("\n── Labeling ──")
for name in reserve_features:
    reserve_features[name] = label_species(reserve_features[name])
    vc = reserve_features[name]['label'].value_counts()
    print(f"{name}: well-represented={vc.get(1,0)}, underrepresented={vc.get(0,0)}")

# ── 4. Train Reserve-Specific Random Forest Models ───────────────────────────

FEATURE_COLS = [
    'obs_count', 'unique_observers', 'unique_dates', 'research_count',
    'research_ratio', 'days_since_last', 'days_since_first',
    'observation_span_days', 'years_observed', 'lat_std', 'lon_std',
    'spatial_spread', 'taxon_encoded', 'n_other_reserves', 'in_other_reserves'
]

models = {}
cv_results = {}

print("\n── Training Models ──")
for name, feats in reserve_features.items():
    X = feats[FEATURE_COLS].fillna(0)
    y = feats['label']
    
    # Skip if only one class (can't train)
    if y.nunique() < 2:
        print(f"{name}: skipping (only one class present)")
        continue
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation (stratified, 5-fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='f1_weighted')
    cv_results[name] = cv_scores
    
    # Fit final model on all reserve data
    rf.fit(X, y)
    models[name] = rf
    
    print(f"{name}: CV F1 = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── 5. Feature Importances ───────────────────────────────────────────────────

print("\n── Top Feature Importances per Reserve ──")
fi_records = []
for name, rf in models.items():
    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    fi_records.append({'Reserve': name, **dict(zip(imp.index[:5], imp.values[:5]))})
    print(f"\n{name}:")
    for feat, val in imp.head(5).items():
        print(f"  {feat:<30} {val:.4f}")

# ── 6. Candidate Prediction: Missing / Underrepresented Species ───────────────

"""
For each reserve, we identify candidate species two ways:
  A) ABSENT candidates  – seen in ≥1 other reserve but NOT in this reserve
  B) PRESENT but sparse – already in reserve with label=0 (underrepresented)
     The model scores these and ranks by predicted probability of being well-represented
     (low prob = truly underrepresented / needs more survey effort).
"""

all_species_global = all_obs['scientific_name'].unique()

print("\n── Generating Candidate Predictions ──")

predictions = {}

for name, rf in models.items():
    feats_reserve = reserve_features[name].copy()
    reserve_species = set(feats_reserve['scientific_name'])
    
    # --- Absent candidates ---
    other_reserve_species = set(all_obs[all_obs['reserve'] != name]['scientific_name'].unique())
    absent_species = other_reserve_species - reserve_species
    
    # Build feature rows for absent species using cross-reserve stats
    absent_rows = []
    for sp in absent_species:
        sp_global = all_obs[all_obs['scientific_name'] == sp]
        taxon = sp_global['iconic_taxon_name'].mode()
        taxon_val = taxon.iloc[0] if not taxon.empty else 'Other'
        le = LabelEncoder(); le.fit(TAXON_ORDER)
        taxon_enc = le.transform([taxon_val])[0] if taxon_val in le.classes_ else len(TAXON_ORDER)-1
        
        n_other = sp_global['reserve'].nunique()  # how many reserves have it
        global_obs_count = len(sp_global)
        
        absent_rows.append({
            'scientific_name': sp,
            'common_name': sp_global['common_name'].mode().iloc[0] if not sp_global['common_name'].mode().empty else '',
            'obs_count': 0,
            'unique_observers': 0,
            'unique_dates': 0,
            'research_count': 0,
            'research_ratio': sp_global['quality_grade'].eq('research').mean(),
            'days_since_last': 9999,
            'days_since_first': 9999,
            'observation_span_days': 0,
            'years_observed': 0,
            'lat_std': 0,
            'lon_std': 0,
            'spatial_spread': 0,
            'taxon_encoded': taxon_enc,
            'taxon_group': taxon_val,
            'n_other_reserves': n_other,
            'in_other_reserves': 1,
            'global_obs_count': global_obs_count,
            'candidate_type': 'Absent from reserve',
            'label': 0,
        })
    
    absent_df = pd.DataFrame(absent_rows) if absent_rows else pd.DataFrame()
    
    # --- Underrepresented present species ---
    under_df = feats_reserve[feats_reserve['label'] == 0].copy()
    # Add common_name
    cn_map = all_obs[all_obs['reserve']==name].drop_duplicates('scientific_name').set_index('scientific_name')['common_name']
    under_df['common_name'] = under_df['scientific_name'].map(cn_map)
    under_df['candidate_type'] = 'Underrepresented in reserve'
    under_df['global_obs_count'] = under_df['scientific_name'].map(
        all_obs.groupby('scientific_name')['id'].count()
    )
    
    # Score all candidates
    candidate_dfs = []
    
    if not absent_df.empty:
        Xa = absent_df[FEATURE_COLS].fillna(0)
        absent_df['prob_well_represented'] = rf.predict_proba(Xa)[:, 1]
        # Priority score for absent: higher n_other_reserves + global obs = higher priority
        absent_df['priority_score'] = (
            absent_df['n_other_reserves'] * 0.5 +
            np.log1p(absent_df['global_obs_count']) * 0.5
        )
        candidate_dfs.append(absent_df[['scientific_name','common_name','taxon_group',
                                         'candidate_type','obs_count','global_obs_count',
                                         'n_other_reserves','prob_well_represented','priority_score']])
    
    if not under_df.empty:
        Xu = under_df[FEATURE_COLS].fillna(0)
        under_df['prob_well_represented'] = rf.predict_proba(Xu)[:, 1]
        under_df['priority_score'] = 1 - under_df['prob_well_represented']
        candidate_dfs.append(under_df[['scientific_name','common_name','taxon_group',
                                        'candidate_type','obs_count','global_obs_count',
                                        'n_other_reserves','prob_well_represented','priority_score']])
    
    if candidate_dfs:
        candidates = pd.concat(candidate_dfs, ignore_index=True)
        candidates = candidates.sort_values(['candidate_type','priority_score'], ascending=[True, False])
        predictions[name] = candidates
        print(f"\n{name}:")
        print(f"  Absent candidates:          {(candidates['candidate_type']=='Absent from reserve').sum()}")
        print(f"  Underrepresented (present): {(candidates['candidate_type']=='Underrepresented in reserve').sum()}")
        print(f"  Top 5 absent candidates:")
        top_absent = candidates[candidates['candidate_type']=='Absent from reserve'].head(5)
        for _, r in top_absent.iterrows():
            print(f"    {r['scientific_name']:<40} ({r['taxon_group']}) - seen in {int(r['n_other_reserves'])} other reserves")

# ── 7. Save Results ───────────────────────────────────────────────────────────

# Per-reserve CSVs
output_paths = []
for name, cand_df in predictions.items():
    path = f'/mnt/user-data/outputs/{name}_species_predictions.csv'
    cand_df.round(4).to_csv(path, index=False)
    output_paths.append(path)
    print(f"Saved: {path}")

# Summary: model performance
summary_rows = []
for name, scores in cv_results.items():
    summary_rows.append({
        'Reserve': name,
        'CV_F1_Mean': round(scores.mean(), 4),
        'CV_F1_Std': round(scores.std(), 4),
        'n_species_in_reserve': len(reserve_features[name]),
        'n_well_represented': int((reserve_features[name]['label']==1).sum()),
        'n_underrepresented': int((reserve_features[name]['label']==0).sum()),
        'n_absent_candidates': int((predictions[name]['candidate_type']=='Absent from reserve').sum()),
    })
summary_df = pd.DataFrame(summary_rows)
summary_path = '/mnt/user-data/outputs/model_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved summary: {summary_path}")

# Feature importance CSV
fi_all = []
for name, rf in models.items():
    for feat, imp in zip(FEATURE_COLS, rf.feature_importances_):
        fi_all.append({'Reserve': name, 'Feature': feat, 'Importance': round(imp, 5)})
fi_df = pd.DataFrame(fi_all).sort_values(['Reserve','Importance'], ascending=[True,False])
fi_path = '/mnt/user-data/outputs/feature_importances.csv'
fi_df.to_csv(fi_path, index=False)
print(f"Saved feature importances: {fi_path}")

print("\n✅ Pipeline complete.")
print(f"Output files: {output_paths + [summary_path, fi_path]}")