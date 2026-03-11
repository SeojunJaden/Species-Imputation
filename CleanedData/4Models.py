import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define the pairs: (Main Reserve Name, Main CSV, Backup CSV)
pairs = [
    ("Scripps", "Scripps_Filtered.csv", "TorreyPines_Filtered.csv"),
    ("ElliottChaparral", "ElliottChaparral_Filtered.csv", "MissionTrails_Filtered.csv"),
    ("LosMonos", "LosMonos_Filtered.csv", "BuenaVista_Filtered.csv"),
    ("MissionBay", "MissionBay_Filtered.csv", "TijuanaRiver_Filtered.csv")
]

def build_advanced_features(main_df, backup_df):
    """
    Aggregates backup reserve observation data to the species level 
    and engineers features for the Machine Learning model.
    """
    backup_species = backup_df[backup_df['scientific_name'].notna()].copy()
    
    # Extract date to calculate unique days observed
    backup_species['observed_on'] = pd.to_datetime(backup_species['observed_on'], errors='coerce')
    
    # Aggregate base features
    features = backup_species.groupby('scientific_name').agg(
        backup_obs_count=('id', 'count'),
        backup_unique_users=('user_id', 'nunique'),
        backup_unique_days=('observed_on', 'nunique'),
        lat_std=('latitude', 'std'),
        lon_std=('longitude', 'std')
    ).reset_index()
    
    # Fill NaN for spatial spread (happens when a species only has 1 observation)
    features['lat_std'].fillna(0, inplace=True)
    features['lon_std'].fillna(0, inplace=True)
    
    # Calculate proportion of observations that are 'research' grade
    research_counts = backup_species[backup_species['quality_grade'] == 'research'].groupby('scientific_name').size()
    features['backup_research_prop'] = features['scientific_name'].map(research_counts).fillna(0) / features['backup_obs_count']
    
    # Extract taxonomic info and common names
    taxons = backup_species[['scientific_name', 'iconic_taxon_name', 'common_name']].dropna(subset=['iconic_taxon_name']).drop_duplicates('scientific_name')
    features = features.merge(taxons, on='scientific_name', how='left')
    features['iconic_taxon_name'].fillna('Unknown', inplace=True)
    features['common_name'].fillna('Unknown', inplace=True)
    
    # Create Target Variable: 1 if present in Main Reserve, 0 if not
    main_species = set(main_df['scientific_name'].dropna().unique())
    features['present_in_main'] = features['scientific_name'].apply(lambda x: 1 if x in main_species else 0)
    
    return features

# Process each pair
for main_name, main_file, backup_file in pairs:
    print(f"\n{'='*50}")
    print(f"Processing {main_name} Reserve (Backup: {backup_file.split('_')[0]})")
    print(f"{'='*50}")
    
    # Load datasets
    main_df = pd.read_csv(main_file)
    backup_df = pd.read_csv(backup_file)
    
    # Engineer Features
    df_features = build_advanced_features(main_df, backup_df)
    
    # Save the feature-engineered dataset for future use
    csv_name = f"{main_name}_Engineered_Features.csv"
    df_features.to_csv(csv_name, index=False)
    print(f"[+] Saved feature matrix to {csv_name}")
    
    # Prepare Data for Modeling
    feature_cols = ['backup_obs_count', 'backup_unique_users', 'backup_unique_days', 'backup_research_prop', 'lat_std', 'lon_std']
    
    # One-Hot Encode Categorical feature (Taxon)
    X_cat = pd.get_dummies(df_features['iconic_taxon_name'], prefix='taxon', drop_first=True)
    X = pd.concat([df_features[feature_cols], X_cat], axis=1)
    y = df_features['present_in_main']
    
    # Initialize and Train Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        min_samples_leaf=5, 
        random_state=42, 
        class_weight='balanced', 
        oob_score=True
    )
    rf.fit(X, y)
    
    # Calculate OOB AUC
    oob_probs = rf.oob_decision_function_[:, 1]
    valid_idx = ~np.isnan(oob_probs)
    auc = roc_auc_score(y[valid_idx], oob_probs[valid_idx])
    
    print(f"[+] Model Evaluation (OOB AUC): {auc:.4f}")
    
    # Predict probabilities for ALL species in the feature set
    df_features['prob_in_main'] = rf.predict_proba(X)[:, 1]
    
    # Filter for unobserved species (Imputation Candidates)
    missing_species = df_features[df_features['present_in_main'] == 0]
    
    # Filter for robustness (minimum 3 observations in backup) to avoid noisy 1-off sightings
    robust_missing = missing_species[missing_species['backup_obs_count'] >= 3]
    top_missing = robust_missing.sort_values('prob_in_main', ascending=False).head(5)
    
    print(f"\nTop 5 Imputed Species (Highly likely to be in {main_name} but not yet observed):")
    for _, row in top_missing.iterrows():
        print(f"  - {row['scientific_name']} ({row['common_name']})")
        print(f"    Taxon: {row['iconic_taxon_name']} | Probability: {row['prob_in_main']:.2f} | Backup Obs: {row['backup_obs_count']}")

print("\nPipeline complete.")