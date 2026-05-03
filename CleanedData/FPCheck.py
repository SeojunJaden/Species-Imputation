import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    """Aggregates backup reserve data and builds features."""
    backup_species = backup_df[backup_df['scientific_name'].notna()].copy()
    backup_species['observed_on'] = pd.to_datetime(backup_species['observed_on'], errors='coerce')
    
    features = backup_species.groupby('scientific_name').agg(
        backup_obs_count=('id', 'count'),
        backup_unique_users=('user_id', 'nunique'),
        backup_unique_days=('observed_on', 'nunique'),
        lat_std=('latitude', 'std'),
        lon_std=('longitude', 'std')
    ).reset_index()
    
    features['lat_std'].fillna(0, inplace=True)
    features['lon_std'].fillna(0, inplace=True)
    
    research_counts = backup_species[backup_species['quality_grade'] == 'research'].groupby('scientific_name').size()
    features['backup_research_prop'] = features['scientific_name'].map(research_counts).fillna(0) / features['backup_obs_count']
    
    taxons = backup_species[['scientific_name', 'iconic_taxon_name']].dropna().drop_duplicates('scientific_name')
    features = features.merge(taxons, on='scientific_name', how='left')
    features['iconic_taxon_name'].fillna('Unknown', inplace=True)
    
    # Define Target: 1 if present in Main Reserve
    main_species = set(main_df['scientific_name'].dropna().unique())
    features['present_in_main'] = features['scientific_name'].apply(lambda x: 1 if x in main_species else 0)
    
    return features, main_species

for main_name, main_file, backup_file in pairs:
    print(f"\n{'='*60}")
    print(f"Comparing Actual vs. Predicted for: {main_name}")
    print(f"{'='*60}")
    
    # Load data
    main_df = pd.read_csv(main_file)
    backup_df = pd.read_csv(backup_file)
    
    # Feature Engineering
    df_features, main_species_set = build_advanced_features(main_df, backup_df)
    
    # Prepare Data for Modeling
    feature_cols = ['backup_obs_count', 'backup_unique_users', 'backup_unique_days', 'backup_research_prop', 'lat_std', 'lon_std']
    X_cat = pd.get_dummies(df_features['iconic_taxon_name'], prefix='taxon', drop_first=True)
    X = pd.concat([df_features[feature_cols], X_cat], axis=1)
    y = df_features['present_in_main']
    
    # Train Model
    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        min_samples_leaf=5, 
        random_state=42, 
        class_weight='balanced'
    )
    rf.fit(X, y)
    
    # Predict using a 0.5 probability threshold
    df_features['predicted_present'] = rf.predict(X)
    
    # ---------------------------------------------------------
    # CALCULATE METRICS
    # ---------------------------------------------------------
    total_main_species = len(main_species_set)
    total_backup_species = len(df_features)
    
    # Species in Main but NOT in Backup (Model can't predict what it hasn't seen in the backup)
    main_not_in_backup = len(main_species_set - set(df_features['scientific_name']))
    
    # Confusion Matrix mapping
    TP = len(df_features[(df_features['present_in_main'] == 1) & (df_features['predicted_present'] == 1)])
    FP = len(df_features[(df_features['present_in_main'] == 0) & (df_features['predicted_present'] == 1)])
    FN = len(df_features[(df_features['present_in_main'] == 1) & (df_features['predicted_present'] == 0)])
    TN = len(df_features[(df_features['present_in_main'] == 0) & (df_features['predicted_present'] == 0)])
    
    # Print Report
    print(f"Total Unique Species Observed in {main_name}: {total_main_species}")
    print(f"Total Unique Species Observed in Backup: {total_backup_species}")
    print(f"  * Note: {main_not_in_backup} species were seen in {main_name} but NEVER in the backup reserve.")
    print(f"          The model cannot evaluate these.\n")
    
    print(f"--- Model Evaluation on Backup Species ---")
    print(f"✅ True Positives (Observed & Predicted Present):       {TP}")
    print(f"❌ False Negatives (Observed but Predicted Absent):     {FN}  <-- (Model missed these)")
    print(f"🌟 False Positives (Not Observed but Predicted Present): {FP}  <-- (Your Imputed/New Species!)")
    print(f"✔️ True Negatives (Not Observed & Predicted Absent):     {TN}  <-- (Correctly ignored)")