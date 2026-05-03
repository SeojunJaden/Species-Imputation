import pandas as pd
import numpy as np
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Assumes execution from within the CleanedData folder
pairs = [
    ("Scripps",          "Scripps_Filtered.csv",          "../ProcessedData/Scripps_with_env_data.csv",          "TorreyPines_Filtered.csv",   "../ProcessedData/TorreyPines_with_env_data.csv"),
    ("ElliottChaparral", "ElliottChaparral_Filtered.csv", "../ProcessedData/ElliottChaparral_with_env_data.csv", "MissionTrails_Filtered.csv", "../ProcessedData/MissionTrails_with_env_data.csv"),
    ("LosMonos",         "LosMonos_Filtered.csv",         "../ProcessedData/LosMonos_with_env_data.csv",         "BuenaVista_Filtered.csv",    "../ProcessedData/BuenaVista_with_env_data.csv"),
    ("MissionBay",       "MissionBay_Filtered.csv",       "../ProcessedData/MissionBay_with_env_data.csv",       "TijuanaRiver_Filtered.csv",  "../ProcessedData/TijuanaRiver_with_env_data.csv"),
]

def load_and_merge(base_csv, env_csv):
    """Loads the original and environmental CSVs and merges them securely."""
    df_base = pd.read_csv(base_csv)
    df_env = pd.read_csv(env_csv)
    
    merged = pd.merge(
        df_base, 
        df_env, 
        on=['scientific_name', 'time_observed_at', 'latitude', 'longitude'], 
        how='inner'
    )
    
    if 'common_name_x' in merged.columns:
        merged.rename(columns={'common_name_x': 'common_name'}, inplace=True)
    if 'iconic_taxon_name_x' in merged.columns:
        merged.rename(columns={'iconic_taxon_name_x': 'iconic_taxon_name'}, inplace=True)
        
    return merged

def build_environmental_features(main_df, backup_df):
    """Aggregates backup reserve data and builds Environmental SDM features."""
    backup_species = backup_df[backup_df['scientific_name'].notna()].copy()
    backup_species['observed_on'] = pd.to_datetime(backup_species['observed_on'], errors='coerce')
    
    features = backup_species.groupby('scientific_name').agg(
        backup_obs_count=('id', 'count'),
        backup_unique_users=('user_id', 'nunique'),
        backup_unique_days=('observed_on', 'nunique'),
        avg_elevation=('elevation', 'mean'),
        avg_slope=('slope', 'mean'),
        avg_ndvi=('ndvi', 'mean'),
        avg_soil_sand=('soil_sand', 'mean'),
        avg_soil_ph=('soil_ph', 'mean'),
        avg_soil_clay=('soil_clay', 'mean')
    ).reset_index()
    
    features.fillna(0, inplace=True)
    features['log_obs_count'] = np.log1p(features['backup_obs_count'])
    
    taxons = backup_species[['scientific_name', 'iconic_taxon_name', 'common_name']].dropna(subset=['iconic_taxon_name']).drop_duplicates('scientific_name')
    features = features.merge(taxons, on='scientific_name', how='left')
    features['iconic_taxon_name'].fillna('Unknown', inplace=True)
    features['common_name'].fillna('Unknown', inplace=True)
    
    main_counts = main_df['scientific_name'].value_counts().reset_index()
    main_counts.columns = ['scientific_name', 'main_obs_count']
    
    features = features.merge(main_counts, on='scientific_name', how='left')
    features['main_obs_count'] = features['main_obs_count'].fillna(0)
    features['present_in_main'] = (features['main_obs_count'] > 0).astype(int)
    
    return features

if __name__ == "__main__":
    # Create the AccuracyResults folder inside CleanedData
    output_dir = "AccuracyResults"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for main_name, main_base, main_env, backup_base, backup_env in pairs:
        print(f"[*] Calculating 5-Fold Cross-Validation AUC for {main_name}...")
        
        main_df = load_and_merge(main_base, main_env)
        backup_df = load_and_merge(backup_base, backup_env)
        
        df_features = build_environmental_features(main_df, backup_df)
        
        feature_cols_ml = [
            'log_obs_count', 'backup_unique_users', 'backup_unique_days',
            'avg_elevation', 'avg_slope', 'avg_ndvi', 
            'avg_soil_sand', 'avg_soil_ph', 'avg_soil_clay'
        ]
        
        X_cat = pd.get_dummies(df_features['iconic_taxon_name'], prefix='taxon', drop_first=True)
        X = pd.concat([df_features[feature_cols_ml], X_cat], axis=1)
        y = df_features['present_in_main']
        
        # Initialize 5-Fold Stratified Cross Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        rf = RandomForestClassifier(
            n_estimators=300, 
            max_depth=8, 
            min_samples_leaf=10, 
            random_state=42, 
            class_weight='balanced'
        )
        
        auc_scores = []
        
        # Train and test the model 5 separate times on different data chunks
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            rf.fit(X_train, y_train)
            
            # Predict probabilities and calculate AUC for this fold
            y_probs = rf.predict_proba(X_test)[:, 1]
            fold_auc = roc_auc_score(y_test, y_probs)
            auc_scores.append(fold_auc)
            
        # Calculate final statistics for the reserve
        mean_auc = np.mean(auc_scores)
        min_auc = np.min(auc_scores)
        max_auc = np.max(auc_scores)
        std_auc = np.std(auc_scores)
        
        results.append({
            "Reserve": main_name,
            "Mean_AUC": round(mean_auc, 4),
            "Min_AUC": round(min_auc, 4),
            "Max_AUC": round(max_auc, 4),
            "Std_Dev_AUC": round(std_auc, 4)
        })
        
    # Compile the final dataframe
    results_df = pd.DataFrame(results)
    print("\n========================================")
    print("        MODEL AUC STATISTICS")
    print("========================================")
    print(results_df.to_string(index=False))
    
    # Export to the new folder
    export_path = os.path.join(output_dir, "Model_AUC_Statistics.csv")
    results_df.to_csv(export_path, index=False)
    print(f"\n[+] AUC statistics saved successfully to {export_path}")