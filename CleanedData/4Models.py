import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
import os

warnings.filterwarnings('ignore')

# Automatically change working directory to where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

pairs = [
    ("Scripps",          "Scripps_Filtered.csv",          "../ProcessedData/Scripps_with_env_data.csv",          "TorreyPines_Filtered.csv",   "../ProcessedData/TorreyPines_with_env_data.csv"),
    ("ElliottChaparral", "ElliottChaparral_Filtered.csv", "../ProcessedData/ElliottChaparral_with_env_data.csv", "MissionTrails_Filtered.csv", "../ProcessedData/MissionTrails_with_env_data.csv"),
    ("LosMonos",         "LosMonos_Filtered.csv",         "../ProcessedData/LosMonos_with_env_data.csv",         "BuenaVista_Filtered.csv",    "../ProcessedData/BuenaVista_with_env_data.csv"),
    ("MissionBay",       "MissionBay_Filtered.csv",       "../ProcessedData/MissionBay_with_env_data.csv",       "TijuanaRiver_Filtered.csv",  "../ProcessedData/TijuanaRiver_with_env_data.csv"),
]

def load_and_merge(base_csv, env_csv):
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
    # Ensure FinalPredictions output directory exists
    output_dir_predictions = "FinalPredictions"
    os.makedirs(output_dir_predictions, exist_ok=True)
    
    # Ensure CleanedData/Features output directory exists
    # Using '..' assumes the script is inside a subfolder alongside CleanedData
    output_dir_features = os.path.join("..", "CleanedData", "Features")
    os.makedirs(output_dir_features, exist_ok=True)
    
    for main_name, main_base, main_env, backup_base, backup_env in pairs:
        print(f"[*] Processing {main_name}...")
        
        main_df = load_and_merge(main_base, main_env)
        backup_df = load_and_merge(backup_base, backup_env)
        
        # Build features
        df_features = build_environmental_features(main_df, backup_df)
        
        # --- NEW ADDITION: Export the Features matrix directly to CleanedData/Features ---
        ordered_feature_cols = [
            'scientific_name', 'common_name', 'iconic_taxon_name',
            'present_in_main', 'main_obs_count', 'backup_obs_count', 'log_obs_count',
            'backup_unique_users', 'backup_unique_days',
            'avg_elevation', 'avg_slope', 'avg_ndvi', 
            'avg_soil_sand', 'avg_soil_ph', 'avg_soil_clay'
        ]
        
        feature_export_df = df_features[ordered_feature_cols].copy()
        feature_export_path = os.path.join(output_dir_features, f"{main_name}_SDM_Engineered_Features.csv")
        feature_export_df.to_csv(feature_export_path, index=False)
        print(f"    -> Saved features to {feature_export_path}")
        # ---------------------------------------------------------------------------------
        
        # Prepare for ML Model
        feature_cols_ml = [
            'log_obs_count', 'backup_unique_users', 'backup_unique_days',
            'avg_elevation', 'avg_slope', 'avg_ndvi', 
            'avg_soil_sand', 'avg_soil_ph', 'avg_soil_clay'
        ]
        
        X_cat = pd.get_dummies(df_features['iconic_taxon_name'], prefix='taxon', drop_first=True)
        X = pd.concat([df_features[feature_cols_ml], X_cat], axis=1)
        y = df_features['present_in_main']
        
        # Train ML Model
        rf = RandomForestClassifier(
            n_estimators=300, 
            max_depth=8, 
            min_samples_leaf=10, 
            random_state=42, 
            class_weight='balanced'
        )
        rf.fit(X, y)
        
        # Predict
        df_features['probability_of_presence'] = rf.predict_proba(X)[:, 1]
        df_features['predicted_present'] = (df_features['probability_of_presence'] >= 0.5).astype(int)
        
        # Format ML Output
        output_cols_ml = [
            'scientific_name', 'common_name', 'iconic_taxon_name', 
            'present_in_main', 'predicted_present', 'probability_of_presence', 
            'main_obs_count', 'backup_obs_count',
            'avg_elevation', 'avg_ndvi', 'avg_soil_clay'
        ]
        output_df = df_features[output_cols_ml].copy()
        
        UNDERREPRESENTED_THRESHOLD = 3
        
        conditions = [
            (output_df['present_in_main'] == 1) & (output_df['predicted_present'] == 1) & (output_df['main_obs_count'] <= UNDERREPRESENTED_THRESHOLD),
            (output_df['present_in_main'] == 1) & (output_df['predicted_present'] == 1) & (output_df['main_obs_count'] > UNDERREPRESENTED_THRESHOLD),
            (output_df['present_in_main'] == 0) & (output_df['predicted_present'] == 1),
            (output_df['present_in_main'] == 1) & (output_df['predicted_present'] == 0),
            (output_df['present_in_main'] == 0) & (output_df['predicted_present'] == 0)
        ]
        
        choices = [
            'True Positive (Underrepresented)', 
            'True Positive (Well Documented)', 
            'False Positive (Imputed/Missing)', 
            'False Negative', 
            'True Negative'
        ]
        output_df['Prediction_Result'] = np.select(conditions, choices, default='Unknown')
        
        # Noise Filter
        noise_filter = (output_df['Prediction_Result'] == 'False Positive (Imputed/Missing)') & (output_df['backup_obs_count'] < 10)
        output_df.loc[noise_filter, 'Prediction_Result'] = 'True Negative (Filtered)'
        output_df.loc[noise_filter, 'predicted_present'] = 0
        
        # Sort Output
        output_df['sort_key'] = output_df['Prediction_Result'].map({
            'False Positive (Imputed/Missing)': 1,
            'True Positive (Underrepresented)': 2,
            'True Positive (Well Documented)': 3,
            'False Negative': 4,
            'True Negative': 5,
            'True Negative (Filtered)': 6
        })
        
        output_df = output_df.sort_values(by=['sort_key', 'probability_of_presence'], ascending=[True, False]).drop(columns=['sort_key'])
        
        # Export Predictions
        export_name = os.path.join(output_dir_predictions, f"{main_name}_Final_Predictions.csv")
        output_df.to_csv(export_name, index=False)
        print(f"    -> Saved SDM predictions to {export_name}")
        
    print("\n[+] All environmental models trained and exported successfully.")