import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

pairs = [
    ("Scripps", "Scripps_Filtered.csv", "TorreyPines_Filtered.csv"),
    ("ElliottChaparral", "ElliottChaparral_Filtered.csv", "MissionTrails_Filtered.csv"),
    ("LosMonos", "LosMonos_Filtered.csv", "BuenaVista_Filtered.csv"),
    ("MissionBay", "MissionBay_Filtered.csv", "TijuanaRiver_Filtered.csv")
]

def build_final_features(main_df, backup_df):
    """Aggregates backup reserve data, tracks main reserve counts, and builds features."""
    backup_species = backup_df[backup_df['scientific_name'].notna()].copy()
    backup_species['observed_on'] = pd.to_datetime(backup_species['observed_on'], errors='coerce')
    
    # Aggregate base features from the backup reserve
    features = backup_species.groupby('scientific_name').agg(
        backup_obs_count=('id', 'count'),
        backup_unique_users=('user_id', 'nunique'),
        backup_unique_days=('observed_on', 'nunique')
    ).reset_index()
    
    # Log Transformation to handle highly abundant species vs rare species
    features['log_obs_count'] = np.log1p(features['backup_obs_count'])
    
    # Extract taxonomic info and common names
    taxons = backup_species[['scientific_name', 'iconic_taxon_name', 'common_name']].dropna(subset=['iconic_taxon_name']).drop_duplicates('scientific_name')
    features = features.merge(taxons, on='scientific_name', how='left')
    features['iconic_taxon_name'].fillna('Unknown', inplace=True)
    features['common_name'].fillna('Unknown', inplace=True)
    
    # TRACK MAIN RESERVE OBSERVATIONS
    # Get the exact count of observations in the MAIN reserve
    main_counts = main_df['scientific_name'].value_counts().reset_index()
    main_counts.columns = ['scientific_name', 'main_obs_count']
    
    # Merge the main counts into the feature set
    features = features.merge(main_counts, on='scientific_name', how='left')
    features['main_obs_count'] = features['main_obs_count'].fillna(0)
    
    # Define Target: 1 if count > 0, else 0
    features['present_in_main'] = (features['main_obs_count'] > 0).astype(int)
    
    return features

if __name__ == "__main__":
    for main_name, main_file, backup_file in pairs:
        print(f"[*] Training final model for {main_name}...")
        
        # Load data
        main_df = pd.read_csv(main_file)
        backup_df = pd.read_csv(backup_file)
        
        # Feature Engineering
        df_features = build_final_features(main_df, backup_df)
        
        # Prepare Data for Modeling
        feature_cols = ['log_obs_count', 'backup_unique_users', 'backup_unique_days']
        X_cat = pd.get_dummies(df_features['iconic_taxon_name'], prefix='taxon', drop_first=True)
        X = pd.concat([df_features[feature_cols], X_cat], axis=1)
        y = df_features['present_in_main']
        
        # Train Random Forest Model with Stronger Regularization
        rf = RandomForestClassifier(
            n_estimators=300, 
            max_depth=8, 
            min_samples_leaf=10, 
            random_state=42, 
            class_weight='balanced'
        )
        rf.fit(X, y)
        
        # Extract Probabilities & Predictions
        df_features['probability_of_presence'] = rf.predict_proba(X)[:, 1]
        df_features['predicted_present'] = (df_features['probability_of_presence'] >= 0.5).astype(int)
        
        # Format Output
        output_cols = [
            'scientific_name', 'common_name', 'iconic_taxon_name', 
            'present_in_main', 'predicted_present', 'probability_of_presence', 
            'main_obs_count', 'backup_obs_count'
        ]
        output_df = df_features[output_cols].copy()
        
        # ==========================================
        # CATEGORIZE PREDICTIONS
        # ==========================================
        UNDERREPRESENTED_THRESHOLD = 3 # Species seen 3 or fewer times in main reserve
        
        conditions = [
            # True Positive, but severely underrepresented!
            (output_df['present_in_main'] == 1) & (output_df['predicted_present'] == 1) & (output_df['main_obs_count'] <= UNDERREPRESENTED_THRESHOLD),
            # True Positive, well-documented
            (output_df['present_in_main'] == 1) & (output_df['predicted_present'] == 1) & (output_df['main_obs_count'] > UNDERREPRESENTED_THRESHOLD),
            # False Positive: Highly probable but completely missing
            (output_df['present_in_main'] == 0) & (output_df['predicted_present'] == 1),
            # False Negative: Model missed it
            (output_df['present_in_main'] == 1) & (output_df['predicted_present'] == 0),
            # True Negative: Correctly ignored
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
        
        # ==========================================
        # ECOLOGICAL NOISE FILTER
        # Only recommend 'False Positives' if they have >= 10 observations in the backup reserve
        # ==========================================
        noise_filter = (output_df['Prediction_Result'] == 'False Positive (Imputed/Missing)') & (output_df['backup_obs_count'] < 10)
        output_df.loc[noise_filter, 'Prediction_Result'] = 'True Negative (Filtered)'
        output_df.loc[noise_filter, 'predicted_present'] = 0
        
        # ==========================================
        # SORTING FOR ACTIONABILITY
        # Missing and Underrepresented at the top, ranked by probability
        # ==========================================
        output_df['sort_key'] = output_df['Prediction_Result'].map({
            'False Positive (Imputed/Missing)': 1,
            'True Positive (Underrepresented)': 2,
            'True Positive (Well Documented)': 3,
            'False Negative': 4,
            'True Negative': 5,
            'True Negative (Filtered)': 6
        })
        
        output_df = output_df.sort_values(by=['sort_key', 'probability_of_presence'], ascending=[True, False]).drop(columns=['sort_key'])
        
        # Export to CSV
        export_name = f"{main_name}_Final_Predictions.csv"
        output_df.to_csv(export_name, index=False)
        print(f"    -> Saved final predictions to {export_name}")
        
    print("\n[+] All final models trained and exported successfully.")