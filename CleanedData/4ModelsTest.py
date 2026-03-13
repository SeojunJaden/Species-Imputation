"""
This script:
    1. Combines observation data with environment data for each observation 
    2. Trains 4 Random Forest models on environmental data. Each model is trained
    on data from a specific ecologically similar area to a UCSD natual reserve. We
    call these "backup reserves."
    3. Makes predictions based on those models of what species live in
    the UCSD Natural Reseves
    4. Outputs those predictions to .csv files

Input:
    X: Each row in the Feature matrix X is a species. Each column in that row 
    contains data about that specific species from the "backup reserve" observations
    and google earth environment data. We utilize feature engineering to come up with
    many of these features.
    y: 1 if the species has been observed at a UCSD natural reserve, 0 if not

Model Notes:
    We use "positive-unlabeled" learning via BaggingPuClassifier from the pulearn 
    package. This treats "0" y values as unlabeled rather than negative. This works
    well given that "negatives" are unknown for our problem. Unlike a normal random
    forest, when making a decision tree, BaggingPuClassifier takes all positive examples 
    and only some of the "0"s. This means that unlabeled species that resemble "positive"
    species will be predicted "positive" more often, as there will be more trees in the
    random forest that have not already taken its "0" label into account.

Feature Engineering:
        - log_obs_count       : log of total times the species was recorded
        - backup_unique_users : number of distinct observers (high = more reliable)
        - backup_unique_days  : number of distinct days observed (high = established resident)
        - obs_per_user        : obs count / unique users (low = many independent confirmations)
        - days_since_last_obs : recency of last observation (low = still actively present)
        - lat_std, lon_std    : std of observation coordinates (high = generalist)
        - avg_elevation, avg_slope, avg_ndvi, avg_soil_sand, avg_soil_ph, avg_soil_clay (environmental niche)
        - std_elevation, std_ndvi, std_slope (high std = tolerates wide range of conditions)
        - iconic_taxon_name — one-hot encoded (Plantae, Aves, Insecta, etc.)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from pulearn import BaggingPuClassifier


RANDOM_STATE = 69

## PARAMETERS TO TUNE
N_ESTIMATORS = 300  # try: 100, 300, 500
MAX_DEPTH = 8 # try: 4, 8, None
MIN_SAMPLES_LEAF = 10 # try: 5, 10, 20
MAX_SAMPLES = 0.5 # try: 0.5, 0.7, 0.9

MIN_BACKUP_OBS = 10 # minimum observations to consider a species
PRESENCE_THRESHOLD = 0.5 # probability cutoff for predicting present
UNDERREPRESENTED_THRESHOLD = 3 # used to flag species that seem to have few observations

# reserve data, backup reserve data + reserve name
RESERVES = [
    {
        "name": "Scripps",
        "main": "DataForModel/Scripps_Combined.csv",
        "backup": "DataForModel/TorreyPines_Combined.csv",
        "min_backup_obs": 10,
        "presence_threshold": 0.55,
    },
    {
        "name": "ElliottChaparral",
        "main": "DataForModel/ElliottChaparral_Combined.csv",
        "backup": "DataForModel/MissionTrails_Combined.csv",
        "min_backup_obs": 15,
        "presence_threshold": 0.6,
    },
    {
        "name": "LosMonos",
        "main": "DataForModel/LosMonos_Combined.csv",
        "backup": "DataForModel/BuenaVista_Combined.csv",
        "min_backup_obs": 5,
        "presence_threshold": 0.55,
    },
    {
        "name": "MissionBay",
        "main": "DataForModel/MissionBay_Combined.csv",
        "backup": "DataForModel/TijuanaRiver_Combined.csv",
        "min_backup_obs": 10,
        "presence_threshold": 0.6,
    },
]

# FEATURE ENGINEERING
def build_features(backup_df, main_df):

    #remove na, convert time string
    df = backup_df[backup_df['scientific_name'].notna()].copy()
    df['observed_on'] = pd.to_datetime(df['observed_on'], errors='coerce')

    # aggregate all features into one row using different functions like mean, std
    features = df.groupby('scientific_name').agg(
        backup_obs_count=('scientific_name', 'count'),
        backup_unique_users=('user_id', 'nunique'),
        backup_unique_days=('observed_on', 'nunique'),
        avg_elevation=('elevation', 'mean'),
        avg_slope=('slope', 'mean'),
        avg_ndvi=('ndvi', 'mean'),
        avg_soil_sand=('soil_sand', 'mean'),
        avg_soil_ph=('soil_ph', 'mean'),
        avg_soil_clay=('soil_clay', 'mean'),
        std_elevation=('elevation', 'std'),
        std_ndvi=('ndvi', 'std'),
        std_slope=('slope', 'std'),
    ).reset_index()

    """
    features not used in this test: 
    lat_std=('latitude', 'std'),
    lon_std=('longitude', 'std'),
    backup_unique_users=('user_id', 'nunique'),
    """

    # take log of observations data is less skewed for model
    features['log_obs_count'] = np.log1p(features['backup_obs_count'])

    #add another feature
    features['obs_per_user'] = features['backup_obs_count'] / features['backup_unique_users'].clip(lower=1)
    features.fillna(0, inplace=True)

    # attach species info
    taxons = (
        df[['scientific_name', 'iconic_taxon_name', 'common_name']]
        .dropna(subset=['iconic_taxon_name'])
        .drop_duplicates('scientific_name')
    )
    features = features.merge(taxons, on='scientific_name', how='left')
    features['iconic_taxon_name'] = features['iconic_taxon_name'].fillna('Unknown')
    features['common_name'] = features['common_name'].fillna('Unknown')

    # puts together full dataset for model, labels species 1 if seen
    # at scripps and 0 if not
    main_species = main_df['scientific_name'].value_counts().reset_index()
    main_species.columns = ['scientific_name', 'main_obs_count']
    features = features.merge(main_species, on='scientific_name', how='left')
    features['main_obs_count'] = features['main_obs_count'].fillna(0)
    features['present_in_main'] = (features['main_obs_count'] > 0).astype(int)

    return features



# MAIN FUNCTION
if __name__ == "__main__":
    os.makedirs("FinalPredictions", exist_ok=True)

    # train for each reserve
    for reserve in RESERVES:
        name = reserve["name"]
        print(f"\nstarting on: {name}")

        main_df = pd.read_csv(reserve["main"])
        backup_df = pd.read_csv(reserve["backup"])

        #engineer features, drop columns with too few observations
        # add reserve specific params
        df_features = build_features(backup_df, main_df)
        df_features = df_features[df_features['backup_obs_count'] >= reserve["min_backup_obs"]]

        # encode taxon group as a feature
        X_cat = pd.get_dummies(df_features['iconic_taxon_name'], prefix='taxon', drop_first=True)

        """
        features not used in this test: 
        'lat_std',
        'lon_std',
        'backup_unique_users',
        'avg_soil_ph',
        'backup_unique_days',
        """
        FEATURE_COLS = [
            'log_obs_count',
            'obs_per_user',
            'avg_elevation',
            'avg_slope',
            'avg_ndvi',
            'avg_soil_sand',
            'avg_soil_clay',
            'std_elevation',
            'std_ndvi',
            'std_slope',
        ]

        # define x, y, for model
        X = pd.concat([df_features[FEATURE_COLS], X_cat], axis=1)
        y = df_features['present_in_main']

        # train model, print results
        model = BaggingPuClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=MAX_DEPTH,
                min_samples_leaf=MIN_SAMPLES_LEAF,
            ),
            n_estimators=N_ESTIMATORS,
            max_samples=MAX_SAMPLES,
            random_state=RANDOM_STATE,
            oob_score=True,
            n_jobs=-1,
        )
        model.fit(X, y)

        # oob score is a way to test our model
        #print(f"OOB score (tests trees where ): {model.oob_score_:.3f}")

        # make predicitons
        df_features['probability_of_presence'] = model.predict_proba(X)[:, 1]
        df_features['predicted_present'] = (df_features['probability_of_presence'] >= reserve["presence_threshold"]).astype(int)

        # classify results in order to improve interpretation of results and better
        # present data to kellie
        conditions = [
            (df_features['present_in_main'] == 1) & (df_features['predicted_present'] == 1) & (df_features['main_obs_count'] <= UNDERREPRESENTED_THRESHOLD),
            (df_features['present_in_main'] == 1) & (df_features['predicted_present'] == 1) & (df_features['main_obs_count'] > UNDERREPRESENTED_THRESHOLD),
            (df_features['present_in_main'] == 0) & (df_features['predicted_present'] == 1),
            (df_features['present_in_main'] == 1) & (df_features['predicted_present'] == 0),
            (df_features['present_in_main'] == 0) & (df_features['predicted_present'] == 0),
        ]
        choices = [
            'True Positive (Underrepresented)',
            'True Positive (Well Documented)',
            'Imputed',
            'False Negative',
            'True Negative',
        ]
        df_features['result'] = np.select(conditions, choices, default='Unknown')

        # output results
        output_cols = [
            'scientific_name',
            'common_name',
            'iconic_taxon_name',
            'present_in_main',
            'predicted_present',
            'probability_of_presence',
            'main_obs_count',
            'backup_obs_count',
            'result',
        ]

        df_features = df_features.sort_values('probability_of_presence', ascending=False)

        export_path = os.path.join("FinalPredictionsTest", f"{name}_Predictions.csv")
        df_features[output_cols].to_csv(export_path, index=False)
        print(f".csv saved to {export_path}")



