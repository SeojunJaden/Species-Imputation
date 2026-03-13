"""
Evaluates model predictions for each reserve by summarizing the result categories
from the output CSVs produced by train.py.
"""

import os
import pandas as pd

PREDICTION_DIR = "FinalPredictionsTest"

RESERVES = ["Scripps", "ElliottChaparral", "LosMonos", "MissionBay"]

for name in RESERVES:
    path = os.path.join(PREDICTION_DIR, f"{name}_Predictions.csv")

    df = pd.read_csv(path)
    counts = df['result'].value_counts()

    total_backup   = len(df)
    imputed        = counts.get('Imputed', 0)
    tp_well        = counts.get('True Positive (Well Documented)', 0)
    tp_under       = counts.get('True Positive (Underrepresented)', 0)
    false_neg      = counts.get('False Negative', 0)
    true_neg       = counts.get('True Negative', 0)
    confirmed      = tp_well + tp_under

    print(f"\n{'='*60}")
    print(f"Results for: {name}")
    print(f"{'='*60}")
    print(f"Total backup species evaluated:         {total_backup}")
    print(f"Confirmed present (well documented):  {tp_well}")
    print(f"Confirmed present (underrepresented): {tp_under}")
    print(f"Imputed (likely present, unrecorded): {imputed}")
    print(f"False negatives (model missed):       {false_neg}")
    print(f"True negatives (correctly ignored):   {true_neg}")
    print(f"---")
    print(f"Total confirmed + imputed species:      {confirmed + imputed}")