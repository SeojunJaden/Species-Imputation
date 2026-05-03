"""
Merges filtered observation CSVs with GEE environmental data CSVs for each reserve.
Run this before train.py. Outputs one combined CSV per reserve to ProcessedData/Combined/.
"""

import os
import pandas as pd

RESERVES = [
    {
        "name": "Scripps",
        "base": "Scripps_Filtered.csv",
        "env": "../ProcessedData/Scripps_with_env_data.csv",
    },
    {
        "name": "TorreyPines",
        "base": "TorreyPines_Filtered.csv",
        "env": "../ProcessedData/TorreyPines_with_env_data.csv",
    },
    {
        "name": "ElliottChaparral",
        "base": "ElliottChaparral_Filtered.csv",
        "env": "../ProcessedData/ElliottChaparral_with_env_data.csv",
    },
    {
        "name": "MissionTrails",
        "base": "MissionTrails_Filtered.csv",
        "env": "../ProcessedData/MissionTrails_with_env_data.csv",
    },
    {
        "name": "LosMonos",
        "base": "LosMonos_Filtered.csv",
        "env": "../ProcessedData/LosMonos_with_env_data.csv",
    },
    {
        "name": "BuenaVista",
        "base": "BuenaVista_Filtered.csv",
        "env": "../ProcessedData/BuenaVista_with_env_data.csv",
    },
    {
        "name": "MissionBay",
        "base": "MissionBay_Filtered.csv",
        "env": "../ProcessedData/MissionBay_with_env_data.csv",
    },
    {
        "name": "TijuanaRiver",
        "base": "TijuanaRiver_Filtered.csv",
        "env": "../ProcessedData/TijuanaRiver_with_env_data.csv",
    },
]

OUTPUT_DIR = "DataForModel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for reserve in RESERVES:
    name = reserve["name"]
    print(f"[*] Processing {name}...")

    df_base = pd.read_csv(reserve["base"])
    df_env = pd.read_csv(reserve["env"])

    merged = pd.merge(
        df_base,
        df_env,
        on=['scientific_name', 'time_observed_at', 'latitude', 'longitude'],
        how='inner'
    )

    # Resolve duplicate column names from the merge (keep _x versions)
    if 'common_name_x' in merged.columns:
        merged.rename(columns={'common_name_x': 'common_name'}, inplace=True)
    if 'iconic_taxon_name_x' in merged.columns:
        merged.rename(columns={'iconic_taxon_name_x': 'iconic_taxon_name'}, inplace=True)

    output_path = os.path.join(OUTPUT_DIR, f"{name}_Combined.csv")
    merged.to_csv(output_path, index=False)
    print(f"    -> Saved to {output_path}")

print("\n[+] Done.")