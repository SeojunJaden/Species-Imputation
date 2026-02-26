import numpy as np
import pandas as pd

# path relative to project root
CSV_PATH = "../carsten-cleaning-pipeline/full_sd_obs_with_env_data.csv"

SPECIES_COL = "taxon_id"

ENV_COLS = [
    "elevation", "slope", "aspect", "ndvi",
    "landcover", "impervious", "bathymetry",
    "soil_sand", "soil_ph", "soil_clay",
]

FEATURE_COLS = ENV_COLS + [SPECIES_COL]

# box for coordinates we have data for
BBOX = [-117.4, 32.7, -116.9, 33.2]

# minimum number of observations for a species to be included in the model
MIN_OBSERVATIONS = 5

# num of background points; used to train model on points where species is absent
N_BACKGROUND = 100_000

# number of samples 
SAMPLE_SIZE = 50_000

RANDOM_SEED = 69

# load data, take 50,000 samples 
def load_data():
    df = pd.read_csv(CSV_PATH)

    # remove species with less than MIN_OBSERVATIONS
    counts = df['taxon_id'].value_counts()
    keep   = counts[counts >= MIN_OBSERVATIONS].index
    df     = df[df['taxon_id'].isin(keep)].copy()

    # sample 50,000 points
    df = df.sample(SAMPLE_SIZE, random_state=RANDOM_SEED)

    return df

# make bg points. we use points where species have been observed, and set presence to 0
def make_background_points(df):
    # generate random environment points
    bg = df[ENV_COLS].sample(n=N_BACKGROUND, random_state=RANDOM_SEED, replace=True).reset_index(drop=True)

    # pair with random species
    bg[SPECIES_COL] = df[SPECIES_COL].sample(n=N_BACKGROUND, random_state=RANDOM_SEED + 1, replace=True).values
    
    bg["presence"] = 0
    return bg

def get_train_data():
    df = load_data()

    # add classification column "presence"
    # set to 1 for species with observations, 0 for background points
    df["presence"] = 1

    bg = make_background_points(df)
    train_data = pd.concat([df[FEATURE_COLS + ["presence"]], bg]).sample(frac=1, random_state=RANDOM_SEED)

    X = train_data[FEATURE_COLS]
    y = train_data["presence"]

    
    return X, y
