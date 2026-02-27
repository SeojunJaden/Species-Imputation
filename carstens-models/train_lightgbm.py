import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier


from utils import get_train_data, write_results

# get train data, convert to numpy
X, y = get_train_data()
X = X.values
y = y.values

RANDOM_SEED = 69



"""
IMPORTANT FOR TUNING: TEST THESE PARAMETERS!
N_ESTIMATORS = [100, 200, 500]
MAX_DEPTH = [3, 6, 9]
LEARNING_RATE = [0.01, 0.1, 0.3]
NUM_LEAVES = [31, 63, 127]
MIN_CHILD_SAMPLES = [20, 50, 100]
N_FOLDS = [5, 10]
"""

# current config, change this to test different parameters
N_FOLDS           = 5
N_ESTIMATORS      = 100
MAX_DEPTH         = 6
LEARNING_RATE     = 0.1
NUM_LEAVES        = 31
MIN_CHILD_SAMPLES = 20

# this gets passed into write_results
current_config = {
    "n_folds":           N_FOLDS,
    "n_estimators":      N_ESTIMATORS,
    "max_depth":         MAX_DEPTH,
    "learning_rate":     LEARNING_RATE,
    "num_leaves":        NUM_LEAVES,
    "min_child_samples": MIN_CHILD_SAMPLES,
}

# get cross validiation indices
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

aucs = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = LGBMClassifier(
        n_estimators      = N_ESTIMATORS,
        max_depth         = MAX_DEPTH,
        learning_rate     = LEARNING_RATE,
        num_leaves        = NUM_LEAVES,
        min_child_samples = MIN_CHILD_SAMPLES,
        random_state      = RANDOM_SEED,
        n_jobs            = -1,
    )
    model.fit(X_train, y_train)

    # test model
    probs = model.predict_proba(X_test)[:, 1]
    # compare probabilities to actual labels
    auc = roc_auc_score(y_test, probs)
    aucs.append(auc)


# write results to output.md
write_results("LightGBM", current_config, aucs)