import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
# used to evaluate performance of model
from sklearn.metrics import roc_auc_score

from utils import get_train_data, write_results

# get train data, convert to numpy
X, y = get_train_data()
X = X.values
y = y.values

RANDOM_SEED = 69


"""
IMPORTANT FOR TUNING: TEST THESE PARAMETERS!
N_ESTIMATORS = [100, 200, 500] ** 500 best **
MAX_DEPTH = [3, 6, 9] ** 6 best **
LEARNING_RATE = [0.01, 0.1, 0.3] ** 0.1 best **
SUBSAMPLE = [0.6, 0.8, 1.0] ** 1.0 best **
COLSAMPLE_BYTREE = [0.6, 0.8, 1.0] ** 1.0 best **
N_FOLDS = [5, 10] ** 10 best **

0.7277 best mean AUC with 290,000 background points and sample of 160,000 (for now)
"""

# current config, change this to test different parameters
N_FOLDS          = 10
N_ESTIMATORS     = 500
MAX_DEPTH        = 6
LEARNING_RATE    = 0.1
SUBSAMPLE        = 1.0
COLSAMPLE_BYTREE = 1.0

# this gets passed into write_results
current_config = {
    "n_folds": N_FOLDS,
    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
    "learning_rate": LEARNING_RATE,
    "subsample": SUBSAMPLE,
    "colsample_bytree": COLSAMPLE_BYTREE,
}

# get cross validiation indices
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# make array to store auc (our version of test error, since we are predicting probabilities)
aucs = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = XGBClassifier(
        n_estimators     = N_ESTIMATORS,
        max_depth        = MAX_DEPTH,
        learning_rate    = LEARNING_RATE,
        subsample        = SUBSAMPLE,
        colsample_bytree = COLSAMPLE_BYTREE,
        random_state     = RANDOM_SEED,
        eval_metric      = "auc",
    )
    model.fit(X_train, y_train)

    # test model
    probs = model.predict_proba(X_test)[:, 1]
    # compare probabilities to actual labels
    auc = roc_auc_score(y_test, probs)
    aucs.append(auc)


# write results to output.md
write_results("XGBoost", current_config, aucs)
