import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from utils import get_train_data, write_results

# get train data, convert to numpy
X, y = get_train_data()
X = X.values
y = y.values

RANDOM_SEED = 69

"""
IMPORTANT FOR TUNING: TEST THESE PARAMETERS!
N_ESTIMATORS = [100, 200, 500] ** 500 slightly better than 200 **
MAX_DEPTH = [None, 10, 20, 30] ** 20 best **
MIN_SAMPLES_LEAF = [1, 5, 10] ** 10 best **
MAX_FEATURES = ["sqrt", "log2"] ** no difference between the two? **
N_FOLDS = [5, 10] ** 10 best **

0.6949 best mean AUC with 290,000 background points and 100,000 sample size
"""

# current config, change this to test different parameters
N_FOLDS          = 10
N_ESTIMATORS     = 500
MAX_DEPTH        = 20
MIN_SAMPLES_LEAF = 10
MAX_FEATURES     = "sqrt"

# this gets passed into write_results
current_config = {
    "n_folds":          N_FOLDS,
    "n_estimators":     N_ESTIMATORS,
    "max_depth":        MAX_DEPTH,
    "min_samples_leaf": MIN_SAMPLES_LEAF,
    "max_features":     MAX_FEATURES,
}

# get cross validiation indices
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

aucs = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = RandomForestClassifier(
        n_estimators     = N_ESTIMATORS,
        max_depth        = MAX_DEPTH,
        min_samples_leaf = MIN_SAMPLES_LEAF,
        max_features     = MAX_FEATURES,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
    )
    model.fit(X_train, y_train)

    # test model
    probs = model.predict_proba(X_test)[:, 1]
    # compare probabilities to actual labels
    auc = roc_auc_score(y_test, probs)
    aucs.append(auc)



#write results to output.md
write_results("Random Forest", current_config, aucs)












