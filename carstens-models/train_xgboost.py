import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
# used to evaluate performance of model
from sklearn.metrics import roc_auc_score

from utils import get_train_data

# get train data, convert to numpy
X, y = get_train_data()
X = X.values
y = y.values

RANDOM_SEED = 69
N_FOLDS = 5

# get cross validiation indices
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# make array to store auc (our version of test error, since we are predicting probabilities)
aucs = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = XGBClassifier(random_state=RANDOM_SEED, eval_metric="auc")
    model.fit(X_train, y_train)

    # test model
    probs = model.predict_proba(X_test)[:, 0]
    # compare probabilities to actual labels
    auc = roc_auc_score(y_test, probs)
    aucs.append(auc)


# calcualte statistics
print(f"\n--- Results ---")
print(f"Mean AUC: {np.mean(aucs):.4f}")
print(f"Std:      {np.std(aucs):.4f}")
print(f"Min AUC:  {np.min(aucs):.4f}")
print(f"Max AUC:  {np.max(aucs):.4f}")
