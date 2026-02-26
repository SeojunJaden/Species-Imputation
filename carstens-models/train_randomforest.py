import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


from utils import get_train_data

# get train data, convert to numpy
X, y = get_train_data()
X = X.values
y = y.values

RANDOM_SEED = 69
N_FOLDS = 5

# get cross validiation indices
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

aucs = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)

    # test model
    probs = model.predict_proba(X_test)[:, 1]
    # compare probabilities to actual labels
    auc = roc_auc_score(y_test, probs)
    aucs.append(auc)


# calculate statistics
print(f"\n--- Stats ---")

print(f"Mean AUC: {np.mean(aucs):.4f}")
print(f"Std:      {np.std(aucs):.4f}")
print(f"Min AUC:  {np.min(aucs):.4f}")
print(f"Max AUC:  {np.max(aucs):.4f}")