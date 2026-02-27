# Instructions to run
1. run utils.py

2. tune models to find best option. for each, tune parameters to minimize AUC. statistics for each model can be found in output.md after running.

## example for tuning parameters in train_randomforest.py. 
we test these:
N_ESTIMATORS = [100, 200, 500]
MAX_DEPTH = [None, 10, 20, 30]
MIN_SAMPLES_LEAF = [1, 5, 10]
MAX_FEATURES = ["sqrt", "log2"]
N_FOLDS = [5, 10]
run the model for each value of N_ESTIMATORS. check results in outputmd, select the best one, and fix that value.
now, move on to MAX_DEPTH. repeat this process for each of param