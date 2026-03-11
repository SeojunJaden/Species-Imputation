# Instructions
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

# Results
After tuning each model, we found that LightGBM model with the following parameters had the best AUC:  
- Sample size: 300000  
- Background points: 600000  
- Folds: 10  
- Mean AUC: 0.7656  
- Std: 0.0025  
- Min AUC: 0.7616  
- Max AUC: 0.7694  
- n_estimators: 500  
- max_depth: -1  
- learning_rate: 0.1  
- num_leaves: 63  
- min_child_samples: 100  
This model is saved in this folder and imported into the script where we output predictions for each reserve.
