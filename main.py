"""
Main Python module launching the pipeline to assess the delay of the TGV.
"""

###############
### Imports ###
###############

### Python imports ###

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm

### Module imports ###

from tools.tools_analysis import (
    display_correlation_graph,
    display_correlation_matrix
)
from tools.tools_constants import (
    TEST_MODE,
    PATH_DATASET,
    DELAY_FEATURE,
    LIST_FEATURES,
    LIST_CAUSE_FEATURES,
    ALPH,
    TOLERANCE,
    ITER_MAX
)
from tools.tools_database import (
    read_data,
    remove_outliers,
    last_month_column
)
from tools.tools_metrics import (
    compute_mse,
    compute_rmse,
    compute_r2,
    compute_bias,
    scores_per_month
)
from tools.tools_models import *
from tools.tools_preprocessing import (
    pipeline_stand
)

#################
### Main code ###
#################

### Read data ###

dataset = read_data(PATH_DATASET)

### Preprocessing ###

print("===================================")
print("Starting the preprocessing pipeline")
print("===================================")

# Removing outliers

start = time()
score_threshold = 3
dataset = remove_outliers(dataset, score_threshold)

# Adding last month delays 

dataset = last_month_column(dataset)

# Spliting data

train_set = dataset[dataset['date'].dt.year != 2023]
test_set = dataset[dataset['date'].dt.year == 2023]

# Scale, normalize and remove the wrong columns

preprocessing_pipeline = pipeline_stand()

### Analysis of the correlation with features of interest ###

if TEST_MODE:
    display_correlation_matrix(dataset)
    display_correlation_graph(dataset)

### Training ###

# Create the pipeline with the model
model_lasso = Lasso_reg()
model_ridge = Ridge_reg(alpha=ALPH, max_iter=ITER_MAX, tol=TOLERANCE)
model_linear_regression = linear_regression()
model_dt = decision_tree_reg(max_depth = 7, min_samples_leaf = 5)
model_rf = random_forest(n_estim = 700, max_depth = 20, min_samples_leaf = 15, min_samples_split = 2)
model_GBR = GBR(n_estim = 1000, max_depth = 5, learning_rate = 0.01, min_samples_split = 140, min_sample_leaf = 5)
model_HGBR = HGBR(max_iter = 50, max_depth = 7, min_samples_leaf = 20, learning_rate = 0.1)
model_ERT = extremely_random_trees(n_estim = 300, max_depth = 25, min_samples_split = 27, min_samples_leaf = 1)
model_XGBReg = XGBR(n_estimators = 100, learning_rate = 0.3, max_depth = 3)

X_train = preprocessing_pipeline.fit_transform(train_set[LIST_FEATURES])
X_test = preprocessing_pipeline.transform(test_set[LIST_FEATURES])
Y_test = np.array(test_set[DELAY_FEATURE])
Y_train = np.array(train_set[DELAY_FEATURE])

p_time = time()
print("Preprocessing time : ", p_time-start)

print("===============================================")
print("Training and fine tuning delay prediction model")
print("===============================================")

### Training and fine tuning ###

params = list(range(100, 1000, 100))
scores_train = []
scores_test = []

for i in tqdm(range(len(params))) :
    model = random_forest(n_estim=params[i], max_depth=10, min_samples_leaf=15)
    model.fit(X_train, Y_train)
    
    y_predicted = model.predict(X_train)
    r2_score_train = compute_r2(y_predicted=y_predicted,y_test=Y_train)
    scores_train.append(r2_score_train)

    y_predicted = model.predict(X_test)
    r2_score_test = compute_r2(y_predicted=y_predicted,y_test=Y_test)
    scores_test.append(r2_score_test)

best_param = params[np.argmax(scores_test)]

print("Best parameters : ", best_param)
print("Best score R2 : ", np.max(scores_test))

plt.plot(params, scores_test, label="Test set R2 score")
plt.plot(params, scores_train, label="Train set R2 score")
plt.xlabel("N_estimators")
plt.ylabel("R2 score")
plt.title("Hyperparameters tuning")
plt.legend()
plt.savefig("./figures/Best_model_tuning.png")

print("Training & prediction time : ", time()-p_time)

delay_model = random_forest(n_estim=best_param, max_depth=10, min_samples_leaf=15)
delay_model.fit(X_train, Y_train)
y_predicted_train = delay_model.predict(X_train)
y_predicted_test = delay_model.predict(X_test)

print('')
Test_frame = dataset[dataset['date'].dt.year == 2023]
y_test = np.array(test_set[DELAY_FEATURE])
scores_per_month(Test_frame, y_predicted, y_test)

#########################
### Causes prediction ###
#########################

print("===================================")
print("Training causes prediction models")
print("===================================")


X_train = np.concatenate((X_train, y_predicted_train.reshape(-1, 1)), axis=1)
X_test = np.concatenate((X_test, y_predicted_test.reshape(-1, 1)), axis=1)

Y_test = np.array(test_set[LIST_CAUSE_FEATURES])
Y_train = np.array(train_set[LIST_CAUSE_FEATURES])

for i in range(Y_test.shape[1]) :
    model = GBR(n_estim=1000, max_depth=5, learning_rate=0.01)
    model.fit(X_train, Y_train[:, i])

    y_predicted = model.predict(X_test)
    r2_score_test = compute_r2(y_predicted=y_predicted,y_test=Y_test[:, i])
    print(f"Cause{i+1} Prediction R2 Score : ", r2_score_test)

### Prediction scores per month ####

Test_frame = dataset[dataset['date'].dt.year == 2023]
y_test = np.array(test_set[DELAY_FEATURE])
scores_per_month(Test_frame, y_predicted, y_test)
