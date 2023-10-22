"""
Main Python module launching the pipeline to assess the delay of the TGV.
"""

###############
### Imports ###
###############

### Python imports ###

from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

### Module imports ###

from tools.tools_analysis import (
    display_correlation_graph,
    display_correlation_matrix
)
from tools.tools_constants import (
    TEST_MODE,
    PATH_DATASET,
    DELAY_FEATURE,
    ALPH,
    ITER_MAX,
    TOLERANCE,
    L1_RATIO,
    LIST_FEATURES, 
    FEATURES_TO_PASS_BINARY, 
    FEATURES_TO_PASS_COORD,
    RANDOM_STATE
)
from tools.tools_database import (
    read_data,
    display_network,
    remove_outliers,
    last_month_column
)
from tools.tools_metrics import (
    compute_mse,
    compute_rmse,
    compute_r2, 
    scores_per_month
)
from tools.tools_models import *
from tools.tools_preprocessing import (
    pipeline_coords_robust,
    pipeline_coords_stand,
    pipeline_coords_minmax,
    pipeline_robust,
    pipeline_stand,
    pipeline_minmax
)

#################
### Main code ###
#################

### Read data ###

dataset = read_data(PATH_DATASET)

### Preprocessing ###

# Removing outliers

score_threshold = 3
dataset = remove_outliers(dataset, score_threshold)

# Adding last month delays

dataset = last_month_column(dataset)

# Spliting data
train_set = dataset[dataset['date'].dt.year != 2023]
test_set = dataset[dataset['date'].dt.year == 2023]
print(list(FEATURES_TO_PASS_BINARY), list(FEATURES_TO_PASS_COORD))

# Scale, normalize and remove the wrong columns

pipeline1 = pipeline_stand()

### Analysis of the correlation with features of interest ###

if TEST_MODE:
    display_correlation_matrix(dataset)
    display_correlation_graph(dataset)

### Training ###

# Create the pipeline with the model
model1 = Lasso_reg()
model_sgd_regressor = sgd_regressor()
model_linear_regression = sgd_regressor()
model_dt = decision_tree_reg(max_depth = 7, min_samples_leaf = 5)
model_rf = random_forest(n_estim = 700, max_depth = 20, min_samples_leaf = 15, min_samples_split = 2)
model_GBR = GBR(n_estim = 1000, max_depth = 5, learning_rate = 0.01, min_samples_split = 140, min_sample_leaf = 5)
model_HGBR = HGBR(max_iter = 50, max_depth = 7, min_samples_leaf = 20, learning_rate = 0.1)
model_ERT = extremely_random_trees(n_estim = 300, max_depth = 25, min_samples_split = 27, min_samples_leaf = 1)
model_XGBReg = XGBRegressor(n_estimators = 100, learning_rate = 0.3, max_depth = 3, random_state = RANDOM_STATE) # extreme Gradient Boosting

complete_pipeline = make_pipeline(pipeline1, model_rf)

print("=========================")
print("Starting the pipeline")
print("=========================")

complete_pipeline.fit(
    train_set[LIST_FEATURES], train_set[DELAY_FEATURE])
y_predicted = complete_pipeline.predict(test_set[LIST_FEATURES])

### Metrics ###

mse = compute_mse(
    y_predicted=y_predicted,
    y_test=test_set[DELAY_FEATURE]
)
print("MSE ERROR = ", mse)
rmse = compute_rmse(
    mse=mse
)
print("RMSE ERROR = ", rmse)
r2_score = compute_r2(
    y_predicted=y_predicted,
    y_test=test_set[DELAY_FEATURE]
)
print("R2 ERROR = ", r2_score)

## Prediction scores per month ###

test_frame = dataset[dataset['date'].dt.year == 2023]
y_test = np.array(test_set[DELAY_FEATURE])
scores_per_month(test_frame, y_predicted, y_test)
