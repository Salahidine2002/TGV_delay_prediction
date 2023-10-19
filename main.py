"""
Main Python module launching the pipeline to assess the delay of the TGV.

Functions
---------
"""

###############
### Imports ###
###############

### Python imports ###

from sklearn.pipeline import make_pipeline

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
    LIST_FEATURES_TRAINING
)
from tools.tools_database import (
    read_data,
    remove_outliers,
    last_month_column
)
from tools.tools_metrics import (
    compute_mse,
    compute_rmse,
    compute_r2
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

dataset = last_month_column(dataset)

# Spliting data
train_set = dataset[dataset['date'].dt.year != 2023]
train_set = train_set[train_set["retard_mois_prec"] != 0]
test_set = dataset[dataset['date'].dt.year == 2023]

# Scale, normalize and remove the wrong columns
pipeline1 = pipeline_coords_stand()

### Analysis of the correlation with features of interest ###

if TEST_MODE:
    display_correlation_matrix(dataset)
    display_correlation_graph(dataset)

### Training ###

# Create the pipeline with the model
model1 = Lasso_reg()
model_sgd_regressor = sgd_regressor()
model_linear_regression = sgd_regressor()
model_dt = decision_tree_reg(max_depth=7, min_samples_leaf=4)
model_rf = random_forest(n_estim=100, max_depth=7, min_samples_leaf=8)
model_GBR = GBR(n_estim=1000, max_depth=5, learning_rate=0.01)
complete_pipeline = make_pipeline(pipeline1, model_GBR)

complete_pipeline.fit(train_set[LIST_FEATURES_TRAINING], train_set[DELAY_FEATURE])
y_predicted = complete_pipeline.predict(test_set[LIST_FEATURES_TRAINING])

### Metrics ###

mse = compute_mse(
    y_predicted=y_predicted,
    y_test=test_set[DELAY_FEATURE]
)
print(mse)
rmse = compute_rmse(
    mse=mse
)
print(rmse)
r2_score = compute_r2(
    y_predicted=y_predicted,
    y_test=test_set[DELAY_FEATURE]
)
print(r2_score)
