"""
Test Laure Python module launching the pipeline to assess the delay of the TGV.

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
    remove_outliers
)
from tools.tools_metrics import (
    compute_mse,
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

# Spliting data
train_set = dataset[dataset['date'].dt.year != 2023]
test_set = dataset[dataset['date'].dt.year == 2023]

# Scale, normalize and remove the wrong columns
pipeline1 = pipeline_coords_stand()

### Analysis of the correlation with features of interest ###

if TEST_MODE:
    display_correlation_matrix(dataset)
    display_correlation_graph(dataset)

### Training ###

# Create the pipeline with the model

model1 = GBR(n_estim = 100, min_sample_leaf = 1, max_depth = 3, learning_rate = 0.2)
model2 = HGBR(max_iter = 100, max_depth = None, min_samples_leaf = 20, learning_rate = 0.2)
model3= random_forest(n_estim = 100, max_depth = 3, min_samples_leaf = 1)
model4 = extremely_random_trees(n_estim = 100, max_depth = None, min_samples_leaf = 1)
model5 = XGBRegressor(n_estimators = 200, learning_rate = 0.3, max_depth = 3, random_state = 42) # extreme Gradient Boosting


model1 = GradientBoostingRegressor(
    min_samples_split = 130,
    n_estimators = 130, 
    min_samples_leaf = 60, 
    learning_rate = 0.05,
    max_depth = 4, 
    subsample = 0.8,
    random_state = RANDOM_STATE)

model6 = ExtraTreesRegressor(
    min_samples_split = 130,
    n_estimators = 160, 
    min_samples_leaf = 1, 
    max_depth = 10, 
    # bootstrap = False,
    warm_start = True,
    # max_samples = None, # If bootstrap is True, the number of samples to draw from X to train each base estimator.
    random_state = RANDOM_STATE)

model5 = XGBRegressor(
    min_samples_split = 130,
    n_estimators = 130, 
    min_samples_leaf = 60, 
    learning_rate = 0.05,
    max_depth = 4, 
    subsample = 0.8,
    random_state = RANDOM_STATE)

complete_pipeline = make_pipeline(pipeline1, model6)

complete_pipeline.fit(train_set[LIST_FEATURES_TRAINING], train_set[DELAY_FEATURE])
y_predicted = complete_pipeline.predict(test_set[LIST_FEATURES_TRAINING])

### Metrics ###

mse = compute_mse(
    y_predicted=y_predicted,
    y_test=test_set[DELAY_FEATURE]
)
print(mse)
r2_score = compute_r2(
    y_predicted=y_predicted,
    y_test=test_set[DELAY_FEATURE]
)
print(r2_score)



