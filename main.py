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
# model1 = Lasso_reg(ALPH, ITER_MAX, TOLERANCE)
model1 = Lasso_reg()
complete_pipeline = make_pipeline(pipeline1, model1)

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
