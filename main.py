"""
Main Python module launching the pipeline to assess the delay of the TGV.

Functions
---------
"""

###############
### Imports ###
###############

### Module imports ###

from tools.tools_analysis import (
    display_correlation_graph,
    display_correlation_matrix
)
from tools.tools_constants import (
    TEST_MODE,
    PATH_DATASET,
    DELAY_FEATURE
)
from tools.tools_database import (
    read_data, 
    remove_outliers
)

#################
### Main code ###
#################

dataset = read_data(PATH_DATASET)

### Preprocessing ###

# Removing outliers 

score_threshold = 3
dataset = remove_outliers(dataset, score_threshold)

# Spliting data 
train_set = dataset[dataset['date'].dt.year != 2023]
test_set = dataset[dataset['date'].dt.year == 2023]

# scaling
# enlever les colonnes qui ne vont pas

### Analysis of the correlation with features of interest ###

if TEST_MODE:
    display_correlation_matrix(dataset)
    display_correlation_graph(dataset)
