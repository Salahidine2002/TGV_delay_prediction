"""
Main Python module launching the pipeline to assess the delay of the TGV.

Functions
---------
"""

###############
### Imports ###
###############

### Module imports ###

from tools.tools_preprocessing import (pipeline_coords_robust, pipeline_coords_stand,
                                       pipeline_coords_minmax, pipeline_robust, pipeline_stand, pipeline_minmax)
from tools.tools_analysis import (
    display_correlation_graph
)
from tools.tools_constants import (
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

# scaling normalizing et enlever les colonnes qui ne vont pas
pipeline = pipeline_coords_stand()

### Analysis of the correlation with features of interest ###

display_correlation_graph(dataset)
