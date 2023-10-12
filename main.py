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
    display_correlation_graph
)
from tools.tools_constants import (
    PATH_DATASET
)
from tools.tools_database import (
    Read_data
)

#################
### Main code ###
#################

dataset = Read_data(PATH_DATASET)

### Analysis of the correlation with features of interest ###

display_correlation_graph(dataset)
