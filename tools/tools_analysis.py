"""
Python module analysing the data before the construction of the model.

Functions
---------
compute_correlation_matrix
    Compute the correlation matrix of the dataset.

compute_correlation
    Compute the correlation between a feature of interest and the other features of the dataset.

display_correlation_graph
    Display the correlation graph between the variable of interest and the other features.
"""

###############
### Imports ###
###############

### Python imports ###

import numpy as np
import matplotlib.pyplot as plt
import numpy as np

### Module imports ###

from tools.tools_constants import (
    DELAY_FEATURE,
    LIST_CAUSE_FEATURES
)

#################
### Functions ###
#################

def compute_correlation_matrix(dataset):
    """
    Compute the correlation matrix of the dataset.

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset where to analyse the correlation.

    Returns
    -------
    correlation_matrix : pandas.core.frame.DataFrame
        Correlation matrix on the whole dataset.
    """
    correlation_matrix = dataset.corr(numeric_only=True)
    return correlation_matrix

def compute_correlation(ref_feature, correlation_matrix):
    """
    Compute the correlation between a feature of interest and the other features of the dataset.

    Parameters
    ----------
    ref_feature : str
        Name of the feature of interest.

    correlation_matrix : pandas.core.frame.DataFrame
        Correlation matrix on the whole dataset.

    Returns
    -------
    correlation_column : pandas.core.series.Series
        Correlation between the feature of interest and the other features.
    """
    correlation_column = correlation_matrix[ref_feature]
    return correlation_column

def display_correlation_graph(dataset):
    """
    Display the correlation graph between the variable of interest and the other features.

    In abscissa, all the features are represented.
    The values correspond to the correlation between the feature of interest and the features in abscissa.
    The features of interest are here:
        - the mean delay of trains at arrival
        - the 6 different delay causes

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset where to analyse the correlation.

    Returns
    -------
    None
    """
    dict_correlation = {}

    # Compute the correlation matrix on the whole dataset
    correlation_matrix = compute_correlation_matrix(
        dataset=dataset)

    # Compute the correlation column for the delay feature
    correlation_column = compute_correlation(
        ref_feature=DELAY_FEATURE,
        correlation_matrix=correlation_matrix)
    dict_correlation[DELAY_FEATURE] = [
        correlation_column.index, correlation_column.values]

    # Compute the correlation column for the causes features
    for cause_feature in LIST_CAUSE_FEATURES:
        correlation_column = compute_correlation(
            ref_feature=cause_feature,
            correlation_matrix=correlation_matrix)
        dict_correlation[cause_feature] = [
        correlation_column.index, correlation_column.values]

    # Display the correlation graph
    for feature_name in dict_correlation:
        plt.scatter(np.arange(len(dict_correlation[feature_name][1])), dict_correlation[feature_name][1], label=feature_name)

    my_xticks = dict_correlation[feature_name][0]
    plt.xticks(np.arange(len(my_xticks)), my_xticks, rotation="vertical")
    plt.subplots_adjust(bottom=0.3)
    plt.legend()
    plt.show()
