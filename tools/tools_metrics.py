"""
Python module containing the metrics to evaluate the results of the models.

Functions
---------
compute_mse
    Compute Mean Square Error.

compute_rmse
    Compute Root Mean Square Error.
"""

###############
### Imports ###
###############

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)
from math import sqrt

#################
### Functions ###
#################

def compute_mse(y_test, y_predicted):
    """
    Compute Mean Square Error.

    Parameters
    ----------
    y_test : numpy.ndarray
        Feature to predict.

    y_predicted : numpy.ndarray
        Feature predicted.

    Returns
    -------
    mse : float
        Mean Square Error.
    """
    return mean_squared_error(y_test, y_predicted)

def compute_rmse(mse):
    """
    Compute Root Mean Square Error.

    Parameters
    ----------
    mse : float
        Mean Square Error.

    Returns
    -------
    rmse : float
        Root Mean Square Error.
    """
    return sqrt(mse)

def compute_r2(y_test, y_predicted):
    """
    Compute R2 score.

    Parameters
    ----------
    y_test : numpy.ndarray
        Feature to predict.

    y_predicted : numpy.ndarray
        Feature predicted.

    Returns
    -------
    r2_score : float
    """
    return r2_score(y_test, y_predicted)
