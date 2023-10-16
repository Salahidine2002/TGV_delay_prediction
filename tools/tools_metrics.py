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
    accuracy_score
)
from math import sqrt

#################
### Functions ###
#################

def accuracy():
    pass


def compute_mse(y_test, y_predicted):
    """
    Compute Mean Square Error.

    Parameters
    ----------
    y_test : TODO
        Feature to predict.

    y_predicted : TODO
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

def accuracy(y_test, y_predicted):
    """
    Compute accuracy_score

    Parameters
    ----------
    y_test :
        Feature to predict

    y_predicted :
        Feature predicted

    Returns
    -------
    accuracy_score : float
        accurary of the model
    """

    return accuracy_score(y_test, y_predicted)
