"""
Python module containing the metrics to evaluate the results of the models.

Functions
---------
compute_mse
    Compute Mean Square Error.

compute_rmse
    Compute Root Mean Square Error.

compute_r2
    Compute R2 score.

scores_per_month
    Decompose the scores for each month.
"""

###############
### Imports ###
###############

### Python imports ###

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

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

def compute_bias(y_test, y_predicted):
    """
    Compute the bias of the solution.

    Parameters
    ----------
    y_test : numpy.ndarray
        Feature to predict.

    y_predicted : numpy.ndarray
        Feature predicted.

    Returns
    -------
    bias : float
    """
    mean_test = np.mean(y_test)
    mean_predicted = np.mean(y_predicted)
    return abs(mean_predicted - mean_test)

def scores_per_month(test_frame, y_predicted, y_test):
    """
    Decompose the scores for each month.

    Parameters
    ----------
    test_frame : pandas.core.frame.DataFrame
        Test dataset.

    y_predicted : numpy.ndarray
        Feature predicted.

    y_test : numpy.ndarray
        Feature to predict.

    Returns
    -------
    None
    """
    for month in range(1, 7) :
        L = test_frame['date'].dt.month==month
        R2 = round(compute_r2(y_test[L], y_predicted[L]), 3)
        MSE = round(compute_mse(y_test[L], y_predicted[L]), 3)
        print(f"Month {month} score : R2={R2}  MSE={MSE}")
        
    maximum = max(np.max(y_predicted), np.max(y_test))
    for month in range(1, 7) :
        L = test_frame['date'].dt.month==month
        plt.figure(figsize=(5, 5))
        plt.scatter(y_test[L], y_predicted[L], s=10)
        plt.title(f"Month : {month}")
        X = np.linspace(0, maximum, 100)
        plt.plot(X, X, c='black', linestyle="--")
        plt.xlabel("Ground truth")
        plt.ylabel("Predictions")
        plt.xlim(0, maximum)
        plt.ylim(0, maximum)
        plt.savefig(f"./figures/Predictions_month_{month}.png")
