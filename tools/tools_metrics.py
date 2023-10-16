"""
Python module containing the metrics to evaluate the results of the models.

Functions
---------
"""
from sklearn.metrics import accuracy_score


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
