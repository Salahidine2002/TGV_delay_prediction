from sklearn.linear_model import Ridge, ElasticNet, Lasso, LinearRegression

from xgboost import XGBRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import cross_val_score, GridSearchCV

"""
Python module to train differents architectures on the train data.

Functions
---------
"""


def Lasso_reg(alpha, max_iter, tol):
    """
    Lasso Regression

    Parameters
    ----------
    alpha : float
        Constant that multiplies the L1 term

    max_iter : int 
        The maximum number of iterations

    tol : Float
        The maximum number of iterations

    Returns
    -------
    sklearn.linear_model.Lasso
    """
    return Lasso(
        alpha, max_iter, tol)


def Ridge_reg(alpha, max_iter, tol):
    """
    Ridge Regression

    Parameters
    ----------
    alpha : float
        Constant that multiplies the L1 term

    max_iter : int 
        The maximum number of iterations

    tol : Float
        The maximum number of iterations

    Returns
    -------
    sklearn.linear_model.Ridge
    """
    return Ridge(
        alpha, max_iter, tol)


def elastic_net(alpha, l1_ratio, max_iter, tol):
    """
    ElasticNet Regression

    Parameters
    ----------
    alpha : float
        Constant that multiplies the L1 term

    l1_ratio :float
        The ElasticNet mixing parameter

    max_iter : int 
        The maximum number of iterations

    tol : Float
        The maximum number of iterations

    Returns
    -------
    sklearn.linear_model.ElasticNet
    """
    return ElasticNet(
        alpha, l1_ratio, max_iter, tol)
