from sklearn.linear_model import Ridge, ElasticNet, Lasso, LinearRegression

from xgboost import XGBRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import cross_val_score, GridSearchCV

"""
Python module to train differents architectures on the train data.

Functions
---------

GBR
    Gradient Boosting Regression 
  
HGBR
    Histogram-based Gradient Boosting Regression Tree

random_forest
    Random Forest Regressor   

extremely_random_trees
    Extremely Random Trees 

Lasso_reg
    Lasso Regression 

Ridge_reg
    Ridge Regression  

elastic_net
    ElasticNet Regression       
"""

###############
### Imports ###
###############

### Python imports ###

from sklearn.linear_model import (
    SGDRegressor,
    LinearRegression
)

from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)


### Module imports ###

from tools.tools_constants import (
    RANDOM_STATE
)

#########################
### Linear regression ###
#########################

def sgd_regressor():
    return SGDRegressor(
        random_state=RANDOM_STATE
    )

def linear_regression():
    return LinearRegression(
        random_state=RANDOM_STATE
    )

######################
### Random forests ###
######################

def GBR(n_estim = 100, min_sample_leaf = 1, max_depth = 3, learning_rate = 0.1):
    """
    Gradient Boosting Regression

    Parameters
    ----------
    n_estim : int
        Number of boosting stages to perform

    min_samples_leaf : int or float
        The minimum number of samples required to be a leaf node

    max_depth : int or None
        Maximum depth of the individual regression estimators

    Returns
    -------
    GradientBoostingRegressor from sklearn.ensemble
    """
    return GradientBoostingRegressor(
        n_estimators = n_estim, 
        min_samples_leaf = min_sample_leaf, 
        learning_rate = learning_rate,
        max_depth = max_depth, 
        random_state = RANDOM_STATE)

def HGBR(max_iter = 100, max_depth = None, min_samples_leaf = 20, learning_rate = 0.1):
    """
    Histogram-based Gradient Boosting Regression Tree

    Parameters
    ----------
    max_iter : int
        maximum nuber of iterations of the boosting process
    
    max_depth : int or None
        Maximum depth of each tree

    min_samples_leaf : int
        The minimum number of samples per leaf

    Returns
    -------
    HistGradientBoostingRegressor from sklearn.ensemble
    """
    return HistGradientBoostingRegressor(
        max_iter = max_iter,
        max_depth = max_depth, 
        learning_rate = learning_rate,
        min_samples_leaf = min_samples_leaf,
        random_state = RANDOM_STATE
    )

def random_forest(n_estim = 100, max_depth = None, min_samples_leaf = 1):
    """
    Random Forest Regressor

    Parameters
    ----------
    n_estim : int
        number of trees in the forest
    
    max_depth : int or None
        Maximum depth of the tree

    min_samples_leaf : int
        The minimum number of samples required to be a leaf node

    Returns
    -------
    RandomForestRegressor from sklearn.ensemble
    """
    return RandomForestRegressor(
        n_estimators = n_estim,
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        random_state = RANDOM_STATE
    )

def extremely_random_trees(n_estim = 100, max_depth = None, min_samples_leaf = 1):
    """
    Extremely Random Trees

    Parameters
    ----------
    n_estim : int
        number of trees in the forest
    
    max_depth : int or None
        Maximum depth off the tree

    min_samples_leaf : int
        The minimum number of samples required to be a leaf node

    Returns
    -------
    ExtraTreesRegressor from sklearn.ensemble
    """
    return ExtraTreesRegressor(
        n_estimators= n_estim,
        max_depth= max_depth,
        min_samples_leaf= min_samples_leaf,
        random_state = RANDOM_STATE
    )


def Lasso_reg():
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
        alpha=1.0)


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
