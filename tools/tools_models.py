"""
Python module to train differents architectures on the train data.

Functions
---------
sgd_regressor
    Stochastic Gradient Descent Regressor

linear_regression
    Linear Regressor

GBR
    Gradient Boosting Regression 
  
HGBR
    Histogram-based Gradient Boosting Regression Tree

XGBR
    Extreme Gradient Boosting

random_forest
    Random Forest Regressor   

extremely_random_trees
    Extremely Random Trees 

decision_tree_reg
    Random Forest Regressor

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
    LinearRegression,
    Ridge,
    ElasticNet,
    Lasso
)
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.tree import (
    DecisionTreeRegressor
)
from xgboost import (
    XGBRegressor
)

### Modules imports ###

from tools.tools_constants import (
    RANDOM_STATE
)

#########################
### Linear regression ###
#########################

def sgd_regressor():
    """
    Stochastic Gradient Descent Regressor.

    Parameters
    ----------
    None

    Returns
    -------
    sklearn.linear_model.SGDRegressor
    """
    return SGDRegressor(
        random_state=RANDOM_STATE
    )

def linear_regression():
    """
    Linear Regressor.

    Parameters
    ----------
    None

    Returns
    -------
    sklearn.linear_model.LinearRegression
    """
    return LinearRegression()

######################
### Random forests ###
######################

def GBR(n_estim = 100, min_sample_leaf = 1, max_depth = 3, learning_rate = 0.1, min_samples_split = 2, subsample = 1.0):
    """
    Gradient Boosting Regression

    Parameters
    ----------
    n_estim : int, optional (default is 100)
        Number of boosting stages to perform

    min_samples_leaf : int or float, optional (default is 1)
        The minimum number of samples required to be a leaf node

    max_depth : int or None, optional (default is 3)
        Maximum depth of the individual regression estimators

    learning_rate : float, optional (default is 0.1)
        The learning rate

    Returns
    -------
    GradientBoostingRegressor from sklearn.ensemble
    """
    return GradientBoostingRegressor(
        n_estimators = n_estim, 
        subsample = subsample,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_sample_leaf, 
        learning_rate = learning_rate,
        max_depth = max_depth, 
        random_state = RANDOM_STATE)

def HGBR(max_iter = 100, max_depth = None, min_samples_leaf = 20, learning_rate = 0.1):
    """
    Histogram-based Gradient Boosting Regression Tree

    Parameters
    ----------
    max_iter : int, optional (default is 100)
        maximum nuber of iterations of the boosting process
    
    max_depth : int or None, optional (default is None)
        Maximum depth of each tree

    min_samples_leaf : int, optional (default is 1)
        The minimum number of samples per leaf

    learning_rate : float, optional (default is 0.1)
        The learning rate

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

def XGBR(n_estimators = 100, learning_rate = 0.3, max_depth = 3):
    """
    Extreme Gradient Boosting

    Parameters
    ----------
    n_estimators : int, optional (default is 100)
        Number of estimators

    learning_rate : float, optional (default is 0.3)
        Learning rate

    max_depth : int, optional (default is 3)
        Maximal depth

    Return
    ------
    XGBRegressor from xgboost
    """
    return XGBRegressor(
        n_estimators = n_estimators,
        learning_rate = learning_rate,
        max_depth = max_depth,
        random_state = RANDOM_STATE)

def random_forest(n_estim = 100, max_depth = None, min_samples_leaf = 1, min_samples_split = 2):
    """
    Random Forest Regressor

    Parameters
    ----------
    n_estim : int, optional (default is 100)
        Number of trees in the forest
    
    max_depth : int or None, optional (default is None)
        Maximum depth of the tree

    min_samples_leaf : int, optional (default is 1)
        The minimum number of samples required to be a leaf node

    Returns
    -------
    RandomForestRegressor from sklearn.ensemble
    """
    return RandomForestRegressor(
        n_estimators = n_estim,
        max_depth = max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf,
        random_state = RANDOM_STATE
    )

def extremely_random_trees(n_estim = 100, max_depth = None, min_samples_leaf = 1, min_samples_split = 2):
    """
    Extremely Random Trees

    Parameters
    ----------
    n_estim : int, optional (default is 100)
        number of trees in the forest
    
    max_depth : int or None, optional (default is None)
        Maximum depth off the tree

    min_samples_leaf : int, optional (default is 1)
        The minimum number of samples required to be a leaf node

    Returns
    -------
    ExtraTreesRegressor from sklearn.ensemble
    """
    return ExtraTreesRegressor(
        n_estimators= n_estim,
        max_depth= max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf= min_samples_leaf,
        random_state = RANDOM_STATE
    )

def decision_tree_reg(max_depth, min_samples_leaf):
    """
    Random Forest Regressor

    Parameters
    ----------  
    max_depth : int or None
        Maximum depth off the tree

    min_samples_leaf : int
        The minimum number of samples required to be a leaf node

    Returns
    -------
    DecisionTreeRegressor from sklearn.tree
    """
    return DecisionTreeRegressor(
        max_depth= max_depth,
        min_samples_leaf= min_samples_leaf,
        random_state = RANDOM_STATE
    )


def Lasso_reg():
    """
    Lasso Regression

    Parameters
    ----------
    None

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
        The precision of the solution

    Returns
    -------
    sklearn.linear_model.Ridge
    """
    return Ridge(
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        random_state=RANDOM_STATE)


def elastic_net(alpha, l1_ratio, max_iter, tol):
    """
    ElasticNet Regression

    Parameters
    ----------
    alpha : float
        Constant that multiplies the L1 term

    l1_ratio : float
        The ElasticNet mixing parameter

    max_iter : int 
        The maximum number of iterations

    tol : float
        The maximum number of iterations

    Returns
    -------
    sklearn.linear_model.ElasticNet
    """
    return ElasticNet(
        alpha, l1_ratio, max_iter, tol)
