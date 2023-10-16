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
"""

###############
### Imports ###
###############

### Python imports ###

from sklearn.linear_model import (
    SGDRegressor
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

######################
### Random forests ###
######################

def GBR(n_estim, min_sample_leaf, max_depth):
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
        max_depth = max_depth, 
        random_state = RANDOM_STATE)

def HGBR(max_iter, max_depth, min_samples_leaf):
    """
    Histogram-based Gradient Boosting Regression Tree

    Parameters
    ----------
    max_iter : int
        maximum nuber of iterations of the boosting process
    
    max_depth : int or None
        Maximum depth off each tree

    min_samples_leaf : int
        The minimum number of samples per leaf

    Returns
    -------
    HistGradientBoostingRegressor from sklearn.ensemble
    """
    return HistGradientBoostingRegressor(
        max_iter = max_iter,
        max_depth = max_depth, 
        min_samples_leaf = min_samples_leaf,
        random_state = RANDOM_STATE
    )

def random_forest(n_estim, max_depth, min_samples_leaf):
    """
    Random Forest Regressor

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
    RandomForestRegressor from sklearn.ensemble
    """
    return RandomForestRegressor(
        n_estimators= n_estim,
        max_depth= max_depth,
        min_samples_leaf= min_samples_leaf,
        random_state = RANDOM_STATE
    )

def extremely_random_trees(n_estim, max_depth, min_samples_leaf):
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
