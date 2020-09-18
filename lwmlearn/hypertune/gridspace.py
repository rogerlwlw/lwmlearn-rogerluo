# -*- coding: utf-8 -*-
"""
use :func:`pipe_grid` to retrieve predefined grid space for parameter tuning

Created on Sun Dec 15 17:04:05 2019

@author: roger luo
"""

import numpy as np
from pandas.core.dtypes import api

from lwmlearn.lwlogging import init_log


def _grid_search_grid(estimator):
    """predefined grid search space
    

    Parameters
    ----------
    estimator : str
        sklearn estimator's name.

    Returns
    -------
    grid : TYPE
        param_grid dict of specified estimator

    """

    LogisticRegression = [{'C': np.logspace(-3, 0, 8)}]

    SGDClassifier = [{
        'early_stopping': [True, False]
    }, {
        'loss': ['log', 'modified_huber', 'perceptron'],
        'alpha': np.logspace(-3, 1, 10)
    }]

    LinearSVC = [{'C': np.logspace(-3, 0, 10)}]

    SVC = [{
        'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    }, {
        'gamma': np.logspace(-5, 5, 5),
    }, {
        'C': np.logspace(-5, 1, 5)
    }]

    XGBClassifier = [
        {
            'learning_rate': np.logspace(-3, 0, 5),
            'n_estimators': np.arange(50, 120, 10).astype(int),
        },
        {
            'colsample_bytree': [0.95, 0.9, 0.8, 0.75],
            'subsample': [0.95, 0.9, 0.8, 0.7, 0.6],
        },
        {
            'scale_pos_weight': np.logspace(0, 1.5, 5)
        },
        {
            'max_depth': [2, 3, 4, 5]
        },
        {
            'gamma': np.logspace(-2, 1, 5)
        },
        {
            'reg_alpha': np.logspace(-2, 3, 3),
            'reg_lambda': np.logspace(-2, 3, 10)
        },
    ]

    AdaBoostClassifier = [
        {
            'learning_rate': np.logspace(-3, 0, 5),
        },
        {
            'n_estimators': np.logspace(1.5, 2.5, 5).astype(int),
        },
    ]

    RandomForestClassifier = [
        {
            'max_depth': range(3, 10),
            'min_samples_leaf': np.logspace(-5, -1, 5),
        },
        {
            'n_estimators': np.logspace(1.5, 2.5, 10).astype(int)
        },
    ]

    GradientBoostingClassifier = [{
        'max_depth': range(2, 5),
        'min_samples_leaf': np.logspace(-5, -1, 5),
    }, {
        'n_estimators':
        np.logspace(1.5, 2.5, 10).astype(int)
    }, {
        'subsample': [1, 0.9, 0.8, 0.75]
    }]

    DecisionTreeClassifier = [{
        'min_samples_leaf': np.logspace(-5, -1, 5),
        'min_impurity_decrease': [1e-5],
    }]

    LabelPropagation = [
        {
            'kernel': ['rbf'],
            'gamma': np.logspace(-5, 1, 5)
        },
        {
            'kernel': ['knn'],
            'n_neighbors': np.logspace(0, 1.2, 5).astype(int)
        },
    ]

    kpca = [{
        'kernel': ['linear', 'sigmoid', 'rbf']
    }, {
        'alpha': np.logspace(0, 2, 5)
    }, {
        'gamma': np.logspace(-5, 0, 5)
    }]

    spca = [{
        'n_components': np.logspace(1.2, 2, 5).astype(int)
    }, {
        'alpha': np.logspace(-1, 3, 5)
    }]

    # kernel approximation
    Nys = [
        {
            'gamma': np.logspace(-5, 1, 5)
        },
    ]

    # feature selection
    frf = [{
        'threshold':
        ['*'.join([str(i), 'mean']) for i in (0.2, 0.5, 0.8, 1, 1.2)]
    }]

    fxgb = [{
        'threshold':
        ['*'.join([str(i), 'mean']) for i in (0.2, 0.5, 0.8, 1, 1.2)]
    }]

    param_grids = locals().copy()

    grid = param_grids.get(estimator)

    # print messages grid space
    _print_warning(grid, estimator)

    return grid


def _bayes_search_grid(estimator):
    """bayesian grid space
    

    Parameters
    ----------
    estimator : str
        sklearn estimator's name.

    Returns
    -------
    grid : TYPE
        param_grid dict of specified estimator

    """

    logscale_C = (1e-5, 2.0, 'log-uniform')
    logscale_gamma = (1e-5, 100.0, 'log-uniform')
    logscale_lr = (1e-3, 2.0, 'log-uniform')
    sample_ratio = (0.3, 1.0, 'uniform')
    col_ratio = (0.75, 1.0, 'uniform')
    pos_ratio = (1.0, 20.0, 'uniform')
    tree_depth = (1, 5, 'uniform')
    nneighbors = (1, 3, 'uniform')
    rebalance = (0.1, 0.5, 'uniform')

    cart_tree_param = {
        'max_depth': (1, 5, 'uniform'),
        'min_samples_leaf': (1e-5, 1e-1, 'log-uniform'),
    }

    # predefined grid space
    LogisticRegression = [{'C': logscale_C}]

    SGDClassifier = [{
        'loss': ['log', 'modified_huber', 'perceptron'],
        'alpha': logscale_C
    }]

    LinearSVC = [{'C': logscale_C}]

    SVC = [{
        'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
        'gamma': logscale_gamma,
        'C': logscale_C
    }]

    XGBClassifier = [
        {
            'learning_rate': logscale_lr,
            'n_estimators': (30, 200, 'uniform'),
        },
        {
            'reg_alpha': logscale_C,
            'reg_lambda': logscale_C,
            'gamma': logscale_C,
            'scale_pos_weight': pos_ratio,
        },
        {
            'colsample_bytree': col_ratio,
            'subsample': sample_ratio,
            'max_depth': tree_depth
        },
    ]

    AdaBoostClassifier = [
        {
            'learning_rate': logscale_lr,
            'n_estimators': (30, 200, 'uniform'),
        },
    ]

    RandomForestClassifier = [
        cart_tree_param,
        {
            'n_estimators': (30, 200, 'uniform')
        },
    ]

    GradientBoostingClassifier = [
        cart_tree_param, {
            'learning_rate': logscale_lr,
            'n_estimators': (30, 200, 'uniform'),
            'subsample': sample_ratio
        }
    ]

    DecisionTreeClassifier = [{
        'min_samples_leaf': (1e-6, 1e-1, 'log-uniform')
    }]

    # kernel approximation
    Nys = [
        {
            'gamma': logscale_gamma
        },
    ]
    # feature selection
    threshold = []
    threshold.extend(
        ['*'.join([str(i), 'mean']) for i in (0.2, 0.4, 0.6, 1, 1.2, 1.68)])
    frf = [{'threshold': threshold}]

    fxgb = [{'threshold': threshold}]

    # resampler operators
    EditedNearestNeighbours = [{
        'n_neighbors': nneighbors,
        'kind_sel': ['all', 'mode']
    }]

    RandomUnderSampler = [{'sampling_strategy': rebalance}]

    BorderlineSMOTE =[{
        'sampling_strategy': rebalance,
    }]

    SMOTENC = [{
        'sampling_strategy': rebalance,
    }]
    param_grids = locals().copy()

    grid = param_grids.get(estimator)

    # print messages grid space
    _print_warning(grid, estimator)

    return grid


def _print_warning(grid, estimator):
    '''
    '''
    logger = init_log()
    
    if grid is None:
        logger.info("key '{}' not found, param_grid not returned".format(estimator))
    else:
        logger.info("param_grid for '{}' returned as : {} ".format(
            estimator, [i for i in grid]))
        


def pipe_grid(estimator, pipe=True, search_type='gridcv'):
    """return pre-defined param_grid space of given operator
    
    notes
    -----
    For a pipeline of operators the param_grid can be extended to form one 
    sequentical grid space for pipeline 

    Parameters
    ----------
    estimator : str
        - str name of estimator or sklearn estimator instance
    pipe : bool, optional
        if False return param_grid, if True return param_grid as embedded
        in pipeline. The default is True.
    search_type : TYPE, optional
        DESCRIPTION. The default is 'gridcv'.

    Returns
    -------
    param_grid : list
        list of dict.

    """
    
    if isinstance(estimator, str):
        keys = estimator
    else:
        keys = estimator.__class__.__name__

    grid_api = {'gridcv': _grid_search_grid, 'bayesiancv': _bayes_search_grid}

    param_grid = grid_api[search_type](keys)

    if param_grid is None:
        return

    if pipe is True:
        return [{'__'.join([keys, k]): i.get(k)
                 for k in i.keys()} for i in param_grid if api.is_dict_like(i)]
    else:
        return param_grid
