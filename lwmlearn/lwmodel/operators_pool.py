# -*- coding: utf-8 -*-
"""Operators pool

this module offers a pool for ML operators like transformers & classifiers. 
Hyper parameters of operators could be predefined otherwise will be initialized
with defautlts.

Created on Thu Dec 12 16:53:52 2019

@author: roger luo
"""

import inspect
import pandas as pd

from lightgbm import LGBMClassifier

from operator import itemgetter

from imblearn import FunctionSampler
from imblearn.base import BaseSampler
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.under_sampling.base import BaseCleaningSampler, BaseUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.base import (ClassifierMixin, TransformerMixin, RegressorMixin,
                          ClusterMixin, OutlierMixin, MetaEstimatorMixin)

from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model._base import LinearClassifierMixin

from sklearn.ensemble import BaseEnsemble

from lwmlearn.utilis.utilis import get_flat_list, getmodules


def predifined_instance():
    '''crawls predefined package collecting predefined operator instances
    
    as a dict returned by all `predefined_ops` functions in `predefined` package
    
    Return 
    ------
    d : dict
        dict of user defined none-default instances of operators
    '''
    from lwmlearn.lwmodel import predefined
    # traverse predefined
    keys_ops = {}
    for module in getmodules(predefined).values():
        for i in inspect.getmembers(module, inspect.isfunction):
            if i[0] == 'predefined_ops':
                keys_ops.update(i[1]())
    return keys_ops


def get_lw_estimators(oper_str=None, type_filter=None):
    '''return instance of estimator given oper_str as `class.__name__`, 
    
    classes included are from packages ['lwmlearn', 'sklearn', 'imblearn'],
    MetaEstimator will be excluded
    
    first look at predefined instances, if not found, initialize one with 
    defaults
    
    Parameters
    ----------
    oper_str : str
        name of estimators
        
    type_filter : str, list of str, None
        available values are 'classifier', 'regressor', 'cluster' and 
        'transformer' to get estimators only of these specific types, or a 
        list of these to get the estimators that fit at least one of the types.
        default is None        
    
    Return
    ---------
    instance :
        if oper_str is None, return dict of available classifers and regressors
        and transformers, classified by class type, 
        categorized as 'default' and 'predefined'
        
        if oper_str not None, return intance of estimators for given 
        class.__name__ 'oper_str'
        
    '''
    error_msg = "invalid operator name '{}' for get_lw_estimators()"
    if oper_str is None:
        return _get_keys()

    # predefined operator instances
    # search predefined instance first
    predefined = predifined_instance()
    if predefined.get(oper_str) is not None:
        return predefined.get(oper_str)

    # all lw estimators with default __init__ parameters
    # search default Classifiers/model Class
    t = lw_all_estimators(type_filter=type_filter)
    try:
        return dict(t).get(oper_str)()
    except:
        raise KeyError(error_msg.format(oper_str))


def _get_keys():
    '''return all available dict of keys for operators, 
    categorized as 'default' and 'predefined'
    '''
    est1 = dict(lw_all_estimators())
    est2 = predifined_instance()
    keys = {}
    keys['default'] = _categorize_operators(est1)
    keys['predefined'] = _categorize_operators(est2)
    return keys


def _categorize_operators(estimator_dict):
    '''to categorize estimators by classes
    
    ['classifier', 'linearclassifier', 'baggingclassifier']
    
    estimator_dict:
        dict of estimators/classifer classes or instances
    '''
    from collections import OrderedDict

    def filter_classes(cls, dict_ops):
        '''
        '''
        lst = []
        for k, v in dict_ops.items():
            try:
                is_true = [issubclass(v, i) for i in get_flat_list(cls)]
            except:
                is_true = [
                    issubclass(v.__class__, i) for i in get_flat_list(cls)
                ]
            if all(is_true):
                lst.append(k)
        return lst

    keys = OrderedDict()
    keys['classifier'] = filter_classes(ClassifierMixin, estimator_dict)
    keys['linearclassifier'] = filter_classes(LinearClassifierMixin,
                                              estimator_dict)

    keys['ensemble'] = filter_classes([ClassifierMixin, BaseEnsemble],
                                      estimator_dict)
    keys['cluster'] = filter_classes(ClusterMixin, estimator_dict)
    keys['regressor'] = filter_classes(RegressorMixin, estimator_dict)

    keys['transformer'] = filter_classes(TransformerMixin, estimator_dict)
    keys['under_sampler'] = filter_classes(BaseUnderSampler, estimator_dict)
    keys['over_sampler'] = filter_classes(BaseOverSampler, estimator_dict)
    keys['cleaning_sampler'] = filter_classes(BaseCleaningSampler,
                                              estimator_dict)
    keys['sampler'] = filter_classes(BaseSampler, estimator_dict)
    keys['function_sampler'] = filter_classes(FunctionSampler, estimator_dict)

    keys['outlier_detection'] = filter_classes(OutlierMixin, estimator_dict)
    keys['selector'] = filter_classes(SelectorMixin, estimator_dict)

    return keys


def count_operators():
    '''count all operators
    
    Return 
    -------
    d : dict
        number of operators for each operator type
        
    '''
    return {k: len(v) for k, v in _get_keys()['default'].items()}


def lw_all_estimators(type_filter=None):
    """Get a list of all estimators from  package 'lwmlearn' and 'sklearn' and
    'imblearn'

    This function crawls the module and gets all classes that inherit
    from BaseEstimator of sklearn. Classes that are defined in test-modules 
    are not included.
    By default meta_estimators such as GridSearchCV are also not included.

    Parameters
    ----------
    type_filter : string, list of string,  or None, default=None:
        available values are 'classifier', 'regressor', 'cluster' and 
        'transformer' to get estimators only of these specific types, or a 
        list of these to get the estimators that fit at least one of the types
        
    pkg_path (path, list of path, or default=None):
        path of package, default lwmlearn.__path__
        
    Return
    -------
    estimators : list of tuples
        (name, class)
        - ``name`` is the class name as string
        - ``class`` is the class object
    """

    from sklearn.base import (BaseEstimator, ClassifierMixin, ClusterMixin,
                              RegressorMixin, TransformerMixin)
    import lwmlearn
    import sklearn
    import imblearn

    def is_abstract(c):
        if not (hasattr(c, '__abstractmethods__')):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True
    
    def is_metaestimator(c):
        if hasattr(c, "_required_parameters"):
            if len(c._required_parameters) > 0:
                return True
            else:
                return False
        else:
            return False

    all_classes = []
    # get parent folder

    for key, module in getmodules(pkg=[sklearn, imblearn, lwmlearn],
                                  subpkgs=True).items():
        classes = inspect.getmembers(module, inspect.isclass)
        all_classes.extend(classes)

    not_test_operators = [
        'BaseEstimator', 'Pipeline', 'LW_model', 'FeatureHasher',
        'DictVectorizer', 'ColumnTransformer', 'SparseRandomProjection',
        'MultiLabelBinarizer', 'SMOTENC', 'SparseCoder', 'MultiTaskLasso',
        'LabelEncoder', 'MultiTaskElasticNetCV', 'HashingVectorizer',
        'KernelCenterer', 'GaussianRandomProjection', 'MultiTaskLassoCV',
        'FeatureUnion', 'LabelBinarizer', 'MultiTaskElasticNet',
        'IsotonicRegression', 'GammaRegressor', 'PoissonRegressor'
    ]

    estimators = [
        c for c in all_classes
        if (issubclass(c[1], BaseEstimator) 
            and c[0] not in not_test_operators
            and not c[0].startswith('_')
            and not is_metaestimator(c[1])
        )
    ]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    all_classes = set(estimators)

    # check for duplicated keys
    df = pd.DataFrame(all_classes)
    dupli = df[df.duplicated(0, False)]
    if not dupli.empty:
        print(dupli)
        raise KeyError("duplicated operators name encountered \n")

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {
            'classifier': ClassifierMixin,
            'regressor': RegressorMixin,
            'transformer': TransformerMixin,
            'cluster': ClusterMixin
        }
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend(
                    [est for est in estimators if issubclass(est[1], mixin)])
        estimators = filtered_estimators
        if type_filter:
            raise ValueError("Parameter type_filter must be 'classifier', "
                             "'regressor', 'transformer', 'cluster' or "
                             "None, got"
                             " %s." % repr(type_filter))

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))


def pipe_main(pipe=None):
    '''pipeline construction using sklearn estimators, final step support only
    classifiers currently
    
    .. note::
        
        data flows through a pipeline consisting of steps as below:
            raw data --> clean --> encoding --> scaling --> feature construction 
            --> feature selection --> resampling --> final estimator
            see scikit-learn preprocess & estimators
    
    Parameters
    ----------
    pipe : str 
        in the format of 'xx_xx' of which 'xx' means steps in pipeline,
        default None
    
    Return
    ------
    instance : 
        pipeline instance of chosen steps
        
        if pipe is None, a dict indicating possible choice of 'steps'
    '''
    def _index_duplicated(string_list):
        '''index duplicated item in a list to make an unique list
        '''
        d = pd.Series(string_list).duplicated(False)
        n = 0
        new_list = []
        for i, j in zip(string_list, d):
            if j:
                new_list.append(str(n) + i)
            else:
                new_list.append(i)
            n += 1
        return new_list

    if pipe is None:
        return get_lw_estimators()
    elif isinstance(pipe, str):
        l = pipe.split('_')
        steps = []
        for i, j in zip(l, _index_duplicated(l)):
            steps.append((j, get_lw_estimators(i)))
        return Pipeline(steps)
    else:
        raise ValueError("input pipe must be a string in format 'xx[_xx]'")


def get_featurenames(pipe_line):
    """get input feature names for fitted pipeline
    

    Parameters
    ----------
    pipe_line : TYPE
        instance returned by pipe_main().

    Return
    -------
    feature names of fitted pipeline.

    """
    if isinstance(pipe_line, Pipeline):
        fn = None
        su = None
        steps = pipe_line.steps
        n = len(steps)
        i = 0
        while i < n:
            tr = steps[i][1]
            if hasattr(tr, 'get_feature_names'):
                fn = pd.Series(tr.get_feature_names())
            if hasattr(tr, 'get_support'):
                su = tr.get_support()
                fn = fn[su]
            i += 1

        if fn is None:
            print('estimator has no feature_names attribute')
            return

        return fn


if __name__ == '__main__':
    pipe_main()
