# -*- coding: utf-8 -*-
"""Discretize Data 

Created on Fri Dec 20 09:23:15 2019

@author: roger luo
"""

import pandas as pd
import numpy as np
from pandas.core.dtypes import api
from sklearn.utils import validation
from sklearn.tree import DecisionTreeClassifier
from lwmlearn.utilis.utilis import get_kwargs, to_num_datetime
import scipy.stats as stats

from lwmlearn.lwlogging import init_log

def _tree_univar_bin(arr_x, arr_y, **kwargs):
    """univariate binning based on binary decision Tree
    
    Parameters
    ----------
    arr_x : 1d array like
        data to be discretized.
    arr_y : 1d binary array
        target class y.
    
    Keyword args
    ------------    
    kwargs : other key words
        key words used by :class:`DecisionTreeClassifier`

    Returns
    -------
    cut_edges : ndarray
        array of cutting points of binning edges, while cut_edges[0] = -inf,
        cut_edges[-1]=inf.
    """
    
    validation.check_consistent_length(arr_x, arr_y)
    clf = DecisionTreeClassifier(
        **get_kwargs(DecisionTreeClassifier, **kwargs))
    X = np.array(arr_x).reshape(-1, 1)
    Y = np.array(arr_y).reshape(-1, 1)

    # tree training
    clf.fit(X, Y)
    thresh = clf.tree_.threshold
    feature = clf.tree_.feature
    thresh = np.unique(thresh[(feature >= 0).nonzero()]).round(
        kwargs.get('decimal', 8))
    cut_edges = np.append(np.append(-np.inf, thresh), np.inf)
    cut_edges = np.unique(cut_edges)
    return cut_edges


def _mono_cut(X, Y):
    """mono binning edges of data 
    
    the binning group increases monotonically with "y" mean value 
    
    Parameters
    ----------
    X : 1d array like
        data to be discretized.
    Y : 1d binary array
        target class y.

    Returns
    -------
    bins : TYPE
        DESCRIPTION.

    """
    
    r = 0
    n = 10
    while np.abs(r) < 1 and n > 2:
        out, bins = pd.qcut(X, n, duplicates='drop', retbins=True)
        d1 = pd.DataFrame({
            "X": X,
            "Y": Y,
            "Bucket": pd.qcut(X, n, duplicates='drop')
        })
        d2 = d1.groupby('Bucket', as_index=False)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    bins[0] = -np.inf
    bins[-1] = np.inf
    return bins


def bin_tree(X,
             y,
             cat_num_lim=0,
             max_leaf_nodes=10,
             min_samples_leaf=0.05,
             random_state=0,
             verbose=0,
             **kwargs):
    '''Discretize features matrix based on Binary DecisionTree classifier
    
    .. note::
    
        CART tree - gini impurity as criterion, not numeric dtype column will 
        be igored, unique number of values less than "cat_num_lim" will be 
        ignored
    
    parameters
    -----------
    X : 2d array or dataframe matrix
        contain feature matrix, should be numerical dtype
    y : str
        col of class label, binary
        
    cat_num_lim
        number of unique vals limit to be treated as continueous feature
        
    max_leaf_nodes
        max number of bins
        
    min_samples_leaf
        minimum number of samples in leaf node
        
    **kwargs
        other tree keywords
    
    return
    -------
    bin_edges : dict
        {'col_name' : bin_edges }
    '''

    bin_edges = {}
    cols = []
    un_split = []
    # reset index
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    for name, col in X.iteritems():
        df = pd.DataFrame({'x': col, 'y': y})
        col_notna = df.dropna().x
        y_notna = df.dropna().y
        if (len(pd.unique(col_notna)) > cat_num_lim
                and api.is_numeric_dtype(col_notna)):
            # call _tree_univar_bin
            bin_edges[name] = _tree_univar_bin(
                col_notna,
                y_notna,
                max_leaf_nodes=max_leaf_nodes,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                **get_kwargs(DecisionTreeClassifier, **kwargs))
            if len(bin_edges[name]) < 3:
                un_split.append(name)
        else:
            cols.append(name)
    
    # log process
    logger = init_log()
    
    msg1 = '''total of {2} unchaged (unique counts less 
           than {1} or categorical dtype) =\n "{0}" 
           '''.format(pd.Index(cols), cat_num_lim, len(cols))
    
    msg2 = '''total of {1} unsplitable features = \n {0} ...
           '''.format(pd.Index(un_split), len(un_split))
           
    msg3 = 'total of {} bin_edges obtained \n'.format(len(bin_edges))
    if cols:
        logger.info(msg1)
    if un_split:
        logger.info(msg2)
    if bin_edges:
        logger.info(msg3)

    return bin_edges


def _check_binning_keywords(bins, q, max_leaf_nodes, mono):
    '''check that only one of (bins, q, max_leaf_nodes, mono) be input with
    value, if not assign q=10 and bins=max_leaf_nodes=mono=None
    
    '''
    logger = init_log()
    if sum([i is not None for i in (bins, q, max_leaf_nodes, mono)]) != 1:
        msg =\
            '''warning: (q={}, bins={}, max_leaf_nodes={}, mono={}) but only one of 
            (bins, q, max_leaf_nodes, mono) can be input, if not, parameters will be 
            reset as q=10 and bins=max_leaf_nodes=mono=None
            '''.format(q, bins, max_leaf_nodes, mono)
        
        logger.warning(msg)
        
        bins = max_leaf_nodes = mono = None
        q = 10

    return bins, q, max_leaf_nodes, mono


def binning(y_pre=None,
             bins=None,
             q=None,
             max_leaf_nodes=None,
             mono=None,
             y_true=None,
             labels=None,
             **kwargs):
    '''supervised binning 
    
    of y_pre based on y_true if y_true is not None
    
    .. _binningmeth:
        
    parameters
    -----------
    y_pre : 1d array_like
         value of y to be cut
    y_true : 1d array like
        binary target y_true for supervised cutting
    
    bins : int 
        number of equal width bins
        
    q : int 
        number of equal frequency bins 
         
    max_leaf_nodes : int
        number of tree nodes bins using tree cut
        if not None use supervised cutting based on decision tree
        
    mono : int
        number of bins that increases monotonically with "y" mean value  
        
        .. note::
            
            arguments [ q, bins, max_leaf_nodes, mono ] control binning method 
            and only 1 of them can be specified. 
            if not valid assign q=10 and bins=max_leaf_nodes=mono=None
            
    labels : bool
        see pd.cut, if False return integer indicator of bins, 
        if True return arrays of labels (or can be manually input)
        
    Keyword args
    -------------
    kwargs : 
        Decision tree keyswords
            
    min_impurity_decrease=0.001
        
    random_state=0 
        
    return 
    --------
    y_binlabel : array      
         bin label of y_pre 
    bin_edge : array
         ndarray of bin edges

    '''
    bins, q, max_leaf_nodes, mono = _check_binning_keywords(
        bins, q, max_leaf_nodes, mono)

    y_pre = to_num_datetime(y_pre)
    y_pre_input = y_pre.copy()
    if y_true is not None:
        y_true = to_num_datetime(y_true)
        y_true = np.array(y_true)
    
    # drop na values for y_pre & y_true pairs in case of supervised cutting
    df = pd.DataFrame({'ypre': np.array(y_pre), 'ytrue': y_true})
    df = df.dropna(subset=['ypre'])
    y_pre = df.pop('ypre')
    y_true = df.pop('ytrue')
    
    # if y_pre is not numeric data type, do not perform cut
    if not api.is_numeric_dtype(y_pre):
        return y_pre_input, y_pre.unique()

    if q is not None:
        bins = np.percentile(y_pre, np.linspace(0, 100, q + 1))
        bins[0] = -np.Inf
        bins[-1] = np.Inf
        bins = np.unique(bins)

    if max_leaf_nodes is not None:
        if y_true.isna().sum() > 0:
            raise ValueError('none nan y_true must be supplied for tree cut')
        y_pre0 = pd.DataFrame(y_pre)
        bins_dict = bin_tree(y_pre0,
                             y_true,
                             max_leaf_nodes=max_leaf_nodes,
                             **kwargs)
        bins = list(bins_dict.values())[0]

    if mono is not None:
        if y_true.isna().sum() > 0:
            raise ValueError('none nan y_true must be supplied for mono cut')
        bins = _mono_cut(Y=y_true, X=y_pre)

    if isinstance(bins, int):
        bins = np.linspace(np.min(y_pre), np.max(y_pre), bins + 1)
        bins[0] = -np.inf
        bins[-1] = np.Inf

    if bins is None:
        raise ValueError('no cutting bins supplied')

    if labels is True:
        labels = None

    y_binlabel, bin_edge = pd.cut(y_pre_input,
                          bins,
                          duplicates='drop',
                          retbins=True,
                          labels=labels)
    return y_binlabel, bin_edge
