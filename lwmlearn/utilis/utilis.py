# -*- coding: utf-8 -*-
""" common functions



Created on Fri Nov  9 11:55:15 2018

@author: roger
"""
import pandas as pd
import numpy as np
import pkgutil
import inspect

from pandas.core.dtypes import api
from functools import wraps, reduce


def ppdf(df, decimals=1):
    """ pretty print a dataframe table
    
    Parameters
    ----------
    df : dataframe
    
    decimals : int, optional
        number of decimals to keep. The default is 1.

    Returns
    -------
    df : dataframe
        data frame with each cell as str.

    """
    def f(s):

        if api.is_numeric_dtype(s):
            if s.apply(abs).max() <= 1:
                s = s.apply(lambda x: str(round(x * 100, decimals)) + '%')
            else:
                fmt = "{" + ":,.{}f".format(decimals) + "}"
                s = s.apply(lambda x: fmt.format(x))
        return s

    df = df.apply(f, axis=0)

    return df


def getmodules(pkg, subpkgs=True):
    '''crawls packages and get all modules
    
    parameters
    ----------
    
    pkg (list of packages): 
        list of packages to look for modules
        
    subpkgs (bool): 
        if True walkthrough all modules recursively,
        if False only sub-modules, not sub-packages
                
    return
    -------
    d : dict
        module object under given path {module name: module object}
    '''
    from importlib import import_module

    module = {}

    for p in get_flat_list(pkg):
        path = p.__path__
        prefix = p.__name__ + '.'
        if not subpkgs:
            gen = pkgutil.iter_modules(path, prefix)
        else:
            gen = pkgutil.walk_packages(path, prefix, onerror=lambda x: None)

        for finder, name, ispkg in gen:
            if ".tests." in name or "externals" in name:
                continue
            if "._" in name or ".__" in name:
                continue
            # module[name] = __import__(name, fromlist='dummy')
            module[name] = import_module(name)

    return module


def join_embed_keys(dictionary, delimiter='_'):
    ''' join keys by delimiter of embedding dicts
    
    like {k1 : {k2 : v}} to {k1_k2 : v}
    
    parameters
    ----------
    dictionary : dict
        
    delimiter : str
        delimiter to use
        
    return
    -------
    d : ditc
        
    '''
    d = dictionary.copy()
    while any([api.is_dict_like(d[k]) for k in d]):
        for k, v in list(d.items()):
            if api.is_dict_like(v):
                v = d.pop(k)
                for kk, vv in v.items():
                    key = '_'.join([str(k), str(kk)])
                    d.update({key: vv})
    return d


def merge_dfs(df_list, how='inner', **kwargs):
    '''merge list of DataFrames recursively
    
    parameters
    ----------
    df_list : list
        list of dataframe
    
    how : str
        merge method
        
    '''
    lamf = lambda left, right: pd.merge(left, right, how=how, **kwargs)
    return reduce(lamf, df_list)


def get_file_path():
    import sys
    import os
    return os.path.abspath(sys._getframe().f_code.co_filename)


def get_flat_list(x):
    '''list and flatten object into one dimension list
    
    parameters
    ----------
    x : list
        input list
    return
    -------
    l :  List
        1d list output list
    '''
    if isinstance(x, list) and not isinstance(x, (str, bytes)):
        return [a for i in x for a in get_flat_list(i)]
    else:
        return [x]


def default_func(func, new_funcname=None, **kwargs_outer):
    '''return function with default keyword arguments specified in
    kwargs_outer where a new_funcname is given to newly initialized func
    '''
    kwargs_outer = get_kwargs(func, **kwargs_outer)

    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs_outer.update(kwargs)
        return func(*args, **kwargs_outer)

    wrapper.__name__ = new_funcname if new_funcname else func.__name__
    return wrapper


def dict_subset(d, keys):
    '''return subset of dict by list of keys
    
    parameters
    -----------
    d : dict
        input dict
    keys : list
        list of keys
        
    return
    -------
    d : dict
        subset of input dict
        
    '''
    return {i: d.get(i) for i in keys}


def dict_diff(d, keys):
    '''subset of dicts with a series of keys removed
 
    parameters
    -----------
    d : dict
        input dict
    keys : list
        list of keys
        
    return
    -------
    d : dict
        subset of input dict with keys removed
        
    '''
    d_keys = set(d.keys())
    in_keys = d_keys.difference(keys)
    return dict_subset(d, in_keys)


def interset_update(d1, d2):
    '''return intersection of d1 and d2 and update with d2's value
    
    parameters
    ----------
    d1 : dict
    
    d2 : dict
    
    return
    --------
    d : dict
        intersection of d1 and d2 update d2's value
    '''
    k_intersect = d1.keys() & d2.keys()
    d3 = dict_subset(d2, k_intersect)
    d1.update(d3)
    return d1


def var_gen(size, kind='p', **kwargs):
    '''generate random variables of specific kind for given size
    
    parameters
    -----------
    size:
        size of ndarray, for example, 1D-100, 2D-(100, 3) 
        
    kind : str 
        ['p', 'bin',  'cat', 'time', 'int', 'float', 'ID', 'w']
    
        'p' :
            probability (0, 1), could be constrained by beta distribution, 
            'alpha=3', 'beta=3'
        
        'float':
             float number, could be constrained by specified 
             'low=0', 'mode=None', 'high=10'
        
        'bin':
            binary 0/1, could be controlled by pos_ratio=0.18
            
        'int':
            integer, could be be constained by low=0, high=10,
            if sequence is specified, select from sequence, egg: 
            sequence_set =[3, 6, 12], weights = [0.2, 0.2, 0.6]
            
        'cat':
            categorical, requires sequence_set=[],  weights=[], 
    
        'time':
            datetime, requires start=, end=, [periods=, freq=], to draw samples
            from date_range
        
        'ID':
            uniqueID, prefix + range, egg['ID0', 'ID1']
            
        'w': weithts
            random positive array sum=1
    
    keyword args
    -------------
    
    'low', 'high', 'mode':
        for continueous dist, lower, upper bound and mode number
    
    'sequence_set', 'weights':
        draw samples from a sequence, for assigned probabilies as weights 
    
    'alpha', 'beta':
        used in beta dist function
    
    'start', 'end','periods', 'freq':
        to generate data_range and from which to draw data samples
    
    'seed':
        random seed, default=None
      
    return
    -------
    array : ndarray
        generated random variable matrix
        
    '''
    def _weights_gen(size):
        '''
        '''
        wei = np.random.random(size)

        return wei / wei.sum()

    defaults = dict(
        alpha=3,
        beta=3,
        low=0,
        mode=None,
        high=10,
        pos_ratio=0.18,
        sequence_set=[],
        weights=[],
        start='2000-01-01',
        periods=None,
        end='2002-01-01',
        freq='D',
        prefix='ID',
    )
    defaults.update(**kwargs)
    if pd.core.dtypes.api.is_array_like(size):
        size = np.shape(size)

    # set random seed
    np.random.seed(defaults.get('seed'))

    if kind == 'p':
        a, b = defaults.get('alpha'), defaults.get('beta')
        vec = np.random.beta(a, b, size=size)
        return vec

    if kind == 'nan':
        return np.where(np.zeros(size), 0, np.nan)

    if kind == 'float':
        l, mode, h = defaults.get('low'), defaults.get('mode'), defaults.get(
            'high')
        s = defaults.get('sequence_set')
        w = defaults.get('weights')
        if len(s) > 0:
            if len(w) == 0: w = _weights_gen(np.shape(s))
            vec = np.random.choice(s, size=size, p=w)
        else:
            if mode is not None:
                vec = np.random.triangular(l, mode, h, size=size)
            else:
                vec = np.random.uniform(l, h, size=size)
        return vec

    if kind == 'bin':
        s = [0, 1]
        p = defaults.get('pos_ratio')
        vec = np.random.choice(s, size=size, p=[1 - p, p])
        return vec

    if kind == 'cat':

        s = defaults.get('sequence_set')
        w = defaults.get('weights')
        if len(s) == 0:
            raise ValueError('sequence_set must be specified')
        if len(w) == 0:
            w = _weights_gen(np.shape(s))
        vec = np.random.choice(s, size=size, p=w)
        return vec

    if kind == 'time':
        st, end, per, freq = [
            defaults.get(i) for i in ['start', 'end', 'periods', 'freq']
        ]
        s = pd.date_range(st, end, per, freq)
        s0 = defaults.get('sequence_set')
        w = defaults.get('weights')
        if len(s0) > 0:
            s = s0
        if len(w) == 0:
            w = _weights_gen(np.shape(s))
        vec = np.random.choice(s, size=size, p=w)
        return vec

    if kind == 'int':
        l, h = defaults.get('low'), defaults.get('high')
        s = defaults.get('sequence_set')
        w = defaults.get('weights')
        if len(s) > 0:
            if len(w) == 0: w = _weights_gen(np.shape(s))
            vec = np.random.choice(s, size=size, p=w)
        else:
            vec = np.random.randint(l, h, size=size)
        return vec

    if kind == 'ID':
        prefix = defaults.get('prefix')
        n = np.prod(size)
        # range number (0, 1, 2, ...)
        vec = np.arange(n).reshape(size)
        # add prefix
        vec = np.frompyfunc(lambda x: ''.join([prefix, str(x)]), 1, 1)(vec)
        return vec

    if kind == 'w':
        return _weights_gen(size)


def inverse_dict(d):
    '''return inversed dict {k, val} as  {val, k} 
    '''
    try:
        return {val: k for k, val in d.items()}
    except Exception:
        raise Exception('dict is not 1 to 1 mapping, cannot be inversed')


def get_current_function_name():
    '''get_current_function_name
    '''
    return inspect.stack()[1][3]


def get_kwargs(func, **kwargs):
    '''return subset of **kwargs that are of func arguments
    
    parameters
    ------------
    func : function
        function object
    
    kwargs : keyword agrs
    
    return
    -------
    d : dict
        keyword args of func
        
    '''
    func_args = set(inspect.signature(func).parameters.keys())
    
    func_args.intersection_update(kwargs)
    return {i: kwargs[i] for i in func_args}


def dec_iferror_getargs(func):
    ''' catch exceptions when calling func and return arguments input '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(repr(e))
            print('func name = ', func.__name__)
            print('args =\n ', args, '...')
            print('kwargs =\n ', kwargs, '...\n')
            raise Exception('''Error encountered in func {0} \n'''.format(
                func.__name__))

    return wrapper


def to_num_datetime(col, name='array', thresh=0.75, **kwargs):
    '''convert col to numeric or datetime if possible, otherwise remain
    unchaged 
    
    parameters
    -----------
    col : series scalar or ndarry
        input sequence
    
    name : str
        name of the col series 
    
    thresh : float
        default 0.8,  
        if more than the thresh percentage of X could be converted, 
          then should commit conversion   
    
    keyword args
    ------------
    other pandas to_datetime key words
    
    errors : {'ignore', 'raise', 'coerce'}
        default 'coerce'
        
        If 'raise', then invalid parsing will raise an exception
        
        If 'coerce', then invalid parsing will be set as NaN
        
        If 'ignore', then invalid parsing will return the input
    
    
    return
    --------
    s : series
        converted col
    '''
    try:
        col = pd.Series(col)
    except Exception:
        raise Exception('col must be 1-d array/list/tuple/dict/Series')

    if api.is_numeric_dtype(col):
        return col
    if api.is_datetime64_any_dtype(col):
        return col
    if api.is_categorical_dtype(col):
        return col
    if col.count() == 0:
        return col
    if col.astype(str).str.contains('^0\d+$').any():
        return col

    is_numeric_convertible = False
    not_null_count = col.count()

    try:
        num = pd.to_numeric(col, errors=kwargs.get('errors', 'coerce'))
        if num.count() / not_null_count >= thresh:
            col = num
            is_numeric_convertible = True
    except:
        pass
    if not is_numeric_convertible:
        params = {'errors': 'coerce', 'infer_datetime_format': True}
        params.update(kwargs)
        try:
            date = pd.to_datetime(col, **params)
            if pd.notnull(date).sum() / not_null_count >= thresh:
                col = date
            else:
                col = col.apply(lambda x: x if pd.isna(x) else str(x))
        except:
            pass

    return col


def to_num_datetime_df(X, thresh=0.8):
    '''convert each column to numeric or datetime if possible, otherwise remain
    unchanged 
    
    parameters
    -----------
    X : dataframe
        input dataframe to convert
    
    thresh : float
        default 0.8,  
        if more than the thresh percentage of col could be converted, 
        then should commit conversion
        
    '''
    try:
        X = pd.DataFrame(X)
    except Exception:
        raise ValueError('X must be df or convertible to df')
    lamf = lambda x: to_num_datetime(x, name=x.name, thresh=thresh)
    rst = X.apply(lamf, axis=0)
    return rst
