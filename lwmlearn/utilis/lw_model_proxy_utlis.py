# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:55:15 2018

@author: roger
"""

import numpy as np
import pandas as pd
from itertools import compress
from scipy.sparse import issparse

from sklearn.utils import validation
from sklearn.model_selection._split import check_cv
from sklearn.utils import indexable


def _reset_index(*array):
    '''reset_index for df or series, return list of *arrays
    '''
    rst = []
    for i in array:
        if isinstance(i, (pd.DataFrame, pd.Series)):
            rst.append(i.reset_index(drop=True))
        else:
            rst.append(i)
    return rst


def _parse_version(version_string):
    version = []
    for x in version_string.split('.'):
        try:
            version.append(int(x))
        except ValueError:
            # x may be of the form dev-1ea1592
            version.append(x)
    return tuple(version)


def _array_indexing(array, key, key_dtype, axis):
    """Index an array or scipy.sparse consistently across NumPy version."""
    np_version = _parse_version(np.__version__)
    if np_version < (1, 12) or issparse(array):
        # FIXME: Remove the check for NumPy when using >= 1.12
        # check if we have an boolean array-likes to make the proper indexing
        if key_dtype == 'bool':
            key = np.asarray(key)
    if isinstance(key, tuple):
        key = list(key)
    return array[key] if axis == 0 else array[:, key]


def _pandas_indexing(X, key, key_dtype, axis):
    """Index a pandas dataframe or a series."""
    if hasattr(key, 'shape'):
        # Work-around for indexing with read-only key in pandas
        # FIXME: solved in pandas 0.25
        key = np.asarray(key)
        key = key if key.flags.writeable else key.copy()
    elif isinstance(key, tuple):
        key = list(key)
    # check whether we should index with loc or iloc
    indexer = X.iloc if key_dtype == 'int' else X.loc
    return indexer[:, key] if axis else indexer[key]


def _list_indexing(X, key, key_dtype):
    """Index a Python list."""
    if np.isscalar(key) or isinstance(key, slice):
        # key is a slice or a scalar
        return X[key]
    if key_dtype == 'bool':
        # key is a boolean array-like
        return list(compress(X, key))
    # key is a integer array-like of key
    return [X[idx] for idx in key]


def _determine_key_type(key, accept_slice=True):
    """Determine the data type of key.

    Parameters
    ----------
    key : scalar, slice or array-like
        The key from which we want to infer the data type.

    accept_slice : bool, default=True
        Whether or not to raise an error if the key is a slice.

    Returns
    -------
    dtype : {'int', 'str', 'bool', None}
        Returns the data type of key.
    """
    err_msg = ("No valid specification of the columns. Only a scalar, list or "
               "slice of all integers or all strings, or boolean mask is "
               "allowed")

    dtype_to_str = {int: 'int', str: 'str', bool: 'bool', np.bool_: 'bool'}
    array_dtype_to_str = {
        'i': 'int',
        'u': 'int',
        'b': 'bool',
        'O': 'str',
        'U': 'str',
        'S': 'str'
    }

    if key is None:
        return None
    if isinstance(key, tuple(dtype_to_str.keys())):
        try:
            return dtype_to_str[type(key)]
        except KeyError:
            raise ValueError(err_msg)
    if isinstance(key, slice):
        if not accept_slice:
            raise TypeError('Only array-like or scalar are supported. '
                            'A Python slice was given.')
        if key.start is None and key.stop is None:
            return None
        key_start_type = _determine_key_type(key.start)
        key_stop_type = _determine_key_type(key.stop)
        if key_start_type is not None and key_stop_type is not None:
            if key_start_type != key_stop_type:
                raise ValueError(err_msg)
        if key_start_type is not None:
            return key_start_type
        return key_stop_type
    if isinstance(key, (list, tuple)):
        unique_key = set(key)
        key_type = {_determine_key_type(elt) for elt in unique_key}
        if not key_type:
            return None
        if len(key_type) != 1:
            raise ValueError(err_msg)
        return key_type.pop()
    if hasattr(key, 'dtype'):
        try:
            return array_dtype_to_str[key.dtype.kind]
        except KeyError:
            raise ValueError(err_msg)
    raise ValueError(err_msg)


def safe_indexing(X, indices, axis=0):
    """Return rows, items or columns of X using indices.

    .. warning::

        This utility is documented, but **private**. This means that
        backward compatibility might be broken without any deprecation
        cycle.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series
        Data from which to sample rows, items or columns. `list` are only
        supported when `axis=0`.
    indices : bool, int, str, slice, array-like
        - If `axis=0`, boolean and integer array-like, integer slice,
          and scalar integer are supported.
        - If `axis=1`:
            - to select a single column, `indices` can be of `int` type for
              all `X` types and `str` only for dataframe. The selected subset
              will be 1D, unless `X` is a sparse matrix in which case it will
              be 2D.
            - to select multiples columns, `indices` can be one of the
              following: `list`, `array`, `slice`. The type used in
              these containers can be one of the following: `int`, 'bool' and
              `str`. However, `str` is only supported when `X` is a dataframe.
              The selected subset will be 2D.
    axis : int, default=0
        The axis along which `X` will be subsampled. `axis=0` will select
        rows while `axis=1` will select columns.

    Returns
    -------
    subset
        Subset of X on axis 0 or 1.

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if indices is None:
        return X

    if axis not in (0, 1):
        raise ValueError(
            "'axis' should be either 0 (to index rows) or 1 (to index "
            " column). Got {} instead.".format(axis))

    indices_dtype = _determine_key_type(indices)

    if axis == 0 and indices_dtype == 'str':
        raise ValueError("String indexing is not supported with 'axis=0'")

    if axis == 1 and X.ndim != 2:
        raise ValueError(
            "'X' should be a 2D NumPy array, 2D sparse matrix or pandas "
            "dataframe when indexing the columns (i.e. 'axis=1'). "
            "Got {} instead with {} dimension(s).".format(type(X), X.ndim))

    if axis == 1 and indices_dtype == 'str' and not hasattr(X, 'loc'):
        raise ValueError(
            "Specifying the columns using strings is only supported for "
            "pandas DataFrames")

    if hasattr(X, "iloc"):
        return _pandas_indexing(X, indices, indices_dtype, axis=axis)
    elif hasattr(X, "shape"):
        return _array_indexing(X, indices, indices_dtype, axis=axis)
    else:
        return _list_indexing(X, indices, indices_dtype)


def split_cv(*arrays, y=None, groups=None, cv=3, random_state=None):
    '''supervise splitting

    *arrays
        - arrays to be splitted, usually X    
    y
        - class label, if None not to stratify
    groups
        - split by groups
    cv
        - number of splits

    return
    ----
    generator of list containing splited arrays,shape = [m*n*k], for 1 fold
    [(0train, 0test), (1train, 1test), ...]

    m - indices of folds [0 : cv-1]
    n - indice of variable/arrays [0 : n_arrays-1]
    k - indice of train(0)/test[1] set [0:1]
    '''

    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    validation.check_consistent_length(*arrays, y, groups)
    arrays = list(arrays)

    if cv == 1:
        if y is not None:
            arrays.append(y)
        return [[(i, i) for i in arrays]]
    # get cross validator
    if y is not None:
        arrays.append(y)
        cv = check_cv(cv, y=y, classifier=True)
    else:
        cv = check_cv(cv, classifier=False)
    # set random state
    if hasattr(cv, 'random_state'):
        cv.random_state = random_state
    # reset_index pandas df or series
    arrays = _reset_index(*arrays)
    arrays = indexable(*arrays)
    # get indexing method
    train_test = ([
        (safe_indexing(i, train_index), safe_indexing(i, test_index))
        for i in arrays
    ] for train_index, test_index in cv.split(arrays[0], y, groups))

    return train_test


def get_splits_combined(xy_splits, ret_type='test'):
    '''return list of combined X&y DataFrame for cross validated test set
    '''
    data_splits_test = [
        pd.concat((pd.DataFrame(i[1]) for i in item), axis=1)
        for item in xy_splits
    ]

    data_splits_train = [
        pd.concat((pd.DataFrame(i[0]) for i in item), axis=1)
        for item in xy_splits
    ]

    if ret_type == 'test':
        return data_splits_test
    if ret_type == 'train':
        return data_splits_train
