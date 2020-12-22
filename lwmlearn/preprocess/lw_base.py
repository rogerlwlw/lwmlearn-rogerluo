# -*- coding: utf-8 -*-
"""Base class for preprocess cleaning

Use example
-----------

.. ipython::
    :okwarning:
    
    In [31]: from lwmlearn.preprocess.lw_base import  Cleaner 
    
    In [32]: import numpy as np
    
    In [3]: data0 = [   [1, '223,000', '+10%'],
       ...:             [10, '1,000,001', '+12%'],
       ...:             [12, '1,023,001', '+13%'],
       ...:             ['--', np.nan, '-50%'],
       ...:             ['-  _ --', 'null', '缺失值']]
    
    In [5]: cln = Cleaner(na1=None, na2=None)    

    In [6]: cln.fit_transform(data0)    
    
Created on Sun Dec 15 16:22:09 2019

@author: roger luo
"""
import pandas as pd
import numpy as np
import re

from pandas.core.dtypes import api
from sklearn.utils import validation
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from lwmlearn.utilis.utilis import get_flat_list, to_num_datetime_df
from lwmlearn.lwlogging import init_log

logger = init_log()


class LW_Base():
    '''base class for all 'lw' developed preprocessing operators

    attributes
    -----------
    out_labels : list
        labels for transformed X columns
        
    input_labels : list
        labels for original input X columns
        
    methods
    -------   
    _fit :
        
        to perform before fit method, to store input_labels
        
    _filter_labels:
        
        - to perform before transform method 
        - filter only stored labels (self.input_labels) 
        
    _check_df
    
        - convert X to DataFrame
        - drop duplicated cols
        - try converting X to numeric or datetime or object dtype
        - recognize na_value strings in data
        - convert values that solely are space strs to np.nan
        
    _check_is_fitted
    
        - check if operator has been fitted
        
    get_feature_names
    
        - return out_labels, to get selected feature names
        
    '''
    def _check_df(self, X, na_values=None):
        '''perform some basic cleaning to input data
        
        convert X to DataFrame, drop duplicated cols, try converting X
        to numeric or datetime or object dtype.
        
        parameters
        -----------
        X : 2d array
            data X will be converted as DataFrame   
        na_values : list
            list of values to recognize as null values, like ['null', 'nan']
            
        return
        ----------
        X: dataframe
            cleaned df
        
        '''
        try:
            X = pd.DataFrame(X)
        except Exception:
            raise ValueError('input must be DataFrame convertible')
        if X.empty:
            raise ValueError('X empty')

        # recognize na_values
        if na_values is not None:
            na_values = get_flat_list(na_values)
            X = X.apply(lambda x: x.where(~x.isin(na_values)))

        # convert accounting number or space strings
        X = basic_cleandata(X)

        # convert to dtype
        X = to_num_datetime_df(self._drop_duplicated_cols(X))
        return X

    def _filter_labels(self, X, na_values=None):
        '''to perform before transform method 
        '''
        self._check_is_fitted(['input_labels'])
        # --filter input_labels
        X = self._check_df(X, na_values)
        X = X.reindex(columns=getattr(self, 'input_labels'))
        if X.isna().all(None):
            raise ValueError(
                'no X column matchs with transfromer input_labels')
        return X

    def _check_is_fitted(self, attributes):
        '''check if self has been fitted by checking after-fit attributes
        '''
        validation.check_is_fitted(self, attributes)

    def _fit(self, X, na_values=None):
        '''to perform before fit method
        '''
        X = self._check_df(X, na_values)
        # -- store input_labels
        self.input_labels = X.columns.tolist()
        return X

    def _drop_duplicated_cols(self, X):
        '''drop duplicated cols 
        '''

        columns = X.columns
        if columns.is_unique:
            return X
        else:
            col_dup = columns[columns.duplicated('first')]
            if getattr(self, 'verbose') > 0:
                msg = "{} duplicated columns '{}' are dropped\n ".format(
                    len(col_dup), col_dup)
                logger.info(msg)
            return X.drop(columns=col_dup)

    def get_feature_names(self, ):
        '''get out_labels, return input_labels if None 
        '''
        try:
            return getattr(self, 'out_labels', self.input_labels)
        except:
            raise ValueError('X not fitted, perform fit  method first')


class Cleaner(BaseEstimator, TransformerMixin, LW_Base):
    '''data cleaning
    
    mainly performs step by step
    
    - store input column labels
    - recognize na_values and replace them with np.nan
    - recognize all space strs values and replace them with np.nan
    - clean accounting number or percentages like 1,234,000 or 20%
    - convert to numeric/str/datetime dtype
    - drop all na/constant/UID columns
    - fill in values for null for both numeric and categorical column
    - filter columns of specific dtypes specified
    - store output columns labels
    
            
    Attributes
    -----------
    obj_na : instance
        imputer instance for categorical data 
    num_na : instance
        imputer instance for numerical data
    objcols : list
        column label for object dtype
    numcols : list
        column label for numeric dtype
        
        
    '''
    def __init__(self,
                 dtype_filter='not_datetime',
                 verbose=0,
                 na1=None,
                 na2=None,
                 na_thresh=1,
                 na_values=[
                     'nan', 'NaN', 'null', 'NULL', 'None', '缺失值', -999, -99999
                 ],
                 uniq_frac=0.95,
                 count_frac=0.01,
                 drop_uid=True):
        """
        Parameters
        ----------- 
    
        dtype_filter : str
            options are ['num', 'obj', 'datetime', 'not_datetime', 'all']
            
            - num - filter only numeric dtype
            - obj - filter only obj dtype
            - datetime - filter only datetime dtype
            - not_datetime - exclude only datetime dtype
            - all - all dtypes
            
            default is 'not_datetime'
            
        na1 : str
            fill in strategy for categorical data column
            
            - default None, don't fill
            - na1='most_frequent', fill in most frequent category 'xxx' 
            - na1='xxx', fill in 'xxx' string
        
        na2 : int or str
            fill in stategy for numeric data column
        
            - default None, don't fill
            - na2='mean', fill in mean value of column
            - na2=int, fill in integer number
            
        na_thresh: int or float(0.0~1.0)
            thresh number of non-null values to drop, 
            default is 1
            
        na_values : list
            default ['nan', 'NaN', 'null', 'NULL','缺失值', -999, -99999]
            strings in na_values will be recognized as null and replaced
            with np.nan
                    
        uniq_frac : int or float(0.0-1.0)
            if drop_uid is True for numerical data, the fraction of number of 
            unique values limit, remove column over the limit
            
            - defaul 0.95
            - the greater (1.0) the value less likely column will be dropped
        
        count_frac : fraction
            for categirical column ,the maximum of fraction of counts for each 
            category must be greater than, otherwise the column will be dropped.
            
            - default is 0.01
            - the less the value, less likely column will be dropped
            
        drop_uid : bool
            whether or not to drop uid columns, 
            defualt True        

        """

        L = locals().copy()
        L.pop('self')
        self.set_params(**L)

    def fit(self, X, y=None):
        '''fit input_labels & out_labels 
        '''
        X = self._fit(X, self.na_values)
        # drop na columns over na_thresh
        na_col = X.columns[X.apply(lambda x: all(x.isna()))]
        length = len(X)

        # drop null column
        thresh = self.na_thresh
        if api.is_integer(thresh):
            pass
        elif api.is_float(thresh):
            thresh = length * thresh
        else:
            msg = "'na_thresh' must be integer or float"
            logger.exception(msg, stack_info=True)
            raise ValueError(msg)
        X.dropna(axis=1, how='any', thresh=thresh, inplace=True)

        # drop constant
        const_col = []
        for k, col in X.iteritems():
            if (api.is_numeric_dtype(col) and col.std()<0.01) \
            or len(pd.unique(col))==1:
                X.drop(k, axis=1, inplace=True)
                const_col.append(k)

        # drop uid cols or too discrete data
        uid_col = []
        if self.drop_uid:
            for k, col in X.iteritems():
                if api.is_object_dtype(col) or api.is_integer_dtype(col):
                    if len(pd.unique(col)) > self.uniq_frac * len(col):
                        X.drop(k, axis=1, inplace=True)
                        uid_col.append(k)

            # drop too small fractions of categorical data
            count_frac = []
            for k, col in X.iteritems():
                if api.is_object_dtype(col):
                    n = len(col)
                    max_frac = col.value_counts().max() / n
                    if max_frac < self.count_frac:
                        X.drop(k, axis=1, inplace=True)
                        count_frac.append(k)

        # filter dtypes
        options = {
            'not_datetime': X.select_dtypes(exclude='datetime').columns,
            'number': X.select_dtypes(include='number').columns,
            'object': X.select_dtypes(include='object').columns,
            'datetime': X.select_dtypes(include='datetime').columns,
            'all': X.columns
        }

        self.objcols = options.get('object')
        self.numcols = options.get('number')
        self.datetimecols = options.get('datetime')

        self.obj_na = _get_imputer(self.na1)
        self.num_na = _get_imputer(self.na2)
        # fill na values for obj dtype
        if self.obj_na is not None and not self.objcols.empty:
            self.obj_na.fit(X.reindex(columns=self.objcols))
        # fill na values for num dtype
        if self.num_na is not None and not self.numcols.empty:
            self.num_na.fit(X.reindex(columns=self.numcols))

        self.out_labels = options.get(self.dtype_filter).tolist()
        # --
        if len(na_col) > 0:
            msg =\
            'columns {} , total {} columns are null , have been dropped'.format(
                na_col, len(na_col))
            logger.info(msg)

        if len(uid_col) > 0:
            msg = '''columns {}, total {} columns are uid, have been dropped
                 '''.format(uid_col, len(uid_col))
            logger.info(msg)

        if len(const_col) > 0:
            msg = '''columns {},  total {} columns are constant , have been dropped
                  '''.format(const_col, len(const_col))
            logger.info(msg)

        # if self.get_params()['verbose'] > 0:
        logger.info(
            'matrix has valid {} columns; {} numeric columns; {} categorical columns; {} datetime columns'
            .format(X.shape[-1], len(self.numcols), len(self.objcols),
                    len(self.datetimecols)))

        return self

    def transform(self, X):
        '''transform X to a cleaned DataFrame of specified filter_dtype
        '''
        X = self._filter_labels(X, self.na_values)
        # --
        obj = X.reindex(columns=self.objcols)
        obj = obj.applymap(lambda x: x if pd.isna(x) else str(x))
        if self.obj_na is not None and not self.objcols.empty:
            obj = self.obj_na.transform(obj)
            obj = pd.DataFrame(obj, columns=self.objcols)

        num = X.reindex(columns=self.numcols)
        num = num.applymap(
            lambda x: x if pd.isna(x) else pd.to_numeric(x, errors='coerce'))
        if self.num_na is not None and not self.numcols.empty:
            num = self.num_na.transform(num)
            num = pd.DataFrame(num, columns=self.numcols)

        cols = set(self.out_labels)
        othercols = cols.difference(self.objcols.union(self.numcols))
        date_col = X.reindex(columns=othercols)

        X = pd.concat((i for i in [obj, num, date_col] if not i.empty), axis=1)
        X = X.reindex(columns=self.out_labels)

        return X


def basic_cleandata(df):
    """Try cleaning mapping value of df to numeric 
    
    convert all space string , like '-._-   ' to np.nan, 
    convert accounting number like 1,232,000 to 1232000 and percentage like
    10% to 0.1.
    
    Parameters
    ----------
    df : data frame.

    Returns
    -------
    df : data frame.

    """
    # -- remove '^\s*$' for each cell
    df = df.replace("^[-\_\s\.]*$", np.nan, regex=True)

    # convert accounting number to digits by remove ',' and percentages
    df = df.applymap(_convert_numeric)

    return df


def _convert_numeric(x_str):
    """check if x_str is accounting or percentage number
    
    convent them to float or integer
    
    integer code string starting with '0' will be ingored, treated as code str

    Parameters
    ----------
    x_str : TYPE
        DESCRIPTION.

    Returns
    -------
    x_str : TYPE
        float(x_str) or x_str.

    """
    # str_pattern to represent numrical data
    numeric_pattern = "^[-+]?\d*(?:\.\d*)?(?:\d[eE][+\-]?\d+)?(\%)?$"
    # integer code starts with 0
    code_pattern = "^0\d+$"

    def _is_numeric_pattern(x_str):
        try:
            float(x_str)
            return True
        except:
            return False

    if isinstance(x_str, str):
        x_str_r = re.sub("[,\%\$]", '', x_str)

        if re.match(numeric_pattern, x_str_r) is not None:
            if re.match(code_pattern, x_str_r) is None:
                try:
                    x_str_r = int(x_str_r)
                except:
                    x_str_r = float(x_str_r)

                if re.match('^.*\%$', x_str):
                    # for percentage
                    return x_str_r / 100
                else:
                    # not percentage
                    return x_str_r
    # return input x_str
    return x_str


def _get_imputer(imput):
    '''return imputer instance for given strategy
    '''
    if api.is_number(imput):
        imputer = SimpleImputer(strategy='constant', fill_value=imput)
    elif imput == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif imput == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
    elif isinstance(imput, str):
        imputer = SimpleImputer(strategy='constant', fill_value=imput)
    else:
        return

    return imputer


if __name__ == '__main__':
    # from lwmlearn.dataset.load_data import get_local_data
    # from sklearn.datasets import  make_classification
    # data = get_local_data('heart.csv')
    # clean = Cleaner(na1='missing', na2='mean')
    # y = data.pop('y')
    # x = data
    # data_clean = clean.fit_transform(x)
    data0 = [[1.0, '223,000', '+10%'], [10, '1,000,001', '+12%'],
             [1.2, '1,023,001', '+13%'], ['-..-', np.nan, '-50%'],
             ['nan', 'null', '缺失值']]
    data0
    cln = Cleaner(na1=None, na2=None)
    cln.fit_transform(data0)
