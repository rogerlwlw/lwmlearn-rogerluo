# -*- coding: utf-8 -*-
"""Categorical encoder

Created on Mon Dec 16 16:53:08 2019

@author: roger luo
"""
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from lwmlearn.preprocess.lw_base import LW_Base
from lwmlearn.lwlogging import init_log


class OhtEncoder(BaseEstimator, TransformerMixin, LW_Base):
    '''transform categorical features to  one-hot encoded
    
    transform only object dtype column, other dtype will remain unchanged
    
    Attributes
    ----------
    encoder : instance
        sklearn transformer instance
    out_labels : list
        column labels of transformed matrix
    encode_mapper : dict
        categories mapper of each column, 
        dict egg. {cloname : array(category names)}
        
    '''
    def __init__(
            self,
            handle_unknown='ignore',
            drop=None,
            sparse=False,
            dtype=np.float64,
    ):
        """
        
        parameters
        -----------
        handle_unknown : str
            default 'ignore', for one-hot encoding, unknown feature category
            will be treated as zeros, 'raise' error encountered unknow category
              
        sparse : bool
            default False, for one-hot encoding, which will return 2D arrays

        """
        L = locals().copy()
        L.pop('self')
        self.set_params(**L)

    def _check_categories(self, X):
        '''check if feature category are out of categories strained, 
        treat them as np.nan
        '''
        if len(self.encode_mapper) == 0: return X
        #
        mapper = self.encode_mapper
        isin_cat = X.apply(lambda x: x.isin(mapper.get(x.name, x)) | x.isna(),
                           axis=0)
        out_c = np.ravel(~isin_cat).sum()
        if out_c > 0:
            logger = init_log()
            logger.warning('''total of {} element out of categories and 
                        will be treated as np.nan '''.format(out_c))
        return X

    def fit(self, X, y=None):
        '''fit X using one-hot encoder 
        '''
        X = self._fit(X)
        self.obj_cols = X.select_dtypes('object').columns
        self.not_obj = X.columns.difference(self.obj_cols)

        self.encoder = OneHotEncoder(**self.get_params())
        self.encoder.fit(X.reindex(columns=self.obj_cols))

        self.encoder_fnames = self.encoder.get_feature_names(self.obj_cols)
        self.encode_mapper = dict(zip(self.obj_cols, self.encoder.categories_))
        self.out_labels = self.encoder_fnames.tolist() + self.not_obj.tolist()

        return self

    def transform(self, X):
        '''transform  X  to dummy variable
        
        transform only object dtype and other columns remain unchaged
        
        return
        -------
        X : dataframe
            encoded dataframe
        '''
        X = self._filter_labels(X)
        # --obj cols
        X0 = X.reindex(columns=self.obj_cols)
        X0 = self._check_categories(X0)
        X0 = self.encoder.transform(X0)
        X0 = pd.DataFrame(X0, columns=self.encoder_fnames)
        # --not obj do nothing
        X1 = X.reindex(columns=self.not_obj)
        rst = pd.concat((i for i in [X0, X1] if not i.empty), axis=1)
        rst = rst.reindex(columns=self.out_labels)
        return rst


class OrdiEncoder(BaseEstimator, TransformerMixin, LW_Base):
    '''transform categorical features to ordinal encoded
    
    out of training categories will be treated as np.nan and encoded as -1,
    transform only object dtype column others will remain unchanged.
     
    Attributes
    ----------       
    encode_mapper : dict
        categories mapper for each column, 
        dict egg. {cloname : {category1 : 0, category2 : 1, ...}}
    out_labels : list
        column labels of output matrix
    '''
    def __init__(self, categories='auto', dtype=np.float64):
        '''
        '''
        L = locals().copy()
        L.pop('self')
        self.set_params(**L)

    def _check_categories(self, X):
        '''check if feature category are out of categories scope, treat them as 
        null
        '''
        if len(self.encode_mapper) == 0: return X
        #
        mapper = self.encode_mapper
        isin_cat = X.apply(lambda x: x.isin(mapper.get(x.name, x)) | x.isna(),
                           axis=0)
        out_c = np.ravel(~isin_cat).sum()
        if out_c > 0:
            logger = init_log()
            logger.warning('''total of {} element out of categories and 
                  will be treated as np.nan '''.format(out_c))
        X = X.where(isin_cat, np.nan)
        return X

    def fit(self, X, y=None):
        '''fit X using ordinal encoder 
        '''
        X = self._fit(X)
        self.obj_cols = X.select_dtypes('object').columns
        self.not_obj = X.columns.difference(self.obj_cols)

        self.encode_mapper = {}
        for name, col in X.iteritems():
            categories_ = col.unique()
            mapper = dict(zip(categories_, range(len(categories_))))
            mapper.update({np.nan: -1})
            self.encode_mapper[name] = mapper

        self.out_labels = self.obj_cols.tolist() + self.not_obj.tolist()

        return self

    def transform(self, X):
        '''transform  X  to oridinal encoded
        
        transform only object dtype column others will remain unchanged.
        '''
        X = self._filter_labels(X)
        # --obj cols
        X0 = X.reindex(columns=self.obj_cols)
        X0 = self._check_categories(X0)

        X0 = X0.apply(lambda col: col.map(self.encode_mapper[col.name]),
                      axis=0)
        # --not obj do nothing
        X1 = X.reindex(columns=self.not_obj)
        rst = pd.concat((i for i in [X0, X1] if not i.empty), axis=1)
        rst = rst.reindex(columns=self.out_labels)
        return rst
