# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:10:17 2020

@author: roger luo
"""

from sklearn.base import BaseEstimator, TransformerMixin
from xgboost.sklearn import XGBClassifier
from lwmlearn.preprocess.lw_base import LW_Base


class XGB_encoder(BaseEstimator, TransformerMixin, LW_Base):
    '''
    
    attributes
    ----
    
    xgbmodels:
        xgb model to encode each column, {col : xgb_model}
        
    '''
    def __init__(self, pos_label=1):
        '''
        '''

        self.pos_label = pos_label

        return

    def fit(self, X, y):
        '''
        X:
            feature matrix
        y:
            target label, binary
        '''
        X = self._fit(X)
        # --
        self.xgbmodel = {}

        for name, col in X.iteritems():
            xgb = XGBClassifier()
            xgb.fit(col.to_frame(), y)
            self.xgbmodel.update({name: xgb})

        return self

    def transform(self, X):
        '''
        X:
            feature matrix
        '''

        X = self._filter_labels(X)
        # --
        xgb_mapper = self.xgbmodel
        X = X.apply(lambda x: xgb_mapper[x.name].predict_proba(x.to_frame())
                    [:, self.pos_label],
                    axis=0)
        return X
