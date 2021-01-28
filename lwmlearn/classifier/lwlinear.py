# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:18:11 2021

@author: roger luo
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.linear_model import LogisticRegression

class __LWLinear(BaseEstimator, ClassifierMixin):
    """
    
    
    """
    
    def __init__(self, mapping, weights):
        """
        

        Parameters
        ----------
        mapping : dict
            {"x1" : {Interval1 : val2, Interval2 : val2 }}.
        weights : dict
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        """
        

        Returns
        -------
        None.

        """
        
        self.mapping = mapping
        self.weights = weights
        
    
    
    def fit(self, X, y):
        """
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        return self
    
    
    def predict(self, X):
        """
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        return
    
    def decision_function(self, X):
        """
        """
        
        return
    

if __name__ == "__main__":
    
    from lwmlearn import  LW_model, get_local_data, pipe_gen
    
    
    data = get_local_data("loan_short.csv")
    y = data.pop("y")
    X = data     
    
    m = LW_model("cleanNA_woe5_LogisticRegression")
    m.fit(X, y)
    coef = m.estimator.named_steps.LogisticRegression.coef_
    
    mapping = m.estimator.named_steps.woe5.woe_map
    
    data0 = X[["2"]]
    from lwmlearn.preprocess.lw_woe_encoder import WoeEncoder
    woe = WoeEncoder(max_leaf_nodes=5)    
    woe.fit(data0, y)    

    p = pipe_gen("cleanNA_woe5_XGBClassifier")    
