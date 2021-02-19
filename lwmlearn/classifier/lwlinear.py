# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:18:11 2021

@author: roger luo
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.linear_model import LogisticRegression

class __LWLinear(BaseEstimator, ClassifierMixin):
    """
    
    
    """
    
    def __init__(self, encoder_map, weights):
        """
        

        Parameters
        ----------
        encoder_map : dict
            {"x1" : {Interval1 : val2, Interval2 : val2 }, 
             "x2" : {Interval1 : val2, Interval2 : val2 }, 
            }
            
        weights : dict
            {"x1" : w1, "x2" : w2}.

        Returns
        -------
        None.

        """
        
        """
        

        Returns
        -------
        None.

        """
        
        self.encoder_map = encoder_map
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
        # encode data
        data = X.apply(lambda x : x.map(self.encoder_map[x.name]), axis=0)
        
        # calculate decision boundary
        weights = pd.Series(self.weights)
        data = data.reindex(columns=weights)
        return np.matmul(data, weights)
    

if __name__ == "__main__":
    
    from lwmlearn import  LW_model, get_local_data, pipe_gen
    
    
    data = get_local_data("loan_short.csv")
    y = data.pop("y")
    X = data     
    
    m = LW_model("cleanNA_woe5_LogisticRegression")
    m.fit(X, y)
    coef = m.estimator.named_steps.LogisticRegression.coef_
    
    encoder_map = m.estimator.named_steps.woe5.encode_map
    


    
    
    
