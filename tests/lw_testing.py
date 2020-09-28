# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:10:59 2019

@author: rogerluo
"""
from sklearn.feature_extraction import FeatureHasher
from lwmlearn.utilis.testing import runlocaldataset

from sklearn.datasets import make_classification
from lwmlearn import LW_model, pipe_main

from lwmlearn.dataset import get_local_data

from lightgbm import LGBMClassifier



if __name__ == '__main__':
    pass
    m = runlocaldataset('loan_short.csv',
                      sample=1000,
                      out_searchstep=False,
                      is_search=True,
                      kind='bayesiancv',
                      model_list=[
    'clean_oht_fxgb_LGBMClassifier',
                      ],
                  )
    
