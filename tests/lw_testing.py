# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:10:59 2019

@author: rogerluo
"""
from sklearn.feature_extraction import FeatureHasher
from lwmlearn.utilis.testing import runlocaldataset

if __name__ == '__main__':

        m = runlocaldataset('adult_data.csv',
                         sample=5000,
                         out_searchstep=False,
                         is_search=True,
                         kind='bayesiancv',
                         model_list=[
        'cleanNA_woe8_fxgb_TomekLinks_XGBClassifier',
        'cleanNA_woe8_fxgb_inlierLocal_NeighbourhoodCleaningRule_SGDClassifier',
                         ],
                     )
    
