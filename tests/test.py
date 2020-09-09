# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:10:59 2019

@author: rogerluo
"""

from lwmlearn.utilis.testing import test_dataset
if __name__ == '__main__':

    m = test_dataset('make_classification',
                     out_searchstep=False,
                     is_search=True,
                     kind='bayesiancv',
                     model_list=[
                         'cleanNA_woe8_XGBClassifier',
        # 'cleanNA_woe8_fxgb_SGDClassifier',
        # 'cleanNA_woe8_fxgb_TomekLinks_SGDClassifier',
        # 'cleanNA_woe8_fxgb_inlierLocal_NeighbourhoodCleaningRule_SGDClassifier',
                       
                         ],
                     )
    
    # m = test_dataset('make_classification',
    #                   out_searchstep=True,
    #                   is_search=True,
    #                   kind='gridcv',
    #                   model_list=['XGBClassifier'],
    #                   )

