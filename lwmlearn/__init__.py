'''
machine learning based on sklearn-estimators and many other python project

@author: rogerluo
'''
__version__ = '0.0.1'

from lwmlearn.lwmodel.operators_pool import pipe_gen, get_lw_estimators
from lwmlearn.lwmodel.lw_model_proxy import LW_model
from lwmlearn.lwmodel.lw_fn import run_CVscores
from lwmlearn.viz.analyze import DataAnalyzer

from lwmlearn.dataset import get_local_data