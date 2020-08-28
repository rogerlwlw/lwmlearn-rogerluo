# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:57:29 2020

@author: rogerluo
"""
from lwmlearn.lwmodel.lw_model_proxy import LW_model
from lwmlearn.dataset import get_local_data
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import os


def get_data(dataset, test_size=0.3, sample=None):
    '''
    '''
    if dataset == 'make_classification':
        n_sample = 5000 if sample is None else sample
        x, y = make_classification(n_sample, n_redundant=1, n_features=30)
    else:
        if sample is None:
            data = get_local_data(dataset)
        else:
            data = get_local_data(dataset).sample(sample)
        y = data.pop('y')
        x = data
    return train_test_split(x, y, test_size=test_size)


def test_dataset(dataset, sample=None, test_size=0.3, **kwargs):
    '''
    '''
    x_train, x_test, y_train, y_test = get_data(dataset,
                                                test_size=test_size,
                                                sample=sample)
    path, file = os.path.splitext(dataset)
    m = LW_model(path=path)
    m.run_autoML(x_train, y_train, (x_test, y_test), **kwargs)

    m.plot_all_auc(False, X=x_test, y=y_test, save_fig=True)
    return m, x_train, x_test, y_train, y_test
