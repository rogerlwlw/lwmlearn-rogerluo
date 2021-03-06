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


def _get_data(dataset, test_size=0.3, sample=None):
    '''
    '''
    if dataset == 'make_classification':
        n_sample = 5000 if sample is None else sample
        x, y = make_classification(n_sample,
                                   n_redundant=5,
                                   n_features=30,
                                   flip_y=0.1)
    else:
        if sample is None:
            data = get_local_data(dataset)
        else:
            data = get_local_data(dataset).sample(sample)
        y = data.pop('y')
        x = data
    return train_test_split(x, y, test_size=test_size)


def runlocaldataset(dataset, sample=None, test_size=0.3, dirs=None, **kwargs):
    """test :meth:`~.LW_model.run_autoML` method on given dataset
    

    Parameters
    ----------
    dataset : str
        options ['make_classification', 'givemesomecredit.csv', ..], data file
        names in dataset folder
    sample : int, optional
        sample given number of records from data. The default is None.
    test_size : float, optional
        fraction to use as testset. The default is 0.3.
    kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    m : instance
        :class:`LW_model` instance.
    x_train : TYPE
        DESCRIPTION.
    x_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.

    """

    x_train, x_test, y_train, y_test = _get_data(dataset,
                                                 test_size=test_size,
                                                 sample=sample)
    path, file = os.path.splitext(dataset)
    if dirs is not None:
        path = os.path.join(dirs, path)

    m = LW_model(path=path)
    m.run_autoML(x_train, y_train, (x_test, y_test), **kwargs)

    m.plot_all_auc(False, X=x_test, y=y_test, save_fig=True)
    return m, x_train, x_test, y_train, y_test
