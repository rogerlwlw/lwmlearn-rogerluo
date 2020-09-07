# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:30:12 2020

@author: Administrator
"""

import numpy as np

from sklearn.metrics import log_loss, roc_curve
from sklearn.metrics import make_scorer


def ks_score(y_true, y_pred, pos_label=1):
    """return K-S score of y_pred over y_true

    Parameters
    ----------
    y_true : 1d array
        binary label of target class. Usually positive==1
    y_pred : 1d array
        predictions made by model. Usually float dtype
    pos_label : int, optional
        1 or 0. The default is 1.

    Returns
    -------
    ks : float
        K-S score.

    """

    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=pos_label)
    ks = (tpr - fpr).max()
    return ks


def psi_score(act, ex):
    '''return psi score
    
    parameters
    -----------
    act:
        array of actual ratios
    ex:
        array of expected ratios
        
    return
    --------
    psi score
    '''
    act = np.array(act)
    ex = np.array(ex)
    ex = np.where(ex == 0, 0.0001, ex)
    if len(act) != len(ex):
        raise ValueError("length of 'act' and 'ex' must match")
    delta = act - ex
    ln = np.log(act / ex)
    return np.sum(delta * ln)


def get_custom_scorer():
    ''' return custom scorer dict
    
    'KS' scorer added
    
    return
    -------
    
    d : dict
        dict of scorer
    '''
    scorer_dict = {}
    scorer_dict['KS'] = make_scorer(ks_score, needs_threshold=True)
    scorer_dict['neg_log_loss'] = make_scorer(log_loss,
                                              greater_is_better=False,
                                              needs_threshold=True)

    return scorer_dict
