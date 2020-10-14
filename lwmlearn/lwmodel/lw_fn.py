# -*- coding: utf-8 -*-
"""
``lw_fn`` module

offers high level functions over `lw_model_proxy`

Created on Mon Apr 13 13:00:23 2020

@author: Rogerluo
"""
import os
import pandas as pd

from lwmlearn.lwmodel.auto_list import get_default_models
from lwmlearn.lwmodel.lw_model_proxy import LW_model


def run_CVscores(
    X=None,
    y=None,
    cv=3,
    scoring=['roc_auc', 'KS', 'neg_log_loss'],
    estimator_lis=None,
):
    """return CV scores of a series of pre-defined piplines as returned by
    :func:`.get_default_models`
    

    Parameters
    ----------
    X : TYPE, optional
        DESCRIPTION. The default is None.
    y : TYPE, optional
        DESCRIPTION. The default is None.
    cv : TYPE, optional
        DESCRIPTION. The default is 3.
    scoring : TYPE, optional
        DESCRIPTION. The default is ['roc_auc', 'KS'].
    estimator_lis : TYPE, optional
        DESCRIPTION. The default is None, use get_default_models().

    Returns
    -------
    df : dataframe 
        cv scores of each pipeline

    """

    if estimator_lis is None:
        l = get_default_models()
    else:
        l = estimator_lis

    if X is None:
        return set(l)
    else:
        lis = []
        for i in l:
            m = LW_model(estimator=i)
            scores = m.cv_validate(X, y, cv=cv, scoring=scoring).mean()
            scores['pipe'] = i
            lis.append(scores)
        m.delete_model()
        df = pd.concat(lis, axis=1, ignore_index=True).T
        return df
