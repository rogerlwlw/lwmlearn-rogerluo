# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:23:58 2020

@author: rogerluo
"""

import pandas as pd
import numpy as np

from lwmlearn.lwmodel.operators_pool import get_lw_estimators

from pandas.core.dtypes.api import is_numeric_dtype


def remove_feature_outlier(data, outlier_detector, contamination=0.1):
    '''remove outlier (replace by np.nan) for each feature in data
    
    Parameters
    ----------
    data : DataFrame
    
    outlier_detector : 
        - if True, use default outlier detector 'IsolationForest'
        - if one of option in ['LocalOutlierFactor', 'IsolationForest', 
                               'EllipticEnvelope', 'OneClassSVM'], 
        use corresponding outlier_detector
        
        - if dict, {'colname' : outlier_detector, ...}, remove outlier for 
        corresponding column in data
        
    contamination : float, [0, 0.5]
        DESCRIPTION. The default is 0.1.

    Returns
    -------
    data frame

    '''

    msg =\
    '''
    wrong input '{}' for outlier_detector, possible vaules are dict, str, bool
    '''

    colname = data.columns.tolist()
    default_detector = 'IsolationForest'
    if outlier_detector is True:
        outlier_detector = {
            col: get_lw_estimators(default_detector)
            for col in colname
        }
    elif isinstance(outlier_detector, dict):
        outlier_detector = {
            k: default_detector if v is True else v
            for k, v in outlier_detector.items()
        }

        outlier_detector = {
            k: get_lw_estimators(v)
            for k, v in outlier_detector.items()
        }
    elif isinstance(outlier_detector, str):
        outlier_detector = {
            col: get_lw_estimators(outlier_detector)
            for col in colname
        }
    else:
        raise ValueError(msg.format(outlier_detector))

    removed = []
    for k, col in data.iteritems():
        if outlier_detector.get(k) is not None:
            detector = outlier_detector.get(k)
            detector.set_params(contamination=contamination)
            inlier = detector.fit_predict(pd.DataFrame(col)) == 1
            removed.append(col.where(inlier, np.nan))
        else:
            removed.append(col)

    return pd.concat(removed, axis=1)


def scale_data(data, scaler, **kwargs):
    '''use a scaler instance to scale numeric column data

    Parameters
    ----------
    data : dataframe
    
    scaler: str 
        name of scaler class
        ['RobustScaler', 'StandardScaler','MaxAbsScaler', 'MinMaxScaler']
    
    **kwargs
    -------
    key words argument pass to scaler
    
    Returns
    -------
    scaled data

    '''
    # numeric data columns
    data_num = data.select_dtypes(exclude=['object'])
    # not numeric data columns
    data_not_num = data.select_dtypes(include=['object'])
    if scaler:
        scaler = 'RobustScaler' if scaler is True else scaler
        tr = get_lw_estimators(scaler)
        tr.set_params(**kwargs)
        data_num = pd.DataFrame(tr.fit_transform(data_num),
                                columns=data_num.columns)

    return pd.concat([data_num, data_not_num], axis=1)


def null_outlier(data, data_range):
    '''
    make data as np.nan that not fall within range 
    for each numemric feature
    
    Parameters
    ---------
    
    data : df
        dataframe data
        
        
    data_range : list of tuple [(low, high, flag, column)]:
        
        - low, lower bound 
        - high, upper bound
        - flag, 'numeric' or 'percentage'
        - column, columns to apply to, default None, apply to all columns
        

    Returns
    -------
    datafrarme

    '''
    def f(ser, kwd):
        '''
        '''
        param = kwd.get(ser.name)
        if param is None or not is_numeric_dtype(ser):
            return ser
        else:
            low, high, flag = param
            if low is not None:
                if flag == 'percentage':
                    low = ser.quantile(low)
                ser = ser.where(low <= ser, np.nan)

            if high is not None:
                if flag == 'percentage':
                    high = ser.quantile(high)
                ser = ser.where(ser <= high, np.nan)
        return ser

    if data_range is not None:
        cols = data.columns
        to_change_cols = {}
        for i in data_range:
            j = i[3]
            if j is None:
                to_change_cols.update({k: i[:3] for k in cols})
            else:
                to_change_cols.update({k: i[:3] for k in j})

        data = data.apply(f, args=(to_change_cols, ), axis=0)

    return data
