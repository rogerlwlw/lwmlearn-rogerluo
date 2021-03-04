# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:21:25 2021

@author: luowen1
"""
import numpy as np
import pandas as pd



def _psi_value(array2d):
    """
    知识:
        
    公式： psi = sum(（实际占比-预期占比）* ln(实际占比/预期占比))
    一般认为psi小于0.1时候变量稳定性很高，0.1-0.25一般，大于0.25变量稳定性差，建议重做。
    相对于变量分布(EDD)而言, psi是一个宏观指标, 无法解释两个分布不一致的原因。但可以通过观察每个分箱的sub_psi来判断。    

    Parameters
    ----------
    nd1 : percentage array1
    nd2 : percentage array2

    Returns
    -------
    None.

    """
    # drop rows that contain nan values
    data = pd.DataFrame(array2d)
    data = data.dropna()
    
    nd1 = np.array(data.iloc[: , 0])
    nd2 = np.array(data.iloc[: , 1])
    psi = np.nansum((nd1 - nd2) * np.log(nd1 / nd2))
    return psi

def psi_percentage_cal(array, bins):
    """
    

    Parameters
    ----------
    array : ndarray
        arrays to count percentage of interval or category element.
        if array is not numeric, count each unique element of array
    bins : edges of intervals
        like [-1, 0, 1, 2, 3, ...]

    Returns
    -------
    pct : Series
        percentage for each interval or category.

    """
    
    
    if pd.core.dtypes.api.is_numeric_dtype(np.array(array)):
        array = pd.cut(array, bins=bins)
    
    counts = pd.value_counts(array, dropna=False)
    pct = counts / counts.sum()
    
    return pct

def sample_psi(array, num, n_bins=8, iter_num=5, method=1):
    """
    return psi value of an array by sampling  pairs of arrays and cutting
    arrays with equal width or equal frequency method 

    Parameters
    ----------
    array : ndarray
        distribution of variable
    num : int
        number of samples to withdraw with replace.
    n_bins : int, optional
        number of bins to group arrays. The default is 8.
        arrays of bin edges could be passed if method is not 1
    iter_num : TYPE, optional
        number of iterations or sampled array. The default is 3.
    method : int, optional
        if 0, use equal width. 
        if 1, use equal frequency
        if 2, use bin edgeds, and num must array of edges
        The default is 1.

    Returns
    -------
    psi : float

    """

    psi = []
    for i in range(iter_num):
        sample = np.random.choice(array, size=(num, 2), replace=True)
        psi_val = pair_psi(sample[:, 0], sample[:, 1],
                               n_bins, method)
        psi.append(psi_val)
        
    return np.mean(psi)

def pair_psi(array, base_array, n_bins, method=0):
    """
    return psi value of two arrays from the same distribution

    Parameters
    ----------
    array : ndarray
        to compare to base_array
    base_array : ndarray
        to cut edges from base_array.
    n_bins : int
        number of intervals.
        arrays of bin edges could be passed if method is not 1
    method : int, optional
        if 0, use equal width. 
        if 1, use equal frequency
        if 2, use bin edgeds, and num must array of edges
        The default is 0.

    Returns
    -------
    None.

    """
    if pd.core.dtypes.api.is_numeric_dtype(np.array(array)) \
    and pd.core.dtypes.api.is_numeric_dtype(np.array(base_array)):
        bins = bucket_array(base_array, n_bins, method)
    else:
        bins = None
    array_pct = psi_percentage_cal(array, bins)
    array_pct.name = "array"
    base_array_pct = psi_percentage_cal(base_array, bins)
    base_array_pct.name = "base_array"
    
    df = pd.concat([array_pct, base_array_pct], axis=1)
    
    psi = _psi_value(df)

    return psi
    
def bucket_array(array, num, method=0):
    """
    

    Parameters
    ----------
    array : TYPE
        DESCRIPTION.
    num : int
        number of intervals
    method : int, optional
        if 0, use equal width. 
        if 1, use equal frequency
        if 2, use bin edgeds, and num must array of edges
        The default is 0.

    Returns
    -------
    bins :
        cutting edges of binning

    """
    
    if method == 1:
        array, bins = pd.qcut(array, q=num, retbins=True)
    else:
        array, bins = pd.cut(array, bins=num, retbins=True)
        
    
    bins[-1] = np.inf
    bins[0] = -np.inf
    return  bins

    

if __name__ == "__main__":
    
    from lwmlearn import get_local_data
    
    X, y = get_local_data("make_classification")
    x1 = X[:, 1]
    x2 = X[: ,2]
    
    
