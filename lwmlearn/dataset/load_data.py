# -*- coding: utf-8 -*-
"""use get_local_data function to retrieve local data
Module Descripiton:

    The :mod:`load_data` module offers function :func:`get_local_data` 
    to load data files as dataframe. Supported formats are ['.csv', '.xlsx', '.pkl']. 
    
    By dault a DataFrame object will be returned. The DataFrame should have a 
    column labled as 'y' as the target class, other columns are features matrix.
    
    local demo data matrix can be dumpped to dataset.data folder to be loaded

Created on Tue Dec 10 17:42:49 2019

@author: roger luo
"""

import inspect
import os

from ..utilis.read_write import Objs_management
from sklearn.datasets import make_classification

file_path = os.path.split(inspect.currentframe().f_code.co_filename)[0]
data_path = os.path.join(file_path, 'data')


def get_local_data(data_name=None, all_datafile=False, rel_path=None):
    '''read data file by filename.csv 
    
    from dataset.data folder, extension suffix included
    
    Parameters
    ----------
    data_name : str
        file name of data to be loaded, by default stored as .csv suffix must 
        be included.
        if "make_classification" use synthetic data (5000, 30), return X, y
    
    all_datafile : bool
        if all_datafile=True then, load all file in `data` folder
                
    rel_path : path
        subfoler of 'data' folder to search from, default None will search all
        folders recursively
        
    Return 
    ---------
    list : dict
        if data_name=None,  dict of {filename: obj} that could  be loaded
    dataframe : DataFrame
        if data_name is not None, load that data with `data_name`
        
    '''
    
    if data_name == 'make_classification':
        x, y = make_classification(5000,
                                   n_redundant=10,
                                   n_features=30,
                                   flip_y=0.1)    
        return x, y
        
    reader = Objs_management(data_path)
    if all_datafile is True:
        d_df = reader.read_all(path=rel_path,
                               suffix=['.csv', '.xlsx', '.pkl'],
                               subfolder=True)
        return d_df

    available_files = reader.list_all(path=rel_path,
                                      suffix=['.csv', '.xlsx', '.pkl'],
                                      subfolder=True)
    if data_name is None:
        return available_files.keys()
    else:
        file_name = available_files.get(data_name)
        if file_name is None:
            raise ValueError(
                "data file '{}' not found \n available files: {}".format(
                    data_name, available_files.keys()))
        return reader.read(file_name)


def _remove_label(df):
    ''' remove column labels of dataframe to make data uninterpretable
    also encode categories as C + 'integer'
    
    target classification label must be 'y'
    '''
    import pandas as pd
    import numpy as np
    from pandas.core.dtypes import api
    from lw_mlearn.utilis.utilis import to_num_datetime_df

    def _mapping_col(col, na_values=['null', '缺失值', -999, -99999, -1]):
        ''' encrypt categorical features
        '''
        col = col.replace(na_values, np.nan)
        if not api.is_numeric_dtype(col):
            uniq = col.unique()
            mapper = dict(
                zip(uniq, [''.join(['C', str(i)]) for i in range(len(uniq))]))

            if mapper.get(np.nan) is not None:
                mapper.pop(np.nan)

            col = col.map(mapper, na_action='ignore')

        return col

    df = to_num_datetime_df(df)
    y = df.pop('y')

    # encrypt categorical features
    X = df.apply(_mapping_col, axis=0)
    # remove column labels
    X = pd.DataFrame(X.values)

    X['y'] = y
    return X


if __name__ == '__main__':
    pass
