# -*- coding: utf-8 -*-
"""WOE and IV

use example
------------

the following code transform a 1000 x 4 feature matrix into woe encoded based on 
tree cutting strategy

.. ipython::
    :okwarning:
    
    In [2]: from lwmlearn.preprocess.lw_woe_encoder import WoeEncoder

    In [3]: from sklearn.datasets import  make_classification
    
    In [4]: woe = WoeEncoder(max_leaf_nodes=5)

    In [5]: woe
    
    In [8]: X, y = make_classification(1000, 4)
    
    In [11]: woe.fit_transform(X, y)
    
Created on Sun Dec 15 16:21:27 2019

@author: roger luo

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.core.dtypes import api
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier

from lwmlearn.preprocess.lw_base import LW_Base
from lwmlearn.utilis.utilis import get_kwargs
from lwmlearn.viz.plotter import plotter_rateVol
from lwmlearn.utilis.binning import binning
from lwmlearn.lwlogging import init_log

logger = init_log()

class WoeEncoder(BaseEstimator, TransformerMixin, LW_Base):
    '''woe encode feature matrix using auto-binning strategy
    
    #. calcualte woe & iv of each feature
    
    #. NaN values will be binned independently.
    
    #. By default binning edges are based on CART tree gini impurity
    or could be specified by input_edges = {col : edges}.
    
    #. For discontinuous columns will have no binning edges, each of the 
    category will be used as bins
    
    
    Parameters
    ----------  
    input_edges : dict
        mannual input cutting edges as 
        {colname : [-inf, point1, point2, .., inf]}
    cat_num_lim : int
        number of unique value limit to be treated as continueous feature,
        default 5
    bins : int
        number of equal width  edges, if not None, use equal width cutting
    q : 
        number of equal frequency edges, if not None, use equal frequency         
    max_leaf_nodes : int
        number of tree nodes using tree cut,
        if not None use supervised cutting based on decision tree
    mono : int
        binning edges that increases monotonically with "y" mean 
        
        .. note::
            only 1 of ( q, bins, max_leaf_nodes, mono ) can be specified, 
            if not valid assign q=10 and bins=max_leaf_nodes=mono=None
    
    Keyword args
    --------------    
    min_samples_leaf :
        default = 0.02
    min_samples_split :
        default = 0.05
    criterion :
        default = 'gini'
    min_impurity_split : TYPE, optional
        DESCRIPTION. The default is None.
    random_state : TYPE, optional
        DESCRIPTION. The default is 0.
    splitter : TYPE, optional
        DESCRIPTION. The default is 'best'.
    min_samples_leaf_num : int, optional
        compare sample number resulting from min_samples_leaf, 
        use the bigger one. The default is 10.
    verbose : TYPE, optional
        DESCRIPTION. The default is 1.
    
    Attributes
    -----------
    edges : dict  
        like {colname : [-inf, point1, point2..., inf]}; 
        fit method will try to get edges by decision Tree algorithm or
        pandas cut method, if not specified
    encode_map : dict
        woe mapping, like {colname : {category : woe, ...}}
    woe_iv : dataframe
        woe & iv table of all features, concatenated in one df
    feature_importances_ : series
        iv value of each feature, NA for iv <0.02 to be used in feature
        selection
    feature_iv : series 
        iv value of each feature (available of iv < 0.02)
      
    '''
    def __init__(self,
                 input_edges={},
                 cat_num_lim=10,
                 q=None,
                 bins=None,
                 mono=None,
                 max_leaf_nodes=None,
                 min_samples_leaf=0.02,
                 min_samples_split=0.05,
                 criterion='gini',
                 min_impurity_decrease=1e-5,
                 min_impurity_split=None,
                 random_state=0,
                 splitter='best',
                 min_samples_leaf_num=10,
                 verbose=1):
        '''
        '''
        L = locals().copy()
        L.pop('self')
        
        for k,v in L.items():
            setattr(self, k, v)
            
        self.params = L

    def _get_binned(self, X, labels=None):
        '''get binned matrix using self edges, cols without cutting edges
        will remain unchaged
        '''
        if self.edges is None:
            raise Exception('no bin edges, perform fit first')
        cols = []
        # return integer or labels
        for name, col in X.iteritems():
            if name in self.edges:
                edges = self.edges.get(name)
                col_binned = pd.cut(col,
                                    edges,
                                    retbins=False,
                                    duplicates='drop',
                                    labels=labels)
                cols.append(col_binned)
            else:
                cols.append(col)
        return pd.concat(cols, axis=1)

    def _min_samples_leaf_check(self, X):
        '''compare min samples leaf specified by the two __init__ parameters,
        use the bigger one
        '''
        n = len(X)
        if self.min_samples_leaf > 1:
            a = int(self.min_samples_leaf)
        else:
            a = int(n * self.min_samples_leaf)

        b = self.min_samples_leaf_num

        return a if a > b else b

    @property
    def feature_importances_(self):
        '''return iv value of features
        '''
        if hasattr(self, 'feature_iv'):

            print('''IV < 0.02 has been forced to 0 due to
                  low predictive power  value''')
            value = self.feature_iv
            iv = self.feature_iv.where(0.02 < value)
            return iv
        else:
            return

    def fit(self, X, y):
        '''fit X to get cutting edges
        
        the cuttign algorithm use CART Tree to cut each feature, or use 
        monotonical increasing method if mono is not None;
        
        cutting edges could be updated by manual input edges 
        categorical features will use category as group
        
        self.edges and self.encode_map will be updated
        
        Parameters
        -----------
        X : 2d array
            feature matrix
         
        y : 1d array 
            class label 
        '''
        X = self._fit(X)
        # --
        params = get_kwargs(_get_binning, **self.params)
        params.update(get_kwargs(DecisionTreeClassifier, **self.params))
        # -- check minimum leaf node samples
        min_samples_leaf = self._min_samples_leaf_check(X)
        params.update(min_samples_leaf=min_samples_leaf)
        self.edges = _get_binning(X, y, **params)
        self.edges.update(self.input_edges)
        # --
        df_binned = self._get_binned(X)
        self.woe_iv, self.encode_map, self.feature_iv = calc_woe(df_binned, y)
        return self

    def transform(self, X):
        '''get woe encoded X using self encode_map
        
        Parameters
        ----------
        X : 2d array
        
            feature matrix
        
        Return
        ------
        x_transformed : dataframe
        
            woe encoded dataframe table
        '''
        X = self._filter_labels(X)
        # --
        encode_map = self.encode_map.copy()
        cols = []
        cols_notcoded = []
        for name, col in X.iteritems():
            if name in encode_map:
                mapper = encode_map.get(name)
                
                #-- use mapper directly
                cols.append(col.map(mapper))
                
                # # -- np.nan cannot be indexed as dict's keys
                # if mapper.get(np.nan) is not None:
                #     na = mapper.get(np.nan)
                #     mapper = {k : v for k, v in mapper.items() 
                #               if k not in {np.nan}}
                #     cols.append(col.map(mapper).fillna(na))
                # else:
                #     cols.append(col.map(mapper).fillna(0))
                    
            else:
                cols_notcoded.append(col.name)

        if cols_notcoded:

            logger.warning(
                "{} have not been woe encoded".format(cols_notcoded))

        return pd.concat(cols, axis=1)

    def plot_event_rate(self, save_path=None, suffix='.png', dw=0.02, up=1.2):
        """plot correlations between cutting edges and positive proportion of 
        samples for each feature
        
        Parameters
        -----------
        save_path : path
            path to save plots of all 
        suffix : str
            suffix of plot file to use
        dw : 
            lower bound of iv value
        up :
            upper bound of iv value
        
        Return
        ---------
        None
            
        """

        import os

        woe_iv = self.woe_iv
        n = 0
        for keys, gb in woe_iv.groupby('FEATURE_NAME'):
            if (gb['IV'].sum() > dw) and (gb['IV'].sum() < up):
                plot_data = gb[['CATEGORY', 'EVENT_RATE', 'COUNT']]
                plot_data.columns = [keys, 'EVENT_RATE', 'COUNT']
                plot_data = plot_data.sort_values(keys)
                plot_data[keys] = plot_data[keys].astype(str)
                plotter_rateVol(plot_data)
                if save_path is not None:
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path, exist_ok=True)
                    path = '/'.join([save_path, keys + suffix])
                    plt.savefig(path, dpi=100, frameon=True)
                n += 1
                plt.show()


def _get_binning(X,
                 y,
                 q=None,
                 bins=None,
                 max_leaf_nodes=None,
                 mono=None,
                 cat_num_lim=0,
                 **kwargs):
    '''use by Woe_encoder to get binning edges
    
    Parameters
    -----------
    
    X :
        DataFrame
    y :
        binary target
    
    return
    -------
    edges:
        {colname : [-inf, point1, point2..., inf]}
    '''
    bin_edges = {}
    # reset index
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    for name, col in X.iteritems():
        df = pd.DataFrame({'x': col, 'y': y})
        col_notna = df.dropna().x
        y_notna = df.dropna().y
        if (len(pd.unique(col_notna)) > cat_num_lim \
            and api.is_numeric_dtype(col_notna)):
            label, bin_edges[name] = binning(col_notna, bins, q,
                                             max_leaf_nodes, mono, y_notna,
                                             **kwargs)
    return bin_edges


def _single_mapping(X, Y, var_name='VAR'):
    """calculate woe mapping table for single vaiable X
    
    use 0.01 instead when event or nonevent count equals 0    
    
    y=1 event; y=0 non_event 
    
    np.nan value in X will be grouped independently
    
    Parameters
    ----------
    X : 1d array
        sequence to be grouped and calculate woe and IV.
    Y : 1d array
        supervising target class, Y==1 indicates positive 
        while Y==0 indicates negative.
    var_name : TYPE, optional
        name label of X. The default is 'VAR'.

    Returns
    -------
    d3 : dataframe
        woe mapping table of X .

    """
    '''calculate woe and iv for single binned X feature, with binary Y target

    '''
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isna()]
    notmiss = df1[['X', 'Y']][df1.X.notna()]
    df2 = notmiss.groupby('X', as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["COUNT"] = df2.count().Y
    d3["CATEGORY"] = df2.sum().index
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({
            "COUNT": [justmiss.count().Y],
            "EVENT": [justmiss.sum().Y],
            "NONEVENT": [justmiss.count().Y - justmiss.sum().Y]
        })
        d3 = pd.concat([d3, d4], axis=0, ignore_index=True, sort=True)

    # add 0.01 when event or nonevent count equals 0
    dc = d3.copy()
    dc.EVENT.replace(0, 0.01, True)
    dc.NONEVENT.replace(0, 0.01, True)
    dc["EVENT_RATE"] = dc.EVENT / dc.COUNT
    dc["NON_EVENT_RATE"] = dc.NONEVENT / dc.COUNT
    dc["DIST_EVENT"] = dc.EVENT / dc.EVENT.sum()
    dc["DIST_NON_EVENT"] = dc.NONEVENT / dc.NONEVENT.sum()
    # add 0.01 when event or nonevent count equals 0

    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.EVENT.sum()
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.NONEVENT.sum()

    # calculate woe values for binned categories
    d3["WOE"] = np.log(dc.DIST_EVENT / dc.DIST_NON_EVENT)
    d3["IV"] = (dc.DIST_EVENT - dc.DIST_NON_EVENT) * np.log(
        dc.DIST_EVENT / dc.DIST_NON_EVENT)

    d3["FEATURE_NAME"] = var_name
    d3 = d3[[
        'FEATURE_NAME', 'CATEGORY', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT',
        'NON_EVENT_RATE', 'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV'
    ]]

    d3['IV_SUM'] = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    return d3


def calc_woe(df_binned, y):
    '''calculate woe and iv used by WoeEncoder internally

    use 0.01 instead when event or nonevent count equals 0    
    
    y=1 event; y=0 non_event 
    
    np.nan value in X will be grouped independently
    
    parameters
    -----------
    df_binned : dataframe
        binned feature_matrix, binned feature should be categorial 
    y : 1d array
        binary 'y' target   
    
    return
    ---------
    df_woe_iv : dataframe
        column names are [ 
        'VAR_NAME', 'CATEGORY', 'COUNT', 'EVENT', 'EVENT_RATE',
        'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT',
        'WOE', 'IV' ]
        
    encode_map : dict
        {'colname' : {category : woe}}
    iv : series
        colname index iv value 
    
    '''

    l = []
    encode_map = {}
    iv = []
    var_names = []
    for name, col in df_binned.iteritems():
        col_iv = _single_mapping(col, y, name)
        l.append(col_iv)
        encode_map[name] = dict(col_iv[['CATEGORY', 'WOE']].values)
        iv.append(col_iv.IV.sum())
        var_names.append(name)
    # print logging info
    logger.info('total of {} cols get woe & iv'.format(len(l)))
    # concatenate col_iv
    woe_iv = pd.concat(l, axis=0, ignore_index=True)

    return woe_iv, encode_map, pd.Series(iv, var_names)


if __name__ == '__main__':
    from lwmlearn.dataset.load_data import get_local_data
    data = get_local_data('givemesomecredit.csv')
    woe = WoeEncoder()
    y = data.pop('y')
    x = data
    data_f = woe.fit_transform(x, y)
    print(woe.woe_iv)
