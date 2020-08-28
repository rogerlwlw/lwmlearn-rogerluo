# -*- coding: utf-8 -*-
"""
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
from lwmlearn.utilis.binning import _binning


class WoeEncoder(BaseEstimator, TransformerMixin, LW_Base):
    '''to woe encode feature matrix using auto-binning based on CART tree
    gini impurity/bins or specified by input_edges = {col : edges};
    transform data matrix to woe encoded matrix;
    calcualte woe & iv of each feature, NaN values will be binned independently
    
    parameters
    ------            
    input_edges={}
        - mannual input cutting edges as 
        {colname : [-inf, point1, point2..., inf]}
    cat_num_lim
        - number of unique value limit to be treated as continueous feature,
        default 5
        
    bins
        - number of equal width or array of edges
    q
        - number of equal frequency              
    max_leaf_nodes
        - number of tree nodes using tree cut
        - if not None use supervised cutting based on decision tree
    mono 
        - binning edges that increases monotonically with "y" mean value
        
    .. note::
        -  only 1 of (q, bins, max_leaf_nodes, mono) can be specified 
        
        
    max_leaf_nodes
        - max number of bins default None
    min_samples_leaf=0.02
        - minimum number of samples in leaf node as fraction
    min_samples_split=0.01
        - the minimum number of samles required to split a node   
    min_samples_leaf_num
        - minimum number of samples in leaf node as numbers
    **tree_params
        - other decision tree keywords
        
    attributes
    -----
    edges 
        - dict={colname : [-inf, point1, point2..., inf]}; 
        - 'fit' method will try to get edges by decision Tree algorithm or
        pandas cut method
    woe_map
        - dict={colname : {category : woe, ...}}
    woe_iv
        - df, woe & iv of all features, concatenated in one df
    feature_importances_ 
        - iv value of each feature, NA for iv <0.02 to be used in feature
        selection
    feature_iv 
        - iv value of each feature (available of iv < 0.02)
      
    method
    -----
    fit 
        - calculate woe & iv values for each col categories, obtain 
        self edges & woe_map
    transform
        - to get woe encoded feature matrix using self woe_map

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
                 min_samples_leaf_num=100,
                 verbose=1):

        L = locals().copy()
        L.pop('self')
        self.set_params(**L)

    def _get_binned(self, X, labels=None):
        '''to get binned matrix using self edges, cols without cutting edges
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
        '''fit X(based on CART Tree) to get cutting edges
        (updated by manual input edges) and  calculate woe & iv for each 
        categorical group, categorical features will use category as group
        
        parameter
        ---------
        X - df
         
        y - class label 
        '''
        X = self._fit(X)
        # --
        params = get_kwargs(_get_binning, **self.get_params())
        params.update(get_kwargs(DecisionTreeClassifier, **self.get_params()))
        # -- check minimum leaf node samples
        min_samples_leaf = self._min_samples_leaf_check(X)
        params.update(min_samples_leaf=min_samples_leaf)
        self.edges = _get_binning(X, y, **params)
        self.edges.update(self.get_params()['input_edges'])
        # --
        df_binned = self._get_binned(X)
        self.woe_iv, self.woe_map, self.feature_iv = calc_woe(df_binned, y)
        return self

    def transform(self, X):
        '''to get woe encoded X using self woe_map
        parameters
        ----------
        X - df
        
        return
        ------
        df --> X woe encoded value
        '''
        X = self._filter_labels(X)
        # --
        woe_map = self.woe_map
        cols = []
        cols_notcoded = []
        for name, col in X.iteritems():
            if name in woe_map:
                mapper = woe_map.get(name)
                if mapper.get(np.nan) is not None:
                    na = mapper.pop(np.nan)
                    cols.append(col.map(mapper).fillna(na))
                else:
                    cols.append(col.map(mapper).fillna(0))
            else:
                cols_notcoded.append(col.name)

        if cols_notcoded:
            print("{} have not been woe encoded".format(cols_notcoded))

        return pd.concat(cols, axis=1)

    def plot_event_rate(self, save_path=None, suffix='.pdf', dw=0.02, up=1.2):
        '''return iv of each column using self.edges
        '''
        plotter_woeiv_event(self.woe_iv, save_path, suffix, dw, up)


class BinEncoder(WoeEncoder):
    '''
    parameters
    ------   

    int_bins bool:
        if true, return integer label for bins
        if false, return [low, upper] edges for bins
        default='False' 
        
    input_edges={}
        - mannual input cutting edges as 
        {colname : [-inf, point1, point2..., inf]}
        
    cat_num_lim
        - number of unique value limit to be treated as continueous feature,
        default 5

    bins
        - number of equal width or array of edges
    q
        - number of equal frequency              
    max_leaf_nodes
        - number of tree nodes using tree cut
        - if not None use supervised cutting based on decision tree
    mono 
        - binning edges that increases monotonically with "y" mean value
        
    .. note::
        -  only 1 of (q, bins, max_leaf_nodes, mono) can be specified 
        
        
    max_leaf_nodes
        - max number of bins default None
    min_samples_leaf=0.02
        - minimum number of samples in leaf node as fraction
    min_samples_split=0.01
        - the minimum number of samles required to split a node   
    min_samples_leaf_num
        - minimum number of samples in leaf node as numbers
    **tree_params
        - other decision tree keywords 
    '''
    def __init__(
            self,
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
            min_samples_leaf_num=100,
            verbose=1,
            int_bins=False,
    ):
        '''
        '''
        L = locals().copy()
        L.pop('self')
        self.set_params(**L)

    def fit(self, X, y):
        '''fit X(based on CART Tree) to get cutting edges
        (updated by manual input edges)
        
        parameter
        ----
        X:
            df
         
        y:
            class label 
        '''
        X = self._fit(X)
        # --
        params = get_kwargs(_get_binning, **self.get_params())
        params.update(get_kwargs(DecisionTreeClassifier, **self.get_params()))
        # -- check minimum leaf node samples
        min_samples_leaf = self._min_samples_leaf_check(X)
        params.update(min_samples_leaf=min_samples_leaf)
        self.edges = _get_binning(X, y, **params)
        self.edges.update(self.get_params()['input_edges'])
        return self

    def transform(self, X):
        """
        return binned dataframe

        Parameters
        ----------
        X : dataframe
            binned

        Returns
        -------
        binned dataframe
            
        """
        X = self._filter_labels(X)
        labels = None if not self.int_bins else False
        return self._get_binned(X, labels)


def _get_binning(X,
                 y,
                 q=None,
                 bins=None,
                 max_leaf_nodes=None,
                 mono=None,
                 cat_num_lim=0,
                 **kwargs):
    '''use by Woe_encoder to get binning edges
    
    X - DataFrame
    y - binary target
    return
    ----
    edges:
        {colname : [-inf, point1, point2..., inf]}
    '''
    bin_edges = {}
    for name, col in X.iteritems():
        df = pd.DataFrame({'x': col, 'y': y})
        col_notna = df.dropna().x
        y_notna = df.dropna().y
        if (len(pd.unique(col_notna)) > cat_num_lim \
            and api.is_numeric_dtype(col_notna)):
            label, bin_edges[name] = _binning(col_notna, bins, q,
                                              max_leaf_nodes, mono, y_notna,
                                              **kwargs)
    return bin_edges


def _single_mapping(X, Y, var_name='VAR'):
    '''calculate woe and iv for single binned X feature, with binary Y target
    (use 0.01 instead when event or nonevent count equals 0)    
    - y=1 event; y=0 non_event 
    - na value in X will be grouped independently
    
    return
    ----
    df, of WOE, IVI and IV_SUM ...
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
    '''calculate woe and iv 
    
    df_binned
        - binned feature_matrix
    y
        - binary 'y' target   
    
    return
    ----
    df_woe_iv =  [
            'VAR_NAME','CATEGORY', 'COUNT', 'EVENT', 'EVENT_RATE',
            'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT',
            'WOE', 'IV' ]
    
    woe_map = {'colname' : {category : woe}}
    
    iv series
        - colname--> iv 
    '''

    l = []
    woe_map = {}
    iv = []
    var_names = []
    for name, col in df_binned.iteritems():
        col_iv = _single_mapping(col, y, name)
        l.append(col_iv)
        woe_map[name] = dict(col_iv[['CATEGORY', 'WOE']].values)
        iv.append(col_iv.IV.sum())
        var_names.append(name)

    # concatenate col_iv
    woe_iv = pd.concat(l, axis=0, ignore_index=True)
    print('---' * 20)
    print('total of {} cols get woe & iv'.format(len(l)))
    print('---' * 20, '\n\n')
    return woe_iv, woe_map, pd.Series(iv, var_names)


def plotter_woeiv_event(woe_iv, save_path=None, suffix='.pdf', dw=0.05,
                        up=1.0):
    '''plot event rate for given woe_iv Dataframe
    see woe_encoder attribute woe_iv
    '''
    n = 0
    for keys, gb in woe_iv.groupby('FEATURE_NAME'):
        if (gb['IV'].sum() > dw) and (gb['IV'].sum() < up):
            plot_data = gb[['CATEGORY', 'EVENT_RATE', 'COUNT']]
            plot_data.columns = [keys, 'EVENT_RATE', 'COUNT']
            plotter_rateVol(plot_data.sort_values(keys))
            if save_path is not None:
                path = '/'.join([save_path, keys + suffix])
                plt.savefig(path, dpi=100, frameon=True)
            n += 1
            print('(%s)-->\n' % n)
            plt.show()
            plt.close()

    return


if __name__ == '__main__':
    from lwmlearn.dataset.load_data import get_local_data
    data = get_local_data('givemesomecredit.csv')
    woe = WoeEncoder(mono=True)
    y = data.pop('y')
    x = data
    data_f = woe.fit_transform(x, y)
    #    woe.plot_event_rate(up=2)
    bb = BinEncoder(max_leaf_nodes=5)
