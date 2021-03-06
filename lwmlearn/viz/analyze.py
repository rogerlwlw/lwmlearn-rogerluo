# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:47:33 2020

@author: roger luo
"""

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from lwmlearn import pipe_gen
from lwmlearn.lwmodel.operators_pool import get_featurenames
from lwmlearn.viz.plotter import (plotter_binlift, _save_fig, plotter_scatter,
                                  plotter_ridge)

from lwmlearn.dataset.load_data import get_local_data
from lwmlearn.preprocess.lw_base import Cleaner
from lwmlearn.utilis.docstring import Appender, dedent

from lwmlearn.viz.utilis import null_outlier, scale_data
from lwmlearn.viz.mlens_plot import (corr_X_y, corrmat, pca_plot,
                                     pca_comp_plot, clustered_corrmap,
                                     exp_var_plot)
import seaborn as sns


def pct_nan(x):
    """
    calculate nan / total number ratio

    Parameters
    ----------
    x : array

    Returns
    -------
    ratio : float

    """
    
    return 1 - pd.Series(x).count() / len(x)

def imp(X, y, estimator="clean_ordi_impXGB"):
    """
    output feature importance ranking of an classifier

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    estimator : TYPE, optional [clean_ordi_impXGB, clean_ordi_impRF]
        DESCRIPTION. The default is "clean_ordi_impXGB".

    Returns
    -------
    imp : TYPE
        DESCRIPTION.

    """

    # estimator pipeline
    pip = pipe_gen(estimator)
    pip.fit(X, y)
    
    imp_label = estimator.split('_')[-1]
    # extract feature importance
    imp = pd.DataFrame(
        {imp_label : pip.steps[-1][1].feature_importances_}, 
        index=get_featurenames(pip)
    )
    imp['imp_rank'] = imp[imp_label].rank(ascending=False)
    return imp

def iv(X, y, estimator="cleanNA_woe5"):
    """
    

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    pip = pipe_gen(estimator)
    pip.fit(X, y)
    
    feature_iv = pd.DataFrame({"iv": pip.steps[-1][-1].feature_iv},
                              index=get_featurenames(pip)
                              )
    feature_iv['iv_rank'] = feature_iv.rank(ascending=False)
    return feature_iv

def corr_clusters(df, n_clusters):
    """
    correlation clusters

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from sklearn.cluster import KMeans
    
    km = KMeans(n_clusters, random_state=0)
    corr = df.corr()
    km.fit(corr)
    
    df = pd.DataFrame({"corr_clusters" : km.labels_}, index=corr.columns)
    
    return df


class DataAnalyzer():
    """
    data analyzer
    
    """
    def __init__(self, data, class_label, encode_featurename=False):
        '''
         
        Parameters
        ----------
        data : 2D array or dataframe
            data matrix to be analyzed.
            
        class_label : str
            name of class label, if no class label, assign class_label=None

        encode_featurename bool:
            if True, encode feature name by 'Fx', where x is index of feature
            
        Returns
        -------
        None.
         
        '''
        # assign class label
        self.class_label = class_label
        
        # clean data
        self.data = self.clean_data(data)
        
        # encode feature name as 'Fx'
        if encode_featurename:
            self.encode_featurename()

    def _add_random(self, df_data):
        """
        add a random column to df_data

        Parameters
        ----------
        df_data : DataFrame
            input data matirx.

        Returns
        -------
        None.

        """
        
        df_data = pd.DataFrame(df_data)
        df_data['random'] = np.random.randn(len(df_data))
        return df_data
    
    def resample_TimeSeries(self, data, date_col, freq, agg):
        """
        resample data by datetime column for specific frequency

        Parameters
        ----------
        date_col : str
            name of column to used as datetime series.
        freq : str
            ['d', 'm', 'w', 'q'].
        agg : TYPE
            ['sum', 'count', 'mean', 'max', 'min'].

        Returns
        -------
        aggregated dataframe

        """

        data = data.copy()
        date_s = pd.to_datetime(data.pop(date_col))
        data = data.set_index(date_s)
        re = data.resample(freq)
        num = getattr(re, agg)()

        return num

    def outlier_scale(self, data_range, scaler, keep_scaled=False):
        """
        preprocess numerical data columns only and update self.data

        Parameters
        ----------
        data_range : list of tuple, [(low, high, flag, column)] 
            to null outlier values    
        
            - filter data within range
            - low, lower bound 
            - high, upper bound
            - flag, 'numeric' or 'percentage'
            - list of columns to apply to, default None, apply to all columns 
              The default is None. the best practice is to scale
              each column one by one.

        scaler : STRING, ['absmax', 'minmax', 'stdscale'], optional
            name of scaler class to perform scaling. 
            
            if None do nothing
        
        keep_scaled : bool
            if True, update self.data as scaled data

        Returns
        -------
        data : data frame

        """
        data = self.data.copy()

        try:
            y_class = data.pop(self.class_label)
        except:
            y_class = None

        # null outlier
        data = null_outlier(data, data_range)
        # scale data
        data = scale_data(data, scaler)

        data = pd.concat([data, y_class], axis=1)
        
        if keep_scaled:
            self.data = data
        
        return data

    def transform_data(self, trans_xx_xx="ordi_stdscale", sample_col=None):
        """
        Transform data maxtrix (X) to other dataframe matrix

        Parameters
        ----------
        trans_xx_xx : str
            string in the form of 'xx_xx_xx' representing transformer.
            
        sample_col : list of columns
            to filter given list of columns. if None, do not filter 
            
        Returns
        -------
        DataFrame :
            dataframe with transformed features

        """
        data = self.data.copy()
        
        if self.class_label:
            y_class = data.pop(self.class_label)
        else:
            y_class = None
            
        if sample_col is not None:
            data = self.data.reindex(columns=sample_col)
            
        if trans_xx_xx is not None:
            pip = pipe_gen(trans_xx_xx)
            pip.fit(data, y=y_class)
            data = pd.DataFrame(pip.transform(data), 
                                columns=get_featurenames(pip))
        
        return pd.concat([data, y_class], axis=1)


    @dedent
    @Appender(Cleaner.__init__.__doc__, join='\n')
    def clean_data(self, data, dtype_filter='all', **kwargs):
        '''
        call Cleaner to clean DataFrame
        
        parameters
        ----------
        data : 2d array
            dataframe or numpy array
        
        Appended
        --------
        
        '''
        clean = Cleaner(dtype_filter=dtype_filter, **kwargs)
        return clean.fit_transform(data)

    def encode_featurename(self):
        '''
        encode feature names of self.data as F{integer}, 
        class label will not change 

        Returns
        -------
        self
        
        '''
        if self.class_label:
            cols = self.data.drop(columns=self.class_label).columns
        else:
            cols = self.data.columns

        self.feature_map = {k: 'F%d' % i for i, k in enumerate(cols)}
        self.data = self.data.rename(columns=self.feature_map)
        return self

    def show_statiscs(self,
                      agg_fun=[
                          np.dtype, 'min', 'mean', 'max', 'std', 'sum',
                          'count', 'size'],
                      add_random=True,
                      n_clusters=5
                      ):
        """
        return dataframe describing statistics for each column

        Parameters
        ----------
        agg_fun : TYPE, optional
            aggregation func. The default is [np.dtype, 'min', 'mean', 'max', 
            'std','sum', 'count', 'size'].
        
        add_random : bool
            add one random column to serve as benchmark

        Returns
        -------
        df : DataFrame
            columns are ["perc_nan", "corr_xy", "clf_imp", "imp_rank", "iv", 
                         "iv_rank", "cov_clusters", "dtype"]
            - perc_nan
                percentage of nan values
            - corr_xy
                if has y label, correlations with y
            - clf_imp
                if has y label, importance values for classifying y class
            - imp_rank
                if has y label, importance rank of feature importance
            - iv
                if has y label, information value
            - iv_rank
                if has y label, information value rank
            - cov_clusters
                cluster group of feature
            - dtype
                data type of feature
            
        """

        data = self.data.copy()
        if add_random:
            data = self._add_random(data)        
         
        # calculate statistics 
        df = data.agg(agg_fun).T
        
        df_pct_nan = data.apply(pct_nan, axis=0)
        df_pct_nan.name = "pct_nan"
        
        # calculate correlation clusters
        df_corr_cluster = corr_clusters(data.corr(), n_clusters)
        
        df = pd.concat([df, df_pct_nan, df_corr_cluster], axis=1)
        
        # y related statistics
        if self.class_label is not None:
            corr_xy = data.corr()[self.class_label]
            df_corr = pd.DataFrame({"corr_xy" : corr_xy})
            
            y = data[self.class_label]
            X = data.drop(columns=self.class_label)
            # importance ranking
            df_imp = imp(X, y)
            # information value
            df_iv = iv(X, y)
            # concatenate statistics 
            df = pd.concat([df, df_corr, df_imp, df_iv], axis=1)            
        return df

    def plot_corr(self,
                  kind='X_y',
                  savefig=None,
                  inflate=True,
                  trans_xx_xx=None,
                  ticks_lim=25,
                  sample_col=None,
                  **kwargs):
        """
        plot correlation coefficients for numeric features in data 

        Parameters
        ----------
        kind : str, ['X_y', 'corrmat', 'clustered'] optional
        
            if 'X_y', plot correlation coefficient between y and other features
            
            if 'corrmat', plot lower triangle correlation matrix, annotate=True
            to show annotations
            
            if 'clustered', plot clustered correlation matrix
            
            The default is 'X_y'.
            
            
        inflate : bool (default = True)
            Whether to inflate correlation coefficients to a 0-100 scale.
            Avoids decimal points in the figure, which often appears very 
            cluttered otherwise.

        trans_xx_xx : TYPE
            string in the form of 'xx_xx_xx', representing transformer to call
            self.transform_data. For example use 'clean_ordi' to covert 
            categorical data column to oridinal encoded integers
            
        savefig : TYPE, optional
            DESCRIPTION. The default is None. 
                
        ticks_lim : int
            if number of features above ticks_lim, show no tikcs
        
        **kwargs : TYPE
            Other optional arguments to sns heatmap.

        Raises
        ------
        KeyError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        plot_fn = {
            'X_y': corr_X_y,
            'corrmat': corrmat,
            'clustered': clustered_corrmap
        }

        data = self.data.copy()
        data = self.transform_data(trans_xx_xx, sample_col)

        plot = plot_fn.get(kind)
        if plot is None:
            raise KeyError("no plot_fn for '{}'".format(kind))

        if kind == 'X_y':
            plot(data, y=self.class_label, top=3, ticks_lim=ticks_lim)

        if kind == 'corrmat':
            if trans_xx_xx is not None:
                title = "Correlation Matrix {}".format('after' + trans_xx_xx)
            else:
                title = "Correlation Matrix"
            plot(data.corr(),
                 inflate=inflate,
                 annot_kws={'fontsize': 12},
                 title=title,
                 **kwargs)

        if kind == 'clustered':
            if trans_xx_xx is not None:
                title = "Clustered Correlation {}".format('after' +
                                                          trans_xx_xx)
            else:
                title = "Clustered Correlation"
            plot(data.corr(),
                 KMeans(),
                 inflate=inflate,
                 annot_kws={'fontsize': 12},
                 **kwargs)

        if savefig:
            if savefig is True: savefig = 'corr_{}.pdf'.format(kind)
            _save_fig(None, savefig)

    def plot_scatter(self,
                     col1,
                     col2,
                     col3,
                     c=None,
                     trans_xx_xx=None,
                     savefig=None):
        """
        scatter plot for at most 3 dimensions

        Parameters
        ----------
        col1 : str
            column to plot as x axis.
        col2 : str
            column to plot as y axis.
        col3 : str
            column to plot as z axis.
        c : str, optional
            column to plot as color. The default is None.
        trans_xx_xx : TYPE, optional
            if not none, peform transformation of data matrix. 
            The default is None.
        savefig : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        ax : TYPE
            DESCRIPTION.

        """

        data = self.data.copy()
        data = self.transform_data(trans_xx_xx)
        # --
        x, y, z = data[col1], data[col2], data[col3]
        if c is not None:
            c = data[c]
        ax = plotter_scatter(x, y, z, c=c)
        if savefig:
            if savefig is True:
                savefig = 'scatter.pdf'
            _save_fig(None, savefig)

        return ax

    def plot_JointScatter(self,
                          col1,
                          col2,
                          savefig=None,
                          kind='hex',
                          **kwargs):
        """
        
        Parameters
        ----------
        col1 : str
            column name to plot as x axis.
        col2 : str
            column name to plot as y axis.

        savefig : TYPE, optional
            DESCRIPTION. The default is None.
        kind:
            { "scatter" | "reg" | "resid" | "kde" | "hex" }, optional
            Kind of plot to draw.

        Returns
        -------
        g : TYPE
            DESCRIPTION.

        """

        data = self.data.copy()
        # --
        g = sns.jointplot(x=data[col1],
                          y=data[col2],
                          data=data,
                          kind=kind,
                          **kwargs)
        if savefig:
            if savefig is True:
                savefig = 'plot_JointScatter.pdf'
            _save_fig(None, savefig)

        return g

    def plot_ridge(self,
                   col,
                   groupings,
                   bw='silverman',
                   cut=0,
                   palette=None,
                   start=0,
                   height=0.5,
                   savefig=None):
        """
        

        Parameters
        ----------
        col : str
            name of data column to plot.

        grouping : series or int
            name of column as groupings to compare distribution.
            
            if int, group column into 'int' folds by sequence and label group
            as integer starting from 0,1,2,3..., usually used by time series
            
        bw : TYPE, optional
            width to estimate density. The default is 'silverman'.
            
        cut : TYPE, optional
            DESCRIPTION. The default is 0.
            
        palette : TYPE, optional
            DESCRIPTION. The default is None, use a sns.cubehelix_palette.
            
        start : TYPE, optional
            startfloat, 0 <= start <= 3
            The hue at the start of the helix The default is 1.    
            
        savefig : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        g : TYPE
            DESCRIPTION.

        """

        data = self.data.copy()
                    
        # --
        if groupings in data.columns:
            groupings = data[groupings]
        
        g = plotter_ridge(data[col],
                          groupings,
                          bw=bw,
                          cut=cut,
                          palette=palette,
                          start=start,
                          height=height)

        if savefig:
            if savefig is True:
                savefig = 'ridge_plot.pdf'
            _save_fig(None, savefig)
        return g

    def plot_catplot(self,
                     groupings=None,
                     trans_xx_xx='clean_ordi',
                     kind='bar',
                     sharex=False,
                     sharey=False,
                     sample_col=None,
                     savefig=None,
                     **kwargs):
        """
        call seaborn catplot to show the relationship between a numerical
        and one or more categorical variables using one of several visual
        representations.       
        
        note
        -----
        data matrix will be unpivoted by class labels plus groupings 
        (``column labels``). numerical data will be labeld as 
        ``value``, column labels will be named as ``colname``, the matrix 
        columns look like [``y``, ``groupings``, ``colname``, ``value``]
        
        Parameters
        ----------
        groupings : str, optional
            column name to use as grouping variable. The default is None.
        
        trans_xx_xx : TYPE, optional
            DESCRIPTION. The default is 'clean_ordi'.
        
        kind str :
            Options are: “point”, “bar”, “strip”, “swarm”, “box”, “violin”, or “boxen”.
        
        sample_col: list
            list of sampled column names. default is None.
        
        **kwargs : TYPE
            key words for sns.catplot().

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        data = self.data.copy()
        data = self.transform_data(trans_xx_xx, sample_col)

        id_vars = [self.class_label]
        if groupings is not None:
            id_vars.extend(groupings)
        # unpivot data
        data = data.melt(id_vars=id_vars,
                         var_name='colname',
                         value_name='value')
        g = sns.catplot(data=data,
                        kind=kind,
                        sharex=sharex,
                        sharey=sharey,
                        **kwargs)
        g.set_xlabels('')
        g.add_legend()
        if savefig:
            if savefig is True:
                savefig = 'cat_plot.pdf'
            _save_fig(None, savefig)
        return g

    def plot_bindist(self,
                     is_supervised=False,
                     bins=None,
                     q=None,
                     max_leaf_nodes=None,
                     mono=None,
                     labels=None,
                     col_grouping_mapper={},
                     col_wrap=3,
                     dropna=False,
                     hue='col_dtype',
                     savefig=None,
                     sample_col=None):
        """
        perform binning for each feature then plot bin dsitribution in grid
        (lift curve if is_supervised=True) 

        Parameters
        ----------
            
        is_supervised bool:
            if True plot positive rate for class label ('y')
            if False plot histgram of each feature
            
        col_grouping_mapper dict:
            {'groupingname' : {'colname : mapped_name, ...}}, to add groupings
            name that 'groupingname' can be used in hue keyword.
            
        savefig : bool or file_path, optional
            save fig to file_path. The default is None.

        col_wrap : TYPE, optional
            number of column in grid. The default is 3.
        
        dropna : bool
            if True, drop na value when plotting a column

        hue str:
            to pass to hue key word

        sample_col: list
            list of sampled column names. default is None to plot all columns.
            
        other args 
        -----------
        
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
            -  only 1 of (q, bins, max_leaf_nodes, mono) can be specified,
            if conflicted q=10 is used


        Returns
        -------
        g : TYPE
            DESCRIPTION.

        """
        # split X, y;
        data = self.data.copy()
        data = self.transform_data(trans_xx_xx=None, sample_col=sample_col)

        if self.class_label is not None:
            y = data.pop(self.class_label)
        else:
            y = None
            
        # dtype ranking of X features
        col_order = data.dtypes.sort_values()

        data['y'] = y

        # unpivot data
        id_vars = 'y' if is_supervised else None
        data = data.melt(id_vars=id_vars,
                         var_name='colname',
                         value_name='value')
        # add groupings
        data['col_dtype'] = data['colname'].map(col_order.to_dict())
        for k, v in col_grouping_mapper.items():
            data[k] = data['colname'].map(v)

        # plot facet grid
        g = sns.FacetGrid(data,
                          col='colname',
                          col_wrap=3,
                          sharex=False,
                          sharey=False,
                          dropna=dropna,
                          aspect=1.5,
                          col_order=col_order.index,
                          hue=hue)
        if is_supervised:
            g.map(plotter_binlift,
                  'value',
                  'y',
                  max_leaf_nodes=max_leaf_nodes,
                  bins=bins,
                  q=q,
                  mono=mono,
                  labels=labels)
        else:
            g.map(plotter_binlift,
                  'value',
                  max_leaf_nodes=max_leaf_nodes,
                  bins=bins,
                  q=q,
                  mono=mono,
                  labels=labels)

        g.set_titles("{col_name}")
        if savefig:
            if savefig is True:
                savefig = 'bindist_plot.pdf'
            _save_fig(None, savefig)

        return g


if __name__ == '__main__':
    data = get_local_data('givemesomecredit.csv')
    data['cat'] = 'A'
    data['cat'].iloc[:15000] = 'B'
    data['cat'].iloc[18000:100000] = np.nan

    an = DataAnalyzer(data, class_label='y', encode_featurename=True)
    # show commonly used statistics
    df = an.show_statiscs()
    
    an.plot_corr('X_y')
    an.plot_corr('corrmat')
    an.plot_corr('clustered')
    
    # plot distribution with na
    an.plot_bindist(max_leaf_nodes=5, 
                    is_supervised=True,
                    dropna=False)
    
    # plot distribution without na
    # not use supervised binning 
    an.plot_bindist(bins=10, 
                    is_supervised=True,
                    dropna=True)
    # cat plot
    an.plot_catplot(x="y", y="value", col_wrap=3, col='colname',
                    kind='box')
    
    
    an.outlier_scale(data_range=[(0.0, 0.95, 'percentage', ["F3", "F5"])], 
                      scaler=None, 
                      keep_scaled=True)
    
    # plot joint scatter
    an.plot_JointScatter("F1", "F3")
    
    # plot ridge plot
    an.plot_ridge("F3", "F5")
