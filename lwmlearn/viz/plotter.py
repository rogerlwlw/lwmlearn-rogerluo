# -*- coding: utf-8 -*-
"""data visualization module

Created on Sun Dec 15 16:22:09 2018

@author: roger
"""
import numpy as np
import pandas as pd
import inspect
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from collections import OrderedDict

from functools import reduce
from pandas.core.dtypes import api
from scipy import interp
from sklearn.metrics import auc, roc_curve

from lwmlearn.utilis.utilis import get_flat_list, get_kwargs, dict_diff
from lwmlearn.utilis.binning import binning
from lwmlearn.utilis.docstring import Appender, dedent

from scipy.interpolate import interp1d
from scipy.misc import derivative

plt.style.use('seaborn')

plt.rcParams.update({
    'figure.dpi': 120.0,
    'axes.unicode_minus': False,
    'font.family': ['sans-serif'],
    'font.sans-serif': ['SimHei'],
    'font.size': 10.0,
    'font.style': 'normal',
    'lines.linewidth': 1.2,
    'xtick.labelsize': 10.0,
    'ytick.labelsize': 10.0,
    'axes.labelsize': 10.0
})


def txt_fontdict(**kwargs):
    '''
    '''
    font = {
        'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 12,
    }
    font.update(**kwargs)
    print(font)
    return font


def _get_snsplot(kind=None):
    '''return plot functions in seaborn catplot module
    '''
    if callable(kind):
        return kind
    lst = inspect.getmembers(sns, inspect.isfunction)
    fn_dict = dict([i for i in lst if i[0].count('plot') > 0])
    fn_dict.update(hist=plt.hist, lift=plotter_binlift)
    if kind is None:
        return fn_dict.keys()
    # Determine the plotting function
    try:
        plot_func = fn_dict[kind]
    except KeyError:
        err = "Plot kind '{}' is not recognized".format(kind)
        raise ValueError(err)
    return plot_func


def _sequence_grouping(x, n):
    """
    group array into 'n' groups by sequence

    Parameters
    ----------
    x : array or int
        DESCRIPTION.
    n : integer
        number of groupings.

    Returns 
    -------
    array
        grouping index of x

    """
    if np.ndim(x) > 0:
        array = np.arange(len(x))
    else:
        array = np.arange(x)
    return pd.cut(array, bins=n, labels=False)


def _save_fig(fig, file, **kwargs):
    '''save fig object to 'path' if it has 'savefig' method
    
    fig:
        fig object
    file:
        file name to save fig
    
    **kwargs:
        key words argument passed to savefig() method
    '''
    import os

    _, suffix = os.path.splitext(file)

    if not suffix in ['.png', '.pdf']:
        raise ValueError('extension name must be one of {}'.format(
            ['.png', '.pdf']))

    if fig is None:
        fig = plt.gcf()

    if hasattr(fig, 'savefig'):
        fig.savefig(file, **kwargs)
    else:
        getattr(fig, 'get_figure')().savefig(file, **kwargs)

    plt.close()


def _rotate_tick_labels(ax, rotation, ha='right', max_nticks=None):
    '''
    '''
    if max_nticks is not None:
        locator = ticker.MaxNLocator(max_nticks)
        ax.xaxis.set_major_locator(locator)
    plt.draw()
    labels = ax.xaxis.get_ticklabels()
    for i in labels:
        i.set_rotation(rotation)
        i.set_ha(ha)

    return ax


def _percent_axis(ax, axis='y'):
    '''
    '''
    fmt = _get_ticks_formatter('percent', decimals=1)
    if axis == 'y':
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)
    return ax


def _autofmt_xdate(ax, minticks=3, maxticks=5, concise_date=False):

    import matplotlib.dates as mdates

    locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    ax.xaxis.set_major_locator(locator)

    if concise_date:
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_formatter(formatter)

    return


def _annotate(x, y, ax):
    ''' plot annotate
    '''
    if api.is_list_like(x):
        for i, j in zip(x, y):
            ax.annotate(s='%.1f%%' % (100 * j),
                        xy=(i, j),
                        xytext=(10, 10),
                        textcoords='offset pixels')
    return ax


def _get_mean_line(fpr, tpr):
    '''return averaged fpr, tpr
    '''
    fpr = get_flat_list(fpr)
    tpr = get_flat_list(tpr)
    n = len(fpr)
    # plot mean tpr line
    if n > 1:
        mean_fpr = np.linspace(0, 1, 100)
        tprs = [interp(mean_fpr, x, y) for x, y in zip(fpr, tpr)]
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        #plot variance
        std_tpr = np.std(tprs, axis=0)
        return mean_fpr, mean_tpr, std_tpr


def _get_ticks_formatter(name, *args, **kwargs):
    ''' return ticks formattor  
    ---
    name - form of ticks formatter
    see matplotlib.ticker        
    '''
    if name == 'percent':
        frm = ticker.PercentFormatter(xmax=1, *args, **kwargs)
    if name == 'func':
        frm = ticker.FuncFormatter(*args, **kwargs)
    if name == 'scalar':
        frm = ticker.ScalarFormatter(*args, **kwargs)

    return frm


def _filter_subset(data, filter_con, **kwargs):
    '''
    data:
        DataFrame
    filter_con (dict):
        filter condition on data's columns, 
        eg: {col1 : [str1, str2, ...], ...}
        
    '''
    if not isinstance(data, pd.DataFrame):
        raise ValueError("'data' must be DataFrame")
    gen = (data[k].isin(v) for k, v in filter_con.items())
    filtered = reduce(lambda x, y: x & y, gen)
    return data.loc[filtered]


def plt_insp(artist):
    """
    inspect setters for matplotlib artist
    
    Parameters
    ----------
    artist : str
        name of artist, like ['axe', 'axis', 'fig'].

    Returns
    -------
    list of setters for given artist

    """
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure
    selector = {}
    selector.update(
        axe=Axes,
        axis=Axis,
        fig=Figure,
    )
    from matplotlib.artist import ArtistInspector
    insp = ArtistInspector(selector.get(artist))
    return insp.get_setters()


def plotter_ridge(x,
                  grouping,
                  bw='silverman',
                  cut=0,
                  palette=None,
                  start=1,
                  height=0.5,
                  **kwargs):
    """
    ridge plot to compare distribution between groups 

    Parameters
    ----------
    x : series
        array of a single variable.
    grouping : series or int
        groupings to compare distribution.
        if int, group x into 'int' folds by sequence 
    bw : TYPE, optional
        width to estimate density. The default is 'silverman'.
    cut : TYPE, optional
        DESCRIPTION. The default is 0.
    palette : TYPE, optional
        DESCRIPTION. The default is None, use a sns.cubehelix_palette.
    start : TYPE, optional
        startfloat, 0 <= start <= 3
        The hue at the start of the helix The default is 1.
    
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    g : TYPE
        DESCRIPTION.

    """

    if not np.ndim(grouping) > 0:
        grouping = _sequence_grouping(x, grouping)

    df = pd.DataFrame({'x': x, 'g': grouping})
    if palette is None:
        n_g = len(pd.unique(grouping))
        palette = sns.cubehelix_palette(n_colors=n_g, start=start, rot=0.165)
    g = sns.FacetGrid(df,
                      row="g",
                      hue="g",
                      aspect=8,
                      height=height,
                      palette=palette)

    # Draw the densities in a few steps
    g.map(sns.kdeplot,
          "x",
          clip_on=False,
          shade=True,
          alpha=1,
          lw=1.5,
          bw=bw,
          cut=cut)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=bw, cut=cut)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0,
                .2,
                label,
                fontweight="bold",
                color='K',
                ha="left",
                va="center",
                transform=ax.transAxes)

    g.map(label, "x")

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    plt.tight_layout()
    return g


def plotter_scatter(x, y, z, c=None, cmap='Spectral', alpha=0.85, **kwargs):
    """
    

    Parameters
    ----------
    *args : TYPE
        DESCRIPTION.
    c : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """

    dim = len([i for i in (x, y, z) if i is not None])
    if dim > 3:
        raise ValueError("len of *args can't be more than 3")
    elif dim > 2:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        scatter = ax.scatter(x, y, z, c=c, cmap=cmap, alpha=alpha, **kwargs)
        ax.set(xlabel=x.name, ylabel=y.name, zlabel=z.name)
    else:
        fig, ax = plt.subplots(1, 1)
        scatter = ax.scatter(x, y, c=c, cmap=cmap, alpha=alpha, **kwargs)
        ax.set(xlabel=x.name, ylabel=y.name)

    # produce a legend with the unique colors from the scatter
    legend = ax.legend(*scatter.legend_elements(), title="Classes")

    plt.tight_layout()
    return ax


@Appender(sns.FacetGrid.__doc__, join='\nsee Facetgrid\n')
@dedent
def plotter_facet(data,
                  plot_args,
                  subset=None,
                  kind='distplot',
                  savefig=None,
                  **kwargs):
    '''plot grids of plots using seaborn Facetgrid
    
    parameter
    -----
    data : DataFrame
    
        Tidy (“long-form”) dataframe where each column is a variable and each 
        row is an observation.

    subset (dict):
        fitler subset of data by column's categorical values
        eg: {col1 : [str1, str2, ...], ...}
        
    kind:
        callable plot function or 'str' to call in _get_plot_fn()
        ['distplot', 'boxplot', 'violinplot']
        
    plot_args (tuple):
        (colname1 as x, colname2 as y ) indexed by DataFrame   
        
    row, col, hue : strings
    
        Variables that define subsets of the data, which will be drawn on 
        separate facets in the grid. See the *_order parameters to control 
        the order of levels of this variable.
    
    col_wrap : int, optional
    
        “Wrap” the column variable at this width, so that the column facets
        span multiple rows. Incompatible with a row facet.
    
    share{x,y} : bool, ‘col’, or ‘row’ optional
    
        If true, the facets will share y axes across columns and/or x axes 
        across rows.
    
    height : scalar, optional
    
        Height (in inches) of each facet. See also: aspect.
    
    aspect : scalar, optional
    
        Aspect ratio of each facet, so that aspect * height gives the width of 
        each facet in inches.
    
    palette : palette name, list, or dict, optional
    
        Colors to use for the different levels of the hue variable. 
        Should be something that can be interpreted by color_palette(), 
        or a dictionary mapping hue levels to matplotlib colors.
    
    {row,col,hue}_order : lists, optional
    
        Order for the levels of the faceting variables. By default, 
        this will be the order that the levels appear in data or, 
        if the variables are pandas categoricals, the category order.

    '''
    if subset is not None:
        data = _filter_subset(data, subset)

    fn_plot = _get_snsplot(kind)
    # get facet kwds
    facet_kws = get_kwargs(sns.FacetGrid, **kwargs)
    # get fn kwds
    plot_fn_kws = get_kwargs(fn_plot, **kwargs)
    # get other than kwds
    ax_kws = dict_diff(kwargs, facet_kws.keys() | plot_fn_kws.keys())
    # generate grid
    g = sns.FacetGrid(data, **facet_kws)
    # map plot function
    g.map(fn_plot, *plot_args, **plot_fn_kws)

    if len(ax_kws) > 0:
        g.set(**ax_kws)

    g.fig.tight_layout()
    g.add_legend()
    if savefig:
        _save_fig(g, savefig)
    return g


@Appender(sns.catplot.__doc__, join='\nsee catplot\n')
@Appender(sns.violinplot.__doc__, join='\nsee violinplot\n')
def plotter_catplot(data,
                    kind='violin',
                    swarm=False,
                    hline=None,
                    subset=None,
                    **kwargs):
    '''make a distr plot through catplot function   

    parameter
    ---------
    data (DataFrame):
    
        Tidy (“long-form”) dataframe where each column is a variable and each 
        row is an observation.

    kind (str): 'violin' default
        ['violin', 'swarm', 'box', 'bar', 'count'] 
    swarm (bool):
        whether to combine a swarmplot, default False
    hline (int):
        add a horizontal base line 
    subset (dict):
        fitler subset of data by column's categorical values
    kwargs:
        other keywords to customize ax and to pass to plot functions
    
    return
    --------    
        g : FacetGrid
            Returns the FacetGrid object with the plot on it for further 
            tweaking.
    '''
    if subset is not None:
        data = _filter_subset(data, subset)

    # get plot function key words
    fn_kws = dict(
        violin=get_kwargs(sns.violinplot, **kwargs),
        box=get_kwargs(sns.boxplot, **kwargs),
        swarm=get_kwargs(sns.swarmplot, **kwargs),
        bar=get_kwargs(sns.barplot, **kwargs),
        count=get_kwargs(sns.countplot, **kwargs),
        cat=get_kwargs(sns.catplot, **kwargs),
        point=get_kwargs(sns.pointplot, **kwargs),
        factor=get_kwargs(sns.factorplot, **kwargs),
    )
    plot_fn_kws = fn_kws.get(kind)
    plot_fn_kws.update(fn_kws.get('cat'))

    if hline is not None:
        plot_fn_kws.update(legend_out=False)
    # plot categorical data
    g = sns.catplot(data=data, kind=kind, **plot_fn_kws)

    if swarm:
        g.map(sns.swarmplot,
              data=data,
              ax=g.ax,
              x=kwargs.get('x'),
              y=kwargs.get('y'),
              size=2.5,
              color='k',
              alpha=0.3)
    if hline is not None:
        g.map(plt.axhline,
              y=hline,
              color='red',
              linestyle='--',
              label='baseline%s' % hline)
        g._legend_out = True
        g.add_legend()

    ax_kws = dict_diff(kwargs, plot_fn_kws.keys())
    if 'savefig' in ax_kws:
        ax_kws.pop('savefig')
    if len(ax_kws) > 0:
        g.set(**ax_kws)
    # save fig to savefig path
    if kwargs.get('savefig') is not None:
        _save_fig(g, kwargs['savefig'])
    return g


def plotter_auc(fpr,
                tpr,
                ax=None,
                alpha=0.95,
                lw=1.2,
                curve_label=None,
                title=None,
                cm=None,
                plot_mean=True):
    '''plot roc_auc curve given fpr, tpr, or list of fpr, tpr
    
    cm:
        color map default 'tab20'
    
    return
    ----
    ax
    '''
    fpr, tpr = get_flat_list(fpr), get_flat_list(tpr)
    if len(fpr) != len(tpr):
        raise ValueError("length of fpr and tpr doesn't match")
    n = len(fpr)
    names = range(n) if curve_label is None else get_flat_list(curve_label)
    if len(names) != n:
        print('n_curve label not match with n_fpr or n_tpr')
        names = range(n)

    # -- plot each line
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    aucs = []
    kss = []
    if cm is None:
        cm = plt.get_cmap('tab20')
    cmlist = [cm(i) for i in np.linspace(0, 1, n)]
    for i in range(n):
        if len(fpr[i]) != len(tpr[i]):
            print("length of {}th fpr and tpr doesn't match".format(i))
            continue
        else:
            auc_score = auc(fpr[i], tpr[i])
            ks_score = max(np.array(tpr[i]) - np.array(fpr[i]))
            aucs.append(auc_score)
            kss.append(ks_score)
            ax.plot(fpr[i],
                    tpr[i],
                    color=cmlist[i],
                    alpha=alpha,
                    lw=lw,
                    label='%s (AUC=%0.2f;KS=%0.2f)' %
                    (names[i], auc_score, ks_score))

    # plot mean tpr line
    if n > 1 and plot_mean:
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_fpr, mean_tpr, std_tpr = _get_mean_line(fpr, tpr)
        ax.plot(mean_fpr,
                mean_tpr,
                'b-.',
                alpha=1,
                lw=1.5,
                label='Mean ROC(AUC=%0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))
        #plot variance
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr,
                        tprs_lower,
                        tprs_upper,
                        color='grey',
                        alpha=.3,
                        label=r'$\pm$ 1 std. dev.')

    # plot chance line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (AUC=0.5)')
    # set property
    if title is None:
        title = 'Receiver operating characteristic'
    plt.setp(ax,
             xlabel='False Positive Rate',
             ylabel='True Positive Rate',
             xlim=[-0.05, 1.05],
             ylim=[-0.05, 1.05],
             title=title)
    ax.legend(loc="lower right", fontsize='medium', bbox_to_anchor=(1, 0))
    plt.tight_layout()
    return ax


def plotter_auc_y(y_pre, y_true, **kwargs):
    '''plot roc_auc curve given y_pre, y_true
    '''
    fpr, tpr, threshhold = roc_curve(y_true, y_pre,
                                     **get_kwargs(roc_curve, **kwargs))
    ax = plotter_auc(fpr, tpr, **get_kwargs(plotter_auc, **kwargs))
    return ax


def plotter_KS(y_pred, y_true, n, asc):
    '''
    # preds is score: asc=1
    # preds is prob: asc=0
    '''
    pred = y_pred  # 预测值
    bad = y_true  # 取1为bad, 0为good
    ksds = pd.DataFrame({'bad': bad, 'pred': pred})
    ksds['good'] = 1 - ksds.bad

    if asc == 1:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, True])
    elif asc == 0:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])
    ksds1.index = range(len(ksds1.pred))
    ksds1['cumsum_good1'] = 1.0 * ksds1.good.cumsum() / sum(ksds1.good)
    ksds1['cumsum_bad1'] = 1.0 * ksds1.bad.cumsum() / sum(ksds1.bad)

    if asc == 1:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, False])
    elif asc == 0:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
    ksds2.index = range(len(ksds2.pred))
    ksds2['cumsum_good2'] = 1.0 * ksds2.good.cumsum() / sum(ksds2.good)
    ksds2['cumsum_bad2'] = 1.0 * ksds2.bad.cumsum() / sum(ksds2.bad)

    # ksds1 ksds2 -> average
    ksds = ksds1[['cumsum_good1', 'cumsum_bad1']]
    ksds['cumsum_good2'] = ksds2['cumsum_good2']
    ksds['cumsum_bad2'] = ksds2['cumsum_bad2']
    ksds['cumsum_good'] = (ksds['cumsum_good1'] + ksds['cumsum_good2']) / 2
    ksds['cumsum_bad'] = (ksds['cumsum_bad1'] + ksds['cumsum_bad2']) / 2

    # ks
    ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
    ksds['tile0'] = range(1, len(ksds.ks) + 1)
    ksds['tile'] = 1.0 * ksds['tile0'] / len(ksds['tile0'])

    qe = list(np.arange(0, 1, 1.0 / n))
    qe.append(1)
    qe = qe[1:]

    ks_index = pd.Series(ksds.index)
    ks_index = ks_index.quantile(q=qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)

    ksds = ksds.loc[ks_index]
    ksds = ksds[['tile', 'cumsum_good', 'cumsum_bad', 'ks']]
    ksds0 = np.array([[0, 0, 0, 0]])
    ksds = np.concatenate([ksds0, ksds], axis=0)
    ksds = pd.DataFrame(ksds,
                        columns=['tile', 'cumsum_good', 'cumsum_bad', 'ks'])

    ks_value = ksds.ks.max()
    ks_pop = ksds.tile[ksds.ks.idxmax()]
    print ('ks_value is ' + str(np.round(ks_value, 4)) + \
           ' at pop = ' + str(np.round(ks_pop, 4)))

    # chart
    plt.plot(ksds.tile,
             ksds.cumsum_good,
             label='cum_good',
             color='blue',
             linestyle='-',
             linewidth=2)

    plt.plot(ksds.tile,
             ksds.cumsum_bad,
             label='cum_bad',
             color='red',
             linestyle='-',
             linewidth=2)

    plt.plot(ksds.tile,
             ksds.ks,
             label='ks',
             color='green',
             linestyle='-',
             linewidth=2)

    plt.axvline(ks_pop, color='gray', linestyle='--')
    plt.axhline(ks_value, color='green', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_good'],
                color='blue',
                linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_bad'],
                color='red',
                linestyle='--')
    plt.title('KS=%s ' % np.round(ks_value, 4) +
              'at Pop=%s' % np.round(ks_pop, 4),
              fontsize=15)
    plt.xlabel('Percentage')

    return ksds


def plotter_cv_results_(results,
                        train_style='mo-',
                        test_style='go-.',
                        title=None):
    '''plot univariate parameter cross validated results after 
    grid search of model
    
    return
    -----
    ax, or tuple of ax
    '''
    scoring = results.filter(like='mean_train_').columns
    scoring = [i.replace('mean_train_', '') for i in scoring]
    df_param = results.filter(like='param_')
    param_array = df_param.columns
    if len(param_array) > 1:
        print('multi-parameter is encountered ... ')
        print(df_param.apply(lambda x: pd.Series(pd.unique(x))))
    # plot
    n = len(scoring)
    i, j = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(n, 1, figsize=(i, j + 2 * (n // 2)))
    ax = get_flat_list(ax) if n == 1 else ax
    for s, ax0 in zip(scoring, ax):
        df = results[['mean_train_' + s, 'mean_test_' + s, 'std_test_' + s]]
        if len(param_array) == 1:
            df.index = results[param_array[0]]
            xlabel = param_array[0]
            num_param = api.is_numeric_dtype(df.index)
            if not num_param:
                df.index = np.arange(len(df.index)) + 1
        else:
            xlabel = ' + '.join([i.split('__')[-1] for i in param_array])

        df.sort_index(inplace=True)
        # plot
        mean = df['mean_test_' + s].values
        std = df.pop('std_test_' + s)
        x = df.index
        space = (x.max() - x.min()) / 20
        df.plot.line(style=[train_style, test_style],
                     ax=ax0,
                     xlim=(x.min() - space, x.max() + space))
        ax0.fill_between(x,
                         mean - std,
                         mean + std,
                         color='grey',
                         alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
        # annotate
        x_max = df.index[np.argmax(mean)]
        best_score = np.max(mean)
        std = np.mean(std)
        h, l = ax0.get_legend_handles_labels()
        ax0.legend(
            [h[-1]],
            ['score_max= %0.4f $\pm$ %0.2f' % (np.max(mean), np.mean(std))])
        ax0.axvline(x_max, linestyle='--', marker='x', color='y')
        ax0.annotate("%0.4f" % best_score, (x_max, best_score))
        plt.setp(ax0, ylabel=s)

    # set title
    ax[0].set_title(title, fontsize=13)
    # use fig legend
    fig.legend(h, ('train', 'test', r'$\pm$ 1 std. dev.'),
               loc='upper right',
               ncol=3,
               bbox_to_anchor=(0.98, 1))
    ax[-1].set_xlabel(xlabel)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return ax


def plotter_dist_thresh(s,
                        thresh,
                        step=1,
                        subplot_kw=None,
                        savefig=None,
                        **fig_kws):
    '''plot distribution of series and percentage above thresh
    
    s - ndarray or series:
        vector to calculate percentile distribution
    step - integer or float:
        step of percentages to plot cummulative distribution
    thresh - integer:
        threshhold/cutoff of decision
    return
    -----
        quantiles of data
    '''
    s = pd.Series(s)
    q = np.arange(100, step=step) / 100
    q = np.append(q, 1)
    perc = s.quantile(q).drop_duplicates()

    x = perc.reset_index(drop=True)
    y = 1 - perc.index
    y = y.rename('percentage above')
    f = interp1d(x, y, kind='slinear')

    fig, axes = plt.subplots(2,
                             1,
                             sharex=True,
                             subplot_kw=subplot_kw,
                             **fig_kws)
    # hist plot
    sns.distplot(s.dropna(), ax=axes[1], kde=False, hist_kws={'rwidth': 0.85})
    # line plot
    sns.lineplot(x, y, ax=axes[0], c='green')
    fmt = _get_ticks_formatter('percent', decimals=1)
    axes[0].yaxis.set_major_formatter(fmt)
    if thresh:
        axes[0].axvline(thresh,
                        linestyle='--',
                        lw=3,
                        color='y',
                        label='%1.0f' % thresh)
        above_thresh = np.array(x >= thresh)
        x0 = np.array(x[np.nonzero(above_thresh)[0]])
        x0[0] = thresh
        y0 = f(x0)
        marginal = derivative(f, thresh, n=1)
        axes[0].annotate(''' %1.0f%% above %1.0f score \n 
                            marginal contribution: %1.2f%%
                         ''' % (100 * f(thresh), thresh, 100 * marginal),
                         xy=(x0[0], y0[0]),
                         xycoords='data',
                         xytext=(1, 0.9),
                         textcoords='axes fraction',
                         ha='right',
                         va='top',
                         size=12,
                         arrowprops=dict(arrowstyle="->",
                                         lw=1.2,
                                         connectionstyle='angle3'))

        axes[0].fill_between(x0, y0, 0, alpha=0.1, color='m')

    axes[1].set_ylabel('count')

    plt.tight_layout()

    if savefig:
        _save_fig(None, savefig)

    return perc.reset_index()


def plotter_k_timeseries(time_rate, subplot_kw=None, **fig_kws):
    '''plot time series of rate and volume
    
    time_rate - df:
        pass rate at different nodes
    time_vol - series:
        vol at different nodes  
        
    .. note ::         
        index of df/series is used as xaxis 
    
    '''

    time_vol = None
    for name, col in time_rate.iteritems():
        if any(col > 1.0):
            time_vol = time_rate.pop(name)
            break
    if time_vol is not None:

        fig, axe = plt.subplots(2,
                                1,
                                sharex=True,
                                subplot_kw=subplot_kw,
                                **fig_kws)
        # plot line plot for rate data
        sns.lineplot(data=time_rate,
                     ax=axe[0],
                     palette='Set1',
                     markers=True,
                     markersize=6)
        fmt = _get_ticks_formatter('percent', decimals=1)
        axe[0].yaxis.set_major_formatter(fmt)
        # plot area plot for volume data
        sns.lineplot(data=time_vol.to_frame(),
                     markers=['o'],
                     markersize=6,
                     markerfacecoloralt='red',
                     ax=axe[1])
        axe[1].fill_between(time_vol.index, 0, time_vol, alpha=.1)
    else:
        fig, ax = plt.subplots(subplot_kw=subplot_kw, **fig_kws)
        sns.lineplot(data=time_rate,
                     ax=ax,
                     palette='Set1',
                     markers=True,
                     markersize=6)
        fmt = _get_ticks_formatter('percent', decimals=1)
        ax.yaxis.set_major_formatter(fmt)

    fig.autofmt_xdate()
    plt.tight_layout()
    return


def plotter_score_path(df_score, title=None, cm=None, style='-.o'):
    '''
    df_score:
        data frame of scores of metrics
    '''
    # plot
    data = df_score.select_dtypes(include='number')
    n = len(data.columns)
    i, j = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(n, 1, figsize=(i, j + 2.5 * (n // 2)))
    ax = get_flat_list(ax) if n == 1 else ax
    if cm is None:
        cm = plt.get_cmap('tab10')
    cmlist = [cm(i) for i in np.linspace(0, 1, n)]

    i = 0
    for ax0, col in zip(ax, data.columns):
        s = data[col]
        if api.is_numeric_dtype(s):
            s.plot(ax=ax0,
                   color=cmlist[i],
                   style=style,
                   xlim=(-0.05, len(s) - 0.95))
            ax0.fill_between(s.index,
                             s - s.std(),
                             s + s.std(),
                             color='grey',
                             alpha=.3,
                             label=r'{} = {}$\pm$ {}'.format(
                                 col, round(s.mean(), 4), round(s.std(), 4)))
            plt.setp(ax0, ylabel=col)
            h, l = ax0.get_legend_handles_labels()
            ax0.legend([h[-1]], [l[-1]])
            i += 1
    ax[0].set_title(title)
    ax[-1].set_xlabel('index')
    plt.tight_layout(rect=(0, 0, 0.98, 0.96))
    return fig


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plotter_contours(ax,
                     clf,
                     x,
                     y,
                     h=0.02,
                     pre_method='predict',
                     pos_label=1,
                     **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    pos_label: index of predicted class
    params: dictionary of params to pass to contourf, optional
    """
    xx, yy = make_meshgrid(x, y, h)
    pre = getattr(clf, pre_method)
    if pre is not None:
        Z = pre(np.c_[xx.ravel(), yy.ravel()])
    if np.ndim(Z) > 1:
        Z = Z[:, pos_label]

    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def color_reference(keys=None):
    '''show color maps of given keys, if keys=None, show a list of reference
    keys
    '''
    cmaps = OrderedDict()

    cmaps['Perceptually Uniform Sequential'] = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    ]

    cmaps['Sequential'] = [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr',
        'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu',
        'PuBuGn', 'BuGn', 'YlGn'
    ]

    cmaps['Diverging'] = [
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn',
        'Spectral', 'coolwarm', 'bwr', 'seismic'
    ]

    cmaps['Miscellaneous'] = [
        'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
        'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow',
        'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'
    ]

    cmaps['Qualitative'] = [
        'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2',
        'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
    ]

    nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps.items())
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    def plot_color_gradients(cmap_category, cmap_list, nrows):
        fig, axes = plt.subplots(nrows=nrows)
        fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
        axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

        for ax, name in zip(axes, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3] / 2.
            fig.text(x_text,
                     y_text,
                     name,
                     va='center',
                     ha='right',
                     fontsize=10)

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axes:
            ax.set_axis_off()

    if keys is None:
        return cmaps.keys()
    else:
        for cmap_category, cmap_list in cmaps.items():
            if cmap_category in keys:
                plot_color_gradients(cmap_category, cmap_list, nrows)
            plt.show()


# BBD plotter
def plotter_k_status(data, savefig=None):
    ''' calculate pass rate at different application-appraisal nodes, and 
    plot data
    
    data - DataFrame (is_passed series, keys as name):
        columns:
            represent application-appraisal nodes name
        values (ndarray or series):
            represent passed applications at different nodes, binary array 
            1 indicate pass, 0 indicate not pass
        egg. (apply, admission, score, amount)
   
    return (df):
        dataframe with columns  [nodes, rate, volume]
        
    '''

    vol = data.sum()
    rate = vol / vol.max()

    plot_data = pd.DataFrame({'节点': vol.index, '百分比': rate, '单量': vol})
    # plot
    plotter_rateVol(plot_data,
                    anno=True,
                    show_mean=False,
                    dpi=100,
                    bar_c=sns.color_palette('Blues_d', n_colors=4))

    if savefig:
        _save_fig(None, savefig)

    return plot_data


def plotter_binlift(x,
                    y=None,
                    bins=None,
                    q=None,
                    max_leaf_nodes=None,
                    mono=None,
                    labels=None,
                    color='g',
                    xlabel=False,
                    ylabel=False,
                    legend=True,
                    show_mean=True,
                    ax=None,
                    label=None,
                    **kwargs):
    """
    plot lift curve for binned series x_bin and y (binary target), np.nan will
    be included as a single bar
    

    Parameters
    ----------
    x : array
        feature arrays.
    y : array
        class labels binary 0/1.
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
        
    labels
        - see pd.cut, if False return integer indicator of bins, 
        - if True return arrays of labels (or can be passed )
        
    data : TYPE, optional
        DESCRIPTION. The default is None.
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    label : TYPE, optional
        DESCRIPTION. The default is None.
    color : TYPE, optional
        DESCRIPTION. The default is 'g'.
    xlabel : TYPE, optional
        DESCRIPTION. The default is False.
    ylabel : TYPE, optional
        DESCRIPTION. The default is False.
    legend : TYPE, optional
        DESCRIPTION. The default is False.
    show_mean : TYPE, optional
        DESCRIPTION. The default is True.
    legend : TYPE, optional
        DESCRIPTION. The default is False.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """

    if ax is None:
        ax = plt.gca()
    # binning x features
    x_bin, bins_edges = binning(x,
                                 y_true=y,
                                 bins=bins,
                                 q=q,
                                 max_leaf_nodes=max_leaf_nodes,
                                 mono=mono,
                                 labels=labels,
                                 **kwargs)
    if y is not None:
        df1 = _cal_rate_vol(x_bin, y)
        # plot
        plotted_data = df1.dropna()
        ax = plotter_rateVol(plotted_data,
                             ax=ax,
                             xlabel=xlabel,
                             ylabel=ylabel,
                             bar_c=color,
                             show_mean=show_mean,
                             legend=legend)
    else:
        # plot bar chart
        x_bin = x_bin.value_counts(dropna=False).sort_index()
        labels = x_bin.index.astype(str)
        ax.bar(labels, x_bin, color=color, alpha=0.85)
        if max([len(i) for i in labels]) > 8:
            _rotate_tick_labels(ax, 30, ha='right')

    return ax


def _cal_rate_vol(x_bin, y, kind='vol'):
    """cal bin counts and percentage
    

    Parameters
    ----------
    x_bin : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    kind : TYPE, optional ['percentage', 'vol']
        if 'percentage' output percentage of count number. 
        if 'vol' output count number
        The default is 'vol'.

    Returns
    -------
    df1 : TYPE
        DESCRIPTION.

    """
    

    sort_index = x_bin.value_counts(
        dropna=False).sort_index().index.astype(str)

    df0 = pd.DataFrame({'y_cut': x_bin.astype(str), 'y_true': y})
    df_gb = df0.groupby('y_cut', sort=False)

    df1 = df_gb.count()['y_true'].to_frame('vol')
    df1['rate'] = df_gb.sum() / df_gb.count()
    df1['xbin'] = df1.index.values
    df1 = df1.reindex(index=sort_index, columns=['xbin', 'rate', 'vol'])

    if kind == 'percentage':
        df1['vol'] = df1['vol'] / df1['vol'].sum()

    return df1


def plotter_rateVol(df,
                    ax=None,
                    lstyle='k-.o',
                    bar_c='g',
                    ymajor_formatter='percent',
                    xlabel_position='bottom',
                    xlabelrotation=30,
                    anno=False,
                    show_mean=True,
                    ylabel=True,
                    xlabel=True,
                    legend=True,
                    **subplot_kw):
    ''' plot rate along with volume
    
    df - 3 cols [D1, rate, denominator]
        --> D1-dimensional label, rate= numerator / denominator
    lstyle
        -->2Dline stype egg '-.ko'
    bar_c
        -->  color of bar chart              
    ylim
        --> left y axis limit
    ymajor_formatter
        -->
    xlabel_position
        --> str, 'top'/'down'/'both'   
    return
    ----
    list of artists drawed in ax
    '''
    L = locals().copy()
    L.pop('df')
    L.pop('ax')

    axe = ax if ax is not None else plt.gca()

    labels = df.iloc[:, 0]
    rate = df.iloc[:, 1]
    vol = df.iloc[:, 2]
    rate_weighted = np.average(rate, weights=vol)
    # plot artists
    out = []
    axe_right = axe.twinx()
    axe.plot(labels, rate, lstyle, label=rate.name, alpha=1)

    # set axe attr
    fmt = _get_ticks_formatter(ymajor_formatter, decimals=1)
    plt.setp(axe, ylim=(-0.001, rate.max() * 1.2))
    if labels.astype(str).apply(len).max() > 8:
        _rotate_tick_labels(axe, xlabelrotation, ha='right')
    if ylabel:
        axe.set_ylabel(rate.name)
    if xlabel:
        axe.set_xlabel(labels.name)
    axe.yaxis.set_major_formatter(fmt)
    axe.xaxis.set_label_position(xlabel_position)
    # plot axe_right
    axe_right.bar(labels, vol, label=vol.name, color=bar_c, alpha=0.4)
    # set axe_right attr
    if ylabel:
        axe_right.set_ylabel(vol.name)
    axe_right.set_ylim((0, vol.max() * 1.1))
    axe_right.grid(False)

    if show_mean:
        axe.axhline(rate_weighted, linestyle='--', color='yellow', lw=1.5)
        bbox = dict(boxstyle="round", fc='w', alpha=1)
        axe.annotate("{}/{} = {}%".format(int(sum(rate * vol)), sum(vol),
                                          round(100 * rate_weighted, 1)),
                     (0, rate_weighted), (0.01, 0.95),
                     xycoords='data',
                     textcoords='axes fraction',
                     bbox=bbox)
    if anno is True:
        _annotate(rate.index.values, rate.values, axe)

    if legend:
        # get legends
        h_r, l_r = axe_right.get_legend_handles_labels()
        h, l = axe.get_legend_handles_labels()
        h.extend(h_r)
        l.extend(l_r)
        axe.legend(h,
                   l,
                   bbox_to_anchor=(1, 1),
                   loc='upper right',
                   fontsize='small',
                   ncol=2)

    plt.tight_layout(pad=1.08, rect=(0, 0, 1, 0.9))
    return axe


def plotter_lift_curve(y_pre,
                       y_true,
                       bins,
                       q,
                       max_leaf_nodes,
                       mono,
                       labels,
                       ax,
                       header,
                       xlabel='modelscore',
                       **kwargs):
    '''return lift curve of y_pre score on y_true binary  
   
    y_pre
        - array_like, value of y to be cut
    y_true
        - true value of y for supervised cutting based on decision tree 
    bins
        - number of equal width or array of edges
    q
        - number of equal frequency              
    max_leaf_nodes
        - number of tree nodes using tree cut
        - if not None use supervised cutting based on decision tree
    **kwargs - Decision tree keyswords, egg:
        - min_impurity_decrease=0.001
        - random_state=0 
    .. note::
        -  only 1 of (q, bins, max_leaf_nodes) can be specified       
    labels
        - see pd.cut, if False return integer indicator of bins, 
        - if True return arrays of labels (or can be passed )
    header
        - title of plot
    xlabel
        - xlabel for xaxis
    '''

    y_cut, bins = binning(y_pre,
                           y_true=y_true,
                           bins=bins,
                           q=q,
                           max_leaf_nodes=max_leaf_nodes,
                           mono=mono,
                           labels=labels,
                           **kwargs)
    df0 = pd.DataFrame({'y_cut': y_cut, 'y_true': y_true})
    df_gb = df0.groupby('y_cut')
    df1 = pd.DataFrame()
    df1[xlabel] = df_gb.sum().index.values
    df1['rate'] = (df_gb.sum() / df_gb.count()).values
    df1['vol'] = df_gb.count().values
    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    plotted_data = df1.dropna()
    ax = plotter_rateVol(plotted_data, ax=ax)
    plt.title(header)
    return ax, y_cut, bins, plotted_data


def plotter_rate_volarea(s_rate,
                         s_vol,
                         area_color='g',
                         line_color='b',
                         linestyle='-',
                         numticks=3,
                         xlabel='',
                         ax=None,
                         show_mean=True):
    """
    plot rate as lineplot, vol as area plot on twinx()

    Parameters
    ----------
    s_rate : TYPE
        rate series.
    s_vol : TYPE
        vol series.
    area_color : TYPE, optional
        color fore area plot. The default is 'g'.
    line_color : TYPE, optional
        color for line. The default is 'b'.
    linestyle : TYPE, optional
        DESCRIPTION. The default is '-'.
    numticks : TYPE, optional
        number of x axis ticks. The default is 3.
    xlabel : TYPE, optional
        The default is ''.
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    show_mean : TYPE, optional
        whether to annotate weighted mean of rate & vol. The default is True.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.
    ax_tw : TYPE
        DESCRIPTION.

    """

    x = s_rate.index
    ax_tw = ax.twinx()
    ax.plot(x,
            s_rate,
            color=line_color,
            linestyle=linestyle,
            marker='o',
            lw=1.5)
    ax.yaxis.set_major_formatter(_get_ticks_formatter('percent'))
    ax.set_ylim(0, min(s_rate.max() + 20 * s_rate.std(), 1.1))

    # area_plot(x, s_vol, ax=ax_tw, color=area_color)
    ax_tw.fill_between(x, 0, s_vol, alpha=0.4, color=area_color)
    ax_tw.grid(False)
    ax_tw.set_ylim(0, s_vol.max() + 10 * s_rate.std())
    ax.grid(lw=0.6, ls='-')
    _autofmt_xdate(ax)
    if not xlabel:
        ax.set_xlabel(xlabel)

    if show_mean:
        bbox = dict(boxstyle="round", alpha=0.5, facecolor='w')
        ax.annotate('mean={}%'.format(round(s_rate.mean() * 100, 1)),
                    (0.05, s_rate.mean()), (0.05, 0.95),
                    xycoords='axes fraction',
                    textcoords='axes fraction',
                    bbox=bbox,
                    fontsize=8)
    return ax, ax_tw


def plotter_timeseries_dfwide_rate_vol(df_wide,
                                       okind='bar',
                                       palette='Set2',
                                       sharex=False,
                                       fig_title='',
                                       savefig=None):
    '''plot time series of rate and volume
    
    df_wide dataframe:
        wide form dataframe, index will be plotted as x axis,
        each rate column(0<col<1) will be plotted as y axis, comparing each 
        other
        
    okind str: ['area', 'bar']
        plotting kind for columns other than rate(col_value > 1), default 'bar'
    
    palette:
        default 'Set2', color set to plot with
        
    fig_title:
        title for fig
        
    savefig: str
        'path/file.pdf' to save fig
             
    '''
    df_wide = df_wide.copy()
    ocols = [
        df_wide.pop(name) for name, col in df_wide.iteritems() if any(col > 1)
    ]

    n_ocols = len(ocols)
    n = n_ocols + 1
    i, j = plt.rcParams['figure.figsize']
    fig, axes = plt.subplots(n,
                             1,
                             figsize=(i, j + 2.8 * (n // 2)),
                             sharex=sharex)
    fig.suptitle(fig_title)
    axes = np.array(axes)

    # plot rate columns
    sns.lineplot(data=df_wide,
                 ax=axes[0],
                 palette=palette,
                 markers=True,
                 markersize=6)
    fmt = ticker.PercentFormatter(xmax=1, decimals=1)
    axes[0].yaxis.set_major_formatter(fmt)
    axes[0].set_ylim((0, 1.2))
    axes[0].legend(ncol=8, loc=0)

    # plot other than rate columns
    if n_ocols > 0:
        sx = df_wide.index
        cm = sns.color_palette(palette)
        for ii in range(n - 1):
            axi = axes[ii + 1]
            if okind == 'area':
                area_plot(sx, ocols[ii], ax=axi, color=cm[ii])
            elif okind == 'bar':
                sns.barplot(sx, ocols[ii], ax=axi, color=cm[ii])
            labels = [i.get_text() for i in axi.xaxis.get_ticklabels()]
            if max([len(i) for i in labels]) > 8:
                if ii < n - 2:
                    axi.get_xaxis().set_ticks([])
                else:
                    axi.tick_params('x', labelrotation=35)

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    # save fig to savefig path
    if savefig is not None:
        _save_fig(fig, savefig)

    return fig


def area_plot(sx,
              sy,
              yformater=None,
              sbase=0,
              ax=None,
              legend=False,
              marker='o',
              linestyle='--',
              color='b',
              linewidth='0.5',
              markersize=1,
              markerfacecolor='w',
              fig_kws={}):
    '''
    sx series:
        to plot as x axis
    sy series:
        to plot as y axis
    sbase series:
        horizontal base line for area plot, default=0
        
    '''
    if ax is None:
        fig, ax = plt.subplots(1, 1, **fig_kws)

    sns.lineplot(sx,
                 sy,
                 ax=ax,
                 alpha=0.3,
                 marker=marker,
                 markersize=markersize,
                 linestyle=linestyle,
                 color=color,
                 markerfacecolor=markerfacecolor,
                 linewidth=linewidth,
                 legend=legend)

    ax.fill_between(sx, sbase, sy, alpha=0.4, color=color)
    if yformater is not None:
        fmt = _get_ticks_formatter(yformater)
        ax.yaxis.set_major_formatter(fmt)
    if pd.core.dtypes.api.is_datetime64_dtype(sx):
        plt.gcf().autofmt_xdate()

    return ax
