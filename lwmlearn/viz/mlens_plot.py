# -*- coding: utf-8 -*-
"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Correlation plots.
"""

from __future__ import division, print_function

import numpy as np
from scipy.stats import pearsonr

import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from seaborn import diverging_palette, heatmap
except ImportError:
    warnings.warn(
        "Matplotlib and Seaborn not installed. Cannot load "
        "visualization module.", ImportWarning)

try:
    from pandas import DataFrame, Series
except ImportError:
    DataFrame = Series = None
    warnings.warn(
        "Pandas not installed. Visualization module may not work "
        "as intended.", ImportWarning)

try:
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import KernelPCA
    from matplotlib.colors import ListedColormap
    from seaborn import color_palette
except:
    warnings.warn(
        "Matplotlib and Seaborn not installed. Cannot load "
        "visualization module.", ImportWarning)


def corrmat(corr,
            figsize=(11, 9),
            annotate=True,
            inflate=True,
            linewidths=.5,
            cbar_kws='default',
            show=True,
            ax=None,
            title='Correlation Matrix',
            title_font_size=14,
            **kwargs):
    """Function for generating color-coded correlation triangle.

    Parameters
    ----------
    corr : array-like of shape = [n_features, n_features]
        Input correlation matrix. Pass a pandas ``DataFrame`` for axis labels.

    figsize : tuple (default = (11, 9))
        Size of printed figure.

    annotate : bool (default = True)
        Whether to print the correlation coefficients.

    inflate : bool (default = True)
        Whether to inflate correlation coefficients to a 0-100 scale.
        Avoids decimal points in the figure, which often appears very cluttered
        otherwise.

    linewidths : float
        with of line separating each coordinate square.

    cbar_kws : dict, str (default = 'default')
        Optional arguments to color bar. The default options, 'default',
        passes the ``shrink`` parameter to fit colorbar standard figure frame.

    show : bool (default = True)
        whether to print figure using :obj:`matplotlib.pyplot.show`.

    title : str
        figure title if shown.

    title_font_size : int
        title font size.

    ax : object, optional
        axis to attach plot to.

    **kwargs : optional
        Other optional arguments to sns heatmap.

    Returns
    -------
    ax : object
        axis object.

    See Also
    --------
    :class:`mlens.visualization.clustered_corrmap`
    """
    if inflate:
        corr *= 100
        fmt = '2.0f'
        v_max = 100.0
        v_min = -100.0
    else:
        fmt = '.2f'
        v_max = 1.0
        v_min = -1.0

    if cbar_kws == "default":
        cbar_kws = {"shrink": 1.0}

    # Determine annotation
    do_annot = {True: corr, False: None}
    annot = do_annot[annotate]

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = diverging_palette(230, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    ax = heatmap(corr,
                 mask=mask,
                 cmap=cmap,
                 vmin=v_min,
                 annot=annot,
                 fmt=fmt,
                 vmax=v_max,
                 square=True,
                 linewidths=linewidths,
                 cbar_kws=cbar_kws,
                 ax=ax,
                 **kwargs)
    plt.title(title, fontsize=title_font_size)

    return ax


def clustered_corrmap(corr,
                      cls,
                      label_attr_name='labels_',
                      figsize=(10, 8),
                      annotate=True,
                      inflate=True,
                      linewidths=.5,
                      cbar_kws='default',
                      show=True,
                      title_fontsize=14,
                      title_name='Clustered correlation heatmap',
                      ax=None,
                      **kwargs):
    """Function for plotting a clustered correlation heatmap.

    Parameters
    ----------
    corr : array-like of shape = [n_features, n_features]
        Input correlation matrix. Pass a pandas ``DataFrame`` for axis labels.

    cls : instance
        cluster estimator with a ``fit`` method and cluster labels stored as an
        attribute as specified by the ``label_attr_name`` parameter.

    label_attr_name : str
        name of attribute that contains cluster labels.

    figsize : tuple (default = (10, 8))
        Size of figure.

    annotate : bool (default = True)
        Whether to print the correlation coefficients.

    inflate : bool (default = True)
        Whether to inflate correlation coefficients to a 0-100 scale.
        Avoids decimal points in the figure, which often appears very cluttered
        otherwise.

    linewidths : float (default = .5)
        with of line separating each coordinate square.

    cbar_kws : dict, str (default = 'default')
        Optional arguments to color bar.

    title_name : str
        Figure title.

    title_fontsize : int
        size of title.

    show : bool (default = True)
        whether to print figure using :obj:`matplotlib.pyplot.show`.

    ax : object, optional
        axis to attach plot to.

    **kwargs : optional
        Other optional arguments to sns heatmap.

    See Also
    --------
    :class:`mlens.visualization.corrmat`
    """
    # find closely associated features
    cls.fit(corr)

    # Sort features on cluster membership
    if corr.__class__.__name__ == 'DataFrame':
        columns_names = corr.columns.tolist()
    else:
        columns_names = [i for i in range(corr.shape[1])]

    corr_list = [
        tup[0]
        for tup in sorted(zip(columns_names, getattr(cls, label_attr_name)),
                          key=lambda x: x[1])
    ]

    if corr.__class__.__name__ == 'DataFrame':
        corr = corr.loc[corr_list, corr_list]
    else:
        corr = corr[np.ix_(corr_list, corr_list)]

    # Prepare figure
    if inflate:
        corr *= 100
        fmt = '2.0f'
        v_max = 100.0
        v_min = -100.0
    else:
        fmt = '.2f'
        v_max = 1.0
        v_min = -1.0

    if cbar_kws == "default":
        cbar_kws = {"shrink": 1.0}

    # Determine annotation
    do_annot = {True: corr, False: None}
    annot = do_annot[annotate]

    # Generate a custom diverging colormap
    cmap = diverging_palette(240, 10, as_cmap=True)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax = heatmap(corr,
                 cmap=cmap,
                 vmin=v_min,
                 annot=annot,
                 fmt=fmt,
                 vmax=v_max,
                 square=True,
                 linewidths=linewidths,
                 cbar_kws=cbar_kws,
                 ax=ax,
                 **kwargs)
    plt.title(title_name, fontsize=title_fontsize)

    return ax


def corr_X_y(X,
             y,
             top=5,
             figsize=(10, 8),
             fontsize=12,
             hspace=None,
             no_ticks=True,
             label_rotation=0,
             show=True):
    """Function for plotting input feature correlations with output.

    Output figure shows all correlations as well as top pos and neg.

    Parameters
    ----------
    X : pandas DataFrame Input data.

    y : class label.

    top : int
        number of features to show in top pos and neg graphs.

    figsize : tuple (default = (10, 8))
        Size of figure.

    hspace : float, optional
        whitespace between top row of figures and bottom figure.

    fontsize : int
        font size of subplot titles.

    no_ticks : bool (default = False)
        whether to remove ticklabels from full correlation plot.

    label_rotation: float (default = 0)
        rotation of labels

    show : bool (default = True)
        whether to print figure using :obj:`matplotlib.pyplot.show`.

    Returns
    -------
    ax : object
        axis object.
    """
    if not X.__class__.__name__ == 'DataFrame':
        raise ValueError("Expected 'X' to be pandas DataFrame.")

    # Prep pairwise correlations
    corr = X.corr()[y]
    corr = corr.drop(y).sort_values(ascending=False)

    # Check that top selections will not be greater than all features
    n = len(corr)
    if top > n:
        top = n

    # Render figure
    names = corr.index.tolist()
    if hspace is None:
        hspace = 2 * fontsize / 100
    if label_rotation > abs(45):
        hspace += max([len(i) for i in names]) / 35 * (fontsize / 10)

    plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, hspace=hspace)

    # Axes
    ax0 = plt.subplot(gs[0, 0])
    ax0.bar(range(top), corr.iloc[:top], align='center')
    ax0.axhline(0, color='black', linewidth=0.5)
    ax0.set_title('Top %i positive pairwise correlation coefficients' % top,
                  fontsize=fontsize)
    plt.xticks(range(top),
               names[:top],
               rotation=label_rotation,
               fontsize=fontsize - 1)

    ax1 = plt.subplot(gs[0, 1])
    ax1.bar(range(top), corr.iloc[-top:], align='center')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_title('Top %i negative pairwise correlation coefficients' % top,
                  fontsize=fontsize)
    plt.xticks(range(top),
               names[-top:],
               rotation=label_rotation,
               fontsize=fontsize - 1)

    ax2 = plt.subplot(gs[1, :])
    ax2.bar(range(len(corr)), corr, align='center')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_title('All pairwise correlation coefficients', fontsize=fontsize)

    if no_ticks:
        ax2.set_xticks([], [])
    else:
        plt.xticks(range(len(names)),
                   names,
                   rotation=label_rotation,
                   fontsize=fontsize - 1)

    return gs


def pca_comp_plot(X,
                  y=None,
                  figsize=(10, 8),
                  title='Principal Components Comparison',
                  title_font_size=14,
                  show=True,
                  **kwargs):
    """Function for comparing PCA analysis.

    Function compares across 2 and 3 dimensions and
    linear and rbf kernels.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        input matrix to be used for prediction.

    y : array-like of shape = [n_samples, ] or None (default = None)
        training labels to be used for color highlighting.

    figsize : tuple (default = (10, 8))
        Size of figure.

    title : str
        figure title if shown.

    title_font_size : int
        title font size.

    show : bool (default = True)
        whether to print figure :obj:`matplotlib.pyplot.show`.

    **kwargs : optional
        optional arguments to pass to :class:`mlens.visualization.pca_plot`.

    Returns
    -------
    ax :
        axis object.

    See Also
    --------
    :class:`mlens.visualization.pca_plot`
    """
    comp = ['linear', 'rbf']
    f = plt.figure(figsize=figsize)

    ax = []
    subplot_kwarg = {}

    for dim, frame in [(2, 221), (3, 223)]:

        if dim is 3:
            # Need to specify projection
            subplot_kwarg = {'projection': '3d'}

        for i, kernel in enumerate(comp):
            # Create subplot
            ax.append(f.add_subplot(frame + i, **subplot_kwarg))

            # Build figure
            ax[-1] = pca_plot(X,
                              KernelPCA(dim, kernel),
                              y,
                              ax=ax[-1],
                              show=False,
                              **kwargs)

            ax[-1].set_title('%s kernel, %i dims' % (kernel, dim))

            # Whiten background if dim is 3
            if dim is 3:
                ax[-1].set_facecolor((1, 1, 1))

    f.suptitle(title, fontsize=title_font_size)

    return ax


def pca_plot(X,
             estimator,
             y=None,
             cmap=None,
             figsize=(10, 8),
             title='Principal Components Analysis',
             title_font_size=14,
             show=True,
             ax=None,
             **kwargs):
    """Function to plot a PCA analysis of 1, 2, or 3 dims.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        matrix to perform PCA analysis on.

    estimator : instance
        PCA estimator. Assumes a Scikit-learn API.

    y : array-like of shape = [n_samples, ] or None (default = None)
        training labels to be used for color highlighting.

    cmap : object, optional
        cmap object to pass to :obj:`matplotlib.pyplot.scatter`.

    figsize : tuple (default = (10, 8))
        Size of figure.

    title : str
        figure title if shown.

    title_font_size : int
        title font size.

    show : bool (default = True)
        whether to print figure :obj:`matplotlib.pyplot.show`.

    ax : object, optional
        axis to attach plot to.

    **kwargs : optional
        arguments to pass to :obj:`matplotlib.pyplot.scatter`.

    Returns
    -------
    ax : optional
        if ``ax`` was specified, returns ``ax`` with plot attached.
    """
    Z = X.values if isinstance(X, (DataFrame, Series)) else X

    Z = estimator.fit_transform(Z)

    # prep figure if no axis supplied
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
        if estimator.n_components == 3:
            ax = f.add_subplot(111, projection='3d')

    if cmap is None:
        cmap = ListedColormap(color_palette('husl'))

    if estimator.n_components == 1:
        ax.scatter(Z, c=y, cmap=cmap, **kwargs)
    elif estimator.n_components == 2:
        ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap=cmap, **kwargs)
    elif estimator.n_components == 3:
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=y, cmap=cmap, **kwargs)
    else:
        raise ValueError("'n_components' is too large to visualize. "
                         "Set to one of [1, 2, 3].")

    plt.title(title, fontsize=title_font_size)

    return ax


def exp_var_plot(X,
                 estimator,
                 figsize=(10, 8),
                 buffer=0.01,
                 set_labels=True,
                 title='Explained variance ratio',
                 title_font_size=14,
                 show=True,
                 ax=None,
                 **kwargs):
    """Function to plot the explained variance using PCA.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        input matrix to be used for prediction.

    estimator : class
        PCA estimator, not initiated, assumes a Scikit-learn API.

    figsize : tuple (default = (10, 8))
        Size of figure.

    buffer : float (default = 0.01)
        For creating a buffer around the edges of the graph. The buffer
        added is calculated as ``num_components`` * ``buffer``,
        where ``num_components`` determine the length of the x-axis.

    set_labels : bool
        whether to set axis labels.

    title : str
        figure title if shown.

    title_font_size : int
        title font size.

    show : bool (default = True)
        whether to print figure using :obj:`matplotlib.pyplot.show`.

    ax : object, optional
        axis to attach plot to.

    **kwargs : optional
        optional arguments passed to the :obj:`matplotlib.pyplot.step`
        function.

    Returns
    -------
    ax : optional
        if ``ax`` was specified, returns ``ax`` with plot attached.
    """
    estimator.set_params(**{'n_components': None})
    ind_var_exp = estimator.fit(X).explained_variance_ratio_
    cum_var_exp = np.cumsum(ind_var_exp)
    x = range(1, len(ind_var_exp) + 1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.step(x, cum_var_exp, **kwargs)

    buffer_ = buffer * len(ind_var_exp)
    ax.set_ylim(0, 1 + buffer_ / 3)
    ax.set_xlim(1 - buffer_, len(ind_var_exp) + buffer_)
    ax.set_xticks([i for i in x])

    if set_labels:
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Explained variance ratio')

    plt.title(title, fontsize=title_font_size)
    return ax
