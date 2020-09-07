# -*- coding: utf-8 -*-
"""
`lw_model_proxy` module

offers a `LW_model` class that provides common methods for ML proccess

Created on Wed Dec 11 18:39:56 2019

@author: Rogerluo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import traceback

from functools import wraps
from shutil import rmtree
from copy import deepcopy

from imblearn.pipeline import Pipeline
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, cross_validate)
from sklearn.base import BaseEstimator
from sklearn.model_selection import _validation
from sklearn.metrics import roc_curve, auc
from sklearn.utils import validation

from skopt import BayesSearchCV

from lwmlearn.lwmodel.auto_list import get_default_models
from lwmlearn.utilis.utilis import get_flat_list, get_kwargs
from lwmlearn.utilis.read_write import Objs_management
from lwmlearn.utilis.docstring import Appender, dedent

from lwmlearn.viz.plotter import (plotter_auc, plotter_cv_results_,
                                  plotter_score_path, plotter_lift_curve)
from lwmlearn.hypertune.gridspace import pipe_grid
from lwmlearn.utilis.binning import binning
from lwmlearn.utilis.utilis import to_num_datetime_df
from lwmlearn.utilis.lw_model_proxy_utlis import (split_cv,
                                                  get_splits_combined)
from lwmlearn.utilis.sklearn_score import get_custom_scorer
from lwmlearn.lwmodel.operators_pool import pipe_main, get_featurenames


@dedent
def run_modellist(
        X,
        y,
        test_set=None,
        model_list=None,
        dirs='auto',
        scoring=['roc_auc', 'KS', 'average_precision', 'neg_log_loss'],
        verbose=1,
        on_error='raise',
        **kwargs):
    '''
    run a series of pre-defined models as returned by get_default_models(),
    list of models can be specified by keyword 'model_list', return evaluation
    table for each model, and dict of models instances
        
    Parameters
    ----------
    X
        feature matrix
    y
        classifiction calss label 0/1
        
    test_set:
        (X_test, y_test) tuple to use as test set, if None, no valuation
        will be performed on testset.
           
    dirs - str:
        directory to dump analyze models, default='auto'
        
    model_list - list:
        list of  string represented stepped pipeline model
        
    scoring (list):
        scoring metrics, 
        like ['roc_auc', 'KS', 'average_precision', 'neg_log_los'] where 'KS' 
        is a custom scorer
            
    on_error - ['raise', 'ignore'], default 'ignore'
        - if 'raise', raise error
        - if 'ignore', continue iteration of models, output error to 
        'error_log.txt'
        
    **Kwargs
    ----------------
    see LW_model.run_analysis() method 
    
    is_search bool: default=True
        if True, predefined param_grid will be tuned automatically
        if False, run default hyper parameters without tuning
        
    out_searchstep bool:
        if true cv results of each search step will be output in gridcvtab
        default=False
    
    cv (int):
       n of cross validation folds, cv should be greater thant 1
       
    kind str: ['gridcv', 'bayesiancv', 'randomcv'], search method to use
    
    default is 'bayesiancv'
        - if kind == 'bayesiancv' use bayesian optimization
        
        - if kind == 'gridcv' sequentially update the best parameter 
        settings in each dict of param_grid by grid search
        
        - if kind == 'randomcv' sequentially update the best parameter 
        settings in each dict of param_grid by random grid search    
        
    Plot lift curve
    ------------------
    q
        - number of equal frequency 
    bins
        - number of equal width or array of edges
    max_leaf_nodes
        - if not None perform supervised cutting, 
        - number of tree nodes using tree cut        
    mono 
        - binning edges that increases monotonically with "y" mean value
        
    .. note::
        -  only 1 of (q, bins, max_leaf_nodes, mono) can be specified 
        if all None, no lift curve plot
        
    return 
    ------------------
    autocv_score: df
        cross validation score for both trainset & testset
    
    models: dict
        dict of model instances evaluated
    '''
    # --
    if model_list is None:
        l = get_default_models()
    else:
        l = model_list

    score_list = []
    models = {}

    for i in l:
        try:
            path = os.path.join(dirs, i)
            model = LW_model(i, path, verbose=verbose)
            model.run_analysis(train_set=(X, y),
                               test_set=test_set,
                               scoring=scoring,
                               **kwargs)
            print("\n '{}' complete \n".format(i))
            # --
            score_metrics = pd.Series()
            score_metrics['Model_represent'] = i

            train = pd.Series(model.kws_attr.get('trainscore'))
            train = train.rename(
                index=lambda x: x.replace('test_', '') + '_Trainset')
            test = pd.Series(model.kws_attr.get('testscore'))
            test = test.rename(
                index=lambda x: x.replace('test_', '') + '_Testset')

            score_metrics = score_metrics.append(train)
            score_metrics = score_metrics.append(test)
            score_list.append(score_metrics)
            models.update({i: model})
        except Exception as e:
            if on_error == 'raise':
                raise e
            else:
                traceback.print_exc()
                traceback.print_exc(file=open('error_log.txt', 'a+'))

    if len(score_list) > 0:
        autocv_score = pd.concat(score_list, axis=1, ignore_index=True).T
        autocv_score = to_num_datetime_df(autocv_score)
        autocv_score.to_excel(os.path.join(dirs, 'autocv_score.xlsx'),
                              index=False)

    return autocv_score, models


class LW_model(BaseEstimator):
    ''' an meta-estimator for quantifying predictions and offers many common
    methods for ML 

    Parameters
    ----------
    estimator (str)
        - operator 'str' seperated by '_', like 'XXX_XXX_XXX'
        - 'XXX' represents instance name of sklearn estimator or predefined
        estimator operator name
        - default='LogisticRegression'
    path
        - dir to place model and other files, default 'model' under current 
        directory
    seed
        - random state seed, 0 default
    pos_label
            - positive label default 1
       
    
    attributes
    ----------
    estimator
        - model instance, usually a pipeline
        - has attributes
            bins(bin edges cut of predictions of estimator)
            
    modelsteps
        - return estimator's operator str representation 'xxx_xxx_xxx'
    
    featurenames
        - series of features that actually used as model input     
   
    loaddump
        - read_write object to load/dump datafiles from self.path
    
    kws_attr attributes
    -------------------  

    NInputFeatures
        - number of features for an fitted estimator
   
    NSamples
        - number of samples for an fitted estimator
        
    ycounts
        - class label counts for an fitted estimator
        
    gridcvtab (list of df) 
        - cv_results after running searchcv
        
    testscore (Series)
        - averaged score for test set returned by run_anlysis
        
    trainscore (Series)
        - averaged score for train set returned by run_anlysis
        
    optional attributes
    -------------------
    autocv_score df
        - cv_score after running auto_run for both training and test dataset  
    automodels dict
        - dict of model instances used in run_autoML
    _plot_all_auc_data
        - data to plot auc comarison curves for fitted estimators for auto_run
    
    method
    ---------
    load:
        load pickled LW_model instance
    fit:
        perform fit of estimator
    predict:
        perform predict of estimator
    predict_proba:
        perform predict_proba of estimator
    opt_sequential
        grid_search of parameter grid space sequentially iterating hyper-param
    run_train
        run CV for train set, return cv score of multiple metrics
    run_test
        run cv splitted test for test set, return score of multiple metrics
    run_analysis
        run gridsearch and trainset and testset together
    run_autoML:
        update self.estimator as best model of predefined pipeline
            
    cv_score:
        return cross score of estimator
    cv_validate:
        return cross score of estimator, allowing multi scorers
    test_score:
        return test_score of estimator, given X_test, y_test data, support
        splitting data into groups
    searchcv:
        perform grid search of param_grid, update self esimator estimator,
        support 3 search method kind=['bayesiancv', 'gridcv', 'randomcv']
      
    plot_auc:
        plot auc of test data
    plot_auc_traincv:
        plot auc of train data
    plot_lift:
        plot lift curve of model
    plot_gridcv:
        plot  grid seach cv results of model
    '''
    @staticmethod
    def load(lwpkl, path='model'):
        ''' load 'LW_model' istance and set self.path to 'path'
        
        Parameters
        ----
        lwpkl:
           LW_model pickled file
        path:
            directory to for model output
        
        return
        ----
            model instance
        '''
        import pickle
        with open(lwpkl, 'rb') as f:
            pkl = pickle.Unpickler(f)
            model = pkl.load()
        model.path = path
        return model

    def __init__(self,
                 estimator='LogisticRegression',
                 path='model',
                 seed=0,
                 verbose=1,
                 pos_label=1,
                 kws_attr={}):
        '''             
        '''
        self.path = path
        self.verbose = verbose
        self.pos_label = pos_label
        self.seed = seed
        self.set_estimator(estimator)
        self.kws_attr = kws_attr

    def _shut_temp_loaddump(self):
        '''shut temp loaddump directory
        '''
        if getattr(self.estimator, 'memory') is not None:
            while os.path.exists(self.estimator.memory):
                rmtree(self.estimator.memory, ignore_errors=True)

            print('%s has been removed' % self.estimator.memory)
            self.estimator.memory = None

    def _check_fitted(self, estimator):
        '''check if estimator has been fitted
        '''
        validation.check_is_fitted(
            estimator,
            ['classes_', 'coef_', 'feature_importances_', 'booster', 'tree_'],
            all_or_any=any)

    def _pre_continueous(self, estimator, X, **kwargs):
        '''make continueous predictions
        '''
        classes_ = getattr(estimator, 'classes_')
        if len(classes_) > 2:
            raise ValueError(' estimator should only output binary classes...')

        for i in ['decision_function', 'predict_proba']:
            if hasattr(estimator, i):
                pre_func = getattr(estimator, i)
                break

        if pre_func is None:
            raise ValueError('estimator have no continuous predictions')

        y_pre = pre_func(X, **kwargs)
        if np.ndim(y_pre) > 1:
            y_pre = y_pre[:, self.pos_label]
        return y_pre

    def _get_automodels(self, suffix='.lwpkl'):
        '''return dict of automodels 
        '''
        d = self.loaddump.read_all(suffix, subfolder=True)

        return {model.modelsteps: model for model in d.values()}

    def _get_scorer(self, scoring):
        ''' return sklearn scorer, including custom scorer
        '''
        scorer = {}
        sk_scoring = []
        custom_scorer = get_custom_scorer()
        scoring = get_flat_list(scoring)
        for i in scoring:
            if i in custom_scorer:
                scorer.update({i: custom_scorer[i]})
            else:
                sk_scoring.append(i)
        if len(sk_scoring) > 0:
            s, _ = _validation._check_multimetric_scoring(self.estimator,
                                                          scoring=sk_scoring)
            scorer.update(s)
        return scorer

    def _cal_fprs_tprs(self,
                       model,
                       X,
                       y,
                       cv,
                       groups=None,
                       refit=False,
                       **fit_params):
        '''return array of fpr, tpr, roc_auc value for given X and  yy (y_true) 
        on a series of threshold
        '''
        estimator = deepcopy(model)
        # split test set by cv
        data_splits = tuple(
            split_cv(X, y=y, cv=cv, groups=groups, random_state=self.seed))
        self._check_fitted(estimator)
        fprs = []
        tprs = []
        aucs = []
        for x_set, y_set in data_splits:
            xx0 = x_set[0]
            yy0 = y_set[0]
            xx = x_set[1]
            yy = y_set[1]
            if refit:
                estimator.fit(xx0, yy0)
            y_pre = self._pre_continueous(model, xx)
            fpr, tpr, threshhold = roc_curve(yy, y_pre, drop_intermediate=True)
            roc_auc = auc(fpr, tpr)
            fprs.append(fpr)
            tprs.append(tpr)
            aucs.append(roc_auc)

        return fprs, tprs, aucs, data_splits

    def _get_gridspace(self, kind):
        '''return predefined gridspace
        '''
        param_grid = []
        for k, v in self.estimator.named_steps.items():
            grid = pipe_grid(k, search_type=kind)
            if grid is not None:
                param_grid.extend(grid)
        return param_grid

    def _combine_gridspace(self, param_grid):
        '''make sequential search into one step search
        
        return a list of one single dict element so that search will finish 
        in one step
        '''
        d = {}
        for i in param_grid:
            d.update(i)
        print('param_grid has been combined into one space: \n {}'.format(d))
        return [d]

    def _update_bestmodel(self, by_metrics='roc_auc_Trainset', **kwargs):
        '''update model's estimator by the best model after running 
        run_autoML, 'by_metrics' is used to select best model
        '''
        argmax = self.autocv_score[by_metrics].argmax()
        best_model = self.autocv_score.loc[argmax, 'Model_represent']
        params = self.automodels[best_model].get_params()
        params.pop('path')
        self.set_params(**params)
        return self

    def _savefig(self,
                 save_fig,
                 fig=None,
                 default_saved_name=None,
                 closefig=True):
        '''
        save current fig to designated cwd/folder_file
        '''

        if save_fig is False:
            return

        if fig is None:
            fig = plt.gcf()

        if save_fig is True:
            if default_saved_name is None:
                raise ValueError('default_saved_name must be given')
            else:
                save_fig = default_saved_name
        self.loaddump.write(fig, save_fig)
        if closefig:
            plt.close(fig)
        return

    def _get_estimator_name(self):
        '''
        return name of estimator
        '''
        estimator = self.estimator
        if hasattr(estimator, '_final_estimator'):
            estimator = estimator._final_estimator
        if hasattr(estimator, '__class__'):
            return estimator.__class__.__name__.replace('Classifier', '')
        else:
            raise TypeError('estimator is not an valid sklearn estimator')

    def _update_fitted_estimator(self, fitted_estimator, X, y):
        """
        update fitted estimator and its input X, y info

        Parameters
        ----------
        fitted_estimator : TYPE
            DESCRIPTION.
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.kws_attr.update(
            NSamples=np.shape(X)[0],
            NInputFeatures=np.shape(X)[1],
            ycounts=pd.value_counts(y),
            input_featurenames=pd.DataFrame(X).columns.tolist())
        self.estimator = fitted_estimator
        return self

    def _get_basic_info(self,
                        out_str=False,
                        prepend='',
                        append='',
                        x_test=None,
                        y_test=None):
        """
        return basic info of a fitted model

        Parameters
        ----------
        out_str : TYPE, optional
            DESCRIPTION. The default is False.
        prepend : TYPE, optional
            str to prepend header_str. The default is ''.
        append : TYPE, optional
            str to append header_str. The default is ''.
        x_test : TYPE, optional
            if not none, to return testset shape info. The default is None.
        y_test : TYPE, optional
            if not none, to return testset shape info. The default is None.

        Returns
        -------
        if out_str=False:
            - NSamples
            - NInputFeatures
            - NUsedFeatures
            - classifer_name
        
        if  out_str=True:
            return header_str

        """

        self._check_fitted(self.estimator)
        if x_test is None:
            NSamples = self.kws_attr.get('NSamples')
            NInputFeatures = self.kws_attr.get('NInputFeatures')
        else:
            NSamples, NInputFeatures = np.shape(x_test)

        if self.featurenames is not None:
            NUsedFeatures = len(self.featurenames)
        else:
            NUsedFeatures = NInputFeatures

        classifier = self._get_estimator_name()
        header_str = '{} samples; {}/{} (used/all)features; {}'.format(
            NSamples, NUsedFeatures, NInputFeatures, classifier)

        if out_str:
            lst = [i for i in [prepend, header_str, append] if len(i) > 0]
            return '; '.join(lst)
        else:
            return NSamples, NInputFeatures, NUsedFeatures, classifier

    def _plot_all_auc_data(self, X, y, suffix='.lwpkl', **kwargs):
        '''update self._plotdata_all_auc data
        '''
        d = self.loaddump.read_all(suffix, subfolder=True)
        fprs = []
        tprs = []
        aucs = []
        names = []
        for i in d.values():

            fpr, tpr, roc_auc, _ = self._cal_fprs_tprs(i.estimator,
                                                       X,
                                                       y,
                                                       cv=1,
                                                       refit=False)
            names.append(i.modelsteps)
            tprs.extend(tpr)
            fprs.extend(fpr)
            aucs.extend(roc_auc)

        n_sample = len(X)
        n_feature = np.shape(X)[-1]
        self._plotdata_all_auc = (fprs, tprs, names, n_sample, n_feature)
        return self

    @property
    def loaddump(self):
        ''' return obj_management instance to load or dump object
        '''
        return Objs_management(self.path)

    @property
    def modelsteps(self):
        '''return string representing steps of self.estimator
        '''
        if hasattr(self.estimator, 'steps'):
            return '_'.join([step[0] for step in self.estimator.steps])
        elif hasattr(self.estimator, '_estimator_type'):
            return self.estimator.__class__.__name__
        else:
            raise NameError('no estimator input')

    def set_estimator(self, estimator):
        '''update self.estimator, input should be name of estimator or an 
        instance
        '''
        if isinstance(estimator, str):
            model = pipe_main(estimator)
        elif hasattr(estimator, '_estimator_type'):
            model = estimator
        else:
            raise ValueError('invalid estimator input type: {}'.format(
                estimator.__class__.__name__))

        self.estimator = model

    def plot_auc(self,
                 X,
                 y,
                 cv=3,
                 groups=None,
                 title=None,
                 ax=None,
                 save_fig=False,
                 fit_train=False):
        '''plot roc_auc curve for given fitted estimator(must have continuous
        predicton, like decision_function or predict_proba) to evaluate model. 
        single pair of X, y can be splited into folds using cv > 1 to assess 
        locality of data, cross validated auc for traindata could be plotted 
        by fit_train=True

        X
            -2D array or list of 2D ndarrays
        y
            -binary or list of class label
        cv 
            -int, cross-validation generator or an iterable
            - if cv>1, generate splits by StratifyKfold method
        title
            - title added to plot header, default ''
            
        fit_train: bool
            if True, refit estimator using other folds data than current k fold
            each time;fit current estimator using entire (X, y)
            if False, only test a fitted estimator
        
        save_fig bool:
            if True, save plotted fig to default pdf file
            if Falsee, do not save fig
            if file_dir, save fig to file_dir
            
        return
        --------
        - ax
        - mean-auc
        - std-auc       
        - data_splits:
            list of test data set in the form of DataFrame (combined X & y)
        '''

        estimator = self.estimator
        if fit_train:
            self.fit(X, y)
        # split test set by cv
        fprs, tprs, aucs, data_splits = self._cal_fprs_tprs(estimator,
                                                            X,
                                                            y,
                                                            cv,
                                                            groups,
                                                            refit=fit_train)

        # -- plot
        # title info
        title = '' if title is None else title
        header = self._get_basic_info(True, title, x_test=X)
        ax = plotter_auc(fprs, tprs, ax=ax, title=header)
        # -- save plot
        self._savefig(save_fig, default_saved_name='plots/auc.pdf')

        rst = (ax, np.mean(aucs), np.std(aucs),
               get_splits_combined(data_splits))
        return rst

    def plot_lift(
            self,
            X,
            y,
            q=None,
            bins=None,
            max_leaf_nodes=None,
            mono=None,
            use_self_bins=False,
            labels=False,
            ax=None,
            title=None,
            save_fig=False,
            tree_kws={},
    ):
        '''plot list curve of (X, y) data, update self bins
        
            given bins(n equal width) or q( n equal frequency) or 
            max_leaf_nodes cut by tree or monotonically cut
        X :
            -2D array
        y :
            -binary class labels , 1D array
            
        bins:
            - number of equal width or array of edges
            
        q:
            - number of equal frequency    
            
        max_leaf_nodes:
            - number of tree nodes using tree cut
            - if not None use supervised cutting based on decision tree
            
        mono :
            - binning edges that increases monotonically with "y" mean value

        use_self_bins:
            - use self.estimator.bins if true
            
        .. note::
            -  only 1 of (q, bins, max_leaf_nodes, mono) can be specified 
            
        labels bool:
            - see pd.cut, if False return integer indicator of bins, 
            - if True return arrays of labels (or can be manually input)
            
        tree_kws dict:
            - Decision tree keyswords, egg:
            - min_impurity_decrease=0.001
            - random_state=0 
            
        title str :
            - title of plot, output format: 'title' + estimator's name +
            NSamples + n_features, default '', usually for 'Train' or 
            'Test' data info indication.

        save_fig bool:
            if True, save plotted fig to default pdf file
            if Falsee, do not save fig
            if file_dir, save fig to file_dir
            
        return
        -------
        ax,  plotted_data;
        '''
        estimator = self.estimator
        self._check_fitted(estimator)
        y_pre = self._pre_continueous(estimator, X)
        y = np.array(y)

        # use_self_bins stored during training
        if use_self_bins is True:
            if getattr(self.estimator, 'bins') is not None:
                bins = self.estimator.bins
                q = None
                max_leaf_nodes = None
                mono = None
            else:
                print("'self.estimator.bins' is None, no lift curve plot \n")
                return
        # title info
        title = '' if title is None else title
        header = self._get_basic_info(True, title, x_test=X)
        # call plot function
        ax, y_cut, bins, plotted_data = plotter_lift_curve(
            y_pre,
            y_true=y,
            bins=bins,
            q=q,
            header=header,
            max_leaf_nodes=max_leaf_nodes,
            mono=mono,
            labels=labels,
            ax=ax,
            xlabel='ModelScore',
            **tree_kws)

        # update self bins
        self.estimator.bins = bins
        # save fig
        self._savefig(save_fig, default_saved_name='plots/lift.pdf')
        return ax, plotted_data

    def plot_all_auc(self, use_selfdata, save_fig=True, **kwargs):
        '''plot roc_auc curves of all model instance under path directory to
        compare model performance by auc metrics, update self._plotdata_all_auc
        
        parameters
        ----------
        
        use_selfdata bool: 
            default True
            
            if True, used stored self._plotdata_all_auc to plot all auc 
            comparisons;
            
            if False, use input X, y to calculate data, 
            update self._plotdata_all_auc, then plot all auc comparisons
        
        **kwargs
        ---------
        X
            -2D array
        y
            -binary class label , 1D array
            
        save_fig bool or str:
            if False do not save fig,
            if True or str, save fig to 'plots' folder giving  a filename as 
            'savae_fig'
        '''

        if not use_selfdata:
            # update self._plotdata_all_auc
            self._plot_all_auc_data(**kwargs)
        # -- plot
        fprs, tprs, names, n_sample, n_feature = self._plotdata_all_auc
        n = len(fprs)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4 + (n - 6) / 9))
        ax1 = plotter_auc(fprs,
                          tprs,
                          ax=ax1,
                          plot_mean=False,
                          curve_label=names)
        ax1.set_title('{} samples; {} input features'.format(
            n_sample, n_feature))

        h, l = ax1.get_legend_handles_labels()
        ax1.legend([])
        ax2.legend(h, l, bbox_to_anchor=(0, 1), loc='upper left')
        ax2.axis('off')
        fig.tight_layout()

        # save fig
        self._savefig(save_fig, default_saved_name='plots/roc_all.pdf')

        return

    def plot_AucLift(self,
                     X,
                     y,
                     fit_train=False,
                     title=None,
                     save_fig=False,
                     **kwargs):
        '''plot AUC curve and Lift curve together
        '''
        # plot roc_auc
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        title = title if title is not None else ''
        train_auc = self.plot_auc(X,
                                  y,
                                  fit_train=fit_train,
                                  ax=ax1,
                                  title=title,
                                  **get_kwargs(self.plot_auc, **kwargs))
        use_self_bins = False if fit_train else True
        lift_data = self.plot_lift(X,
                                   y,
                                   ax=ax2,
                                   title=title,
                                   use_self_bins=use_self_bins,
                                   **get_kwargs(self.plot_lift, **kwargs))

        # save fig
        self._savefig(save_fig, default_saved_name='plots/AucLift.pdf')

        return train_auc, lift_data

    def plot_cvscore(self,
                     X,
                     y,
                     is_train,
                     scoring=['roc_auc', 'KS'],
                     fit_params={},
                     cv=3,
                     save_fig=False,
                     title=None):
        '''
        calculate cv metrics scores and plot metrics path as lineplot

        Parameters
        ----------
        X : 2D array or DataFrame
            features.
        y : 1d array, or Series
            class label.
        is_train : bool
            if True, refit model using other folds each time.
        scoring : list of strings, optional
            metrics to use. The default is ['roc_auc', 'KS'].
        fit_params : dict, optional
            key word argument passed to fit method. The default is {}.
        cv : int, optional
            number of folds to use. The default is 3.

        Returns
        -------
        DataFrame: 
             cv score

        '''
        L = locals().copy()
        L.pop('self')

        if is_train:
            param = get_kwargs(self.cv_validate, **L)
            if cv < 3:
                cv = 3
            cv_score = self.cv_validate(X,
                                        y,
                                        cv=cv,
                                        scoring=scoring,
                                        **fit_params)
        else:
            cv_score = self.test_score(X, y, cv=cv, scoring=scoring)

        cv_score['kfolds'] = len(cv_score)
        cv_score['NSamples'] = len(X)
        cv_score['NInputFeatures'] = np.shape(X)[-1]
        cv_score['NUsedFeatures'] = len(self.featurenames)
        # plot cv_score path
        k_folds = '{} folds'.format(len(cv_score))
        # title info
        title = '' if title is None else title
        header = self._get_basic_info(True, title, k_folds)
        plot_data = cv_score.drop(
            columns=['kfolds', 'NSamples', 'NInputFeatures', 'NUsedFeatures'])
        plotter_score_path(plot_data, title=header)
        # save fig
        self._savefig(save_fig, default_saved_name='plots/ScorePath.pdf')
        return cv_score

    def plot_gridcv(self, gridcvtab, title=None, save_fig=False):
        '''plot grid seatch cv results
        '''
        if gridcvtab is None:
            print('no cv_result table output')
            return

        # title info
        title = '' if title is None else title
        header = self._get_basic_info(True, title)
        plotter_cv_results_(gridcvtab, title=header)
        # save fig
        self._savefig(save_fig, default_saved_name='plots/searchcv.pdf')
        return

    @wraps(cross_val_score)
    def cv_score(self, X, y, scoring='roc_auc', cv=5, **kwargs):
        '''
        return cross validated score of estimator (see cross_val_score)
        ---------
        '''
        scorer = self._get_scorer(scoring)
        return cross_val_score(self.estimator,
                               X=X,
                               y=y,
                               scoring=scorer,
                               cv=cv,
                               **get_kwargs(cross_val_score, **kwargs))

    @wraps(cross_validate)
    def cv_validate(self,
                    X,
                    y,
                    scoring='roc_auc',
                    cv=3,
                    return_estimator=False,
                    return_train_score=False,
                    **kwargs):
        '''       
        return cross_validate results of estimator(see cross_validate)
        -----
        cv_results: 
            (as DataFrame, allowing for multi-metrics) in the form of
            'test_xxx', train_xxx' where  'xxx' is scorer name
        '''
        estimator = self.estimator
        L = locals().copy()
        L.pop('self')
        L.pop('scoring')
        scorer = self._get_scorer(scoring)
        # --
        cv_results = cross_validate(scoring=scorer,
                                    **get_kwargs(cross_validate, **L,
                                                 **kwargs))
        return pd.DataFrame(cv_results)

    def test_score(self, X, y, cv, scoring):
        '''return test scores of estimator 
        '''
        # test scores
        data_splits = split_cv(X, y=y, cv=cv, random_state=self.seed)
        scorer = self._get_scorer(scoring)
        scores = []
        for item in data_splits:
            x0 = item[0][1]
            y0 = item[1][1]
            scores.append(_validation._score(self.estimator, x0, y0, scorer))
        scores = pd.DataFrame(scores).reset_index(drop=True)
        return scores

    @Appender(GridSearchCV.__doc__, join='\n')
    @dedent
    def searchcv(self,
                 X,
                 y,
                 param_grid,
                 kind='bayesiancv',
                 combine_param_space=None,
                 scoring='roc_auc',
                 cv=3,
                 n_iter=20,
                 refit=['roc_auc'],
                 error_score=-999,
                 iid=False,
                 return_train_score=True,
                 n_jobs=-1,
                 fit_params={},
                 **kwargs):
        '''
        The parameters of the estimator are optimized
        by cross-validated search over a parameter grid. 
        update self.estimator as best model and update self.gridcvtab
    
        Parameters
        ----------
        X
            feature matrix
        y
            classifiction calss label 0/1
        
        param_grid:  dict or list of dictionaries
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.the same parameter for 'search_spaces' 
            in bayesian and for 'param_distributions'  in RandomizedSearchCV

        kind: ['gridcv', 'bayesiancv', 'randomcv'], search method to use
        
        combine_param_space bool:
            if True make param_grid into list of single one dict,
            thus only one cv_result table could be output.        
            by default, for 'gridcv' combine_param_space is set to False
            for 'bayesiancv' combine_param_space is set to True
            
        scoring: string, callable, list/tuple, dict or None, default: None
            A single string (see :ref:`scoring_parameter`) or a callable
            (see :ref:`scoring`) to evaluate the predictions on the test set.
    
            For evaluating multiple metrics, either give a list of (unique) strings
            or a dict with names as keys and callables as values.
    
            NOTE that when using custom scorers, each scorer should return a single
            value. Metric functions returning a list/array of values can be wrapped
            into multiple scorers that return one value each.
            for bayesian search multi-metrics are not supported refit scoring are
            used
    
            See :ref:`multimetric_grid_search` for an example.
    
            If None, the estimator's score method is used.

        n_iter: int, default=30 (for bayesian and randomcv)
            Number of parameter settings that are sampled.
            n_iter trades off runtime vs quality of the solution.
            Consider increasing n_points if you want to try more parameter 
            settings in parallel.

        optimizer_kwargsdict, optional (for bayesian)
            Dict of arguments passed to Optimizer. For example, 
            {'base_estimator': 'RF'} would use a Random Forest surrogate 
            instead of the default Gaussian Process.    
            
        n_jobs: int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
                            
        return
        -----
        cv_results as DataFrame
        
        Appended see GridSearchCV()
        --------
        '''
        L = locals().copy()
        L.pop('self')
        L.pop('fit_params')
        L.pop('scoring')
        L.pop('param_grid')

        # --
        estimator = self.estimator
        if combine_param_space is None:
            combine_param_space = True if kind == 'bayesiancv' else False

        if combine_param_space:
            param_grid = self._combine_gridspace(param_grid)

        if kind == 'bayesiancv':
            scorer = self._get_scorer(refit).get(refit)
        else:
            scorer = self._get_scorer(scoring)

        search_method = {
            'gridcv': GridSearchCV,
            'bayesiancv': BayesSearchCV,
            'randomcv': RandomizedSearchCV
        }
        api = search_method[kind]
        params = get_kwargs(api, **L)
        params.update(get_kwargs(api, **kwargs))
        if cv < 3:
            params.update(cv=3)

        grid = api(estimator, param_grid, scoring=scorer, **params)
        grid.fit(X, y, **fit_params)
        self._update_fitted_estimator(grid.best_estimator_, X, y)

        # check if lengths of values are consistent
        length_ = (len(v) for k, v in grid.cv_results_.items())
        if len(np.unique(length_)) == 1:
            cv_results = pd.DataFrame(grid.cv_results_)
            if kind == 'bayesiancv':
                cv_results.columns = [
                    col.replace('score', refit) for col in cv_results.columns
                ]
            self.kws_attr.get('gridcvtab').append(cv_results)
            return cv_results
        else:
            print('''no gridcvtab output because multi-step search, 
                  set out_searchstep to 'True' if gridcvtab is desired ''')

    def fit(self, X, y, **fit_params):
        '''perform fit of estimator
        '''
        self.estimator.fit(X, y, **fit_params)
        self._update_fitted_estimator(self.estimator, X, y)
        return self

    def predict(self,
                X,
                pre_method=['decision_function', 'predict_proba', 'predict'],
                pre_level=False,
                pos_label=1,
                pre_1d_positive=True,
                **kwargs):
        '''return predictions of estimator
        
        pre_method: list of str
            priority of sklearn estimator method to use: ['decision_function',
            'predict_proba', 'predict']
        pre_level: bool
             if true, output score as integer rankings starting from 0
        pos_label
            index of predicted class
        '''
        estimator = self.estimator

        for i in pre_method:
            if hasattr(estimator, i):
                pre_func = getattr(estimator, i)
                break

        if pre_func is None:
            print('{} has no {} method'.format(self._get_estimator_name(),
                                               pre_method))

        y_pre = pre_func(X, **kwargs)
        if pre_1d_positive:
            if np.ndim(y_pre) > 1:
                y_pre = y_pre[:, pos_label]

            if pre_level:
                y_pre, bins = binning(y_pre,
                                      bins=self.estimator.bins,
                                      labels=False)
        return y_pre

    def predict_proba(self, X):
        '''
        '''
        if hasattr(self.estimator, 'predict_proba'):
            return self.estimator.predict_proba(X)

    def run_cvscore(
            self,
            data_set,
            is_train,
            scoring=['roc_auc', 'KS'],
            fit_params={},
            cv=3,
            q=None,
            bins=None,
            max_leaf_nodes=None,
            mono=None,
            save_fig=True,
            labels=False,
            tree_kws={},
    ):
        '''run cross validation of estimator on data_set, output 
        1) evaluation AucLift plot
        2) score path plot
        
        parameters
        ----
        
        dataset: 
            2 element tuple, (X, y) of train data
        
        is_train: bool
            type of data, True 'Train' dataset, 'False' 'Test' dataset
            
        cv (int):
           n of cross validation folds, cv should be greater thant 1
           
        fit_params
            -other fit parameters of estimator
            
        bins
            - number of equal width or array of edges
        q
            - number of equal frequency              
        max_leaf_nodes
            - number of tree nodes using tree cut
            - if not None use supervised cutting based on decision tree
        mono 
            - binning edges that increases monotonically with "y" mean value

        use_self_bins
            - use self.estimator.bins if true
            
        .. note::
            -  only 1 of (q, bins, max_leaf_nodes, mono) can be specified 
            
        save_fig (bool):
            if true, save plot as .pdf to 'plots' under self.loaddump.path_
            
        labels
            - see pd.cut, if False return integer indicator of bins, 
            - if True return arrays of labels (or can be manually input)
            
        optionally 
        if self.verbose > 0
            - plot AucLift curve for cv splitted train data 
             dump .pdf to 'plots' under self.loaddump.path_
        if self.verbose > 1     
            - dump spreadsheets of calculated data
            
        return
        ----
        series: averaged score for each scoring metrics
        
        AucLift chart for train dataset

        '''
        L = locals().copy()
        L.pop('self')
        L.pop('save_fig')
        loaddump = self.loaddump
        # --

        X, y = data_set
        # --plot TrainAucLift
        fit_train = True if is_train else False
        data_title = 'Train' if is_train else 'Test'
        if save_fig is True:
            pdf_file1 = 'plots/{}AucLift.pdf'.format(data_title)
            pdf_file2 = 'plots/{}ScorePath.pdf'.format(data_title)
        else:
            pdf_file1 = pdf_file2 = False

        auclift_param = get_kwargs(self.plot_AucLift, **L)
        auccv, lift_data = self.plot_AucLift(X,
                                             y,
                                             fit_train=fit_train,
                                             title=data_title,
                                             save_fig=pdf_file1,
                                             **auclift_param)

        # plot cv_score path & return cvscore
        cv_score = self.plot_cvscore(
            X,
            y,
            is_train,
            scoring,
            title=data_title,
            save_fig=pdf_file2,
            cv=cv,
        )

        if self.verbose > 1:
            # dump excel
            lift = lift_data[-1]
            print('lift data & cv_score & cv_splits data are being saved...')
            loaddump.write([lift, cv_score],
                           'spreadsheet/{}Perfomance.xlsx'.format(data_title),
                           sheet_name=['liftcurve', 'score'])
            loaddump.write(auccv[3],
                           'spreadsheet/{}Splits.xlsx'.format(data_title))

        return cv_score.mean()

    def opt_sequential(self,
                       train_set,
                       param_grid=-1,
                       kind='bayesiancv',
                       out_searchstep=False,
                       refit='roc_auc',
                       scoring=['roc_auc', 'KS'],
                       fit_params={},
                       save_fig=True,
                       memory_cache=True,
                       title=None,
                       **kwargs):
        '''
        run sequential model-based optimization of param_grid space, using
        cross-validated performance(if param_grid=-1, use pre-difined space)    
        - update self estimator as best estimator and update self gridcvtab
        
        - dump plots/spreadsheets optional
        
        .. note::
            hyper parameters of each operator are tuned sequentially. for each
            operator the best of each parameter is updated sequentially 
            too
            
        
        parmameters
        ----
        train_set: 
            2 element tuple, (X, y) of train data
        title (str):
            title prefix for Gridsearch results plot
        param_grid:
            parameter grid space, if -1, use pipe_grid() to return predifined 
            param_grid

        kind str: ['gridcv', 'bayesiancv', 'randomcv'], search method to use
            - if kind == 'bayesiancv' use bayesian optimization
            
            - if kind == 'gridcv' sequentially update the best parameter 
            settings in each dict of param_grid by grid search
            
            - if kind == 'randomcv' sequentially update the best parameter 
            settings in each dict of param_grid by random grid search

        out_searchstep bool:
            if True, output grid search cv result for each step
            default False
            
        memory_cache:
            Used to cache the fitted transformers of the pipeline, 
            Caching the transformers is advantageous when fitting is time 
            consuming.

                  
        **kwargs:
            GridSearchCV keywords
        '''

        L = locals().copy()
        L.pop('self')
        L.pop('param_grid')
        loaddump = self.loaddump
        #--
        X, y = train_set

        # get predefined grid space
        if param_grid is -1:
            param_grid = self._get_gridspace(kind)

        if len(param_grid) == 0:
            print('no param_grid found, fit and skip grid search')
            self.fit(X, y, **fit_params)
            return

        # memory cache
        if hasattr(self.estimator, 'memory') and memory_cache:
            self.estimator.memory = os.path.relpath(
                os.path.join(self.loaddump.path_, 'temploaddump'))
        # reinitialize self.gridcvtab
        # store cv results in each sequential step
        self.kws_attr.update(gridcvtab=[])
        params = get_kwargs(self.searchcv, **L)
        params.update(kwargs)

        if out_searchstep:
            for i, grid in enumerate(get_flat_list(param_grid)):
                gdcv = self.searchcv(X,
                                     y,
                                     param_grid=grid,
                                     combine_param_space=False,
                                     **params)
                file = "plots/{}_{}.pdf".format(kind, i)
                self.plot_gridcv(gdcv, save_fig=file, title=kind + str(i))
        else:
            gdcv = self.searchcv(X, y, param_grid, **params)
            self.plot_gridcv(gdcv, save_fig=save_fig, title=kind)

        self._shut_temp_loaddump()

        if self.verbose > 0:
            print('sensitivity results are being saved... ')
            title = 0 if title is None else str(title)
            loaddump.write(self.kws_attr.get('gridcvtab'),
                           'spreadsheet/GridcvResults{}.xlsx'.format(title))

    def run_analysis(self,
                     train_set,
                     test_set=None,
                     is_search=True,
                     save_fig=True,
                     **kwargs):
        '''
        run analysis of estimator
        
        1. run self.opt_sequential(if grid_search=True)
        2. run self.run_cvscore for train_set if not None,
        3. run self.run_cvscpre for test_set of not None
        4. store self trainscore & testscore
        
        train_set:
            (X, y) tuple to use as train set
        
        test_set:
            (X_test, y_test) tuple to use as test set, if None, no valuation
            will be performed on testset.
           
        is_search bool:
            if true, perform opt_search
            if false, skip opt_search

        cv (int):
           n of cross validation folds, cv should be greater thant 1
        
        scoring (list):
            scoring metrics, like ['roc_auc', 'KS', 'average_precision']
        
        q
            - number of equal frequency 
        
        bins
            - number of equal width or array of edges
        
        max_leaf_nodes
            - if not None perform supervised cutting, 
            - number of tree nodes using tree cut        
        
        mono 
            - binning edges that increases monotonically with "y" mean value
            
        .. note::
            -  only 1 of (q, bins, max_leaf_nodes, mono) can be specified 
            if all None, no lift curve plot
        
        return
        -------
            self instance
        '''
        L = locals().copy()
        L.pop('self')

        # --tuning hyper parameters by searchcv
        if is_search:
            self.opt_sequential(**L, **kwargs)

        # cross_validated performance of dataset
        params = {}
        params.update(**get_kwargs(self.run_cvscore, **L),
                      **get_kwargs(self.run_cvscore, **kwargs))
        if train_set is not None:
            trainscore = self.run_cvscore(train_set, is_train=True, **params)
            self.kws_attr.update(trainscore=trainscore)
        # splitted test performance of test_set
        if test_set is None:
            print('no test_set, skip run_test method ...\n')
        else:
            testscore = self.run_cvscore(test_set, is_train=False, **params)
            self.kws_attr.update(testscore=testscore)

        self.save()
        return self

    def save(self):
        '''save current estimator instance, self instance 
        and self construction settings
        '''
        loaddump = self.loaddump
        # save model instance
        loaddump.write(self, ''.join([self.modelsteps, '.lwpkl']))

    def delete_model(self):
        '''delete self.loaddump.path_ loaddump containing model
        '''
        del self.loaddump.path_

    @property
    def featurenames(self):  #need update
        '''get input feature names of fited final estimator
        '''
        pipe_line = self.estimator
        fn = get_featurenames(pipe_line)
        if fn is None:
            return self.kws_attr['input_featurenames']
        else:
            return fn

    @Appender(run_modellist.__doc__, join='\n')
    @dedent
    def run_autoML(self,
                   X,
                   y,
                   test_set=None,
                   verbose=1,
                   by_metrics='roc_auc_Trainset',
                   **kwargs):
        '''
        run_autoML & update estimator as best model
        
        Parameters
        ----------
        X
            feature matrix
        y
            classifiction calss label 0/1     
            
        test_set:
            (X_test, y_test) tuple to use as test set, if None, no valuation
            will be performed on testset.
            
        by_metrics str:
            metrics used to select best model default='roc_auc_Trainset',
            [_'Trainset', _'Test_set'] suffix to indicate dataset from which
            metrics is calculated
        
        Appended see run_analy()'s doc
        ------------------------------
        
        '''
        auto_path = os.path.join(self.path, 'auto')
        autocv_score, models = run_modellist(X,
                                             y,
                                             test_set,
                                             verbose=verbose,
                                             dirs=auto_path,
                                             **kwargs)
        self.autocv_score = autocv_score
        self.automodels = models
        # update best model
        self._update_bestmodel(by_metrics=by_metrics)
        self.run_analysis((X, y), test_set, **kwargs)
        return autocv_score


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    X, y = make_classification(5000, n_redundant=5, n_features=50)
    m = LW_model('clean_oht_frf_OneSidedSelection_XGBClassifier', verbose=1)
    m.fit(X, y)
