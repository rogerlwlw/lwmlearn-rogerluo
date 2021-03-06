# -*- coding: utf-8 -*-
"""
save predefined estimator chain pattern to run autoML

Created on Fri Dec 20 11:54:07 2019

@author: roger luo
"""


def get_default_classifier():
    '''return default list of classifiers
    '''
    lis = [
        # linear
        'LogisticRegression',
        'SGDClassifier',
        # tree based
        'LGBMClassifier',
        'XGBClassifier',
        'HistGradientBoostingClassifier',
        'AdaBoostClassifier',
        'RandomForestClassifier',
        'EasyEnsembleClassifier',
        'RUSBoostClassifier',
        'GradientBoostingClassifier',

        # weak learners
        'DecisionTreeClassifier',
        'KNeighborsClassifier',
    ]
    return lis


def get_default_models(rtype=0):
    """
    

    Parameters
    ----------
    rtype : int, [0, 1]
        return type. The default is 0.

    Returns
    -------
    l : list
        list of estimators

    """
    if rtype == 0:
        return get_default_pipeline()
    elif rtype == 1:
        return get_default_classifier()
    else:
        raise ValueError('invalid rtype input')


def get_default_pipeline():
    '''return default predefined estimators to run AutoML
    '''
    lis = [
        # default SVM, grid search log/hinge/huber/perceptron

        # linear models
        'cleanNA_woe5_fxgb_NeighbourhoodCleaningRule_SGDClassifier',
        # 'cleanNA_woe8_fxgb_TomekLinks_SGDClassifier',
        # 'cleanNA_woe8_fxgb_inlierLocal_NeighbourhoodCleaningRule_SGDClassifier',

        # tree based models
        # 'clean_oht_fxgb_XGBClassifier',
        # 'clean_oht_fxgb_inlierForest_XGBClassifier',
        'clean_oht_frf_NeighbourhoodCleaningRule_XGBClassifier',
        'clean_ordi_fxgb_NeighbourhoodCleaningRule_LGBMClassifier',
        'clean_oht_fxgb_NeighbourhoodCleaningRule_LGBMClassifier',
        # 'clean_oht_fxgb_RandomForestClassifier',
        'clean_oht_fxgb_NeighbourhoodCleaningRule_RandomForestClassifier',
        # 'clean_oht_fxgb_GradientBoostingClassifier',
        # 'clean_oht_fxgb_NeighbourhoodCleaningRule_HistGradientBoostingClassifier',
        # 'clean_oht_frf_AdaBoostClassifier',

        # balance samples on each iteration
        'clean_oht_frf_RUSBoostClassifier',

        # The classifier is an ensemble of AdaBoost learners trained on
        # different balanced boostrap samples.
        # The balancing is achieved by random under-sampling.
        'clean_oht_fxgb_EasyEnsembleClassifier',
        #
        'clean_oht_fxgb_DecisionTreeClassifier',
        # 'clean_oht_fxgb_inlierLocal_NeighbourhoodCleaningRule_DecisionTreeClassifier',
        # 'clean_oht_frf_KNeighborsClassifier',
    ]

    return lis


def sampler_comparison():
    '''
    '''

    lst = [
        'clean_oht_frf_RandomUnderSampler_DecisionTreeClassifier',
        'clean_oht_frf_EditedNearestNeighbours_DecisionTreeClassifier',
        'clean_oht_frf_ALLKNN_DecisionTreeClassifier',
        'clean_oht_frf_CondensedNearestNeighbour_DecisionTreeClassifier',
        'clean_oht_frf_OneSidedSelection_DecisionTreeClassifier',
        'clean_oht_frf_NeighbourhoodCleaningRule_DecisionTreeClassifier',
        'clean_oht_frf_TomekLinks_DecisionTreeClassifier',
        'clean_oht_frf_SMOTEENN_DecisionTreeClassifier',
    ]

    return lst


def get_default_preprocessors():
    '''
    '''
    lis = [
        # linear models
        'cleanNA_woe5_frf',
        'cleanNA_woe5_frf_NeighbourhoodCleaningRule',
        'cleanNA_woe5_frf_EditedNearestNeighbours',
        'cleanNA_woe5_frf_runder',
        'cleanNA_woe5_fxgb_inlierForest_NeighbourhoodCleaningRule',
        'cleanNA_woe5_fxgb20_inlierLocal_NeighbourhoodCleaningRule',
        'clean_oht_fxgb_inlierForest_NeighbourhoodCleaningRule',
        'clean_oht_fxgb_inlierLocal_NeighbourhoodCleaningRule',
        'clean_oht_fxgb_inlierLocal_OneSidedSelection',

        # apply cleaning after SMOTE over sampling
        'cleanNA_woe5_frf_SMOTEENN',
        'cleanNA_woe5_frf_SMOTETomek',

        # under sample
        # 'ClusterCentroids',
    ]
    return lis
