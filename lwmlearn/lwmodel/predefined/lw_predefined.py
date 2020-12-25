# -*- coding: utf-8 -*-
"""lw_predefined instances to put in operators pool

this module pre-defines some operators by initializing without defaults.

predefined_ops():
    return a dict of user defined none-default instances of operators
    
Created on Fri Dec 20 10:24:27 2019

@author: roger luo
"""
from imblearn import FunctionSampler
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks, NearMiss, CondensedNearestNeighbour,
    OneSidedSelection, NeighbourhoodCleaningRule, EditedNearestNeighbours,
    AllKNN, InstanceHardnessThreshold, ClusterCentroids)

from imblearn.over_sampling import (
    ADASYN,
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
)
from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.preprocessing import (PolynomialFeatures, StandardScaler,
                                   MinMaxScaler, RobustScaler, Normalizer,
                                   QuantileTransformer, PowerTransformer,
                                   MaxAbsScaler)
from sklearn.feature_selection import (SelectFromModel,
                                       GenericUnivariateSelect, chi2,
                                       f_classif, mutual_info_classif, RFE)
from sklearn.decomposition import (
    DictionaryLearning, FastICA, IncrementalPCA, KernelPCA,
    MiniBatchDictionaryLearning, MiniBatchSparsePCA, NMF, PCA, SparseCoder,
    SparsePCA, dict_learning, dict_learning_online, fastica,
    non_negative_factorization, randomized_svd, sparse_encode, FactorAnalysis,
    TruncatedSVD, LatentDirichletAllocation)

from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import (SGDClassifier, LogisticRegression)
from sklearn.ensemble import RandomTreesEmbedding

from sklearn.ensemble import IsolationForest, ExtraTreesClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from lwmlearn.preprocess.lw_base import Cleaner
from lwmlearn.preprocess.lw_woe_encoder import WoeEncoder
from lwmlearn.preprocess.lw_cat_encoder import OhtEncoder, OrdiEncoder

from xgboost.sklearn import XGBClassifier


def _outlier_rejection(X=None,
                       y=None,
                       method='IsolationForest',
                       contamination=0.1):
    """This will be function used to resample our dataset, removing outliers
    supposing contamination=0.1.
    """
    outlier_model = (
        IsolationForest(contamination=contamination),
        LocalOutlierFactor(contamination=contamination),
        OneClassSVM(nu=contamination),
        EllipticEnvelope(contamination=contamination),
    )

    outlier_model = {i.__class__.__name__: i for i in outlier_model}

    if X is None:
        return outlier_model.keys()
    model = outlier_model.get(method)
    if model is None:
        raise ValueError("method '{}' is invalid".format(method))
    y_pred = model.fit_predict(X)
    return X[y_pred == 1], y[y_pred == 1]


def predefined_ops():
    '''return dict of user defined none-default instances of operators
    '''
    clean = {
        'clean':
        Cleaner(dtype_filter='not_datetime',
                na1='null',
                na2='mean',
                drop_uid=True),
        'cleanNA':
        Cleaner(dtype_filter='not_datetime', na1=None, na2=None),
        'cleanMean':
        Cleaner(dtype_filter='not_datetime', na1='most_frequent', na2='mean'),
        'cleanMn':
        Cleaner(dtype_filter='not_datetime', na1='missing', na2='mean'),
    }
    #
    encode = {
        'woe8': WoeEncoder(max_leaf_nodes=8),
        'woe5': WoeEncoder(max_leaf_nodes=5),
        'woeq8': WoeEncoder(q=8),
        'woeq5': WoeEncoder(q=5),
        'woeb5': WoeEncoder(bins=5),
        'woem': WoeEncoder(mono=True),
        'oht': OhtEncoder(),
        'ordi': OrdiEncoder(),

        # 'bin10': BinEncoder(bins=10, int_bins=True),  # 10 bin edges encoder
        # 'bin5': BinEncoder(bins=5, int_bins=True),  # 5 bin edges encoder
        # 'binm10': BinEncoder(max_leaf_nodes=10,
        #                      int_bins=True),  # 10 bin tree cut edges encoder
        # 'binm5': BinEncoder(max_leaf_nodes=5,
        #                     int_bins=True),  # 5 bin tree cut edges encoder
    }

    resample = {
        # over_sampling
        # under sampling controlled methods
        'runder':
        RandomUnderSampler(),
        'nearmiss':
        NearMiss(version=3),
        'pcart':
        InstanceHardnessThreshold(),
        # clean outliers
        'inlierForest':
        FunctionSampler(_outlier_rejection,
                        kw_args={
                            'method': 'IsolationForest',
                            'contamination': 0.1
                        }),
        'inlierLocal':
        FunctionSampler(_outlier_rejection,
                        kw_args={
                            'method': 'LocalOutlierFactor',
                            'contamination': 0.1
                        }),
        'inlierEllip':
        FunctionSampler(_outlier_rejection,
                        kw_args={
                            'method': 'EllipticEnvelope',
                            'contamination': 0.1
                        }),
        'inlierOsvm':
        FunctionSampler(_outlier_rejection,
                        kw_args={
                            'method': 'OneClassSVM',
                            'contamination': 0.1
                        }),
    }

    scale = {
        'stdscale': StandardScaler(),
        'minmax': MinMaxScaler(),
        'absmax': MaxAbsScaler(),
        'rscale': RobustScaler(quantile_range=(10, 90)),
        'quantile': QuantileTransformer(),  # uniform distribution
        'power': PowerTransformer(),  # Gaussian distribution
        'norm': Normalizer(),  # default L2 norm

        # scale sparse data
        'maxabs': MaxAbsScaler(),
        'stdscalesp': StandardScaler(with_mean=False),
    }
    # feature construction
    feature_c = {
        'pca': PCA(whiten=True),
        'spca': SparsePCA(n_jobs=-1),
        'ipca': IncrementalPCA(whiten=True),
        'kpca': KernelPCA(kernel='rbf', n_jobs=-1),
        'poly': PolynomialFeatures(degree=2),
        # kernel approximation
        'Nys': Nystroem(random_state=0),
        'rbf': RBFSampler(random_state=0),
        'rfembedding': RandomTreesEmbedding(n_estimators=10),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
    }
    # select from model
    feature_m = {
        'fwoe':
        SelectFromModel(WoeEncoder(max_leaf_nodes=5)),
        'flog':
        SelectFromModel(LogisticRegression(penalty='l1', solver='saga',
                                           C=1e-2)),
        'fsgd':
        SelectFromModel(SGDClassifier(penalty="l1")),
        'fxgb':
        SelectFromModel(XGBClassifier(n_jobs=-1, booster='gbtree'), 
                        max_depth=2,
                        n_estimators=50),
        'frf':
        SelectFromModel(ExtraTreesClassifier(n_estimators=50, max_depth=2)),

        # fixed number of features
        'fxgb20':
        SelectFromModel(XGBClassifier(n_jobs=-1, booster='gbtree'),
                        max_features=20),
        'frf20':
        SelectFromModel(ExtraTreesClassifier(n_estimators=100, max_depth=5),
                        max_features=20),
        'frf10':
        SelectFromModel(ExtraTreesClassifier(n_estimators=100, max_depth=5),
                        max_features=10),
        'fRFElog':
        RFE(LogisticRegression(penalty='l1', solver='saga', C=1e-2), step=0.1),
        'fRFExgb':
        RFE(XGBClassifier(n_jobs=-1, booster='gbtree'), step=0.1),
    }
    # Univariate feature selection
    feature_u = {
        'fchi2':
        GenericUnivariateSelect(chi2, 'percentile', 25),
        'fMutualclf':
        GenericUnivariateSelect(mutual_info_classif, 'percentile', 25),
        'fFclf':
        GenericUnivariateSelect(f_classif, 'percentile', 25),
    }
        
    imp = {
        "impXGB" : XGBClassifier(n_jobs=-1, booster='gbtree', max_depth=2,
                                 n_estimators=50),
        "impRF" : ExtraTreesClassifier(n_estimators=100, max_depth=2)
        }

    instances = {}
    instances.update(**clean, **encode, **scale, **feature_c, **feature_m,
                     **feature_u, **resample, **imp)
    return instances
