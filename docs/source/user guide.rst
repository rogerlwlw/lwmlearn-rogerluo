User Guide
==========


Main Helper Functions & Classes
--------------------------------
* :class:`~.LW_model`
* :func:`~.pipe_gen`
* :func:`~.run_modellist`
* :func:`~.get_lw_estimators`


Use pipe_gen
-------------
``pipe_gen`` is the main function to retrieve pipeline instance, in the pattern
of ``XX_XX_XX`` where XX stands for operator string name.

To retrieve a pipeline connecting ``cleanNA`` and ``woe5`` transformer ::

    pipe_gen("cleanNA_woe5") 


To retrieve a pipeline connecting one additional 
``LogisticRegressionClassifier`` ::
    
    pipe_gen("cleanNA_woe5_LogisticRegressionClassifier")
    
To get reference of available pipeline steps by calling ``pipe_gen()`` with no
arguments

.. ipython::
    :okwarning:
    
    @suppress
    In [1]: import warnings
    @suppress
    In [2]: warnings.filterwarnings("ignore")
      
    In [3]: from lwmlearn import pipe_gen
    
    # get available classifiers with default settings
    In [4]: pipe_gen()['default']['classifier']
    
    # get available transformers with default settings
    In [5]: pipe_gen()['default']['transformer']


Use LW_model
------------
``LW_model`` could be constructed the same way as pipe_gen, and use pipeline 
instance as an inner estimator. In addition, ``LW_model`` instance has 
common evaluation methods to analyze model performance. 

To initialize an instance like below ::
    
    LW_model("cleanNA_woe5_LogisticRegressionClassifier")

.. rubric:: Example

the following runs a ``clean_oht_frf_OneSidedSelection_XGBClassifier`` pipeline
on fake classification data. The pipe line performs ``data cleanning`` --> 
``one hot encoding`` --> ``randomforest feature selection`` --> ``OneSidedSelection`` 
on training dataset and finally use a XGBClassifier to train processed data. 
Then the fitted pipeline could be used to make predictions on test data. 

.. ipython::
    :okwarning:
    
    @suppress   
    In [1]: import warnings
       ...: warnings.filterwarnings("ignore")
       
    In [2]: from lwmlearn import LW_model
       ...: from sklearn.datasets import make_classification
       ...: from sklearn.model_selection import train_test_split

.. ipython::
    :okwarning:  
    
    In [4]: X, y = make_classification(10000, n_redundant=20, n_features=50, flip_y=0.1)

    In [5]: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    In [6]: m = LW_model('clean_oht_frf_OneSidedSelection_XGBClassifier', verbose=1)
    
    In [7]: m.fit(X_train, y_train)
    
    In [9]: m.predict(X_test)   
    
    In [11]: m.test_score(X_test, y_test, cv=1, scoring=['KS', 'roc_auc'])
    
    In [12]: m.cv_validate(X_train, y_train, scoring=['roc_auc', 'KS'])

    # auto tuning parameters by bayesian search and update model
    In [13]: m.opt_sequential((X, y), kind='bayesiancv')

.. ipython::
    :okwarning:
        
    # plot search learning curve
    @savefig plot_learning_curve.png
    In [14]: m.plot_gridcv(m.kws_attr['gridcvtab'][0]) 
    
.. ipython::
    :okwarning:
        
    # plot lift curve for trainset
    @savefig plot_lift_train.png
    In [46]: m.plot_lift(X_train, y_train, max_leaf_nodes=10)

.. ipython::
    :okwarning:
        
    # plot lift curve for test set with bins cut by trainset
    @savefig plot_lift_test.png
    In [45]: m.plot_lift(X_test, y_test, use_self_bins=True)

.. ipython::
    :okwarning:
        
    # plot two plots together
    @savefig plot_auclift.png
    In [47]: m.plot_AucLift(X_test, y_test, fit_train=False)  


Use get_lw_estimators
-----------------------
it retrieves an isntance of ML model(``Classifier/Regressor``), help write
code quickly and keep script simple.
    


Command Line Entry
------------------
Enter entry point ``lwmlearn`` to access AutoML api 

.. ipython::
    :okwarning:
    
    In [0]: ! lwmlearn -h

After running ``autoML``, lots of model outputs can be analyzed:

    #. roc_auc and lift curve for singe pipeline 
    #. learning curve for hyper-parameter tuning process
    #. performance comparison of different ML algorithm
    #. performance comparison table 