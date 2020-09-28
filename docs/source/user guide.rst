User Guide
==========


Main Helper Functions & Classes
--------------------------------
* :class:`~.LW_model`
* :func:`~.pipe_gen`
* :func:`~.run_modellist`
* :func:`~.get_lw_estimators`


Command Line Entry
------------------

this package has one entry point ``lwmlearn`` to access AutoML api 

.. ipython::
    :okwarning:
    
    In [0]: ! lwmlearn -h

After running ``autoML``, lots of model outputs can be analyzed:

    #. roc_auc and lift curve for singe pipeline 
    #. learning curve for hyper-parameter tuning process
    #. performance comparison of different ML algorithm
    #. performance comparison table 

Use pipe_gen
-------------
``pipe_gen`` is the main function to retrieve pipeline instance, like below
retrieves a pipeline connecting ``cleanNA`` and ``woe5`` transformer ::

    pipe_gen("cleanNA_woe5") 


this retrieves a pipeline connecting one additional 
``LogisticRegressionClassifier`` ::
    
    pipe_gen("cleanNA_woe5_LogisticRegressionClassifier")
    

Use LW_model
------------

``LW_model`` could be constructed the same way as pipe_gen, and use pipeline 
instance as an inner estimator. in addition, ``LW_model`` instance has 
common evaluation methods to analyze model performance. to initialize
an instance like below ::
    
    LW_model("cleanNA_woe5_LogisticRegressionClassifier")
    

Use get_lw_estimators
-----------------------

it retrieves an isntance of ML model(``Classifier/Regressor``), it helps write
code quickly and keep script simple.
    