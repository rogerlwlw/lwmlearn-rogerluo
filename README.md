<<<<<<< Updated upstream
# build status
=======
<<<<<<< Updated upstream
=======
# Build status
>>>>>>> Stashed changes
[![Build Status](https://travis-ci.org/rogerlwlw/lwmlearn-rogerluo.svg?branch=master)](https://travis-ci.org/rogerlwlw/lwmlearn-rogerluo)


# Documentation
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://lwmlearn-rogerluo.readthedocs.io/en/latest/)

<<<<<<< Updated upstream
=======
>>>>>>> Stashed changes
>>>>>>> Stashed changes
# lw-mlearn

A Python package that wraps sklearn and many other estimators into pipelines and faciliates workflow 
such as 

1) data cleaning:
    - try converting data to numeric dtype; 
        - dropna columns; 
        - drop uid columns;
        - drop constant columns;
        - filter specific dtypes;
        - convert solely '\t\n' to np.na 
        
2) data encoding: 
        - oridinal/one-hot encoding; 
        - woe encoding; 
        - binning by cart tree/equal frequency/equal width;
        
3) feature selection:
        - select from model(svc, xgb, cart, random forest); 
        - select from test statisics (chi2, mutual-info, woe);
        - pca/LDA/QDA decomposition
        - RFE
        
4) data resampling (over/under sampling for imbalanced dataset)
5) model training / evaluation / cross validation
6) hyper parameter tuning (grid_search, Bayesian Optimization)
7) production integration


Contact
=============
If you have any questions or comments about lwmlearn, please feel free to 
contact me via:
E-mail: coolww@outlook.com


