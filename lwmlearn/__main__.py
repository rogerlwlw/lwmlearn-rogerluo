# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:08:28 2020

@author: Administrator
"""
from __future__ import print_function

import argparse
import sys
import traceback

import pandas as pd
import os 

import lwmlearn
from lwmlearn import LW_model, run_CVscores
from sklearn.model_selection import train_test_split

def _ParseArguments(argv):
    '''Parse the command line arguments.
    
    return 
    ------
    commandline parsed input
    '''
    description =\
    '''
    Main function entry_point for lwmlearn
    '''
    parser = argparse.ArgumentParser(description=description)

    # # positional argument
    parser.add_argument('data_path',
                        help="csv file train data")
    # optional argument
    parser.add_argument('-v',
                        '--version',
                        action='store_true',
                        help='show version number and exit')
    
    parser.add_argument('-y',
                        '--ylabel',
                        type=str,
                        default='y',
                        help='class label, default is y')

    
    parser.add_argument('-p',
                        '--pipeline',
                        action='store',
                        default='clean_ordi_fxgb_NeighbourhoodCleaningRule_LGBMClassifier',
                        help='pipeline to run model, default is XXX')
    
     # mutually exclusive argument
    me_group0 = parser.add_mutually_exclusive_group()
    
    me_group0.add_argument('-ra',
                        '--automl',
                        action='store_true',
                        help='run autoML on data ')

    me_group0.add_argument('-cv',
                        '--cv_only',
                        action='store_true',
                        help='run CV score only to save you some time')
    
    # mutually exclusive argument
    mutually_exclusive_group = parser.add_mutually_exclusive_group()
    
    mutually_exclusive_group.add_argument('-t',
                        '--testsize',
                        type=float,
                        help='split data file as train and test set')
    
    mutually_exclusive_group.add_argument('-td',
                        '--test_path',
                        help='csv file test data')
    

    
    return parser.parse_args(argv[1:])

def _get_model_data(data_path, testsize, test_path, ylabel,
                    **kwargs):
    """
    

    Parameters
    ----------
    data_path : TYPE
        DESCRIPTION.
    testsize : TYPE
        DESCRIPTION.
    test_path : TYPE
        DESCRIPTION.
    ylabel : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    trainset : TYPE
        DESCRIPTION.
    testset : TYPE
        DESCRIPTION.

    """
    print("read in data file ..")

    if testsize:
        X = pd.read_csv(data_path)
        y = X.pop(ylabel)
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=testsize)

    elif test_path:
        x_train = pd.read_csv(data_path)
        y_train = x_train.pop(ylabel)
        x_test = pd.read_csv(test_path)
        y_test = x_test.pop(ylabel)
    else:
        x_train = pd.read_csv(data_path)
        y_train = x_train.pop(ylabel)
        x_test = None
        y_test = None
    
    trainset = (x_train, y_train)
    if x_test is None:
        testset = None
    else:
        testset = (x_test, y_test)
    
    return trainset, testset

def main(argv):
    """Main program.
    Arguments:
      argv: command-line arguments, such as sys.argv (including the program name
        in argv[0]).
      
    Returns:
      Zero on successful program termination, non-zero otherwise.
    """

    args = _ParseArguments(argv)
    file_path = args.data_path
    pipeline = args.pipeline
    # -- run program
    if args.version:
        print("lwmlearn version : {}".format(lwmlearn.__version__))
        return 0
    
    # 
    trainset, testset = _get_model_data(**args.__dict__)
    
    file_name = os.path.splitext(file_path)[0]
    
    if args.automl:
        print("run autoML on {}".format(args.data_path))
        m = LW_model(path=file_name)
        kwargs = {}
        X, y = trainset
        m.run_autoML(X, y, testset, **kwargs)
        
        print("autoML run complete .. \n")
    elif args.cv:
        X, y = trainset
        run_CVscores(X, y)
    else:
        print("run tuning and fitting on {}".format(args.pipeline))
        m = LW_model(estimator=pipeline, path=file_name)
        m.run_analysis(trainset, testset)


def run_main():
    '''
    '''
    try:
        sys.exit(main(sys.argv))
    except SystemExit:
        pass
        # do some cleanup
    except:
        traceback.print_exc()


if __name__ == '__main__':
    run_main()
     