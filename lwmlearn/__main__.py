# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:08:28 2020

@author: Administrator
"""
from __future__ import print_function

import argparse
import sys
import traceback

import pprint
import pandas as pd
import os 

import lwmlearn
from lwmlearn import LW_model, run_CVscores
from sklearn.model_selection import train_test_split
from argparse import ArgumentDefaultsHelpFormatter, REMAINDER

from lwmlearn.preprocess.lw_base import _convert_numeric

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
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=ArgumentDefaultsHelpFormatter)

    # # positional argument
    parser.add_argument('data_path',
                        help="csv file train data")
    # optional argument
    parser.add_argument('-y',
                        '--ylabel',
                        type=str,
                        default='y',
                        help='class label')
    
    parser.add_argument('-s',
                        '--sample',
                        type=int,
                        const=1000,
                        nargs="?",
                        help='sample number of dataset, if not specified directly use 1000')
    
    
    parser.add_argument('-is',
                        '--is_search',
                        action="store_true",
                        help='perform hyper-tuning')
    
    parser.add_argument('-k',
                        '--kind',
                        action="store",
                        default="bayesiancv",
                        help="hyper-tuning method ['bayesiancv', 'gridcv']")
    
    
    parser.add_argument('-v',
                        '--version',
                        action='store_true',
                        help='show version number and exit')
    
    # mutually exclusive argument
    me_group0 = parser.add_mutually_exclusive_group()

    me_group0.add_argument('-rp',
                        '--pipeline',
                        nargs='?',
                        const='clean_ordi_fxgb_LGBMClassifier',
                        help='run pipeline analysis, if not specified directly, use clean_ordi_fxgb_LGBMClassifier')
        
    me_group0.add_argument('-ra',
                        '--automl',
                        action='store_true',
                        help='run autoML on data ')

    me_group0.add_argument('-rcv',
                        '--cv_only',
                        action='store_true',
                        help='run CV score only to save you some time')
    
    # mutually exclusive argument
    mutually_exclusive_group = parser.add_mutually_exclusive_group()
    
    mutually_exclusive_group.add_argument('-ts',
                        '--testsize',
                        type=float,
                        const=0.3,
                        nargs="?",
                        help='split data file as train and test set')
    
    mutually_exclusive_group.add_argument('-tp',
                        '--test_path',
                        help='csv file test data')
    
    parser.add_argument("--set",
                        metavar="KEY=VALUE",
                        nargs='+',
                        help="Set a number of key-value pairs "
                             "(do not put spaces before or after the = sign). "
                             "If a value contains spaces, you should define "
                             "it with double quotes: "
                             'foo="this is a sentence". Note that '
                             "values are always treated as strings.")
    # parse sys args as dict
    args, unknown = parser.parse_known_args(argv[1:])
    
    kwargs = vars(args)
    set_kwargs = dict(map(lambda s: _parse_var(s), kwargs.pop('set')))
    kwargs.update(set_kwargs)
    return kwargs

def _parse_var(s):
    """
    Parse a key, value pair, separated by '='
    That's the reverse of ShellArgs.

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    items = s.split('=')
    key = items[0].strip() # we remove blanks around keys, as is logical
    if len(items) == 2:
        val = _convert_numeric(items[1]) 
        return (key, val)
    else:
        raise ValueError("{} wrong input syntax".format(s))

def _get_model_data(data_path, testsize, test_path, ylabel, sample,
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
    data = pd.read_csv(data_path)
    if sample:
        data = data.sample(sample)
       
    if testsize:
        X = data
        y = X.pop(ylabel)
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=testsize)
    elif test_path:
        x_train = data
        y_train = x_train.pop(ylabel)
        x_test = pd.read_csv(test_path)
        y_test = x_test.pop(ylabel)
    else:
        x_train = data
        y_train = x_train.pop(ylabel)
        x_test = None
        y_test = None
    
    if x_test is None:
        testset = None
    else:
        testset = (x_test, y_test)
    trainset = (x_train, y_train)
    return trainset, testset

def _main(data_path, testsize, test_path, ylabel, sample, 
           version, pipeline, automl, cv_only, is_search, kind, 
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
    sample : TYPE
        DESCRIPTION.
    is_search : TYPE
        DESCRIPTION.
    kind : TYPE
        DESCRIPTION.
    version : TYPE
        DESCRIPTION.
    pipeline : TYPE
        DESCRIPTION.
    automl : TYPE
        DESCRIPTION.
    cv_only : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # -- run program
    if version:
        print("lwmlearn version : {}".format(lwmlearn.__version__))
        return 0
    
    # 
    trainset, testset = _get_model_data(data_path, testsize, test_path, ylabel,
                                        sample)
    
    file_name = os.path.splitext(data_path)[0]
    
    if automl:
        print("run autoML on {} .. \n".format(data_path))
        m = LW_model(path=file_name)
        X, y = trainset
        m.run_autoML(X, y, testset, 
                     is_search=is_search,
                     kind=kind,
                     **kwargs)
        if testset is not None:
            x_test, y_test = testset
            m.plot_all_auc(False, X=x_test, y=y_test, save_fig=True)
        print("autoML run complete .. \n")
    elif cv_only:
        X, y = trainset
        df = run_CVscores(X, y)
        df.to_excel('cv_score.xlsx', index=False)
    elif pipeline:
        print("run tuning and fitting on {} .. \n".format(pipeline))
        path = os.path.join(file_name, pipeline)
        m = LW_model(estimator=pipeline, path=path)
        m.run_analysis(trainset, testset, **kwargs)    
        
    return


def main(argv):
    """Main program.
    Arguments:
      argv: command-line arguments, such as sys.argv (including the program name
        in argv[0]).
      
    Returns:
      Zero on successful program termination, non-zero otherwise.
    """

    kwargs = _ParseArguments(argv)
    # -- run program
    pprint.pp(kwargs, depth=2, width=60)
    _main(**kwargs)


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
     