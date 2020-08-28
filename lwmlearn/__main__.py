# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:08:28 2020

@author: Administrator
"""
from __future__ import print_function

import argparse
import sys
import traceback

import lwmlearn


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
    # parser.add_argument(
    #         'arg1',
    #         help='help message XXXX')

    parser.add_argument('-v',
                        '--version',
                        action='store_true',
                        help='show version number and exit')
    
    # optional argument
    parser.add_argument(
            '-op1',
            '--option1',
            action='store_true',
            default=None,
            help='help message XXXX')
    
    # mutually exclusive argument
    mutually_exclusive_group = parser.add_mutually_exclusive_group()
    mutually_exclusive_group.add_argument(
            '-m_opt1',
            '--m_option1',
            default=None,
            choices=[1,2,3],
            type=int,
            help='help message XXX')
    
    return parser.parse_args(argv[1:])

def main(argv):
    """Main program.
    Arguments:
      argv: command-line arguments, such as sys.argv (including the program name
        in argv[0]).
      
    Returns:
      Zero on successful program termination, non-zero otherwise.
    """
    
    args = _ParseArguments(argv)
    # -- run program
    if args.version:
        print("lwmlearn version : {}".format(lwmlearn.__version__))
        return 0    
    
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



