# -*- coding: utf-8 -*-
"""init logging instance logger

* Module Description:
    
    init logger instance
    to generate log files for deployed projects based on Python Standard Library
    :mod:`logging <python:logging>`
    
    See also the tech blog 
    `python logging <https://juejin.im/post/6844903692915703815/>`_

* Key conceipts:
    
    one root logger --> many loggers --> each has many handlers and filters 
    --> each has :ref:`Log Level <log level>` and formatters

* Logger objects:
    
    method: debug, info, warning, error, critical, exception, log
    
         All these methods have arguments (msg, *args, **kwargs);
         
         There are four keyword arguments in kwargs which are inspected: 
         exc_info, stack_info, stacklevel and extra.
             
         exc_info=True it causes exception information to be added 
         (returned by sys.exc_info())
         
         stack_info=true, stack information is added to the logging message, 
         including the actual logging call

* Best Practice:
    
    * use custom logger to output logging messages (logger.name other than 'root')
    
    * use logger method [debug, info, warning, error, critical, exception] to 
    control log level
    
    * use handler settings to distribute logging to differenct  log files 
    
    * use logging.disable(lvl) to mute logging level output

.. _log level:  
    
* Logging level

    #.    NOTSET(0)
    #.    DEBUG(10)
    #.    INFO(20)
    #.    WARNING(30)
    #.    ERROR(40)
    #.    CRITICAL(50)

Created on Wed Aug 26 16:13:35 2020

@author: roger luo
"""
import logging
import os


class Filter_loglevel(logging.Filter):
    '''reconstruct filter method to filter logrecord by a loglevel
    
    Logrecord has attributes that could be used as filters: [name, levelname]
    
    '''
    def __init__(self, loglevel):
        '''

        Parameters
        ----------
        loglevel : string
            log level name, options are [INFO, DEBUG, ERROR, WARNING]

        '''
        super().__init__()

        self.loglevel = loglevel

    def filter(self, record):
        if record.levelname == self.loglevel:
            return True
        else:
            return False


def init_log(logger_name='lwmlearn',
             error_log='error.log',
             all_log='all.log',
             file_mode='w'):
    """init and return logger instance
    

    Parameters
    ----------
    logger_name : str
        logger name other than 'root'. The default is 'lw'
        
    error_log : path
        The default is 'error.log'. output above ERROR level msg
        
    all_log : path
        The default is 'all.log'. output above DEBUG level msg
        
    file_mode : str 
        The default is "w", options are ["w", "a"]

    Return
    -------
    logger : instance
        logger instance.

    """

    # get logger instance by logger_name, if not exist, create one
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    LOG_FMT = "%(levelname)s - logger: %(name)s - %(asctime)s: \n\t %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    fmt = logging.Formatter(LOG_FMT, datefmt=DATE_FORMAT)

    if not logger.handlers:
        # make dirs for log files
        for item in [all_log, error_log]:
            dirs = os.path.split(item)[0]
            if len(dirs) > 0:
                os.makedirs(dirs, exist_ok=True)
        # Error above handler
        e_h = logging.FileHandler(error_log, file_mode)
        e_h.setLevel(logging.ERROR)
        e_h.setFormatter(fmt)
        # add error handler
        logger.addHandler(e_h)

        # info handler only
        info_h = logging.FileHandler(all_log, file_mode)
        info_h.setLevel(logging.DEBUG)
        info_h.setFormatter(fmt)
        # add info handler
        logger.addHandler(info_h)

        # streamhandler output all record to sys.stdout
        s_h = logging.StreamHandler()
        s_h.setLevel(logging.INFO)
        logger.addHandler(s_h)

    return logger


if __name__ == '__main__':
    pass

    logging.disable(logging.NOTSET)

    logger = init_log(logger_name='mylog',
                      error_log='log/error.log',
                      all_log='log/all_log.log')
    logger.info('this is a info message')
    logger.error("this is an error", exc_info=True)
    logger.exception('this is an exception')
