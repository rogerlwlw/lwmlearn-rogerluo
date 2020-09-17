# -*- coding: utf-8 -*-
"""init logging instance logger

* Module Description:
    
    init logger instance
    to generate log files for deployed projects based on Python Standard Library
    :mod:`logging <python:logging>`
    
    See also the tech blog 
    `python logging <https://juejin.im/post/6844903692915703815/>`_
    
    `loguru<https://loguru.readthedocs.io/en/stable/index.html>`_

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
import sys

from os import makedirs
from os.path import dirname, exists

USE_LOG = 'loguru' # loguru for loguru logger; logging for python logger

loggers = {}
logger_guru = None

LOG_ENABLED = True  # 
LOG_TO_CONSOLE = True  # 
LOG_TO_FILE = True  #
    
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

def init_log():
    '''
    '''
    if USE_LOG == 'logging':
        return python_logger()
    
    if USE_LOG == 'loguru':
        return loguru_logger()
    
def python_logger():
    """init and return logger instance of python standard logging
    

    Parameters
    ----------
    name : str
        logger name other than 'root'. The default is 'lwmlearn'.
    log_path : path, optional
        DESCRIPTION. The default is './runtime.log'.
    log_level : str, optional
        DESCRIPTION. The default is 'DEBUG'.
    file_mode : str, optional
        DESCRIPTION. The default is 'w'.

    Return
    -------
    logger : instance
        logger instance.

    """

    global loggers

    LOG_FMT = '%(asctime)s - %(levelname)s - process: %(process)d - %(filename)s - %(name)s - %(lineno)d - %(module)s - %(message)s'  # 每条日志输出格式
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    file_mode='a'
    name='lwmlearn'
    log_path='./runtime.log'
    log_level='DEBUG'

    if not name: name = __name__
 
    if loggers.get(name):
        return loggers.get(name)
 
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    fmt = logging.Formatter(LOG_FMT, datefmt=DATE_FORMAT)
    fmt_console = logging.Formatter(
        "%(asctime)s - %(levelname)s: \n\t %(message)s", 
        datefmt=DATE_FORMAT)             
    # out to console
    if LOG_ENABLED and LOG_TO_CONSOLE:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=log_level)
        stream_handler.setFormatter(fmt_console)
        logger.addHandler(stream_handler)       
    
    # out to file
    if LOG_ENABLED and LOG_TO_FILE:
        log_dir = dirname(log_path)
        if not exists(log_dir): makedirs(log_dir)
        # FileHandler
        file_handler = logging.FileHandler(log_path, 
                                           file_mode, 
                                           encoding='utf-8')
        file_handler.setLevel(level=log_level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
                 
    loggers[name] = logger    
    return logger

def loguru_logger():
    '''init and return loguru logger
    
    return
    ------
    logger : instance
        - sys.stderr handler
        - runtime_{time}.log file handler, retention 10 days rotation 3 days
    '''
    global logger_guru
    
    # date_fmt = "time:YYYYMMDD"
    # stderr_time = "time:HH:mm:ss A"
    if logger_guru is None:
        from loguru import logger
        logger.remove()
        # out to console
        logger.add(sys.stderr, format='{time} {level}: \n  {message}',
                   level='INFO')
        if LOG_TO_FILE:
            # out to runtime.log file
            logger.add('./logs/runtime_{time:YYYYMMDD}.log', 
                       level='DEBUG',
                       rotation="3 days", 
                       retention='10 days',
                       backtrace=True, 
                       diagnose=True,
                       enqueue=True, # Multiprocess-safe
                       )
    
    return logger

if __name__ == '__main__':
    pass
    logger = init_log()
    logger.info('this is a info message')
    logger.error("this is an error")
    logger.exception('this is an exception')
    logging.shutdown()
