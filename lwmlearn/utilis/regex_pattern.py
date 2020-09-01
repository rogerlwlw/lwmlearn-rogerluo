# -*- coding: utf-8 -*-
"""Common used regex pattern

Created on Tue Feb 12 13:47:28 2019

@author: roger
"""

import re
from pandas.core.dtypes import api

def common_regex_pattern(key):
    '''retrieve common regex pattern by key name
    

    Parameters
    ----------
    key : str
        string name of regex pattern. if key=None, return all available keys

    Returns
    -------
    reg : 
        regex pattern 
        
    '''
    d = {
        "numeric" : "^[-+]?\d*(?:\.\d*)?(?:\d[eE][+\-]?\d+)?(\%)?$",
        "empty" : "^[-\s\_]*$",
        "email" :"^\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*$",
        "ID18" : "^((\d{18})|([0-9x]{18})|([0-9X]{18}))$",
        "IP" : "((?:(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d)\\.){3}(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d))",
        
 
       }
    
    if key is None:
        return d.keys()
    else:
        reg = d.get(key)
        return reg

if __name__ == '__main__':
    
    number = common_regex_pattern('numeric')
    empty = common_regex_pattern('empty')
    re.match(number, '1e-3')
    re.match(empty, '  _-  \t')
