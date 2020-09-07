# -*- coding: utf-8 -*-
"""Common used regex pattern


use example
------------

.. ipython::
    :okwarning:
        
    In [35]: from lwmlearn.utilis.regex_pattern import common_regex_pattern

    In [47]: import re

    In [48]: number = common_regex_pattern('numeric')
    
    In [49]: email  = common_regex_pattern('email')

    In [43]: common_regex_pattern(None)
    
    In [37]: re.match(number, '1e-3')   
    
    In [38]: re.match(number, '-1e+3')
    
    In [39]: re.match(number, '10%')
    # not matched
    In [40]: re.match(number, 'N1990')
    
    In [41]: re.match(email, 'coolww@outlook.com')

Created on Tue Feb 12 13:47:28 2019

@author: roger
"""

import re

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
        # 强密码(必须包含大小写字母和数字的组合，不能使用特殊字符，长度在8-10之间)：
        "pass" : "^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,10}$",          
 
       }
    
    if key is None:
        return d.keys()
    else:
        reg = d.get(key)
        return reg

if __name__ == '__main__':
    
    from lwmlearn.utilis.regex_pattern import common_regex_pattern
    import re
    number = common_regex_pattern('numeric')
    email  = common_regex_pattern('email')
    
    common_regex_pattern(None)
    
    re.match(number, '1e-3')
    re.match(number, '-1e+3')
    re.match(number, '10%')
    re.match(number, 'N1990')
    re.match(email, 'coolww@outlook.com')
