# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:55:15 2018

@author: roger

to format python script using yapf

yapf_allfile:
    codeformat all filies under given rootdir
"""
import yapf
import os


def yapf_allfile(rootdir='.',
                 style='pep8',
                 subfolder=False,
                 suffix=['.py'],
                 **knobs):
    ''' code format all filies under rootdir
    
    parameters
    ----------
    rootdir : path
        dir of files
    style : str
        'pep8'/'google'/'chromium'/'facebook'
    subfolder : bool
        default True, including subfolders
    suffix : suffix
        default '.py'
    
    return
    -------
    None
    
    '''
    styles = {'google', 'pep8', 'chromium', 'facebook'}
    if style not in styles:
        raise KeyError('invalid style input')

    if subfolder:
        files = (file for k, file in _file_traverse(rootdir).items())
    else:
        files = (file for k, file in _listDir(rootdir).items())

    for file in files:
        try:

            if os.path.splitext(file)[1] in suffix:
                config = {
                    'based_on_style': style,
                    'column_limit': 79,
                    'BLANK_LINES_AROUND_TOP_LEVEL_DEFINITION': 2
                }
                config.update(**knobs)
                yapf.yapf_api.FormatFile(file,
                                         style_config=config,
                                         in_place=True)
                print("{} has been code formated as '{}' ...\n".format(
                    file, style))
        except Exception:
            print(file, 'code formation failed')


def _listDir(rootDir):
    '''
    
    return
    ------
    dict : dict
        {filename : filepath} under rootDir not including subfolder
    '''
    file_dict = {}
    for filename in os._listDir(rootDir):
        pathname = os.path.join(rootDir, filename)
        if os.path.isfile(pathname):
            file_dict[filename] = pathname
        else:
            pass
    return file_dict


def _file_traverse(rootDir):
    '''
    return
    ------
    dict : dict
        {filename : filepath} under rootDir including subfolders
    '''
    file_dict = dict([file, os.path.join(dirpath, file)]
                     for dirpath, dirnames, filenames in os.walk(rootDir)
                     for file in filenames)
    return file_dict
