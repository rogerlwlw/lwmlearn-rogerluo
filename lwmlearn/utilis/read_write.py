# -*- coding: utf-8 -*-
"""IO functionality

Module Description:
    
    offers :class:`Objs_management` class to load or dump data objects to file
    
    supported file formats are [.pkl, .csv, .xlsx, .json]
    
    supported data formats are [dataframe, dict, python class instance, text]


Created on Tue Dec 18 14:36:20 2018

@author: roger luo

"""
import pandas as pd
import numpy as np
import os
import pickle
import shutil
import json

from sklearn.utils import check_consistent_length

from lwmlearn.utilis.utilis import get_flat_list, get_kwargs

from ..lwlogging import init_log

logger = init_log()


class Path_File():
    '''descriptor to initialize file attributes
    
    used internally, logging will record the IO process
    
    path: str
        check existance, if not create one

    file: dirs/filename
        check file existance, if not raise FileNotFoundError Error
 
    Newfile: 
        if not exist create one
        
    '''
    @property
    def path_(self):
        return self._p

    @path_.setter
    def path_(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                logger.info("info: path '{}' created ...".format(path))

            self._p = os.path.relpath(path)
        except Exception:

            logger.exception("invalid path input '%s' " % path,
                             stack_info=True)
            raise NotADirectoryError()

    @path_.deleter
    def path_(self):
        for root, dirnames, file in os.walk(self._p, topdown=False):
            for i in file:
                os.remove(os.path.join(root, i))
        shutil.rmtree(self._p, ignore_errors=True)

        logger.info("info: path '{}' removed... \n".format(self._p))

    # file
    @property
    def file_(self):
        return self._f

    @file_.setter
    def file_(self, file):
        if os.path.isfile(file):
            self._f = os.path.relpath(file)
        else:
            logger.exception("file not found '%s' " % file, stack_info=True)
            raise FileNotFoundError()

    @file_.deleter
    def file_(self):
        os.remove(self._f)
        logger.info("info: file '{}' removed".format(self._f))

    ## newfile
    @property
    def newfile_(self):
        return self._nf

    @newfile_.setter
    def newfile_(self, file):
        try:
            if os.path.isfile(file):
                os.remove(file)
                logger.info("info: old file '{}' deleted...\n ".format(file))

            dirs, filename = os.path.split(file)
            if not os.path.exists(dirs) and len(dirs) > 0:
                os.makedirs(dirs, exist_ok=True)
                logger.info("info: path '{}' created...\n".format(dirs))
            self._nf = file
        except Exception:
            logger.exception('invalid path input {}'.format(file),
                             stack_info=True)
            raise NotADirectoryError()

    @newfile_.deleter
    def newfile_(self):
        os.remove(self._nf)
        logger.info("info: file '{}' removed".format(self._nf))


class Reader(Path_File):
    '''read in python objects contained in file object
    
    supported file formats are ['.xlsx', '.csv', '.pkl', '.txt', '.sql']
    
    .. note..
        '.sql' file means sql query script not data file
    
    method:
        
        read : return obj read from file
            
        read_all : return dict of read in objs for dirs
        
    '''
    def __init__(self, path):
        ''' init path and logger variable 
        '''
        self.path_ = path

    def read(self, file, **kwargs):
        '''read obj from file
        
        parameters
        -----------
        
        file : str or file object
        
            file to read, files options include 
            ['.xlsx', '.csv', '.pkl', '.txt', '.sql']   
        
        kwargs
            other key words arguments used by read api, 
            such as :func:`pandas.read_csv`, :func:`pandas.read_excel`
        
        return
        -------
        
        obj : obj
            data object such as pandas DataFrame
            
        '''
        self.file_ = file
        read_api = _rd_apis(self.file_)
        try:
            kw = get_kwargs(read_api, **kwargs)
            rst = read_api(self.file_, **kw)
            logger.info("<obj>: '{}' read from '{}\n".format(
                rst.__class__.__name__, self.file_))
            return rst
        except Exception:
            msg = "fail to read file '{}' ".format(self.file_)
            logger.exception(msg, stack_info=True)
            raise IOError()

    def read_all(self, suffix=None, path=None, subfolder=False, **kwargs):
        '''return dict of all objects in given dirs as {filename : objects}
        
        suffix could be specified
        
        parameters
        -----------
        
        suffix: str 
        
            file suffix or list of suffix to read in
            
        path: path 
        
            relative path to read from, default current self.path_
        
        return
        -------
        d: dict
            dict of read in object {filename : objects}
            
        '''
        if path is None:
            path = self.path_
        else:
            path = os.path.join(self.path_, path)

        file_dict = _get_files(path, suffix, subfolder)

        d = {}
        for k, v in file_dict.items():
            load = self.read(v, **kwargs)
            if load is not None:
                d[k] = load
        return d

    def list_all(self,
                 suffix=None,
                 path=None,
                 subfolder=False,
                 keep_suffix=True):
        '''return dict of {filename: filepath} in given dirs
        '''
        if path is None:
            path = self.path_
        else:
            path = os.path.join(self.path_, path)

        if keep_suffix:
            return _get_files(path, suffix, subfolder)
        else:
            file_d = _get_files(path, suffix, subfolder)
            return {os.path.splitext(k)[0]: v for k, v in file_d.items()}


def _load_pkl(file):
    '''return unpickled obj from 'pkl' file
    '''
    with open(file, 'rb') as f:
        pkl = pickle.Unpickler(f)
        obj = pkl.load()
    return obj


def _read_file(file):
    '''return 'str' obj from file by calling f.read() method
    '''
    with open(file, 'r') as f:
        obj = f.read()
    return obj


def _get_files(dirpath, suffix=None, subfolder=False):
    '''return file dict {filename : filepath}
    
    parameters
    -----------
    
    dirpath - str
        dir_x to traverse
    
    suffix : str or list of extension names, like ['.xlsx', 'csv']
        to include file extensions, default None, to include all extensions
    
    subfolder : bool
        true to traverse subfolders, False only the given dirpath
        
    '''
    if subfolder:
        get_dirs = traverse_all_dirs
    else:
        get_dirs = traverse_dir

    rst = {
        k: v
        for k, v in get_dirs(dirpath).items()
        if os.path.splitext(v)[1] in get_flat_list(suffix) or not suffix
    }
    return rst


def _rd_csv(file, **kwargs):
    '''try to read csv file using "utf-8" or "gbk" encoding
    
    Parameters
    ----------
    file : 
        file path or file object 
    
    **kwargs :
        key word arguments for :func:`pandas.read_csv`

    Returns
    -------
    df : dataframe
    
    '''

    try:
        kwargs.update(encoding='utf-8')
        return pd.read_csv(file, **kwargs)
    except UnicodeDecodeError:
        kwargs.update(encoding='gbk')
        return pd.read_csv(file, **kwargs)


def _read_json(file):
    '''return dict obj from 'json' file
    '''
    with open(file, 'r') as f:
        d = json.load(f)
    return d


def _rd_apis(file):
    '''return read api for given suffix of file
    
    suffix options are ['.xlsx', '.csv', '.txt' , '.sql', '.json', '.pkl']
    default unpickle method '_load_pkl' will be used
    
    parameters
    -----------
    file 
        - file to read obj from
    '''
    api_collections = {
        '.xlsx': pd.read_excel,
        '.csv': _rd_csv,
        '.txt': _read_file,
        '.sql': _read_file,
        '.json': _read_json,
    }

    suffix = os.path.splitext(file)[1]
    rst = api_collections.get(suffix, _load_pkl)
    return rst


class Writer(Path_File):
    '''write objects data to file
    
    method
    -----------
    write:
        write obj data into file
    '''
    def __init__(self, path):
        '''init path and logger variable   
        '''
        self.path_ = path

    def write(self, obj, file, **kwargs):
        '''dump obj into file under self.path_

        parameters
        ------------
        
        file : path
        
            like 'dirs/filename.pkl', supported suffix are
            [.pkl, .xlsx, .csv, .pdf, .png], 
        
        kwargs : other keys arguments for suffix specified api
                
            * sheet_name : list
                for dumping to excel file '.xlsx',    
                if list of dataframe are passed as obj, then dump them into list of 
                sheet_names 
        '''
        file = os.path.join(self.path_, file)
        file = os.path.relpath(file)
        self.newfile_ = file
        wr_api = _wr_apis(self.newfile_)
        try:
            wr_api(obj, self.newfile_, **kwargs)
            msg = "<obj>: '{}' dumped into '{}...\n".format(
                obj.__class__.__name__, file)
            logger.info(msg)
        except Exception:
            msg = "<failure>: '{}' written failed ...".format(file)
            logger.exception(msg, stack_info=True)
            raise IOError()


def _wr_apis(file):
    '''return write api of given suffix of file
    
    default will use _dump_pkl
    
    parameters
    ----------
    obj : obj
        obj to be written
    file : path
        file to wirte into
        
    '''
    api_collections = {
        '.xlsx': _dump_df_excel,
        '.csv': _dump_df_csv,
        '.pdf': _save_plot,
        '.png': _save_plot,
        '.json': _dump_json
    }

    suffix = os.path.splitext(file)[1]
    rst = api_collections.get(suffix, _dump_pkl)
    return rst


def _dump_json(obj, file):
    '''dump obj as json file
    '''
    with open(file, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)


def _dump_pkl(obj, file, **kwargs):
    '''dump obj as pickle file
    
    parameters
    -----------
    
    obj : obj
        python objects
    
    file : path
        file to dump obj into
        
    '''
    with open(file, 'wb') as f:
        pkl = pickle.Pickler(f)
        pkl.dump(obj)


def _dump_df_excel(obj, file, **kwargs):
    '''dump df to excel file
    
    parameters
    ----------
    obj: 
        2d array like data
    file:
        str or file obj
    
    **kwargs
    ----------
    sheet_name : list
        if list of dataframe are passed as obj, then dump them into list of 
        sheet_names 
        
    '''
    writer = pd.ExcelWriter(file)
    obj = get_flat_list(obj)

    sheet_name = kwargs.get('sheet_name')

    if sheet_name is None:
        sheet_name = ['sheet' + str(i + 1) for i in range(len(obj))]
    else:
        sheet_name = get_flat_list(sheet_name)
        check_consistent_length(obj, sheet_name)

    for data, name in zip(obj, sheet_name):
        try:
            data = pd.DataFrame(data)
            kw = get_kwargs(data.to_excel, **kwargs)
            kw.update({
                'sheet_name': name,
                'index': kwargs.get('index', False)
            })
            data.to_excel(writer, **kw)
        except Exception as e:
            print(repr(e))
            continue
    writer.save()


def _dump_df_csv(obj, file, index=False, **kwargs):
    ''' dump df to csv
    '''
    try:
        data = pd.DataFrame(obj)
        data.to_csv(file, index=index, **get_kwargs(data.to_csv, **kwargs))
    except Exception as e:
        print(repr(e))


def _save_plot(fig, file, **kwargs):
    '''save the figure obj , if fig is None, save the current figure
    '''
    if hasattr(fig, 'savefig'):
        fig.savefig(file, **kwargs)
    else:
        getattr(fig, 'get_figure')().savefig(file, **kwargs)


class Objs_management(Reader, Writer):
    '''manage read & write of objects from/into file
    '''
    def __init__(self, path):
        '''init path and logger variable 
        
        parametrs
        ---------
        path : path
            default path of Object_management instance
            
        '''
        super().__init__(path)

    def _remove_path(self):
        '''remove path and all files within
        '''
        del self.path_

    def set_path(self, path):
        '''set path variable
        '''
        self.path_ = path
        return self


def traverse_dir(rootDir):
    '''traverse files under rootDir not including subfolder
   
    parameters
    ------------
    rootDir : path
        directory to traverse
    
    return
    -------
    dict : {filename : file}
    '''
    file_dict = {}
    for filename in os.listdir(rootDir):
        pathname = os.path.join(rootDir, filename)
        if os.path.isfile(pathname):
            file_dict[filename] = pathname
    return file_dict


def traverse_all_dirs(rootDir):
    '''traverse files under rootDir including subfolders
    
    parameters
    ------------
    
    rootDir : path
        directory to traverse
        
    return
    -------
    
    dict : {filename : file}
    '''
    file_dict = dict([file, os.path.join(dirpath, file)]
                     for dirpath, dirnames, filenames in os.walk(rootDir)
                     for file in filenames)
    return file_dict


def search_file(filename,
                search_path,
                suffix=None,
                subfolder=False,
                include_suffix=False):
    '''return searched file path by searching search_path
    
    parameters
    ------------
    filename : str
        file name, may or may not include suffix
    
    search_path : list of path
        dirs to search from
    
    indclude_suffix : bool
        if True, include the suffix extension in filename
        
    return
    --------
    filepath : path
    
    '''
    file_collector = {}
    for i in np.array(search_path):
        file_collector.update(_get_files(i, suffix, subfolder))

    if not include_suffix:
        file_collector = {
            os.path.splitext(k)[0]: v
            for k, v in file_collector.items()
        }

    return file_collector.get(filename)
