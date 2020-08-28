# -*- coding: utf-8 -*-
"""
Module Description:
    
    offers :class:`Objs_management` class to load or dump objects data to file
    supported file formats are [.pkl, .csv, .xlsx, .json]


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


class _Obj():
    pass


class Path_File():
    '''descriptor to init path, file and newfile attributes
    
    logging will record IO process
    
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
                print("info: path '{}' created...".format(path))

            self._p = os.path.relpath(path)
        except Exception as e:
            print(repr(e))
            raise NotADirectoryError("invalid path input '%s' " % path)

    @path_.deleter
    def path_(self):
        for root, dirnames, file in os.walk(self._p, topdown=False):
            for i in file:
                os.remove(os.path.join(root, i))
        shutil.rmtree(self._p, ignore_errors=True)
        print("info: path '{}' removed... \n".format(self._p))

    @property
    def file_(self):
        return self._f

    @file_.setter
    def file_(self, file):
        if os.path.isfile(file):
            self._f = os.path.relpath(file)
        else:
            raise FileNotFoundError("file '{}' does not exist".format(file))

    @file_.deleter
    def file_(self):
        os.remove(self._f)
        print("info: file '{}' removed".format(self._f))

    ## ----
    @property
    def newfile_(self):
        return self._nf

    @newfile_.setter
    def newfile_(self, file):
        try:
            if os.path.isfile(file):
                os.remove(file)
                print("info: old file '{}' deleted...\n ".format(file))

            dirs, filename = os.path.split(file)
            if not os.path.exists(dirs) and len(dirs) > 0:
                os.makedirs(dirs, exist_ok=True)
                print("info: path '{}' created...\n".format(dirs))
            self._nf = file
        except Exception as e:
            print(repr(e))
            raise NotADirectoryError('invalid path input {}'.format(file))

    @newfile_.deleter
    def newfile_(self):
        os.remove(self._nf)
        print("info: file '{}' removed".format(self._nf))


class Reader(Path_File):
    '''read in python objects contained in file object
    
    supported suffix of file are ['.xlsx', '.csv', '.pkl', '.txt', '.sql']
    
    method
    -------
    read : 
        return obj read from file
        
    read_all :
        return dict of read in objs
        
    '''
    def __init__(self, path):
        ''' init path variable 
        '''
        self.path_ = path

    def read(self, file, **kwargs):
        '''read obj from file
        
        supported suffix of file are
        - ['.xlsx', '.csv', '.pkl', '.txt', '.sql']       
        file - str or file object
            - file to read
        '''
        self.file_ = file
        read_api = _rd_apis(self.file_)
        try:
            kw = get_kwargs(read_api, **kwargs)
            rst = read_api(self.file_, **kw)
            print("<obj>: '{}' read from '{}\n".format(rst.__class__.__name__,
                                                       self.file_))
            return rst
        except Exception as e:
            print("<failure>: file '{}' read failed".format(self.file_))
            print(repr(e), '\n')

    def read_all(self, suffix=None, path=None, subfolder=False, **kwargs):
        '''read in all dataframe objects
        
        parameters
        -----------
        
        suffix: str 
            file suffix to read in
            
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

    def list_all(self, suffix=None, path=None, subfolder=False, 
                 keep_suffix=True):
        '''return all available file under given 'path' as dict
        '''
        if path is None:
            path = self.path_
        else:
            path = os.path.join(self.path_, path)
        
        if keep_suffix:
            return _get_files(path, suffix, subfolder)
        else:
            file_d = _get_files(path, suffix, subfolder)
            return {os.path.splitext(k)[0] : v for k, v in file_d.items()}


def _load_pkl(file):
    '''return unpickled obj from 'pkl' file
    '''
    with open(file, 'rb') as f:
        pkl = pickle.Unpickler(f)
        obj = pkl.load()
    return obj


def _read_file(file):
    ''' return 'str' obj from file by calling f.read() method
    '''
    with open(file, 'r') as f:
        obj = f.read()
    return obj


def _get_files(dirpath, suffix=None, subfolder=False):
    ''' return file dict {filename : file}

    dirpath - str
        - dir_x to traverse
    suffix -->extension name, or list of extension names, egg ['.xlsx', 'csv']
        - to include file extensions, default None, to include all extensions
    subfolder --> bool
        - true to traverse subfolders, False only the given dirpath
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
    '''try to read csv file using "utf-8" or "gbk"
    
    Parameters
    ----------
    file : 
        file path or file object 

    Returns
    -------
    None.

    '''
    try:
        return pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding='gbk')

    return


def _read_json(file):
    '''return dict obj from 'json' file
    '''
    with open(file, 'r') as f:
        d = json.load(f)
    return d


def _rd_apis(file):
    '''return read api for given suffix of file, default _load_pkl will be
    used
    
    api parameters
    ----
    file 
        - file to read obj from
    **kwargs
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
    '''write objects into file
    
    method
    -----
    write:
        write obj into file
    '''
    def __init__(self, path):
        ''' init path variable '''
        self.path_ = path

    def write(self, obj, file, **kwargs):
        '''dump obj into file under self.path_

        file
            - filename + suffix egg 'filename.pkl'
            - supported suffix are [.pkl, .xlsx, .csv, .pdf, .png], 
            see _wr_apis
        
        **kwargs
            - other keys arguments for suffix specified api
        '''
        file = os.path.join(self.path_, file)
        file = os.path.relpath(file)
        self.newfile_ = file
        wr_api = _wr_apis(self.newfile_)
        try:
            wr_api(obj, self.newfile_, **kwargs)
            print("<obj>: '{}' dumped into '{}...\n".format(
                obj.__class__.__name__, file))
        except Exception as e:
            print(repr(e))
            print("<failure>: '{}' written failed ...".format(file))


def _wr_apis(file):
    ''' return write api of given suffix of file, default will use _dump_pkl
    
    api parameters
    ---
    obj
        - obj to be written
    file
        - file to wirte into
    **kwargs
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
    '''
    '''
    with open(file, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)


def _dump_pkl(obj, file, **kwargs):
    '''
    obj - python objects
    file - file to dump obj into
    '''
    with open(file, 'wb') as f:
        pkl = pickle.Pickler(f)
        pkl.dump(obj)


def _dump_df_excel(obj, file, **kwargs):
    '''dump df to excel
    
    obj: 
        2d array like data
    file:
        str or file obj:        
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
    def __init__(self, path):
        '''manage read & write of objects from/into file
        '''
        super().__init__(path)

    def _remove_path(self):
        '''remove path and all files within
        '''
        del self.path_

    def set_path(self, path):
        '''set path 
        '''
        self.path_ = path
        return self


def traverse_dir(rootDir):
    '''traverse files under rootDir not including subfolder
    return
    ----
    dict - {filename : file}
    '''
    file_dict = {}
    for filename in os.listdir(rootDir):
        pathname = os.path.join(rootDir, filename)
        if os.path.isfile(pathname):
            file_dict[filename] = pathname
    return file_dict


def traverse_all_dirs(rootDir):
    '''traverse files under rootDir including subfolders
    return
    ----
    dict - {filename : file}
    '''
    file_dict = dict([file, os.path.join(dirpath, file)]
                     for dirpath, dirnames, filenames in os.walk(rootDir)
                     for file in filenames)
    return file_dict

def search_file(filename, search_path, suffix=None, subfolder=False, 
                include_suffix=False):
    '''
    return file_dirpath for filename
    '''
    file_collector = {}
    for i in np.array(search_path):
        file_collector.update(_get_files(i, suffix, subfolder))
    if not include_suffix:
        file_collector = {os.path.splitext(k)[0] : v 
                          for k, v in file_collector.items()}    
    return file_collector.get(filename)