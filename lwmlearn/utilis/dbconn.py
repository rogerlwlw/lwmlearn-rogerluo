# -*- coding: utf-8 -*-
"""

Created on Wed Dec 11 18:39:56 2019

@author: rogerluo

"""
import os
import pandas as pd
from sqlalchemy.types import Integer, Float, String
from sqlalchemy import create_engine

from lwmlearn.lwlogging import init_log
from lwmlearn.utilis.docstring import Substitution, dedent, Appender

logger = init_log()

def _read_sqlfile(sql):
    """
    

    Parameters
    ----------
    sql : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import os
    
    if os.path.isfile(sql):
        with open(sql, 'r') as f:
            sql = f.read()
    
    return sql

class EngineConn():
    '''Get data base engine to down/upload data or excute query
        
    Parameters
    ----------
    url : str
        first positional argument, usually a string
        that indicates database dialect and connection arguments::
        
        
            engine = create_engine("postgresql://scott:tiger@localhost/test")
        
        Additional keyword arguments may then follow it which
        establish various options on the resulting :class:`_engine.Engine`
        and its underlying :class:`.Dialect` and :class:`_pool.Pool`
        constructs::
        
            engine = create_engine("mysql://scott:tiger@hostname/dbname",
                                        encoding='latin1', echo=True)
        
        The string form of the URL is
        ``dialect[+driver]://user:password@host/dbname[?key=value..]``, where
        ``dialect`` is a database name such as ``mysql``, ``oracle``,
        ``postgresql``, etc., and ``driver`` the name of a DBAPI, such as
        ``psycopg2``, ``pyodbc``, ``cx_oracle``, etc.  Alternatively,
        the URL can be an instance of :class:`~sqlalchemy.engine.url.URL`.
     
    '''
    def __init__(self, url, **kwargs):
        '''
        '''
        #Oracle client encoding
        os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
        self._engine = create_engine(url, **kwargs)
    
    @dedent
    @Substitution(to_sql=pd.DataFrame.to_sql.__doc__)
    def upload(self, df, name, dtype=None, **kwargs):
        """
        upload df table data to database

        Parameters
        ----------
        df : df
            DESCRIPTION.
        name : str
            name of table
        
        dtype : bool, optional
            dict = [col : sqlTypes], if None, guess from dtypes of df    
        
        keyword args
        ------------
        
        {to_sql}

        """
        
        if dtype is None:
            dtype = self._get_map_df_types(df)
            
        logger.info('begin uploading data ...\n ')
        df.to_sql(con=self._engine, name=name, **kwargs)
        logger.info(
            "successfullly upload data to table '%s' :\n%s"  % (name, df.head(5)))
        return
    
    @dedent
    @Appender(pd.read_sql_query.__doc__)
    def read_df(self, sql, **kwargs):
        """
        """
        
        logger.info('begin reading sql...\n')
        engine = self._engine
        sql = _read_sqlfile(sql)
        df = pd.read_sql_query(sql, engine, **kwargs)
        logger.info('successfully read data: ... \n', df.head(5), '\n',
              'from database: %s \n' % engine)
        return df

    def execute(self, sql):
        """

        Parameters
        ----------
        sql : TYPE
            sql query to be executed.

        Returns
        -------
        None.

        """
        sql = _read_sqlfile(sql)
        self._engine.execute(sql)
        logger.info("successfully execute SQL:'%s' ..." % sql[:40])
        return

    def _get_map_df_types(self, df):
        '''mapper of df dtype to db datatype
        
        return
        -------
        {colname : sqldtype}
        {object : String(length=255), float : Float(6), int : Integer()}
        '''
        dtypedict = {}
        for i, j in zip(df.columns, df.dtypes):
            if "object" in str(j) or "category" in str(j):
                max_length = df[i].apply(lambda x: len(str(x))).max()
                dtypedict.update(
                    {i: String(length=255 * (max_length // 255) + 255)})
            if "float" in str(j):
                dtypedict.update({i: Float(precision=6, asdecimal=True)})
            if "int" in str(j):
                dtypedict.update({i: Integer()})
        return dtypedict

if __name__ == '__main__':

    #--sqlite默认建立的对象只能让建立该对象的线程使用，
    #而sqlalchemy是多线程的所以我们需要指定check_same_thread=False
    #来让建立的对象任意线程都可使用。否则不时就会报错：
    #sqlalchemy.exc.ProgrammingError:     
    url = 'sqlite:///test.db?check_same_thread=False'
    db = EngineConn(url)
    data = pd.DataFrame([[1,2], [1,2]])
    db.upload(data, name='test', if_exists='append')
