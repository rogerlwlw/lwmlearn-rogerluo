# -*- coding: utf-8 -*-
"""Decorate docstring 

Created on Thu Aug  8 18:33:00 2019

inherited from matplotlib.doctring

@author: rogerluo
"""
from inspect import cleandoc
import re

_whitespace_only_re = re.compile('^[ \t]+$', re.MULTILINE)
_leading_whitespace_re = re.compile('(^[ \t]*)(?:[^ \t\n])', re.MULTILINE)


class Substitution(object):
    """
    A decorator to take a function's docstring and perform string
    substitution on it.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter)

    Usage: construct a docstring.Substitution with a sequence or
    dictionary suitable for performing substitution; then
    decorate a suitable function with the constructed object. e.g.

    sub_author_name = Substitution(author='Jason')

    @sub_author_name
    def some_function(x):
        "%(author)s wrote this function"

    # note that some_function.__doc__ is now "Jason wrote this function"

    """
    def __init__(self, *args, **kwargs):
        '''
        

        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        '''
        assert not (len(args) and len(kwargs)), \
                "Only keyword args are allowed"
        self.params = args or kwargs

    def __call__(self, func):
        func.__doc__ = func.__doc__ and func.__doc__.format(**self.params)
        return func

    def update(self, *args, **kwargs):
        "Assume self.params is a dict and update it with supplied args"
        self.params.update(*args, **kwargs)

    @classmethod
    def from_params(cls, params):
        """
        In the case where the params is a mutable sequence (list or
        dictionary) and it may change before this class is called, one may
        explicitly use a reference to the params rather than using *args or
        **kwargs which will copy the values and not reference them.
        """
        result = cls()
        result.params = params
        return result


class Appender(object):
    """
    A function decorator that will append an addendum to the docstring
    of the target function.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).

    Usage: construct a docstring.Appender with a string to be joined to
    the original docstring. An optional 'join' parameter may be supplied
    which will be used to join the docstring and addendum. e.g.

    add_copyright = Appender("Copyright (c) 2009")

    @add_copyright
    def my_dog(has='fleas'):
        "This docstring will have a copyright below"
        
        pass
        
    """
    def __init__(self, addendum, join=''):
        '''

        Parameters
        ----------
        
        addendum : str
            docstring to be appended.
            
        join : str, optional
            may be supplied which will be used to join the docstring 
            and addendum. e.g.. The default is ''.

        Returns
        -------
        None.

        '''
        self.addendum = addendum
        self.join = join

    def __call__(self, func):
        docitems = [func.__doc__, self.addendum]
        func.__doc__ = func.__doc__ and self.join.join(docitems)
        return func


def dedent(func):
    "Dedent a docstring (if present)"
    func.__doc__ = func.__doc__ and cleandoc(func.__doc__)
    return func


def copy(source):
    "Copy a docstring from another source function (if present)"

    def do_copy(target):
        if source.__doc__:
            target.__doc__ = source.__doc__
        return target

    return do_copy


def copy_dedent(source):
    """A decorator that will copy the docstring from the source and
    then dedent it"""
    # note the following is ugly because "Python is not a functional
    # language" - GVR. Perhaps one day, functools.compose will exist.
    #  or perhaps not.
    #  http://mail.python.org/pipermail/patches/2007-February/021687.html
    return lambda target: dedent(copy(source)(target))


def dedent_str(text):
    """Remove any common leading whitespace from every line in `text`.

    This can be used to make triple-quoted strings line up with the left
    edge of the display, while still presenting them in the source code
    in indented form.

    Note that tabs and spaces are both treated as whitespace, but they
    are not equal: the lines "  hello" and "\\thello" are
    considered to have no common leading whitespace.

    Entirely blank lines are normalized to a newline character.
    """
    # Look for the longest leading string of spaces and tabs common to
    # all lines.
    margin = None
    text = _whitespace_only_re.sub('', text)
    indents = _leading_whitespace_re.findall(text)
    for indent in indents:
        if margin is None:
            margin = indent

        # Current line more deeply indented than previous winner:
        # no change (previous winner is still on top).
        elif indent.startswith(margin):
            pass

        # Current line consistent with and no deeper than previous winner:
        # it's the new winner.
        elif margin.startswith(indent):
            margin = indent

        # Find the largest common whitespace between current line and previous
        # winner.
        else:
            for i, (x, y) in enumerate(zip(margin, indent)):
                if x != y:
                    margin = margin[:i]
                    break

    # sanity check (testing/debugging only)
    if 0 and margin:
        for line in text.split("\n"):
            assert not line or line.startswith(margin), \
                   "line = %r, margin = %r" % (line, margin)

    if margin:
        text = re.sub(r'(?m)^' + margin, '', text)
    return text


if __name__ == '__main__':
    key = \
    '''
    keys
    ----
        key word to repalce 
    '''
    keya = \
    '''
    appending
    ----
    appended_key
        append some explainations here
    '''

    @dedent
    @Appender(keya, join='\n')
    @Substitution(author='roger', date='20190812', key=key)
    def fn():
        '''
        author of this function is {author}, edited on {date},
        
        {key}
        
        '''
