# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.append(os.path.abspath('../../.'))

#%% -- Project information -----------------------------------------------------

project = 'lwmlearn'
copyright = '2020, rogerluo'
author = 'rogerluo'

# The full version, including alpha/beta/rc tags
from lwmlearn import __version__
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# %% -- substitution settings --
# replace text could be refered by |name| everywhere wihin sphyinx docs
rst_epilog = """
.. |author| replace:: Rogerluo www.baidu.com

.. |pkg| replace:: `lwmlearn`

"""

#%% -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# The files to include

# master_doc
# The document name of the “master” document, that is, 
# the document that contains the root toctree directive. 
# Default is 'index'.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['./autoapi/index.rst', ]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# exclude_patterns = ['./index']

# rst_epilog
# A string of reStructuredText that will be included at the 
# end of every source file that is read. This is a possible 
# place to add substitutions that should be available in every 
# file (another being rst_prolog). An example:

# rst_epilog = """
# .. |psf| replace:: Rogerluo open source lib
# """


# rst_prolog
# A string of reStructuredText that will be included at the beginning 
# of every source file that is read. This is a possible place to add 
# substitutions that should be available in every file
#  (another being rst_epilog). An example:

# rst_prolog = """
# .. |psf| replace:: Python Software Foundation
# """


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
# language = 'zh_CN'
language = None

html_scaled_image_link = True

#%% -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'lwmlearn doc'

#%% -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# html_theme_options = {
#     "rightsidebar": "true",
#     "relbarbgcolor": "black"
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

#%% -- Extension configuration -------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'recommonmark',
    'sphinx.ext.intersphinx',
    'autoapi.extension',
    "sphinx_rtd_theme",
    "sphinx.ext.todo",
    'sphinx.ext.graphviz',
]

#%% -- source_suffix --
# The file extensions of source files. Sphinx considers the files 
# with this suffix as sources. The value can be a dictionary mapping 
# file extensions to file types. For example:
source_suffix = ['.rst', '.md']


# %% -- Configuring AutoStructify
from recommonmark.transform import AutoStructify
github_doc_root =\
'''https://github.com/rogerlwlw/lw_project_template/tree/master/doc/source/
'''
def setup(app):
    app.add_config_value('recommonmark_config', {
            'url_resolver': lambda url: github_doc_root + url,
            'auto_toc_tree_section': 'Contents',
            }, True)
    app.add_transform(AutoStructify)
    
#%% -- intersphinx_mapping
# This extension can generate automatic links to the documentation of 
# objects in other projects.


intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sklearn' : ('https://scikit-learn.org/stable', None),
    
    }
# %% -- autoapi configuration --
# Autodoc-Style Directives
# You can opt to write API documentation yourself using autodoc style directives. 
# These directives work similarly to autodoc, but docstrings are retrieved through 
# static analysis instead of through imports.

autoapi_dirs = ['../../lwmlearn']
# autoapi documentation root directory
autoapi_root = 'tech/autoapi'

autoapi_template_dir = '_template'

# To remove the index page altogether, turn off the autoapi_add_toctree_entry 
# configuration option:

autoapi_add_toctree_entry = True

# turning the automatic documentation generation off is as easy as 
# disabling the autoapi_generate_api_docs configuration option:

autoapi_generate_api_docs = True

# get AutoAPI to keep its generated files around as a base to start from 
# using the autoapi_keep_files option:

# autoapi_keep_files = True

# configuration options
autoapi_options = ['members', 
                   'show-inheritance', 
                   'show-module-summary', 
                   'inherited-members',
                   # 'show-inheritance-diagram',
                   ]

autoapi_python_class_content = 'class'

autoapi_member_order = 'groupwise'


# %% --latex settings

latex_engine = 'xelatex'

latex_show_urls = 'footnote'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'lwmlearn.tex', u'lwmlearn Documentation',
     'rogerluo', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
latex_use_parts = True

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# -- Options for manual page output ---------------------------------------

# If false, no module index is generated.
# latex_domain_indices = True

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [('index', 'lwmlearn', 'Documentation',
              ["Rogerluo"], 1)]

# -- Options for Texinfo output -------------------------------------------
# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', 'lwmlearn', 'Documentation',
     'Rogerluo', 'lwmlearn',
     'Toolbox for machine learning.', 'Miscellaneous'),
]

latex_use_xindy = True

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
# texinfo_documents = [
#     (master_doc, 'test', u'test Documentation',
#      author, 'test', 'One line description of project.',
#      'Miscellaneous'),
# ]

# Documents to append as an appendix to all manuals.
#
# texinfo_appendices = []

# If false, no module index is generated.
#
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#
# texinfo_no_detailmenu = False

# -- A random example -----------------------------------------------------
#%% -- autosectionlabel --

# True to prefix each section label with the name of the document it is in, 
# followed by a colon. For example, index:Introduction for a section called 
# Introduction that appears in document index.rst. Useful for avoiding 
# when the same section heading appears in different documents.
# autosectionlabel_prefix_document = True

#  If set, autosectionlabel chooses the sections for labeling by its depth.
#  For example, when set 1 to autosectionlabel_maxdepth, 
#  labels are generated only for top level sections, and deeper sections 
#  are not labeled. It defaults to None (disabled).
# autosectionlabel_maxdepth = None

# %% -- todo options
todo_include_todos = True
todo_link_only = False

# %% doctest
# extensions.append('sphinx.ext.doctest')
# doctest_global_setup = '''
# try:
#     import pandas as pd
# except ImportError:
#     pd = None
# '''
