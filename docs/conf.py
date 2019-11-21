# CLMM documentation build configuration file, created by
import os
import sphinx_nbexamples


# -- General configuration ------------------------------------------------
extensions = ['sphinx.ext.autodoc',
              # 'sphinx.ext.autosummary',
              'sphinx.ext.napoleon']#,
              # 'sphinx.ext.nbexamples']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'CLMM'
copyright = '2018-2019, LSST DESC CLMM Contributors'
author = 'LSST DESC CLMM Contributors'

# version is short X.Y, release is full including alpha/beta/rc
version = '0.0.1'
release = '0.0.1'

# Language of the documentation
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'api/clmm.rst']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# HTML Theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {'prev_next_buttons_location': None,
                      'collapse_navigation': False,
                      'titles_only': True}
html_static_path = []


# -- Options for Napoleon-------------------------------------------------
# Napoleon compiles the docstrings into .rst

# If True, include class __init__ docstrings separately from class
napoleon_include_init_with_doc = False
# If True, include docstrings of private functions
napoleon_include_private_with_doc = False
# Detail for converting docstrings to rst
napoleon_use_ivar = True


# -- Options for nbexamples ----------------------------------------------


# -- Options for Autodoc--------------------------------------------------
# Autodoc collects docstrings and builds API pages

def run_apidoc(_):
    from sphinx.ext.apidoc import main as apidoc_main
    cur_dir = os.path.normpath(os.path.dirname(__file__))
    output_path = os.path.join(cur_dir, 'api')
    modules = os.path.normpath(os.path.join(cur_dir, "../clmm"))
    paramlist = ['--separate', '--no-toc', '-f', '-M', '-o', output_path, modules]
    apidoc_main(paramlist)

def setup(app):
    app.connect('builder-inited', run_apidoc)


# -- Set up the API table of contents ------------------------------------
apitoc = \
"""API Documentation
=================

Information on specific functions, classes, and methods.

.. toctree::
   :glob:

   api/clmm.constants.rst
   api/clmm.galaxycluster.rst
   api/clmm.gcdata.rst
   api/clmm.lsst.rst
   api/clmm.modeling.rst
   api/clmm.plotting.rst
   api/clmm.polaraveraging.rst
   api/clmm.utils.rst
"""

with open('api.rst', 'w') as apitocfile:
    apitocfile.write(apitoc)
