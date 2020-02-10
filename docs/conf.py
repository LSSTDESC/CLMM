# CLMM documentation build configuration file, created by
import os
import subprocess
import sys
sys.path.insert(0, os.path.abspath('../clmm'))
sys.path.insert(0, os.path.abspath('..'))

import clmm


# -- RTD Fix for cluster_toolkit -----------------------------------------
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# This code will execute only on readthedocs
if on_rtd:
    try:
        from unittest.mock import MagicMock
    except ImportError:
        from mock import Mock as MagicMock

    class Mock(MagicMock):
        @classmethod
        def __getattr__(cls, name):
            return MagicMock()

    # For these modules, do a mock import
    MOCK_MODULES = ['cluster_toolkit']
    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Load the version number ----------------------------------------------
version = clmm.__version__
release = version

# -- General configuration ------------------------------------------------
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'sphinx.ext.githubpages',
              'IPython.sphinxext.ipython_console_highlighting']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'CLMM'
copyright = '2018-2019, LSST DESC CLMM Contributors'
author = 'LSST DESC CLMM Contributors'
language = 'en'

# Files to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
                    'api/clmm.rst', 'source/index_body.rst',
                    'api/clmm.cluster_toolkit_patches.rst']

# Some style options
highlight_language = 'python3'
pygments_style = 'sphinx'
todo_include_todos = True
add_function_parentheses = True
add_module_names = True


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


# -- Load from the config file -------------------------------------------
config = open('doc-config.ini').read().strip().split('\n')
apilist, demofiles, examplefiles = [], [], []
apion, demoon, exon = False, False, False
for entry in config:
    if not entry or entry[0] == '#':
        continue
    if entry == 'APIDOC':
        apion, demoon, exon = True, False, False
        continue
    elif entry == 'DEMO':
        apion, demoon, exon = False, True, False
        continue
    elif entry == 'EXAMPLE':
        apion, demoon, exon = False, False, True
        continue
    if apion:
        apilist += [entry]
    elif demoon:
        demofiles += [entry]
    elif exon:
        examplefiles += [entry]


# -- Compile the examples into rst----------------------------------------
outdir = 'compiled-examples/'
nbconvert_opts = ['--to rst',
                  '--ExecutePreprocessor.kernel_name=python3',
                  # '--execute',
                  f'--output-dir {outdir}']

for demo in [*demofiles, *examplefiles]:
    com = ' '.join(['jupyter nbconvert'] + nbconvert_opts + [demo])
    subprocess.run(com, shell=True)


# -- Build index.html ----------------------------------------------------
index_examples_toc = \
""".. toctree::
   :maxdepth: 1
   :caption: Examples

"""
for example in examplefiles:
    fname = ''.join(example.split('.')[:-1]).split('/')[-1] + '.rst'
    index_examples_toc += f"   {outdir}{fname}\n"

# This is automatic
index_demo_toc = \
"""
.. toctree::
   :maxdepth: 1
   :caption: Usage Demos

"""
for demo in demofiles:
    fname = ''.join(demo.split('.')[:-1]).split('/')[-1] + '.rst'
    index_demo_toc += f"   {outdir}{fname}\n"

index_api_toc = \
"""
.. toctree::
   :maxdepth: 1
   :caption: Reference

   api
"""

subprocess.run('cp source/index_body.rst index.rst', shell=True)
with open('index.rst', 'a') as indexfile:
    indexfile.write(index_examples_toc)
    indexfile.write(index_demo_toc)
    indexfile.write(index_api_toc)


# -- Set up the API table of contents ------------------------------------
apitoc = \
"""API Documentation
=================

Information on specific functions, classes, and methods.

.. toctree::
   :glob:

"""
for onemodule in apilist:
    apitoc += f"   api/clmm.{onemodule}.rst\n"
with open('api.rst', 'w') as apitocfile:
    apitocfile.write(apitoc)
