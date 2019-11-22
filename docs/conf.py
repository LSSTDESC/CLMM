# CLMM documentation build configuration file, created by
import os
import subprocess


# -- General configuration ------------------------------------------------
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'IPython.sphinxext.ipython_console_highlighting']

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
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'api/clmm.rst', 'source/index_body.rst']

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


# -- Compile the examples into rst----------------------------------------
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

outdir = 'compiled-examples/'
nbconvert_opts = ['--to rst',
                  '--execute',
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
