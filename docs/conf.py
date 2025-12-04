# CLMM documentation build configuration file, created by
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath("../clmm"))
sys.path.insert(0, os.path.abspath(".."))

from unittest.mock import MagicMock

MOCK_MODULES = [
    "gi",
    "gi.repository",
    "gi.repository.NumCosmoMath",
    "gi.repository.NumCosmo",
    "pyccl",
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

# Fix for ccl
sys.modules["pyccl"].Cosmology = MagicMock

# Fix for numcosmo
sys.modules["gi.repository"].NumCosmo.Distance = MagicMock
sys.modules["gi.repository"].NumCosmo.Distance.new = MagicMock
sys.modules["gi.repository"].NumCosmo.Distance.new.prepare_if_needed = MagicMock

import clmm

# -- RTD Fix for cluster_toolkit -----------------------------------------
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

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
    MOCK_MODULES = ["cluster_toolkit"]
    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Load the version number ----------------------------------------------
version = clmm.__version__
release = version

# -- General configuration ------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "IPython.sphinxext.ipython_console_highlighting",
]

apidoc_module_dir = "../clmm"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "CLMM"
copyright = "2018-2021, LSST DESC CLMM Contributors"
author = "LSST DESC CLMM Contributors"
language = "en"

# Files to ignore when looking for source files
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "api/clmm.rst",
    "source/index_body.rst",
    "api/clmm.cluster_toolkit_patches.rst",
    "api/clmm.modbackend.*",
    ".precompiled-fixed-examples/*",
]

# Some style options
highlight_language = "python3"
pygments_style = "sphinx"
todo_include_todos = True
add_function_parentheses = True
add_module_names = True
smartquotes = False


# -- Options for HTML output ----------------------------------------------

# HTML Theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "prev_next_buttons_location": None,
    "collapse_navigation": False,
    "titles_only": True,
}
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

# def run_apidoc(_):
#    from sphinxcontrib.apidoc import main as apidoc_main
#    cur_dir = os.path.normpath(os.path.dirname(__file__))
#    output_path = os.path.join(cur_dir, 'api')
#    modules = os.path.normpath(os.path.join(cur_dir, "../clmm"))
#    paramlist = ['--separate', '--no-toc', '-f', '-M', '-o', output_path, modules]
#    apidoc_main(paramlist)

# def setup(app):
#    app.connect('builder-inited', run_apidoc)


# -- Load from the config file -------------------------------------------
config = open("doc-config.ini").read().strip().split("\n")
doc_files = {
    "APIDOC": [],
    "DEMO": [],
    "EXAMPLE": [],
    "OTHER": [],
}
key = None
for entry in config:
    if not entry or entry[0] == "#":
        continue
    elif entry in doc_files:
        key = entry
    else:
        doc_files[key].append(entry)
# -- Compile the examples into rst----------------------------------------
run_nb = False

outdir = "compiled-examples/"
nbconvert_opts = [
    "--to rst",
    "--ExecutePreprocessor.kernel_name=python3",
    "--execute",
    f"--output-dir {outdir}",
]
nb_skip_run = [
    #    '../examples/DC2/data_and_model_demo_DC2.ipynb',
    #    '../examples/mass_fitting/Example4_Fit_Halo_mass_to_HSC_data.ipynb',
    #    '../examples/mass_fitting/Example5_Fit_Halo_mass_to_DES_data.ipynb',
]

for lists in [v for k, v in doc_files.items() if k != "APIDOC"]:
    for demo in lists:
        com = " ".join(["jupyter nbconvert"] + nbconvert_opts + [demo])
        if demo in nb_skip_run or not run_nb:
            com = com.replace(" --execute ", " ")
        subprocess.run(com, shell=True)

for nb in nb_skip_run:
    pref = nb.split("/")[-1].replace(".ipynb", "")
    com = f"cp -rf .precompiled-fixed-examples/{pref}* compiled-examples/"
    print(f"* Fix for publication (use precompiled version of {pref} from older version)")
    subprocess.run(com, shell=True)


# -- Build index.html ----------------------------------------------------
doc_captions = {
    "DEMO": "Usage Demos",
    "EXAMPLE": "Mass Fitting Examples",
    "OTHER": "Other",
}
index_toc = ""
for CASE in ("DEMO", "EXAMPLE", "OTHER"):
    index_toc += f"""
.. toctree::
   :maxdepth: 1
   :caption: {doc_captions[CASE]}

"""
    for example in doc_files[CASE]:
        fname = "".join(example.split(".")[:-1]).split("/")[-1] + ".rst"
        index_toc += f"   {outdir}{fname}\n"

subprocess.run("cp source/index_body.rst index.rst", shell=True)
with open("index.rst", "a") as indexfile:
    indexfile.write(index_toc)
    indexfile.write(
        """
.. toctree::
   :maxdepth: 1
   :caption: Reference

   api
"""
    )

# -- Set up the API table of contents ------------------------------------
apitoc = """API Documentation
-----------------

Information on specific functions, classes, and methods.

.. toctree::
   :glob:

"""
for onemodule in doc_files["APIDOC"]:
    apitoc += f"   api/clmm.{onemodule}.rst\n"
with open("api.rst", "w") as apitocfile:
    apitocfile.write(apitoc)
