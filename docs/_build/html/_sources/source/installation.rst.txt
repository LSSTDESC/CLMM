**************
Installation
**************

To install CLMM you currently need to build it from source::
  
  git clone https://github.com/DESC/CLMM.git
  cd CLMM
  python setup.py install

To run the tests you can do::

  pytest
  
Requirements
============

CLMM requires Python version 3.6 or later.  To run the code, there are the following dependencies:

- `Numpy <http://www.numpy.org/>`_: 1.16 or later

- `scipy <http://www.numpy.org/>`_: 1.3 or later

- `astropy <https://www.astropy.org/>`_: 3.x or later for units and cosmology dependence

- `matplotlib <https://matplotlib.org/>`_: for plotting and going through tutorials

- `cluster-toolkit <https://cluster-toolkit.readthedocs.io/en/latest/source/installation.html>`_: for halo functionality
  
All but cluster-toolkit are pip installable,::

  pip install numpy scipy astropy matplotlib

Ultimately, CLMM will depend on `CCL <https://github.com/LSSTDESC/CCL>`_, but until cluster_toolkit is incorporated into CCL, we have an explicit dependency.  Note: While cluster-toolkit mentions the potential need to install CAMB/CLASS for all cluster-toolkit functionality, you do not need to install these to run CLMM.


For developers, you will also need to install:

- `pytest <https://docs.pytest.org/en/latest/>`_: 3.x or later for testing

- `sphinx <https://www.sphinx-doc.org/en/master/usage/installation.html>`_: for documentation

These are also pip installable,::

  pip install pytest sphinx sphinx_rtd_theme

Note, the last item, sphinx_rtd_theme is to make the docs.
