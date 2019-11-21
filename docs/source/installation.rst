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
Ultimately, CLMM will depend on `CCL <https://github.com/LSSTDESC/CCL>`_, but until cluster_toolkit is incorporated into CCL, we have an explicit dependency. cluster_toolkit's installation instructions can be found `here <https://cluster-toolkit.readthedocs.io/en/latest/source/installation.html>`_.  The additional Python dependencies that you can get with pip are:

- `Numpy <http://www.numpy.org/>`_: 1.16 or later

- `scipy <http://www.numpy.org/>`_: 1.3 or later

- `pytest <https://docs.pytest.org/en/latest/>`_: 3.x or later for testing

- `sphinx <https://www.sphinx-doc.org/en/master/usage/installation.html>`_: for documentation

- `astropy <https://www.astropy.org/>`_: 3.x or later for units and cosmology dependence

- `matplotlib <https://matplotlib.org/>`_: for plotting and going through tutorials



  
