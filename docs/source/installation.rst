**************
Installation
**************

Requirements
============
CLMM requires Python version 3.6 or later.  CLMM has the following dependencies:

- `NumPy <https://www.numpy.org/>`_: 1.17 or later
- `sciPy <https://www.scipy.org/>`_: 1.3 or later
- `Astropy <https://www.astropy.org/>`_: 4.x or later
- `Matplotlib <https://matplotlib.org/>`_

These are pip installable::

  pip install numpy scipy astropy matplotlib


For the theoretical predictions of the signal, CLMM relies on existing libraries and **at least one of the following must be installed as well**:

- `cluster-toolkit <https://cluster-toolkit.readthedocs.io/en/latest/>`_ 
- `CCL <https://ccl.readthedocs.io/en/latest/>`_
- `NumCosmo <https://numcosmo.github.io/>`_

See the `INSTALL documentation <https://github.com/LSSTDESC/CLMM/blob/master/INSTALL.md>`_ for more detailed installation instructions.

For developers, you will also need to install:

- `pytest <https://docs.pytest.org/en/latest/>`_ (3.x or later for testing)
- `sphinx <https://www.sphinx-doc.org/en/master/usage/installation.html>`_ (for documentation)

These are also pip installable::

  pip install pytest sphinx sphinx_rtd_theme

Installation
============
To install CLMM you currently need to build it from source::

  git clone https://github.com/LSSTDESC/CLMM.git 
  cd CLMM
  python setup.py install

To run the tests you can do::

  pytest
