
# CLMM
[![Build Status](https://travis-ci.org/LSSTDESC/CLMM.svg?branch=master)](https://travis-ci.org/LSSTDESC/CLMM) 
[![Build and Check](https://github.com/LSSTDESC/CLMM/workflows/Build%20and%20Check/badge.svg)](https://github.com/LSSTDESC/CLMM/actions?query=workflow%3A%22Build+and+Check%22)
[![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/CLMM/badge.svg?branch=master)](https://coveralls.io/github/LSSTDESC/CLMM?branch=master)

The LSST-DESC Cluster Lensing Mass Modeling (CLMM) code is a Python library for performing galaxy cluster mass reconstruction from weak lensing observables. CLMM is associated with Key Tasks _DC1 SW+RQ_ and _DC2 SW_ of the LSST-DESC [Science Roadmap](https://lsstdesc.org/sites/default/files/DESC_SRM_V1_4.pdf) pertaining to absolute and relative mass calibration.
<!---CLMM is descended from [clmassmod](https://github.com/deapplegate/clmassmod) but distinguished by its modular structure and scope, which encompasses both simulated data sets with a known truth and observed data from which we aim to discover the truth.--->
The documentation of the code can be found [here](http://lsstdesc.org/CLMM/).

## Table of contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Overview](#overview)
4. [Contributing](#contributing)
5. [Contact](#contact)

## Requirements <a name="requirements"></a>

CLMM requires Python version 3.6 or later.  CLMM has the following dependencies:

- [numpy](http://www.numpy.org/) (1.17 or later)
- [scipy](http://www.numpy.org/) (1.3 or later)
- [astropy](https://www.astropy.org/) (3.x or later for units and cosmology dependence)
- [matplotlib](https://matplotlib.org/) (for plotting and going through tutorials)

```
  pip install numpy scipy astropy matplotlib
```

For the theoretical predictions of the signal, CLMM relies on existing libraries and **at least one of the following must be installed as well**:

- [cluster-toolkit](https://cluster-toolkit.readthedocs.io/en/latest/)
- [CCL](https://ccl.readthedocs.io/en/v2.0.0/)
- [NumCosmo](https://numcosmo.github.io/) 

(See the CONTRIBUTING documentation for detailed installation instructions.)

For developers, you will also need to install:

- [pytest](https://docs.pytest.org/en/latest/) (3.x or later for testing)
- [sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html) (for documentation)

These are also pip installable:
```
  pip install pytest sphinx sphinx_rtd_theme
```
Note, the last item, `sphinx_rtd_theme` is to make the docs.

## Installation <a name="installation"></a>

To install CLMM you currently need to build it from source:

```
  git clone https://github.com/LSSTDESC/CLMM.git
  cd CLMM
  python setup.py install --user   # Add --user flag to install it locally
```

To run the tests you can do:

  `pytest`

## Code overview and main features <a name="overview"></a>

Overview of the code can be found [here](OVERVIEW.md)

## Contributing <a name="contributing"></a>

Contributing documentation can be found [here](CONTRIBUTING.md)

## Contact <a name="contact"></a>
If you have comments, questions, or feedback, please [write us an issue](https://github.com/LSSTDESC/CLMM/issues).
