# CLMM
[![Build Status](https://travis-ci.org/LSSTDESC/CLMM.svg?branch=master)](https://travis-ci.org/LSSTDESC/CLMM) 
[![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/CLMM/badge.svg)](https://coveralls.io/github/LSSTDESC/CLMM)

The LSST-DESC Cluster Lensing Mass Modeling (CLMM) code is a Python library for performing galaxy cluster weak lensing analyses.
clmm is associated with Key Tasks _DC1 SW+RQ_ and _DC2 SW_ of the LSST-DESC [Science Roadmap](https://lsstdesc.org/sites/default/files/DESC_SRM_V1_4.pdf) pertaining to absolute and relative mass calibration.
CLMM is descended from [clmassmod](https://github.com/deapplegate/clmassmod) but distinguished by its modular structure and scope, which encompasses both simulated data sets with a known truth and observed data from which we aim to discover the truth.
The documentation of the code can be found [here](http://lsstdesc.org/CLMM/).

## Table of contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Overview](#overview)
3. [Contributing](#contributing)

## Requirements <a name="requirements"></a>

CLMM requires Python version 3.6 or later.  To run the code, there are the following dependencies:

- [numpy](http://www.numpy.org/) (1.16 or later)

- [scipy](http://www.numpy.org/) (1.3 or later)

- [astropy](https://www.astropy.org/) (3.x or later for units and cosmology dependence)

- [matplotlib](https://matplotlib.org/) (for plotting and going through tutorials)

- [cluster-toolkit](https://cluster-toolkit.readthedocs.io/en/latest/source/installation.html) (for halo functionality)
  
All but cluster-toolkit are pip installable:
```
  pip install numpy scipy astropy matplotlib
```

Ultimately, CLMM will depend on [CCL](https://github.com/LSSTDESC/CCL), but until [cluster_toolkit](https://github.com/tmcclintock/cluster\_toolkit) is [incorporated into CCL](https://github.com/LSSTDESC/CCL/issues/291), we have an explicit dependency.
cluster\_toolkit's installation instructions can be found [here](https://cluster-toolkit.readthedocs.io/en/latest/). 
**Note**: While cluster-toolkit mentions the potential need to install CAMB/CLASS for all cluster-toolkit functionality, you do not need to install these to run CLMM.

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

## Overview <a name="overview"></a>

Overview of the code can be found [here](OVERVIEW.md)

## Contributing <a name="contributing"></a>

Contributing documentation can be found [here](CONTRIBUTING.md)
