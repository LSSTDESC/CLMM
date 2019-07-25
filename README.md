# CLMM [![Documentation Status](https://readthedocs.org/projects/clmm/badge/?version=master)](https://clmm.readthedocs.io/en/master/?badge=master) [![Build Status](https://travis-ci.org/LSSTDESC/CLMM.svg?branch=master)](https://travis-ci.org/LSSTDESC/CLMM)

The LSST-DESC Cluster Lensing Mass Modeling (CLMM) code is a Python library for performing galaxy cluster weak lensing analyses.
clmm is associated with Key Tasks _DC1 SW+RQ_ and _DC2 SW_ of the LSST-DESC [Science Roadmap](https://lsstdesc.org/sites/default/files/DESC_SRM_V1_4.pdf) pertaining to absolute and relative mass calibration.
CLMM is descended from [clmassmod](https://github.com/deapplegate/clmassmod) but distinguished by its modular structure and scope, which encompasses both simulated data sets with a known truth and observed data from which we aim to discover the truth.

## Installation

To install CLMM you currently need to build it from source::

  `git clone https://github.com/LSSTDESC/CLMM.git
  cd CLMM
  python setup.py install`

To run the tests you can do::

  `pytest`

### Requirements

Ultimately, CLMM will depend on [CCL](https://github.com/LSSTDESC/CCL), but until [cluster_toolkit](https://github.com/tmcclintock/cluster\_toolkit) is [https://github.com/LSSTDESC/CCL/issues/291](incorporated into CCL), we have an explicit dependency.
cluster\_toolkit's installation instructions can be found [here](https://cluster-toolkit.readthedocs.io/en/latest/).

## Overview

## Contact

(see CONTRIBUTING.md for now)
