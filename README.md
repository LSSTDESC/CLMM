# CLMM [![Documentation Status](https://readthedocs.org/projects/clmm/badge/?version=master)](https://clmm.readthedocs.io/en/master/?badge=master) [![Build Status](https://travis-ci.org/LSSTDESC/CLMM.svg?branch=master)](https://travis-ci.org/LSSTDESC/CLMM)

The LSST-DESC Cluster Lensing Mass Modeling (CLMM) code is a Python library for performing galaxy cluster weak lensing analyses.
clmm is associated with Key Tasks _DC1 SW+RQ_ and _DC2 SW_ of the LSST-DESC [Science Roadmap](https://lsstdesc.org/sites/default/files/DESC_SRM_V1_4.pdf) pertaining to absolute and relative mass calibration.
CLMM is descended from [clmassmod](https://github.com/deapplegate/clmassmod) but differs in scope and structure.


clmm is a general code for performing individual- and population-level inference on galaxy cluster weak lensing data.
It will serve to enable the CLMassMod Key Task of the LSST-DESC SRM and will be used as a framework for future CL WG activities.
clmm aims to be modular in (at least) three respects:

    clmm will be able to run on real data as well as simulations, and it will not be restricted to any particular datasets.
    clmm will support multiple modes of inference of the cluster mass function and other relevant distributions, such as the mass-concentration relation.
    clmm will enable evaluation of results on the basis of a number of different metrics, some of which will not require a notion of truth from a simulation.

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
