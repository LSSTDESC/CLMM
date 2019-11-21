# CLMM
[![Build Status](https://travis-ci.org/LSSTDESC/CLMM.svg?branch=master)](https://travis-ci.org/LSSTDESC/CLMM) 
[![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/CLMM/badge.svg)](https://coveralls.io/github/LSSTDESC/CLMM)
[![Documentation Status](https://readthedocs.org/projects/clmm/badge/?version=master)](https://clmm.readthedocs.io/en/master/?badge=master)

The LSST-DESC Cluster Lensing Mass Modeling (CLMM) code is a Python library for performing galaxy cluster weak lensing analyses.
clmm is associated with Key Tasks _DC1 SW+RQ_ and _DC2 SW_ of the LSST-DESC [Science Roadmap](https://lsstdesc.org/sites/default/files/DESC_SRM_V1_4.pdf) pertaining to absolute and relative mass calibration.
CLMM is descended from [clmassmod](https://github.com/deapplegate/clmassmod) but distinguished by its modular structure and scope, which encompasses both simulated data sets with a known truth and observed data from which we aim to discover the truth.

## Installation

To install CLMM you currently need to build it from source::

```
  git clone https://github.com/LSSTDESC/CLMM.git
  cd CLMM
  python setup.py install --user   # Add --user flag to install it locally
```

To run the tests you can do::

  `pytest`

### Requirements

Ultimately, CLMM will depend on [CCL](https://github.com/LSSTDESC/CCL), but until [cluster_toolkit](https://github.com/tmcclintock/cluster\_toolkit) is [incorporated into CCL](https://github.com/LSSTDESC/CCL/issues/291), we have an explicit dependency.
cluster\_toolkit's installation instructions can be found [here](https://cluster-toolkit.readthedocs.io/en/latest/).

## Overview

CLMM (Cluster Lensing Mass Modeling) provides a tool to estimate cluster masses based on weak lensing data.
It also includes a routine to make mock catalogs based on cluster_toolkit.
CLMM consists of the building blocks for an end-to-end weak lensing cosmology pipeline that can be validated on mock data and run on real data from LSST or other telescopes.
We provide [examples](https://github.com/LSSTDESC/CLMM/tree/issue/115/readme/examples) of its usage in this repository.

### The `GalaxyCluster` object

  * The GalaxyCluster object contains the galaxy cluster metadata (unique_id, ra, dec, z) as well as the background galaxy data
  * Background galaxy data: astropy Table containing galaxy_id, ra, dec, e1, e2, z, kappa
  * ra/dec are in decimal degrees

## Mock data generation
  * examples/generate_mock_data.ipynb

### Weak lensing signal measurement with `polaraveraging.py`

  * The function `computeshear` calculates tangential shear, cross shear, and angular separation of each source galaxy relative to the (ra, dec) coordinates of the center of the cluster.
  * A shear profile may be constructed with the user's choice of binning via `make_bins`.
  * `make_shear_profile` takes the average over shear of each source galaxy over radial bins in rad, deg, arcmin, arcsec, kpc, or Mpc.
  * See demo_of_polaraveraging.ipynb for detailed examples.

### Profile and cosmology models with `modeling.py`

  * modeling.py holds functions for evaluating theoretical models.
  * The default is to use an NFW profile, but more halo profile parameterizations will be added soon.
  * See examples/modeling_demo.ipynb for example usage.

### Galaxy cluster mass estimation
  * See examples/demo-pipeline.ipynb for example usage of an end-to-end measurement.
