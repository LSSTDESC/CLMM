# CLMM [![Documentation Status](https://readthedocs.org/projects/clmm/badge/?version=master)](https://clmm.readthedocs.io/en/master/?badge=master) [![Build Status](https://travis-ci.org/LSSTDESC/CLMM.svg?branch=master)](https://travis-ci.org/LSSTDESC/CLMM)

The LSST-DESC Cluster Lensing Mass Modeling (CLMM) code is a Python library for performing galaxy cluster weak lensing analyses.
clmm is associated with Key Tasks _DC1 SW+RQ_ and _DC2 SW_ of the LSST-DESC [Science Roadmap](https://lsstdesc.org/sites/default/files/DESC_SRM_V1_4.pdf) pertaining to absolute and relative mass calibration.
CLMM is descended from [clmassmod](https://github.com/deapplegate/clmassmod) but distinguished by its modular structure and scope, which encompasses both simulated data sets with a known truth and observed data from which we aim to discover the truth.

=======
There are several examples in the examples/ directory to get you
started. A simple example including a simple simulated cluster and
lensed galaxies, binning of the data and modeling is given here: ADD
PATH TO RELEVANT EXAMPLE

## Installation

To install CLMM you currently need to build it from source::

  ```git clone https://github.com/LSSTDESC/CLMM.git
  cd CLMM
  python setup.py install
```

To run the tests you can do::

  `pytest`

### Requirements

Ultimately, CLMM will depend on [CCL](https://github.com/LSSTDESC/CCL), but until [cluster_toolkit](https://github.com/tmcclintock/cluster\_toolkit) is [https://github.com/LSSTDESC/CCL/issues/291](incorporated into CCL), we have an explicit dependency.
cluster\_toolkit's installation instructions can be found [here](https://cluster-toolkit.readthedocs.io/en/latest/).

## Overview

CLMM (Cluster Lensing Mass Modelling) provides a tool to estimate
cluster masses based on weak lensing data. It also includes a routine
to make mock catalogs based on Cluster Toolkit. By running CLMM, the
whole process is able to be conducted from making source galaxies for
a given mass and to estimate mass from the measured weak lensing
signal.

There are several examples in the examples/ directory to get you
started. A simple example including a simple simulated cluster and
lensed galaxies, binning of the data and modeling is given here: ADD
PATH TO RELEVANT EXAMPLE

### Mock data generation
* examples/generate_mock_data.ipynb

### Weak lensing signal measurement 

  * The function "computeshear" calculates tangential shear, cross shear, and angular separation of each source galaxy relative to the center of the cluster (ra,dec)
  * If you want a costomized binning for shear profile, prepare bins by running "make_bins". You can choose rmin, rmax, number of bins, and linear or log scale.  
  * "make_shear_profile" takes average over shear of each source galaxy with radial bins. The unit of radius can be rad, deg, arcmin, arcsec, kpc, Mpc.    
  * Check "demo_of_polaraveraging.ipynb" for the detailed examples

## Mock data generation
  * examples/generate_mock_data.ipynb

## Cluster object parameters
The GalaxyCluster object contains the galaxy cluster metadata
(uniqe_id, ra, dec, z) as well as the background galaxy data. The
latter is an astropy Table containing galaxy_id, ra, dec, e1, e2, z,
kappa. RA and Dec are in decimal degrees.

ADD PATH TO RELEVANT EXAMPLE(?)

### Profile model option
* examples/modeling_demo.ipynb
* model.py holds functions for evaluating theoretical models. The
  default is to use an NFW profile. currently this is the only
  supported halo profile parameterization. 

### Mass estimation
  * ADD PATH TO RELEVANT EXAMPLE 


## Contact

(see CONTRIBUTING.md for now)
