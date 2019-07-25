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

Ultimately, CLMM will depend on [CCL](https://github.com/LSSTDESC/CCL), but until [cluster_toolkit](https://github.com/tmcclintock/cluster\_toolkit) is [incorporated into CCL](https://github.com/LSSTDESC/CCL/issues/291), we have an explicit dependency.
cluster_toolkit's installation instructions can be found [here](https://cluster-toolkit.readthedocs.io/en/latest/).

## Overview

CLMM (Cluster Lensing Mass Modelling) provides a tool to estimate cluster masses based on weak lensing data.
It also includes a routine to make mock catalogs based on cluster_toolkit.
CLMM consists of the building blocks for an end-to-end weak lensing cosmology pipeline that can be validated on mock data and run on real data from LSST or other telescopes.
We provide [examples](https://github.com/LSSTDESC/CLMM/tree/issue/115/readme/examples) of its usage in this repository.

### Mock data generation

### The `GalaxyCluster` object

  * The GalaxyCluster object contains the galaxy cluster metadata (unique_id, ra, dec, z) as well as the background galaxy data
  * Background galaxy data: astropy Table containing galaxy_id, ra, dec, e1, e2, z, kappa
  * ra/dec are in decimal degrees

### Weak lensing signal measurement with `polaraveraging.py`

  * The function `computeshear` calculates tangential shear, cross shear, and angular separation of each source galaxy relative to the (ra, dec) coordinates of the center of the cluster.
  * A shear profile may be constructed with the user's choice of binning via `make_bins`.
  * `make_shear_profile` takes the average over shear of each source galaxy over radial bins in rad, deg, arcmin, arcsec, kpc, or Mpc.    

A simple example including a simple simulated cluster and
lensed galaxies, binning of the data and modeling is given here: ADD
PATH TO RELEVANT EXAMPLE

### Profile and cosmology models with `modeling.py`

  * modeling.py holds functions for evaluating theoretical models.
  * The default is to use an NFW profile, but more halo profile parameterizations will be added soon.

### Galaxy cluster mass estimation

## Contact

  * [Michel Aguena](https://github.com/m-aguena) (LIneA)
  * [Doug Applegate](https://github.com/deapplegate) (Novartis)
  * [Camille Avestruz](https://github.com/cavestruz) (UChicago)
  * [Lucie Baumont](https://github.com/lbaumo) (SBU)
  * [Miyoung Choi](https://github.com/mchoi8739) (UTD)
  * [Celine Combet](https://github.com/combet) (LSPC)
  * [Matthew Fong](https://github.com/matthewwf2001) (UTD)
  * [Shenming Fu](https://github.com/shenmingfu)(Brown)
  * [Matthew Ho](https://github.com/maho3) (CMU)
  * [Matthew Kirby](https://github.com/matthewkirby) (Arizona)
  * [Brandyn Lee](https://github.com/brandynlee) (UTD)
  * [Anja von der Linden](https://github.com/anjavdl) (SBU)
  * [Binyang Liu](https://github.com/rbliu) (Brown)
  * [Alex Malz](https://github.com/aimalz) (NYU --> RUB)
  * [Tom McClintock](https://github.com/tmcclintock) (BNL)
  * [Hironao Miyatake](https://github.com/HironaoMiyatake) (Nagoya)
  * [Mariana Penna-Lima](https://github.com/pennalima) (UBrasilia)
  * [Marina Ricci](https://github.com/marina-ricci) (LAPP)
  * [Cristobal Sifon](https://github.com/cristobal-sifon) (Princeton)
  * [Melanie Simet](https://github.com/msimet) (JPL)
  * [Martin Sommer](https://github.com/sipplund) (Bonn)
  * [Heidi Wu](https://github.com/hywu) (Ohio)
  * [Mijin Yoon](https://github.com/mijinyoon) (RUB)

  The current administrators of the repository are Michel Aguena, Camille Avestruz, Matthew Kirby, and Alex Malz.
