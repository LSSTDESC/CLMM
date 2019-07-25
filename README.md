# CLMM [![Documentation Status](https://readthedocs.org/projects/clmm/badge/?version=master)](https://clmm.readthedocs.io/en/master/?badge=master) [![Build Status](https://travis-ci.org/LSSTDESC/CLMM.svg?branch=master)](https://travis-ci.org/LSSTDESC/CLMM)

A new and improved cluster mass modeling code descended from [clmassmod](https://github.com/LSSTDESC/clmassmod)

clmm is a general code for performing individual- and population-level inference on galaxy cluster weak lensing data. It will serve to enable the CLMassMod Key Task of the LSST-DESC SRM and will be used as a framework for future CL WG activities. clmm aims to be modular in (at least) three respects:

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

CLMM (Cluster Lensing Mass Modelling) provides a tool to estimate cluster masses based on weak lensing data. It also includes a routine to make mock catalogs based on Cluster Toolkit. By running CLMM, the whole process is able to be conducted from making source galaxies for a given mass and to estimate mass from the measured weak lensing signal.    

== Mock data generation ==

== Cluster object parameters ==
  * The GalaxyCluster object contains the galaxy cluster metadata (uniqe_id, ra, dec, z) as well as the background galaxy data
  * Background galaxy data: astropy Table containing galaxy_id, ra, dec, e1, e2, z, kappa
  * ra/dec are in decimal degrees

== Weak lensing signal measurement ==

== Profile model option ==

== Mass estimation == 


## Contact

(see CONTRIBUTING.md for now)
