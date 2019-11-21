******************
CLMM Documentation
******************

The LSST-DESC Cluster Lensing Mass Modeling (CLMM) code is a Python
library for performing galaxy cluster weak lensing analyses. clmm is
associated with Key Tasks DC1 SW+RQ and DC2 SW of the LSST-DESC
`Science Roadmap
<https://lsstdesc.org/sites/default/files/DESC_SRM_V1_4.pdf>`_
pertaining to absolute and relative mass calibration. CLMM is
descended from `clmassmod <https://github.com/deapplegate/clmassmod>`_
but distinguished by its modular structure and scope, which
encompasses both simulated data sets with a known truth and observed
data from which we aim to discover the truth.

The core functionality of CLMM provides tools to estimate cluster
masses based on weak lensing data. It also includes a routine to make
mock catalogs based on cluster_toolkit. CLMM consists of the building
blocks for an end-to-end weak lensing cosmology pipeline that can be
validated on mock data and run on real data from LSST or other
telescopes. We provide examples of usage in this repository.
Functionality includes:

 * The GalaxyCluster object
 * Mock data generation
 * Weak lensing signal measurement with polaraveraging.py
 * Profile and cosmology models with modeling.py
 * Galaxy cluster mass estimation

The source code is publically available at https://github.com/LSSTDESC/CLMM


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   source/installation
   source/citing

