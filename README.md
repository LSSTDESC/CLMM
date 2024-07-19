# CLMM
[![Build and Check](https://github.com/LSSTDESC/CLMM/workflows/Build%20and%20Check/badge.svg)](https://github.com/LSSTDESC/CLMM/actions?query=workflow%3A%22Build+and+Check%22)
[![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/CLMM/badge.svg?branch=main)](https://coveralls.io/github/LSSTDESC/CLMM?branch=main)

The LSST-DESC Cluster Lensing Mass Modeling (CLMM) code is a DESC tool consisting of a Python library for performing galaxy cluster mass reconstruction from weak lensing observables. CLMM is associated with Key Tasks _DC1 SW+RQ_ and _DC2 SW_ of the LSST-DESC [Science Roadmap](https://lsstdesc.org/sites/default/files/DESC_SRM_V1_4.pdf) pertaining to absolute and relative mass calibration.
<!---CLMM is descended from [clmassmod](https://github.com/deapplegate/clmassmod) but distinguished by its modular structure and scope, which encompasses both simulated data sets with a known truth and observed data from which we aim to discover the truth.--->
The documentation of the code can be found [here](http://lsstdesc.org/CLMM/) and the overview of the code can be found [here](OVERVIEW.md).
The journal paper that describes the development and validation of `CLMM` v1.0 can be found [here](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.6092A/abstract). If you make use of the ideas or software here, please cite that paper and provide a
link to this repository: https://github.com/LSSTDESC/CLMM. Please follow the guidelines listed below to install, use and contribute to CLMM.

## Table of contents
1. [Installing CLMM](#installing)
2. [Using CLMM](#using)
3. [Contributing to CLMM](#contributing)
5. [Contact](#contact)
6. [Acknowledgements](#acknowledgements)

# Installing CLMM <a name="installing"></a>

## Requirements <a name="requirements"></a>

CLMM requires Python version 3.8 or later.  CLMM has the following dependencies:

- [NumPy](https://www.numpy.org/) (v1.17 or later)
- [SciPy](https://scipy.org/) (v1.6 or later)
- [Astropy](https://www.astropy.org/) (v4.0 or later for units and cosmology dependence)
(Please avoid Astropy v5.0 since there is bug breaking CCL backend. It has been fixed in Astropy v5.0.1.)
- [Matplotlib](https://matplotlib.org/) (for plotting and going through tutorials)

```
  pip install numpy scipy astropy matplotlib
```

For the theoretical predictions of the signal, CLMM relies on existing libraries and **at least one of the following must be installed as well**:

- [cluster-toolkit](https://cluster-toolkit.readthedocs.io/en/latest/)
- [CCL](https://ccl.readthedocs.io/en/latest/) (versions between 2.7.1.dev10+gf81b59a4 and 3)
- [NumCosmo](https://numcosmo.github.io/) (v0.19 or later)


(See the [INSTALL documentation](INSTALL.md) for more detailed installation instructions.)

For developers, you will also need to install:

- [pytest](https://docs.pytest.org/en/latest/) (3.x or later for testing)
- [Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html) (for documentation)

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
See the [INSTALL documentation](INSTALL.md) for more detailed installation instructions.

To run the tests you can do:

  `pytest`

# Using CLMM <a name="using"></a>

This code has been released by DESC, although it is still under active
development. You are welcome to re-use the code, which is open source and available under
terms consistent with our
[LICENSE](https://github.com/LSSTDESC/CLMM/blob/main/LICENSE) ([BSD
3-Clause](https://opensource.org/licenses/BSD-3-Clause)). In this case,
don't forget to reference the [paper](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.6092A/abstract)
and the [repository](https://github.com/LSSTDESC/CLMM).
If you use CLMM for a project, please see the guidelines below, depending on your case.

**DESC Projects**: External contributors and DESC members wishing to
use CLMM for DESC projects should consult with the DESC Clusters analysis
working group (CL WG) conveners, ideally before the work has started, but
definitely before any publication or posting of the work to the arXiv.

**Non-DESC Projects by DESC members**: If you are in the DESC
community, but planning to use CLMM in a non-DESC project, it would be
good practice to contact the CL WG co-conveners and/or the CLMM Topical
Team leads as well (see Contact section).  A desired outcome would be for your
non-DESC project concept and progress to be presented to the working group,
so working group members can help co-identify tools and/or ongoing development
that might mutually benefit your non-DESC project and ongoing DESC projects.

**External Projects by Non-DESC members**: If you are not from the DESC
community, you are also welcome to contact CLMM Topical Team leads to introduce
your project and share feedback.


For free use of the `NumCosmo` library, the `NumCosmo` developers
require that the `NumCosmo` publication be cited: NumCosmo: Numerical
Cosmology, S. Dias Pinto Vitenti and M. Penna-Lima, Astrophysics
Source Code Library, record ascl:1408.013.  See citation info
[here](https://ui.adsabs.harvard.edu/abs/2014ascl.soft08013D/exportcitation).
The `NumCosmo` repository can be found [here](https://github.com/NumCosmo/NumCosmo).

For free use of the `CCL` library, the `CCL` developers require that
the `CCL` publication be cited.  See details
[here](https://github.com/LSSTDESC/CCL).

The `Cluster Toolkit` documentation can be found
[here](https://cluster-toolkit.readthedocs.io/en/latest/#).

The data for the notebook test_coordinate.ipynb is available at https://www.dropbox.com/scl/fo/dwsccslr5iwb7lnkf8jvx/AJkjgFeemUEHpHaZaHHqpAg?rlkey=efbtsr15mdrs3y6xsm7l48o0r&st=xb58ap0g&dl=0

# Contributing to CLMM <a name="contributing"></a>

You are welcome to contribute to the code. To do so, please follow the guidelines described [here](CONTRIBUTING.md).
If you are not part of the DESC CLMM topical team, it is good to also contact us (see below).

# Contact <a name="contact"></a>

If you have comments, questions, or feedback, please [write us an
issue](https://github.com/LSSTDESC/CLMM/issues).

The current leads of the LSST DESC CLMM Topical Team are Michel Aguena
(m-aguena, aguena@apc.in2p3.fr) and Marina Ricci (mricci,
marina.ricci@apc.in2p3.fr)


# Acknowledgements <a name="acknowledgements"></a>

The DESC acknowledges ongoing support from the Institut National de
Physique Nucl\'eaire et de Physique des Particules in France; the
Science \& Technology Facilities Council in the United Kingdom; and
the Department of Energy, the National Science Foundation, and the
LSST Corporation in the United States.  DESC uses resources of the
IN2P3 Computing Center (CC-IN2P3--Lyon/Villeurbanne - France) funded
by the Centre National de la Recherche Scientifique; the National
Energy Research Scientific Computing Center, a DOE Office of Science
User Facility supported by the Office of Science of the U.S.
Department of Energy under Contract No. DE-AC02-05CH11231; STFC DiRAC
HPC Facilities, funded by UK BIS National E-infrastructure capital
grants; and the UK particle physics grid, supported by the GridPP
Collaboration.  This work was performed in part under DOE Contract
DE-AC02-76SF00515.

The authors express gratitude to the LSSTC for the 2018 and 2019
Enabling Science grants, hosted by CMU and RUB respectively, that
supported the development of `CLMM` and its developer community.  CA
acknowledges support from the LSA Collegiate Fellowship at the
University of Michigan, the Leinweber Foundation, and DoE Award
DE-FOA-0001781.  AIM acknowledges support from the Max Planck Society
and the Alexander von Humboldt Foundation in the framework of the Max
Planck-Humboldt Research Award endowed by the Federal Ministry of
Education and Research. During the completion of this work, AIM was
advised by David W. Hogg and supported by National Science Foundation
grant AST-1517237.  CS acknowledges support from the Agencia Nacional
de Investigaci\'on y Desarrollo (ANID) through FONDECYT grant no.
11191125.  AvdL, RH, LB, and HF acknowledge support by the US
Department of Energy under award DE-SC0018053.  SF acknowledges
support from DOE grant DE-SC0010010.  HM is supported by the Jet
Propulsion Laboratory, California Institute of Technology, under a
contract with the National Aeronautics and Space Administration.

