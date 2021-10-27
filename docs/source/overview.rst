******************
Rapid overview
******************
CLMM is a tool to estimate cluster masses from weak lensing data. 
CLMM provides the initial building blocks for an end-to-end cluster weak 
lensing pipeline that can be validated on mock data and run on real data 
from LSST or other telescopes.

It is based on a data operations package to prepare the data vector, 
a theory package to predict the signal from cluster properties and also 
includes ways to generate mock catalogs of various complexity levels.
A set of examples is also provided, ranging from demos of base functionalities 
to end-to-end WL mass estimations. All are available in the `examples` folder of the project and some have been included in
this documentation (see below).

The `GalaxyCluster` object
==========================

The Galaxy cluster object is the core data structure in CLMM. It contains at least
 * The galaxy cluster metadata (unique_id, ra, dec, z)
 * A table of background galaxies: astropy Table containing at least for each galaxy galaxy_id, ra, dec, e1, e2, z
 * Additional attributes (e.g. binned radial shear profiles) are then added to the GalaxyCluster object at various steps of the analysis.

Weak lensing signal measurement with the `dataops` package
============================================================

All the functions of the `dataops` package are also methods of the `GalaxyCluster` object. In a nutshell, the main functions are:
 * `compute_tangential_and_cross_components` calculates tangential shear, cross shear, and angular separation of each source galaxy relative to the (ra, dec) coordinates of the center of the cluster.
 * `make_radial_profile` averages the tangential and cross shear of galaxies in user-defined bins and support bins in rad, deg, arcmin, arcsec, kpc, or Mpc. The latter are easily generated thanks to the make_bins function.

See `demo_dataops_functionality.ipynb` to see all functionalities and possible options.

Profile models and cosmology with `theory` and `cosmology` packages
=========================================================================

The `theory` package holds modules for evaluating theoretical models, whatever backend (cluster-toolkit, CCL or NumCosmo) the user has chosen to use. All is transparent to the user, but some backend will support more functionality than others. The default, that all backends support, is to use an NFW density profile for the cluster, with a M200,m mass definition.

Each theory backend relies on a cosmology object, of a different type, depending on the backend. The CLMM cosmology superclass wraps these various types of object to make it transparent for the user.

Finally, the theory package has both a functional and object-oriented interface and these may be used whatever the selected theory backend.

See `examples/demo_theory_functionality.ipynb` and `examples/demo_theory_functionality_oo.ipynb` for detailed examples of the functional and object-oriented interfaces respectively.


Mock data generation
========================

In order to test/develop the code but also help new users or scientists new to the field to explore some of the effects affecting cluster WL mass reconstruction, a CLMM module allows us to generate mock datasets from a variety of ingredients (w/wo shape noise, w/wo photoz errors, etc.).

See `examples/demo_generate_mock_cluster.ipynb` for all possible use cases.

Galaxy cluster mass estimation
==================================

CLMM was not designed to provide an end-to-end mass estimation pipeline, but to provide the building blocks to do so. How to use these blocks to make a mass estimation is exemplified in a series of notebooks using either simple scipy tools for the minimization or Numcosmo's more sophisticated statistical framework. Look for the notebooks called `ExampleXX_Fit_Halo_Mass_to_Shear_Catalog*` in the examples folder, e.g, `examples/Example3_Fit_Halo_Mass_to_Shear_Catalog.ipynb`.
