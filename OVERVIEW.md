# Overview
CLMM (Cluster Lensing Mass Modeling) provides a tool to estimate cluster masses based on weak lensing data.
It also includes a routine to make mock catalogs based on cluster_toolkit.
CLMM consists of the building blocks for an end-to-end weak lensing cosmology pipeline that can be validated on mock data and run on real data from LSST or other telescopes.
We provide [examples](https://github.com/LSSTDESC/CLMM/tree/master/examples) of its usage in this repository.

* [Main readme](README.md)

## Table of contents
1. [The `GalaxyCluster` object](#the_galaxycluster_object)
2. [Weak lensing signal measurement with `polaraveraging.py`](#weak_lensing_signal_measurement_with_polaraveraging)
3. [Profile and cosmology models with `modeling.py`](#profile_and_cosmology_models_with_modeling)
4. [Mock data generation](#mock_data_generation)
5. [Galaxy cluster mass estimation](#galaxy_cluster_mass_estimation)


## The `GalaxyCluster` object <a name="the_galaxycluster_object"></a>

  * The GalaxyCluster object contains the galaxy cluster metadata (unique_id, ra, dec, z) as well as the background galaxy data
  * Background galaxy data: astropy Table containing at least galaxy_id, ra, dec, e1, e2, z
  * ra/dec are in decimal degrees

## Weak lensing signal measurement with `polaraveraging.py` <a name="weak_lensing_signal_measurement_with_polaraveraging"></a>

  * The function `compute_shear` calculates tangential shear, cross shear, and angular separation of each source galaxy relative to the (ra, dec) coordinates of the center of the cluster.
  * A shear profile may be constructed with the user's choice of binning via `make_bins`.
  * `make_shear_profile` averages the shear of galaxies in each radial bin in rad, deg, arcmin, arcsec, kpc, or Mpc.
  * See [examples/demo_polaraveraging_functionality.ipynb](examples/demo_polaraveraging_functionality.ipynb) for detailed examples.

## Profile and cosmology models with `modeling.py` <a name="profile_and_cosmology_models_with_modeling"></a>

  * modeling.py holds functions for evaluating theoretical models.
  * The default is to use an NFW profile, but more halo profile parameterizations will be added soon.
  * See examples/modeling_demo.ipynb for example usage.
  * See [examples/demo_modeling_functionality.ipynb](examples/demo_modeling_functionality.ipynb) for detailed examples.

## Mock data generation <a name="mock_data_generation"></a>
  * See [examples/demo_generate_mock_cluster.ipynb](examples/demo_generate_mock_cluster.ipynb).


## Galaxy cluster mass estimation <a name="galaxy_cluster_mass_estimation"></a>
  * See [examples/Example2_Fit_Halo_Mass_to_Shear_Catalog.ipynb](examples/Example2_Fit_Halo_Mass_to_Shear_Catalog.ipynb) for example usage of an end-to-end measurement.
