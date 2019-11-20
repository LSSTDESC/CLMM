*****************
Mock Data Example
*****************

We recommend using mock data of a galaxy cluster to test your
pipeline. This repository contains tools to create both ideal and
noisy mock data.

In this example we generate mock data with a variety of systematic
effects including photometric redshifts, source galaxy distributions,
and shape noise. We then populate a galaxy cluster object

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

.. code:: ipython3

    import clmm
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

Import mock data module and setup the configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import sys
    sys.path.append('./support')
    import mock_data as mock

Mock data generation requires a defined cosmology

.. code:: ipython3

    from astropy.cosmology import FlatLambdaCDM
    mock_cosmo = FlatLambdaCDM(H0=70, Om0=0.27, Ob0=0.045)

Mock data generation requires some cluster information

.. code:: ipython3

    cosmo = mock_cosmo
    cluster_id = "Awesome_cluster"
    cluster_m = 1.e15
    cluster_z = 0.3
    src_z = 0.8
    concentration = 4
    ngals = 1000 # number of source galaxies
    Delta = 200 # mass definition with respect to critical overdensity
    cluster_ra = 0.0
    cluster_dec = 0.0


Generate the mock catalog with different options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clean data: no noise, all galaxies at the same redshift

.. code:: ipython3

    ideal_data = mock.generate_galaxy_catalog(cluster_m, cluster_z, concentration,
                                              cosmo, ngals, Delta, src_z)

Noisy data: shape noise, all galaxies at the same redshift

.. code:: ipython3

    noisy_data_src_z = mock.generate_galaxy_catalog(cluster_m,
                                                cluster_z,
                                                concentration,
                                                cosmo,
                                                ngals,
                                                Delta,
                                                src_z,
                                                shapenoise=0.005,
                                                photoz_sigma_unscaled=0.05)


Noisy data: photo-z errors (and pdfs!), all galaxies at the same
redshift

.. code:: ipython3

    noisy_data_photoz = mock.generate_galaxy_catalog(cluster_m,
                                                cluster_z,
                                                concentration,
                                                cosmo,
                                                ngals,
                                                Delta,
                                                'chang13',
                                                shapenoise=0.005,
                                                photoz_sigma_unscaled=0.05)

Clean data: source galaxy redshifts drawn from Chang et al. 2013

.. code:: ipython3

    ideal_with_src_dist = mock.generate_galaxy_catalog(cluster_m, cluster_z, concentration,
                                              cosmo, ngals, Delta, 'chang13',zsrc_max=7.0)


Noisy data: galaxies following redshift distribution, redshift error,
shape noise

.. code:: ipython3

    allsystematics = mock.generate_galaxy_catalog(cluster_m, cluster_z, concentration,
                                              cosmo, ngals, Delta, 'chang13',zsrc_max=7.0,
                                                  shapenoise=0.005, photoz_sigma_unscaled=0.05)

Inspect the catalog data
========================

Ideal catalog first entries: no noise on the shape measurement, all galaxies at z=0.8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    ideal_data[0:3]

With photo-z errors
^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    noisy_data_photoz[0:3]

.. code:: ipython3

    # Histogram of the redshift distribution of bkg galaxies (starting at z_cluster + 0.1)
    hist = plt.hist(allsystematics['z'], bins=50)
    plt.xlabel('Source Redshift')

.. code:: ipython3

    # pdz for the first galaxy in the catalog
    plt.plot(allsystematics['pzbins'][0], allsystematics['pzpdf'][0])
    plt.xlabel('Redshift')
    plt.ylabel('Photo-z Probability Distribution')

Populate in a galaxy cluster object

.. code:: ipython3

    # At the moment mock data only allow for a cluster centred on (0,0)
    cluster_ra = 0.0
    cluster_dec = 0.0
    gc_object = clmm.GalaxyCluster(cluster_id, cluster_ra, cluster_dec, 
                                   cluster_z, allsystematics)

Plot source galaxy ellipticities

.. code:: ipython3

    plt.scatter(gc_object.galcat['e1'],gc_object.galcat['e2'])
    
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.xlabel('Ellipticity 1',fontsize='x-large')
    plt.ylabel('Ellipticity 2',fontsize='x-large')
