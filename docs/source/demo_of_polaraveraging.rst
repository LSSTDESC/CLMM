
Notebook to serve as example of how to use polaraveraging
=========================================================

--------------

.. code:: ipython3

    import matplotlib.pyplot as plt
    from clmm.polaraveraging import compute_shear, make_shear_profile, make_bins
    from clmm.plotting import plot_profiles
    from clmm.galaxycluster import GalaxyCluster
    import sys
    sys.path.append('./support')
    import mock_data as mock

Define cosmology object
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from astropy.cosmology import FlatLambdaCDM
    mock_cosmo = FlatLambdaCDM(H0=70., Om0=0.3, Ob0=0.025)

1) Generate cluster object with mock data with shape noise, galaxies from redshift distribution and a pdz for each source galaxy
--------------------------------------------------------------------------------------------------------------------------------

Define toy cluster parameters for mock data generation

.. code:: ipython3

    cosmo = mock_cosmo
    cluster_id = "Awesome_cluster"
    cluster_m = 1.e15
    cluster_z = 0.3
    concentration = 4
    ngals = 1000
    Delta = 200
    
    noisy_data_z = mock.generate_galaxy_catalog(cluster_m,
                                                cluster_z,
                                                concentration,
                                                cosmo,
                                                ngals,
                                                Delta,
                                                'chang13',
                                                shapenoise=0.005,
                                                photoz_sigma_unscaled=0.05)

Loading this into a CLMM cluster object centered on (0,0)

.. code:: ipython3

    cluster_ra = 0.0
    cluster_dec = 0.0
    gc_object = GalaxyCluster(cluster_id, cluster_ra, cluster_dec, 
                                   cluster_z, noisy_data_z)

2) Load cluster object containing:
----------------------------------

    Lens properties (ra\_l, dec\_l, z\_l)

    Source properties (ra\_s, dec\_s, e1, e2) ### Note, if loading from
    mock data, use: > cl = gc.load\_cluster("GC\_from\_mock\_data.pkl")

.. code:: ipython3

    cl = gc_object
    print("Cluster info = ID:", cl.unique_id, "; ra:", cl.ra,
          "; dec:", cl.dec, "; z_l :", cl.z)
    print("The number of source galaxies is :", len(cl.galcat))

Plot cluster and galaxy positions

.. code:: ipython3

    plt.scatter(cl.galcat['ra'], cl.galcat['dec'], color='blue', s=1, alpha=0.3)
    plt.plot(cl.ra, cl.dec, 'ro')
    plt.ylabel('dec', fontsize="large")
    plt.xlabel('ra', fontsize="large")

Check the ellipticities

.. code:: ipython3

    fig, ax1 = plt.subplots(1, 1)
    
    ax1.scatter(cl.galcat['e1'], cl.galcat['e2'], s=1, alpha=0.2)
    ax1.set_xlabel('e1')
    ax1.set_ylabel('e2')
    ax1.set_aspect('equal', 'datalim')
    ax1.axvline(0, linestyle='dotted', color='black')
    ax1.axhline(0, linestyle='dotted', color='black')

3) Compute and plot shear profiles
----------------------------------

Compute angular separation, cross and tangential shear for each source
galaxy

.. code:: ipython3

    theta, g_t, g_x = compute_shear(cl, geometry="flat")

Plot tangential and shear distributions for verification, which can be
accessed in the galaxy cluster object, cl.

.. code:: ipython3

    plt.hist(cl.galcat['gt'],bins=100)
    plt.xlabel('$\\gamma_t$',fontsize='xx-large')

.. code:: ipython3

    plt.hist(cl.galcat['gx'],bins=100)
    plt.xlabel('$\\gamma_x$',fontsize='xx-large')
    plt.yscale('log')

Compute transversal and cross shear profiles in units defined by user,
using defaults binning

.. code:: ipython3

    profiles = make_shear_profile(cl, "radians", "kpc", cosmo=cosmo)

Use function to plot the profiles

.. code:: ipython3

    fig, ax = plot_profiles(cl)

Shear Profile example in degrees

.. code:: ipython3

    new_profiles = make_shear_profile(cl, "radians", "degrees",
                                         cosmo=cosmo)
    fig1, ax1 = plot_profiles(cl, "degrees")

With user defined binning, compute transversal and cross shear profiles
in units defined by user, plot the new profiles

.. code:: ipython3

    new_bins = make_bins(1, 6, 20)
    
    new_profiles = make_shear_profile(cl, "radians", "Mpc",
                                         bins=new_bins, cosmo=cosmo)
    
    fig1, ax1 = plot_profiles(cl, "Mpc", r_units='Mpc')

You can also access the individual profile quantities

.. code:: ipython3

    plt.title('cross shear test')
    plt.errorbar(new_profiles['radius'], new_profiles['gx'],
                 new_profiles['gx_err'])
    plt.title('cross shear test')
    plt.axhline(0, linestyle='dotted', color='black')
    plt.xlabel("Radius [Mpc]")
    plt.ylabel('$\\gamma_x$')
