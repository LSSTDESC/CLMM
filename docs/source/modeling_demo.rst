
.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

.. code:: ipython3

    import sys
    sys.path.append('./support')
    import clmm
    import clmm.modeling as m

Define a cosmology
^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from astropy.cosmology import FlatLambdaCDM
    astropy_cosmology_object = FlatLambdaCDM(H0=70, Om0=0.27, Ob0=0.045)
    cosmo_ccl = m.cclify_astropy_cosmo(astropy_cosmology_object)

Define the galaxy cluster model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # Select density profile and profile parametrization options 
    density_profile_parametrization = 'nfw'
    mass_Delta = 200
    cluster_mass = 1.e15
    cluster_concentration = 4
    z_cl = 1.
    z_source = 2.

Quick test of all modeling functionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    r3d = np.logspace(-2, 2, 100)

.. code:: ipython3

    rho = m.get_3d_density(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, 
                           z_cl=z_cl, cosmo=cosmo_ccl)

.. code:: ipython3

    Sigma = m.predict_surface_density(r3d, cluster_mass, cluster_concentration, z_cl, cosmo=cosmo_ccl, 
                                      delta_mdef=mass_Delta, 
                                      halo_profile_model=density_profile_parametrization)

.. code:: ipython3

    DeltaSigma = m.predict_excess_surface_density(r3d, cluster_mass, cluster_concentration, z_cl, cosmo=cosmo_ccl, 
                                                  delta_mdef=mass_Delta, 
                                                  halo_profile_model=density_profile_parametrization)

.. code:: ipython3

    Sigmac = m.get_critical_surface_density(cosmo_ccl, z_cluster=z_cl, z_source=z_source)

.. code:: ipython3

    gammat = m.predict_tangential_shear(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=z_cl, 
                                        z_source=z_source, cosmo=cosmo_ccl, delta_mdef=mass_Delta, 
                                        halo_profile_model=density_profile_parametrization, 
                                        z_src_model='single_plane')

.. code:: ipython3

    kappa = m.predict_convergence(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, 
                                   z_cluster=z_cl, z_source=z_source,
                                   cosmo=cosmo_ccl, delta_mdef=mass_Delta, 
                                   halo_profile_model=density_profile_parametrization, 
                                   z_src_model='single_plane')

.. code:: ipython3

    gt = m.predict_reduced_tangential_shear(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, 
                                            z_cluster=z_cl, z_source=z_source, cosmo=cosmo_ccl, 
                                            delta_mdef=mass_Delta, 
                                            halo_profile_model=density_profile_parametrization, 
                                            z_src_model='single_plane')
            

Plot the predicted profiles

.. code:: ipython3

    def plot_profile(r, profile_vals, profile_label='rho'):
        plt.loglog(r, profile_vals)
        plt.xlabel('r [Mpc]', fontsize='xx-large')
        plt.ylabel(profile_label, fontsize='xx-large')

.. code:: ipython3

    plot_profile(r3d, rho, '$\\rho_{\\rm 3d}$')

.. code:: ipython3

    plot_profile(r3d, Sigma, '$\\Sigma_{\\rm 2d}$')

.. code:: ipython3

    plot_profile(r3d, DeltaSigma, '$\\Delta\\Sigma_{\\rm 2d}$')

.. code:: ipython3

    plot_profile(r3d, kappa, '$\\kappa$')

.. code:: ipython3

    plot_profile(r3d, gammat, '$\\gamma_t$')

.. code:: ipython3

    plot_profile(r3d, gt, '$g_t$')
