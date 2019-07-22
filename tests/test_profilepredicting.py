"""
Tests for profilepredicting
"""

def test_set_omega_m(cosmo):
    # check that Om_b, Om_c exist
    # check type of cosmo
    # use numpy asserts (numpy.testing) when possible
    pass

# others: test that inputs are as expected, values from demos

# AIM: I'm removing these hardcoded things from the notebook.
# # Define CCL cosmology object
# cosmo_ccl = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)

# # Select density profile and profile parametrization options
# density_profile_parametrization = 'nfw'
# mass_Delta = 200
# cluster_mass = 1.e15
# cluster_concentration = 4

# AIM: Let's use these as a starting point for unit tests
# # Quick test of functions
#
# r3d = np.logspace(-2,2,100)
#
# rho = get_3d_density_profile(r3d,mdelta=cluster_mass, cdelta=cluster_concentration, cosmo=cosmo_ccl)
#
# Sigma = calculate_surface_density(r3d, cluster_mass, cluster_concentration, cosmo=cosmo_ccl, Delta=200,
#                                   halo_profile_parameterization='nfw')
#
# DeltaSigma = calculate_excess_surface_density(r3d, cluster_mass, cluster_concentration, cosmo=cosmo_ccl, Delta=200,
#                                               halo_profile_parameterization='nfw')
#
# Sigmac = get_critical_surface_density(cosmo_ccl, z_cluster=1.0, z_source=2.0)
#
# gammat = compute_tangential_shear_profile(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=1.0,
#                                           z_source=2.0, cosmo=cosmo_ccl, Delta=200,
#                                           halo_profile_parameterization='nfw', z_src_model='single_plane')
#
# kappa = compute_convergence_profile(r3d, mdelta=cluster_mass, cdelta=cluster_concentration,
#                             z_cluster=1.0, z_source=2.0,
#                                  cosmo=cosmo_ccl, Delta=200,
#                                      halo_profile_parameterization='nfw',
#                                     z_src_model='single_plane')
#
# gt = compute_reduced_tangential_shear_profile(r3d, mdelta=cluster_mass, cdelta=cluster_concentration,
#                                          z_cluster=1.0, z_source=2.0, cosmo=cosmo_ccl, Delta=200,
#                                          halo_profile_parameterization='nfw', z_src_model='single_plane')
