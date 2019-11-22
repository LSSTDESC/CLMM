"""Tests for modeling.py"""
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
from astropy.cosmology import FlatLambdaCDM, LambdaCDM
import clmm.modeling as md


TOLERANCE = {'rtol': 1.0e-6, 'atol': 1.0e-6}


def test_cclify_astropy_cosmo():
    # Make some base objects
    truth = {'H0': 70., 'Om0': 0.3, 'Ob0': 0.05}
    apycosmo_flcdm = FlatLambdaCDM(**truth)
    apycosmo_lcdm = LambdaCDM(Ode0=1.0-truth['Om0'], **truth)
    cclcosmo = {'Omega_c': truth['Om0'] - truth['Ob0'], 'Omega_b': truth['Ob0'],
                'h': truth['H0']/100., 'H0': truth['H0']}

    # Test for exception if missing baryon density (everything else required)
    missbaryons = FlatLambdaCDM(H0=truth['H0'], Om0=truth['Om0'])
    assert_raises(KeyError, md.cclify_astropy_cosmo, missbaryons)

    # Test output if we pass FlatLambdaCDM and LambdaCDM objects
    x = md.cclify_astropy_cosmo(apycosmo_flcdm)
    assert_equal(md.cclify_astropy_cosmo(apycosmo_flcdm), cclcosmo)
    assert_equal(md.cclify_astropy_cosmo(apycosmo_lcdm), cclcosmo)

    # Test output if we pass a CCL object (a dict right now)
    assert_equal(md.cclify_astropy_cosmo(cclcosmo), cclcosmo)

    # Test for exception if anything else is passed in
    assert_raises(TypeError, md.cclify_astropy_cosmo, 70.)
    assert_raises(TypeError, md.cclify_astropy_cosmo, [70., 0.3, 0.25, 0.05])


def test_scale_factor_redshift_conversion():
    # Convert from a to z - scalar, list, ndarray
    assert_allclose(md._get_a_from_z(0.5), 2./3., **TOLERANCE)
    assert_allclose(md._get_a_from_z([0.1, 0.2, 0.3, 0.4]),
                    [10./11., 5./6., 10./13., 5./7.], **TOLERANCE)
    assert_allclose(md._get_a_from_z(np.array([0.1, 0.2, 0.3, 0.4])),
                    np.array([10./11., 5./6., 10./13., 5./7.]), **TOLERANCE)

    # Convert from z to a - scalar, list, ndarray
    assert_allclose(md._get_z_from_a(2./3.), 0.5, **TOLERANCE)
    assert_allclose(md._get_z_from_a([10./11., 5./6., 10./13., 5./7.]),
                    [0.1, 0.2, 0.3, 0.4], **TOLERANCE)
    assert_allclose(md._get_z_from_a(np.array([10./11., 5./6., 10./13., 5./7.])),
                    np.array([0.1, 0.2, 0.3, 0.4]), **TOLERANCE)

    # Some potential corner-cases for the two funcs
    assert_allclose(md._get_a_from_z(np.array([0.0, 1300.])),
                    np.array([1.0, 1./1301.]), **TOLERANCE)
    assert_allclose(md._get_z_from_a(np.array([1.0, 1./1301.])),
                    np.array([0.0, 1300.]), **TOLERANCE)

    # Test for exceptions when outside of domains
    assert_raises(ValueError, md._get_a_from_z, -5.0)
    assert_raises(ValueError, md._get_a_from_z, [-5.0, 5.0])
    assert_raises(ValueError, md._get_a_from_z, np.array([-5.0, 5.0]))
    assert_raises(ValueError, md._get_z_from_a, 5.0)
    assert_raises(ValueError, md._get_z_from_a, [-5.0, 5.0])
    assert_raises(ValueError, md._get_z_from_a, np.array([-5.0, 5.0]))

    # Convert from a to z to a (and vice versa)
    testval = 0.5
    assert_allclose(md._get_a_from_z(md._get_z_from_a(testval)), testval, **TOLERANCE)
    assert_allclose(md._get_z_from_a(md._get_a_from_z(testval)), testval, **TOLERANCE)


def test_get_3d_density():
    pass

def test_predict_surface_density():
    pass

def test_predict_excess_surface_density():
    pass

def test_get_angular_diameter_distance_a():
    pass

def test_get_critical_surface_density():
    pass

def test_predict_tangential_shear():
    pass

def test_predict_convergence():
    pass

def test_predict_reduced_tangential_shear():
    pass






# import astropy
# from astropy import cosmology
# from numpy import testing as tst
# import numpy as np
# import clmm
# # from clmm import modeling as pp
# # from modeling import *

# density_profile_parametrization = 'nfw'
# mass_lims = (1.e12, 1.e16)
# mass_Delta = 200
# cluster_mass = 1.e15
# cluster_concentration = 4
# z_cluster = 1.
#
# r3d = np.logspace(-2, 2, 100)
# r3d_one = r3d[-1]
#
# rho = clmm.get_3d_density(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, z_cl=z_cluster, cosmo=cosmo_ccl)
# rho_one = clmm.get_3d_density(r3d_one, mdelta=cluster_mass, cdelta=cluster_concentration, z_cl=z_cluster, cosmo=cosmo_ccl)
#
# def test_rho():
#     tst.assert_equal(rho[-1], rho_one)
#
# Sigma = clmm.predict_surface_density(r3d, cluster_mass, cluster_concentration, z_cl=z_cluster, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw')
# Sigma_one = clmm.predict_surface_density(r3d_one, cluster_mass, cluster_concentration, z_cl=z_cluster, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw')
#
# def test_Sigma():
#     assert(np.all(Sigma > 0.))
#     tst.assert_equal(Sigma[-1], Sigma_one)
#
# DeltaSigma = clmm.predict_excess_surface_density(r3d, cluster_mass, cluster_concentration, z_cl=z_cluster, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw')
# DeltaSigma_one = clmm.predict_excess_surface_density(r3d_one, cluster_mass, cluster_concentration, z_cl=z_cluster, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw')
#
# def test_DeltaSigma():
#     tst.assert_equal(DeltaSigma[-1], DeltaSigma_one)
#     assert(np.all(DeltaSigma > 0.))
#     assert(DeltaSigma_one > 0.)
#
# Sigmac = clmm.get_critical_surface_density(cosmo_ccl, z_cluster=1.0, z_source=2.0)
#
# # def test_Sigmac():
#     # not sure what to put here yet
#
# gammat = clmm.predict_tangential_shear(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=1.0, z_source=2.0, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw', z_src_model='single_plane')
# gammat_one = clmm.predict_tangential_shear(r3d_one, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=1.0, z_source=2.0, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw', z_src_model='single_plane')
#
# def test_gammat():
#     tst.assert_equal(gammat[-1], gammat_one)
#     tst.assert_equal(gammat, DeltaSigma / Sigmac)
#     tst.assert_equal(gammat_one, DeltaSigma_one / Sigmac)
#
# kappa = clmm.predict_convergence(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=1.0, z_source=2.0, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw', z_src_model='single_plane')
# kappa_one = clmm.predict_convergence(r3d_one, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=1.0, z_source=2.0, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw', z_src_model='single_plane')

# def test_kappa():
#     tst.assert_equal(kappa[-1], kappa_one)
#     assert(kappa_one > 0.)
#     assert(np.all(kappa > 0.))
#
# gt = clmm.predict_reduced_tangential_shear(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=1.0, z_source=2.0, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw', z_src_model='single_plane')
# gt_one = clmm.predict_reduced_tangential_shear(r3d_one, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=1.0, z_source=2.0, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw', z_src_model='single_plane')
#
# def test_gt():
#     tst.assert_equal(gt[-1], gt_one)
#     tst.assert_equal(gt, gammat / (1. - kappa))
#     tst.assert_equal(gt_one, gammat_one / (1. - kappa_one))
#
# # others: test that inputs are as expected, values from demos
# # positive values from sigma onwards
