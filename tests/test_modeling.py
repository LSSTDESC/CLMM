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
    assert_equal(md.cclify_astropy_cosmo(apycosmo_flcdm), cclcosmo)
    assert_equal(md.cclify_astropy_cosmo(apycosmo_lcdm), cclcosmo)

    # Test output if we pass a CCL object (a dict right now)
    assert_equal(md.cclify_astropy_cosmo(cclcosmo), cclcosmo)

    # Test for exception if anything else is passed in
    assert_raises(TypeError, md.cclify_astropy_cosmo, 70.)
    assert_raises(TypeError, md.cclify_astropy_cosmo, [70., 0.3, 0.25, 0.05])


def test_astropyify_ccl_cosmo():
    # Make a bse object
    truth = {'H0': 70., 'Om0': 0.3, 'Ob0': 0.05}
    apycosmo_flcdm = FlatLambdaCDM(**truth)
    apycosmo_lcdm = LambdaCDM(Ode0=1.0-truth['Om0'], **truth)
    cclcosmo = {'Omega_c': truth['Om0'] - truth['Ob0'], 'Omega_b': truth['Ob0'],
                'h': truth['H0']/100., 'H0': truth['H0']}

    # Test output if we pass FlatLambdaCDM and LambdaCDM objects
    assert_equal(md.astropyify_ccl_cosmo(apycosmo_flcdm), apycosmo_flcdm)
    assert_equal(md.astropyify_ccl_cosmo(apycosmo_lcdm), apycosmo_lcdm)

    # Test output if we pass a CCL object, compare the dicts
    assert_equal(md.cclify_astropy_cosmo(md.astropyify_ccl_cosmo(cclcosmo)),
                 md.cclify_astropy_cosmo(apycosmo_lcdm))

    # Test for exception if anything else is passed in
    assert_raises(TypeError, md.astropyify_ccl_cosmo, 70.)
    assert_raises(TypeError, md.astropyify_ccl_cosmo, [70., 0.3, 0.25, 0.05])


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


def test_get_reduced_shear():
    # Make some base objects
    shear = [0.5, 0.75, 1.25, 0.0]
    convergence = [0.75, -0.2, 0.0, 2.3]
    truth = [2., 0.625, 1.25, 0.0]

    # Test for exception if shear and convergence are not the same length
    assert_raises(ValueError, md.get_reduced_shear_from_convergence, shear[:3], convergence[:2])
    assert_raises(ValueError, md.get_reduced_shear_from_convergence, shear[:2], convergence[:3])

    # Check output including: float, list, ndarray
    assert_allclose(md.get_reduced_shear_from_convergence(shear[0], convergence[0]),
                 truth[0], **TOLERANCE)
    assert_allclose(md.get_reduced_shear_from_convergence(shear, convergence),
                 truth, **TOLERANCE)
    assert_allclose(md.get_reduced_shear_from_convergence(np.array(shear), np.array(convergence)),
                 np.array(truth), **TOLERANCE)


def helper_profiles(func):
    """ A helper function that is used to repeat all of the unit tests
    on the profile functions below.
    - get_3d_density
    - predict_surface_density
    - predict_excess_surface_density
    """
    # Make some base objects
    r3d = np.logspace(-2, 2, 100)
    mdelta = 1.0e15
    cdelta = 4.0
    z_cl = 0.2
    cclcosmo = {'Omega_c': 0.25, 'Omega_b': 0.05}

    # Test for exception if other profiles models are passed
    assert_raises(ValueError, func, r3d, mdelta, cdelta, z_cl, cclcosmo, 200, 'bleh')

    # Test defaults
    defaulttruth = func(r3d, mdelta, cdelta, z_cl, cclcosmo, delta_mdef=200,
                        halo_profile_model='nfw')
    assert_allclose(func(r3d, mdelta, cdelta, z_cl, cclcosmo, halo_profile_model='nfw'),
                    defaulttruth, **TOLERANCE)
    assert_allclose(func(r3d, mdelta, cdelta, z_cl, cclcosmo, delta_mdef=200),
                    defaulttruth, **TOLERANCE)


def test_profiles_unittests():
    # TODO: Revise docstring, not clear what parameters are
    helper_profiles(md.get_3d_density)
    # TODO: Revise docstring, not clear what parameters are
    helper_profiles(md.predict_surface_density)
    # TODO: Why do we hard code sigma_r_proj in here? I moved it out of the NFW block
    # TODO: Revise docstring, not clear what parameters are
    helper_profiles(md.predict_excess_surface_density)


def test_profiles_validation():
    # TODO: Validation tests for `get_3d_density` for the NFW profile
    # TODO: Validation tests for `predict_surface_density` for the NFW profile
    # TODO: Validation tests for `test_predict_excess_surface_density`
    pass


def test_angular_diameter_dist_a1a2():
    # TODO: Can we rename the parameters as well? asource, alens or something
    # Make some base objects
    truth = {'H0': 70., 'Om0': 0.3, 'Ob0': 0.05}
    apycosmo = FlatLambdaCDM(**truth)
    cclcosmo = {'Omega_c': truth['Om0'] - truth['Ob0'], 'Omega_b': truth['Ob0'],
                'h': truth['H0']/100., 'H0': truth['H0']}

    # Test if we pass in CCL cosmo or astropy cosmo
    sf1, sf2 = 0.56, 0.78
    assert_allclose(md.angular_diameter_dist_a1a2(cclcosmo, sf1, sf2),
                    md.angular_diameter_dist_a1a2(apycosmo, sf1, sf2), **TOLERANCE)

    # Test default values
    assert_allclose(md.angular_diameter_dist_a1a2(cclcosmo, sf1),
                    md.angular_diameter_dist_a1a2(apycosmo, sf1, a2=1.),
                    **TOLERANCE)

    # TODO: Validation test


def test_get_critical_surface_density():
    # TODO: Validation test
    pass


def helper_physics_functions(func):
    """ We have several functions with identical call signitures.
    Rather than repeat the same exact code to test each one, I am
    writing this helper that can be used to test things like
    defaults and exceptions """
    # Make some base objects
    rproj = np.logspace(-2, 2, 100)
    mdelta = 1.0e15
    cdelta = 4.0
    z_cl = 0.2
    z_src = 0.45
    cosmo = {'Omega_c': 0.25, 'Omega_b': 0.05, 'H0': 70.}

    # Test defaults
    defaulttruth = func(rproj, mdelta, cdelta, z_cl, z_src, cosmo, delta_mdef=200,
                        halo_profile_model='nfw', z_src_model='single_plane')
    assert_allclose(func(rproj, mdelta, cdelta, z_cl, z_src, cosmo, halo_profile_model='nfw',
                         z_src_model='single_plane'), defaulttruth, **TOLERANCE)
    assert_allclose(func(rproj, mdelta, cdelta, z_cl, z_src, cosmo, delta_mdef=200,
                         z_src_model='single_plane'), defaulttruth, **TOLERANCE)
    assert_allclose(func(rproj, mdelta, cdelta, z_cl, z_src, cosmo, delta_mdef=200,
                         halo_profile_model='nfw'), defaulttruth, **TOLERANCE)

    # Test for exception on unsupported z_src_model and halo profiles
    assert_raises(ValueError, func, rproj, mdelta, cdelta, z_cl, z_src, cosmo,
                  200, 'bleh', 'single_plane')
    assert_raises(ValueError, func, rproj, mdelta, cdelta, z_cl, z_src, cosmo,
                  200, 'nfw', 'bleh')


def test_shear_convergence_unittests():
    helper_physics_functions(md.predict_tangential_shear)
    helper_physics_functions(md.predict_convergence)
    helper_physics_functions(md.predict_reduced_tangential_shear)


def test_shear_convergence_validation():
    """ These tests should test our functions against an external code base
    to ensure that our physical results are valid. 
    - predict_tangential_shear
    - predict_convergence
    - predict_reduced_tangential_shear
    """
    # TODO: Validation tests for predict_tangential_shear
    # TODO: Validation tests for predict_convergence
    # TODO: Validation tests for predict_reduced_tangential_shear
    pass
