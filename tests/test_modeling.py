"""Tests for modeling.py"""
import json
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
from astropy.cosmology import FlatLambdaCDM, LambdaCDM
import clmm.modeling as md
from clmm.constants import Constants as clc


TOLERANCE = {'rtol': 1.0e-6, 'atol': 1.0e-6}

# ----------- Some Helper Functions for the Validation Tests ---------------
def compute_sigmac_physical_constant(lightspeed, gnewt, msun, pc_to_m):
    """ Computes physical constant used to in Sigma_crit

    Parameters
    ----------
    lightspeed,: float
        Lightspeed in km/s
    gnewt: float
        Gravitational constant in m^3/(km s^2)
    msun: float
        Solar mass in kg
    pc_to_m: float
        Value of 1 parsec in meters

    Returns
    -------
    float
        lightspeed^2/G[Msun/pc]
    """
    return (lightspeed*1000./pc_to_m)**2/(gnewt*msun/pc_to_m**3)


def load_validation_config():
    """ Loads values precomputed by numcosmo for comparison """
    numcosmo_path = 'tests/data/numcosmo/'
    with open(numcosmo_path+'config.json', 'r') as fin:
        testcase = json.load(fin)
    numcosmo_profile = np.genfromtxt(numcosmo_path+'radial_profiles.txt', names=True)

    # Physical Constants
    CLMM_SIGMAC_PCST = compute_sigmac_physical_constant(clc.CLIGHT_KMS.value,
                                                        clc.GNEWT.value,
                                                        clc.SOLAR_MASS.value,
                                                        clc.PC_TO_METER.value)
    testcase_SIGMAC_PCST = compute_sigmac_physical_constant(testcase['lightspeed[km/s]'],
                                                            testcase['G[m3/km.s2]'],
                                                            testcase['Msun[kg]'],
                                                            testcase['pc_to_m'])
    SIGMAC_PHYSCONST_CORRECTION = testcase_SIGMAC_PCST/CLMM_SIGMAC_PCST

    # Cosmology
    cosmo = md.cclify_astropy_cosmo(FlatLambdaCDM(H0=testcase['cosmo_H0'],
                                                  Om0=testcase['cosmo_Om0'],
                                                  Ob0=testcase['cosmo_Ob0']))

    # Sets of parameters to be used by multiple functions
    RHO_PARAMS = {
        'r3d': np.array(numcosmo_profile['r3d']),
        'mdelta':testcase['cluster_mass'],
        'cdelta':testcase['cluster_concentration'],
        'z_cl':testcase['z_cluster'],
        }
    SIGMA_PARAMS = {
        'r_proj': np.array(numcosmo_profile['r3d']),
        'mdelta':testcase['cluster_mass'],
        'cdelta':testcase['cluster_concentration'],
        'z_cl':testcase['z_cluster'],
        'delta_mdef':testcase['mass_Delta'],
        'halo_profile_model':testcase['density_profile_parametrization'],
        }
    GAMMA_PARAMS = {
        'r_proj': np.array(numcosmo_profile['r3d']),
        'mdelta': testcase['cluster_mass'],
        'cdelta': testcase['cluster_concentration'],
        'z_cluster': testcase['z_cluster'],
        'z_source': testcase['z_source'],
        'delta_mdef': testcase['mass_Delta'],
        'halo_profile_model': testcase['density_profile_parametrization'],
        'z_src_model': 'single_plane',
        }

    return {'TEST_CASE': testcase, 'z_source': testcase['z_source'],
            'SIGMAC_PHYSCONST_CORRECTION': SIGMAC_PHYSCONST_CORRECTION, 'cosmo': cosmo,
            'RHO_PARAMS': RHO_PARAMS, 'SIGMA_PARAMS': SIGMA_PARAMS, 'GAMMA_PARAMS': GAMMA_PARAMS,
            'numcosmo_profiles': numcosmo_profile, 'TEST_CASE_SIGMAC_PCST': testcase_SIGMAC_PCST,
            'CLMM_SIGMAC_PCST': CLMM_SIGMAC_PCST}
# --------------------------------------------------------------------------

def test_physical_constants():
    """ Test physical values of physical_constants

    Notes
    -----
        The precision set for these tests put in here right now are somewhat arbitrary,
        has to be improved to values provided by CCL
    """
    cfg = load_validation_config()
    assert_allclose(cfg['TEST_CASE']['lightspeed[km/s]'], clc.CLIGHT_KMS.value, 1e-3)
    assert_allclose(cfg['TEST_CASE']['G[m3/km.s2]'], clc.GNEWT.value, 1e-3)
    assert_allclose(cfg['TEST_CASE']['pc_to_m'], clc.PC_TO_METER.value, 1e-6)
    assert_allclose(cfg['TEST_CASE']['Msun[kg]'], clc.SOLAR_MASS.value, 1e-2)
    assert_allclose(cfg['TEST_CASE_SIGMAC_PCST'], cfg['CLMM_SIGMAC_PCST'], 1e-2)


def test_cclify_astropy_cosmo():
    """ Unit tests for md.cllify_astropy_cosmo """
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
    """ Unit tests for astropyify_ccl_cosmo """
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
    """ Unit tests for redshift and scalefactor conversion """
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
    """ Unit tests for get_reduced_shear """
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
    """ A helper function to repeat a set of unit tests on several functions
    that expect the same inputs.

    Tests the following functions: get_3d_density, predict_surface_density,
                                   predict_excess_surface_density

    Tests that the functions:
    1. Throw an error if an invalid profile model is passed
    2. Test each default parameter to ensure that the defaults are not changed.
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


def test_profiles():
    """ Tests for profile functions, get_3d_density, predict_surface_density,
    and predict_excess_surface_density """
    helper_profiles(md.get_3d_density)
    helper_profiles(md.predict_surface_density)
    helper_profiles(md.predict_excess_surface_density)

    # Validation tests
    # NumCosmo makes different choices for constants (Msun). We make this conversion
    # by passing the ratio of SOLAR_MASS in kg from numcosmo and CLMM
    cfg = load_validation_config()
    constants_conversion = clc.SOLAR_MASS.value/cfg['TEST_CASE']['Msun[kg]']
    cosmo = cfg['cosmo']
    cosmo['Omega_c'] = cosmo['Omega_c']*constants_conversion
    cosmo['Omega_b'] = cosmo['Omega_b']*constants_conversion

    assert_allclose(md.get_3d_density(cosmo=cosmo, **cfg['RHO_PARAMS']),
                    cfg['numcosmo_profiles']['rho'], 2.0e-9)
    assert_allclose(md.predict_surface_density(cosmo=cosmo, **cfg['SIGMA_PARAMS']),
                    cfg['numcosmo_profiles']['Sigma'], 2.0e-9)
    assert_allclose(md.predict_excess_surface_density(cosmo=cosmo, **cfg['SIGMA_PARAMS']),
                    cfg['numcosmo_profiles']['DeltaSigma'], 2.0e-9)


def test_angular_diameter_dist_a1a2():
    """ Test function that computes angular diameter distance between
    two scale factors. """
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

    # Validation tests
    cfg = load_validation_config()
    assert_allclose(md.angular_diameter_dist_a1a2(cfg['cosmo'], cfg['TEST_CASE']['aexp_cluster']),
                    cfg['TEST_CASE']['dl'], 1.0e-8)
    assert_allclose(md.angular_diameter_dist_a1a2(cfg['cosmo'], cfg['TEST_CASE']['aexp_source']),
                    cfg['TEST_CASE']['ds'], 1.0e-8)
    assert_allclose(md.angular_diameter_dist_a1a2(cfg['cosmo'], cfg['TEST_CASE']['aexp_source'],
                                                  cfg['TEST_CASE']['aexp_cluster']),
                    cfg['TEST_CASE']['dsl'], 1.0e-8)


def test_get_critical_surface_density():
    """ Validation test for critical surface density """
    cfg = load_validation_config()
    assert_allclose(cfg['SIGMAC_PHYSCONST_CORRECTION']*\
                    md.get_critical_surface_density(cfg['cosmo'],
                                                    z_cluster=cfg['TEST_CASE']['z_cluster'],
                                                    z_source=cfg['TEST_CASE']['z_source']),
                    cfg['TEST_CASE']['nc_Sigmac'], 1.0e-8)


def helper_physics_functions(func):
    """ A helper function to repeat a set of unit tests on several functions
    that expect the same inputs.

    Tests the following functions: predict_tangential_shear, predict_convergence,
                                   predict_reduced_tangential_shear

    Tests that the functions:
    1. Test each default parameter to ensure that the defaults are not changed.
    2. Test that exceptions are thrown for unsupported zsource models and profiles
    """
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
    """ Unit and validation tests for the shear and convergence calculations """
    helper_physics_functions(md.predict_tangential_shear)
    helper_physics_functions(md.predict_convergence)
    helper_physics_functions(md.predict_reduced_tangential_shear)

    # Validation Tests =========================
    # NumCosmo makes different choices for constants (Msun). We make this conversion
    # by passing the ratio of SOLAR_MASS in kg from numcosmo and CLMM
    cfg = load_validation_config()
    constants_conversion = clc.SOLAR_MASS.value/cfg['TEST_CASE']['Msun[kg]']
    
    # First compute SigmaCrit to correct cosmology changes
    cosmo = cfg['cosmo']
    sigma_c = md.get_critical_surface_density(cosmo, cfg['GAMMA_PARAMS']['z_cluster'],
                                              cfg['z_source'])

    # Patch a conversion for cluster_toolkit computations
    cosmo['Omega_c'] = cosmo['Omega_c']*constants_conversion
    cosmo['Omega_b'] = cosmo['Omega_b']*constants_conversion

    # Compute sigma_c in the new cosmology and get a correction factor
    sigcrit_corr = cfg['SIGMAC_PHYSCONST_CORRECTION']
    sigma_c_undo = md.get_critical_surface_density(cosmo, cfg['GAMMA_PARAMS']['z_cluster'],
+                                               cfg['z_source'])
    sigmac_corr = sigma_c_undo/sigma_c/sigcrit_corr

    # Chech error is raised if too small radius   
    assert_raises(ValueError, md.predict_tangential_shear, 1.e-12, 1.e15, 4, 0.2, 0.45, cosmo)

    # Validate tangential shear
    gammat = md.predict_tangential_shear(cosmo=cosmo, **cfg['GAMMA_PARAMS'])
    assert_allclose(gammat*sigmac_corr, cfg['numcosmo_profiles']['gammat'], 1.0e-8)

    # Validate convergence
    kappa = md.predict_convergence(cosmo=cosmo, **cfg['GAMMA_PARAMS'])
    assert_allclose(kappa*sigmac_corr, cfg['numcosmo_profiles']['kappa'], 1.0e-8)

    # Validate reduced tangential shear
    assert_allclose(md.predict_reduced_tangential_shear(cosmo=cosmo, **cfg['GAMMA_PARAMS']),
                    gammat/(1.0 - kappa), 1.0e-10)
    assert_allclose(gammat/(1./sigmac_corr - kappa), cfg['numcosmo_profiles']['gt'], 1.0e-6)
