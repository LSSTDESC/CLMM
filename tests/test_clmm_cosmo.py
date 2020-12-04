"""Tests for clmm_cosmo.py"""
import json
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
import clmm.modeling as md
from clmm.cosmology.parent_class import CLMMCosmology
# ----------- Some Helper Functions for the Validation Tests ---------------


def load_validation_config():
    """ Loads values precomputed by numcosmo for comparison """
    numcosmo_path = 'tests/data/numcosmo/'
    with open(numcosmo_path+'config.json', 'r') as fin:
        testcase = json.load(fin)
    numcosmo_profile = np.genfromtxt(numcosmo_path+'radial_profiles.txt', names=True)
    # Cosmology
    cosmo = md.Cosmology(H0=testcase['cosmo_H0'], Omega_dm0=testcase['cosmo_Om0']-testcase['cosmo_Ob0'], Omega_b0=testcase['cosmo_Ob0'])
    return cosmo, testcase
# --------------------------------------------------------------------------


def test_class(modeling_data):
    """ Unit tests abstract class and unimplemented methdods """
    # Test basic
    assert_raises(TypeError, CLMMCosmology.__getitem__, None, None)
    assert_raises(TypeError, CLMMCosmology.__setitem__, None, None, None)
    # Unimplemented methods
    assert_raises(NotImplementedError, CLMMCosmology._init_from_cosmo, None, None)
    assert_raises(NotImplementedError, CLMMCosmology._init_from_params, None)
    assert_raises(NotImplementedError, CLMMCosmology._set_param, None, None, None)
    assert_raises(NotImplementedError, CLMMCosmology._get_param, None, None)
    assert_raises(AttributeError, CLMMCosmology.set_be_cosmo, None, None)
    assert_raises(NotImplementedError, CLMMCosmology.get_Omega_m, None, None)
    assert_raises(NotImplementedError, CLMMCosmology.eval_da_z1z2, None, None, None)
    assert_raises(AttributeError, CLMMCosmology.eval_da, None, None)
    assert_raises(NotImplementedError, CLMMCosmology.eval_sigma_crit, None, None, None)
TOLERANCE = {'rtol': 1.0e-15}


def test_z_and_a(modeling_data, cosmo_init):
    """ Unit tests abstract class z and a methdods """

    cosmo = md.Cosmology()

    z = np.linspace(0.0, 10.0, 1000)

    assert_raises(ValueError, cosmo._get_a_from_z, z-1.0)

    a = cosmo._get_a_from_z(z)

    assert_raises(ValueError, cosmo._get_z_from_a, a*2.0)

    z_cpy = cosmo._get_z_from_a(a)

    assert_allclose(z_cpy, z, **TOLERANCE)

    a_cpy = cosmo._get_a_from_z(z_cpy)

    assert_allclose(a_cpy, a, **TOLERANCE)

    # Convert from a to z - scalar, list, ndarray
    assert_allclose(cosmo._get_a_from_z(0.5), 2./3., **TOLERANCE)
    assert_allclose(cosmo._get_a_from_z([0.1, 0.2, 0.3, 0.4]),
                    [10./11., 5./6., 10./13., 5./7.], **TOLERANCE)
    assert_allclose(cosmo._get_a_from_z(np.array([0.1, 0.2, 0.3, 0.4])),
                    np.array([10./11., 5./6., 10./13., 5./7.]), **TOLERANCE)

    # Convert from z to a - scalar, list, ndarray
    assert_allclose(cosmo._get_z_from_a(2./3.), 0.5, **TOLERANCE)
    assert_allclose(cosmo._get_z_from_a([10./11., 5./6., 10./13., 5./7.]),
                    [0.1, 0.2, 0.3, 0.4], **TOLERANCE)
    assert_allclose(cosmo._get_z_from_a(np.array([10./11., 5./6., 10./13., 5./7.])),
                    np.array([0.1, 0.2, 0.3, 0.4]), **TOLERANCE)

    # Some potential corner-cases for the two funcs
    assert_allclose(cosmo._get_a_from_z(np.array([0.0, 1300.])),
                    np.array([1.0, 1./1301.]), **TOLERANCE)
    assert_allclose(cosmo._get_z_from_a(np.array([1.0, 1./1301.])),
                    np.array([0.0, 1300.]), **TOLERANCE)

    # Test for exceptions when outside of domains
    assert_raises(ValueError, cosmo._get_a_from_z, -5.0)
    assert_raises(ValueError, cosmo._get_a_from_z, [-5.0, 5.0])
    assert_raises(ValueError, cosmo._get_a_from_z, np.array([-5.0, 5.0]))
    assert_raises(ValueError, cosmo._get_z_from_a, 5.0)
    assert_raises(ValueError, cosmo._get_z_from_a, [-5.0, 5.0])
    assert_raises(ValueError, cosmo._get_z_from_a, np.array([-5.0, 5.0]))

    # Convert from a to z to a (and vice versa)
    testval = 0.5
    assert_allclose(cosmo._get_a_from_z(cosmo._get_z_from_a(testval)), testval, **TOLERANCE)
    assert_allclose(cosmo._get_z_from_a(cosmo._get_a_from_z(testval)), testval, **TOLERANCE)


def test_cosmo_basic(modeling_data, cosmo_init):
    """ Unit tests abstract class z and a methdods """
    cosmo = md.Cosmology(**cosmo_init)
    # Test get_<PAR>(z)
    Omega_m0 = cosmo['Omega_m0']
    assert_allclose(cosmo.get_Omega_m(0.0), Omega_m0, **TOLERANCE)
    assert_allclose(cosmo.get_E2Omega_m(0.0), Omega_m0, **TOLERANCE)
    # Test params
    for param in cosmo_init.keys():
        assert_allclose(cosmo_init[param], cosmo[param], **TOLERANCE)
    # Test for NumCosmo
    if cosmo.backend == 'nc':
        for param in cosmo_init.keys():
            cosmo[param] = cosmo_init[param]*1.01
        assert_raises(ValueError, cosmo._set_param, "nonexistent", 0.0)
    else:
        assert_raises(NotImplementedError, cosmo._set_param, "nonexistent", 0.0)
    # Test missing parameter
    assert_raises(ValueError, cosmo._get_param, "nonexistent")
    # Test da(z) = da12(0, z)
    z = np.linspace(0.0, 10.0, 1000)
    assert_allclose(cosmo.eval_da(z), cosmo.eval_da_z1z2(0.0, z), rtol=8.0e-15)
    assert_allclose(cosmo.eval_da_z1z2(0.0, z), cosmo.eval_da_z1z2(0.0, z), rtol=8.0e-15)
    # Test da(a1, a1)
    cosmo, testcase = load_validation_config()
    assert_allclose(cosmo.eval_da_a1a2(testcase['aexp_cluster']),
                    testcase['dl'], 1.2e-8)
    assert_allclose(cosmo.eval_da_a1a2(testcase['aexp_source']),
                    testcase['ds'], 1.2e-8)
    assert_allclose(cosmo.eval_da_a1a2(testcase['aexp_source'],
                                       testcase['aexp_cluster']),
                    testcase['dsl'], 1.2e-8)

    # Test initializing cosmo
    test_cosmo = md.Cosmology(be_cosmo=cosmo.be_cosmo)


def _rad2mpc_helper(dist, redshift, cosmo, do_inverse):
    """ Helper function to clean up test_convert_rad_to_mpc. Truth is computed using
    astropy so this test is very circular. Once we swap to CCL very soon this will be
    a good source of truth. """
    d_a = cosmo.eval_da(redshift) #Mpc
    if do_inverse:
        assert_allclose(cosmo.mpc2rad(dist, redshift), dist/d_a, **TOLERANCE)
    else:
        assert_allclose(cosmo.rad2mpc(dist, redshift), dist*d_a, **TOLERANCE)


def test_convert_rad_to_mpc():
    """ Test conversion between physical and angular units and vice-versa. """
    # Set some default values if I want them
    redshift = 0.25
    cosmo = md.Cosmology(H0=70.0, Omega_dm0=0.3-0.045, Omega_b0=0.045)
    # Test basic conversions each way
    _rad2mpc_helper(0.003, redshift, cosmo, do_inverse=False)
    _rad2mpc_helper(1.0, redshift, cosmo, do_inverse=True)
    # Convert back and forth and make sure I get the same answer
    midtest = cosmo.rad2mpc(0.003, redshift)
    assert_allclose(cosmo.mpc2rad(midtest, redshift),
                    0.003, **TOLERANCE)
    # Test some different redshifts
    for onez in [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]:
        _rad2mpc_helper(0.33, onez, cosmo, do_inverse=False)
        _rad2mpc_helper(1.0, onez, cosmo, do_inverse=True)
    # Test some different H0
    for oneh0 in [30., 50., 67.3, 74.7, 100.]:
        _rad2mpc_helper(0.33, 0.5, md.Cosmology(H0=oneh0, Omega_dm0=0.3-0.045, Omega_b0=0.045), do_inverse=False)
        _rad2mpc_helper(1.0, 0.5, md.Cosmology(H0=oneh0, Omega_dm0=0.3-0.045, Omega_b0=0.045), do_inverse=True)
    # Test some different Omega_M
    for oneomm in [0.1, 0.3, 0.5, 1.0]:
        _rad2mpc_helper(0.33, 0.5, md.Cosmology(H0=70.0, Omega_dm0=oneomm-0.045, Omega_b0=0.045), do_inverse=False)
        _rad2mpc_helper(1.0, 0.5, md.Cosmology(H0=70.0, Omega_dm0=oneomm-0.045, Omega_b0=0.045), do_inverse=True)
