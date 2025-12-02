"""Tests for clmm_cosmo.py"""
import json
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
import clmm.theory as theo
from clmm.cosmology.parent_class import CLMMCosmology
from clmm.utils.constants import Constants as const

# ----------- Some Helper Functions for the Validation Tests ---------------


def load_validation_config():
    """Loads values precomputed by numcosmo for comparison"""
    numcosmo_path = "tests/data/numcosmo/"
    with open(numcosmo_path + "config.json", "r") as fin:
        testcase = json.load(fin)
    numcosmo_ps = np.genfromtxt(numcosmo_path + "matter_power_spectrum.txt", names=True)
    # Cosmology
    cosmo = theo.Cosmology(
        H0=testcase["cosmo_H0"], Omega_dm0=testcase["cosmo_Odm0"], Omega_b0=testcase["cosmo_Ob0"]
    )

    return cosmo, testcase, numcosmo_ps


# --------------------------------------------------------------------------


def test_class(modeling_data):
    """Unit tests abstract class and unimplemented methdods"""
    # Test basic
    assert_raises(TypeError, CLMMCosmology.__getitem__, None, None)
    assert_raises(TypeError, CLMMCosmology.__setitem__, None, None, None)
    # Unimplemented methods
    assert_raises(NotImplementedError, CLMMCosmology._init_from_cosmo, None, None)
    assert_raises(NotImplementedError, CLMMCosmology._init_from_params, None)
    assert_raises(NotImplementedError, CLMMCosmology._set_param, None, None, None)
    assert_raises(NotImplementedError, CLMMCosmology._get_param, None, None)
    assert_raises(AttributeError, CLMMCosmology.set_be_cosmo, None, None)
    assert_raises(NotImplementedError, CLMMCosmology._get_Omega_m, None, None)
    assert_raises(NotImplementedError, CLMMCosmology._get_rho_c, None, None)
    assert_raises(NotImplementedError, CLMMCosmology._get_E2, None, None)
    assert_raises(NotImplementedError, CLMMCosmology._eval_da_z1z2_core, None, None, None)
    assert_raises(NotImplementedError, CLMMCosmology._eval_sigma_crit_core, None, None, None)
    assert_raises(NotImplementedError, CLMMCosmology._get_E2Omega_m, None, None)
    assert_raises(
        NotImplementedError, CLMMCosmology._eval_linear_matter_powerspectrum, None, None, None
    )


TOLERANCE = {"rtol": 1.0e-12}


def test_z_and_a(modeling_data, cosmo_init):
    """Unit tests abstract class z and a methdods"""

    reltol = modeling_data["cosmo_reltol"]

    cosmo = theo.Cosmology()

    z = np.linspace(0.0, 10.0, 1000)

    assert_raises(ValueError, cosmo.get_a_from_z, z - 1.0)

    a = cosmo.get_a_from_z(z)

    assert_raises(ValueError, cosmo.get_z_from_a, a * 2.0)

    z_cpy = cosmo.get_z_from_a(a)

    assert_allclose(z_cpy, z, **TOLERANCE)

    a_cpy = cosmo.get_a_from_z(z_cpy)

    assert_allclose(a_cpy, a, **TOLERANCE)

    # Convert from a to z - scalar, list, ndarray
    assert_allclose(cosmo.get_a_from_z(0.5), 2.0 / 3.0, **TOLERANCE)
    assert_allclose(
        cosmo.get_a_from_z([0.1, 0.2, 0.3, 0.4]),
        [10.0 / 11.0, 5.0 / 6.0, 10.0 / 13.0, 5.0 / 7.0],
        **TOLERANCE
    )
    assert_allclose(
        cosmo.get_a_from_z(np.array([0.1, 0.2, 0.3, 0.4])),
        np.array([10.0 / 11.0, 5.0 / 6.0, 10.0 / 13.0, 5.0 / 7.0]),
        **TOLERANCE
    )

    # Convert from z to a - scalar, list, ndarray
    assert_allclose(cosmo.get_z_from_a(2.0 / 3.0), 0.5, **TOLERANCE)
    assert_allclose(
        cosmo.get_z_from_a([10.0 / 11.0, 5.0 / 6.0, 10.0 / 13.0, 5.0 / 7.0]),
        [0.1, 0.2, 0.3, 0.4],
        **TOLERANCE
    )
    assert_allclose(
        cosmo.get_z_from_a(np.array([10.0 / 11.0, 5.0 / 6.0, 10.0 / 13.0, 5.0 / 7.0])),
        np.array([0.1, 0.2, 0.3, 0.4]),
        **TOLERANCE
    )

    # Some potential corner-cases for the two funcs
    assert_allclose(
        cosmo.get_a_from_z(np.array([0.0, 1300.0])), np.array([1.0, 1.0 / 1301.0]), **TOLERANCE
    )
    assert_allclose(
        cosmo.get_z_from_a(np.array([1.0, 1.0 / 1301.0])), np.array([0.0, 1300.0]), **TOLERANCE
    )

    # Test for exceptions when outside of domains
    assert_raises(ValueError, cosmo.get_a_from_z, -5.0)
    assert_raises(ValueError, cosmo.get_a_from_z, [-5.0, 5.0])
    assert_raises(ValueError, cosmo.get_a_from_z, np.array([-5.0, 5.0]))
    assert_raises(ValueError, cosmo.get_z_from_a, 5.0)
    assert_raises(ValueError, cosmo.get_z_from_a, [-5.0, 5.0])
    assert_raises(ValueError, cosmo.get_z_from_a, np.array([-5.0, 5.0]))

    # Convert from a to z to a (and vice versa)
    testval = 0.5
    assert_allclose(cosmo.get_a_from_z(cosmo.get_z_from_a(testval)), testval, **TOLERANCE)
    assert_allclose(cosmo.get_z_from_a(cosmo.get_a_from_z(testval)), testval, **TOLERANCE)


def test_cosmo_basic(modeling_data, cosmo_init):
    """Unit tests abstract class z and a methdods"""
    reltol = modeling_data["cosmo_reltol"]
    cosmo = theo.Cosmology(**cosmo_init)
    # Test get_<PAR>(z)
    Omega_m0 = cosmo["Omega_m0"]
    assert_allclose(cosmo.get_Omega_m(0.0), Omega_m0, **TOLERANCE)
    assert_allclose(cosmo.get_E2Omega_m(0.0), Omega_m0, **TOLERANCE)
    assert_allclose(cosmo.get_E2Omega_m(0.0) / cosmo.get_E2(0.0), Omega_m0, **TOLERANCE)
    # Test getting all parameters
    for param in ("Omega_m0", "Omega_b0", "Omega_dm0", "Omega_k0", "h", "H0"):
        cosmo[param]
    # Test params values
    for param in cosmo_init.keys():
        assert_allclose(cosmo_init[param], cosmo[param], **TOLERANCE)
    # Test for NumCosmo
    if cosmo.backend == "nc":
        for param in ("Omega_b0", "Omega_dm0", "Omega_k0", "h", "H0"):
            cosmo[param] *= 1.01
        assert_raises(ValueError, cosmo._set_param, "nonexistent", 0.0)
        # Initializing a cosmology from a dist argument
        theo.Cosmology(dist=cosmo.dist)
    else:
        assert_raises(NotImplementedError, cosmo._set_param, "nonexistent", 0.0)
    # Test missing parameter
    assert_raises(ValueError, cosmo._get_param, "nonexistent")
    # Test da(z) = da12(0, z)
    z = np.linspace(0.0, 10.0, 1000)
    assert_allclose(cosmo.eval_da(z), cosmo.eval_da_z1z2(0.0, z), rtol=8.0e-15)
    assert_allclose(cosmo.eval_da_z1z2(0.0, z), cosmo.eval_da_z1z2(0.0, z), rtol=8.0e-15)
    # Test da(a1, a1)
    cosmo, testcase, _ = load_validation_config()
    assert_allclose(cosmo.eval_da_a1a2(testcase["aexp_cluster"]), testcase["dl"], reltol)
    assert_allclose(cosmo.eval_da_a1a2(testcase["aexp_source"]), testcase["ds"], reltol)
    assert_allclose(
        cosmo.eval_da_a1a2(testcase["aexp_source"], testcase["aexp_cluster"]),
        testcase["dsl"],
        reltol,
    )
    assert_allclose(
        cosmo.eval_da_a1a2(testcase["aexp_source"], [testcase["aexp_cluster"]] * 5),
        [testcase["dsl"]] * 5,
        reltol,
    )

    # Test initializing cosmo
    theo.Cosmology(be_cosmo=cosmo.be_cosmo)

    # Test get rho matter
    rhocrit_mks = 3.0 * 100.0 * 100.0 / (8.0 * np.pi * const.GNEWT.value)
    rhocrit_cd2018 = (
        rhocrit_mks * 1000.0 * 1000.0 * const.PC_TO_METER.value * 1.0e6 / const.SOLAR_MASS.value
    )
    for z in np.linspace(0.0, 2.0, 5):
        assert_allclose(
            cosmo.get_rho_m(z),
            rhocrit_cd2018 * (z + 1) ** 3 * cosmo["Omega_m0"] * cosmo["h"] ** 2,
            rtol=1e-5,
        )
        assert_allclose(cosmo.get_rho_c(z), cosmo.get_rho_m(z) / cosmo.get_Omega_m(z), rtol=1e-5)

    # Test pk - just consistency! A better test must be implemented
    if cosmo.backend in ("ccl", "nc"):
        k = np.logspace(-2, 1, 20)
        assert_allclose(
            cosmo.eval_linear_matter_powerspectrum(k, 0.1),
            cosmo.eval_linear_matter_powerspectrum(k, 0.1),
            rtol=1e-5,
        )


def test_matter_power_spectrum(modeling_data):
    cosmo_ps, testcase, ps = load_validation_config()
    if cosmo_ps.backend in ("ccl", "nc"):
        reltol = modeling_data["ps_reltol"]
        kvals = ps["k"]
        assert_allclose(
            cosmo_ps.eval_linear_matter_powerspectrum(kvals, testcase["z_cluster"]),
            ps["P_of_k"],
            reltol,
        )


def _rad2mpc_helper(dist, redshift, cosmo, do_inverse):
    """Helper function to clean up test_convert_rad_to_mpc. Truth is computed using
    astropy so this test is very circular. Once we swap to CCL very soon this will be
    a good source of truth."""
    d_a = cosmo.eval_da(redshift)  # Mpc
    if do_inverse:
        assert_allclose(cosmo.mpc2rad(dist, redshift), dist / d_a, **TOLERANCE)
    else:
        assert_allclose(cosmo.rad2mpc(dist, redshift), dist * d_a, **TOLERANCE)


def test_convert_rad_to_mpc():
    """Test conversion between physical and angular units and vice-versa."""
    # Set some default values if I want them
    redshift = 0.25
    cosmo = theo.Cosmology(H0=70.0, Omega_dm0=0.3 - 0.045, Omega_b0=0.045)
    # Test basic conversions each way
    _rad2mpc_helper(0.003, redshift, cosmo, do_inverse=False)
    _rad2mpc_helper(1.0, redshift, cosmo, do_inverse=True)
    # Convert back and forth and make sure I get the same answer
    midtest = cosmo.rad2mpc(0.003, redshift)
    assert_allclose(cosmo.mpc2rad(midtest, redshift), 0.003, **TOLERANCE)
    # Test some different redshifts
    for onez in [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]:
        _rad2mpc_helper(0.33, onez, cosmo, do_inverse=False)
        _rad2mpc_helper(1.0, onez, cosmo, do_inverse=True)
    # Test some different H0
    for oneh0 in [30.0, 50.0, 67.3, 74.7, 100.0]:
        _rad2mpc_helper(
            0.33,
            0.5,
            theo.Cosmology(H0=oneh0, Omega_dm0=0.3 - 0.045, Omega_b0=0.045),
            do_inverse=False,
        )
        _rad2mpc_helper(
            1.0,
            0.5,
            theo.Cosmology(H0=oneh0, Omega_dm0=0.3 - 0.045, Omega_b0=0.045),
            do_inverse=True,
        )
    # Test some different Omega_M
    for oneomm in [0.1, 0.3, 0.5, 1.0]:
        _rad2mpc_helper(
            0.33,
            0.5,
            theo.Cosmology(H0=70.0, Omega_dm0=oneomm - 0.045, Omega_b0=0.045),
            do_inverse=False,
        )
        _rad2mpc_helper(
            1.0,
            0.5,
            theo.Cosmology(H0=70.0, Omega_dm0=oneomm - 0.045, Omega_b0=0.045),
            do_inverse=True,
        )


def test_eval_sigma_crit(modeling_data):
    """Validation test for critical surface density"""

    reltol = modeling_data["cosmo_reltol"]

    cosmo, testcase, _ = load_validation_config()

    assert_allclose(
        cosmo.eval_sigma_crit(testcase["z_cluster"], testcase["z_src"]),
        testcase["nc_Sigmac"],
        reltol,
    )
    # Check errors for z<0
    assert_raises(ValueError, cosmo.eval_sigma_crit, -0.2, 0.3)
    assert_raises(ValueError, cosmo.eval_sigma_crit, 0.2, -0.3)
    # Check behaviour when sources are in front of the lens
    z_cluster = 0.3
    z_src = 0.2
    assert_allclose(cosmo.eval_sigma_crit(z_cluster, z_src), np.inf, 1.0e-10)
    z_src = [0.2, 0.12, 0.25]
    assert_allclose(cosmo.eval_sigma_crit(z_cluster, z_src), [np.inf, np.inf, np.inf], 1.0e-10)
