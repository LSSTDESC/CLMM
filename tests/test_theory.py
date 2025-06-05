"""Tests for theory/"""

import json
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
from astropy.cosmology import FlatLambdaCDM, LambdaCDM
import clmm.theory as theo
from clmm.utils.constants import Constants as clc
from clmm.galaxycluster import GalaxyCluster
from clmm import GCData
from clmm.utils import (
    compute_beta_s_square_mean_from_distribution,
    compute_beta_s_mean_from_distribution,
    compute_beta_s_func,
    redshift_distributions as zdist,
)

TOLERANCE = {"rtol": 1.0e-8}

# ----------- Some Helper Functions for the Validation Tests ---------------


def compute_sigmac_physical_constant(lightspeed, gnewt, msun, pc_to_m):
    """Computes physical constant used to in Sigma_crit

    Parameters
    ----------
    lightspeed: float
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
    return (lightspeed * 1000.0 / pc_to_m) ** 2 / (gnewt * msun / pc_to_m**3)


def load_validation_config(halo_profile_model=None):
    """Loads values precomputed by numcosmo for comparison"""
    numcosmo_path = "tests/data/numcosmo/"

    if halo_profile_model == "einasto":
        with open(numcosmo_path + "config_einasto_benchmarks.json", "r") as fin:
            testcase = json.load(fin)
        numcosmo_profile = np.genfromtxt(numcosmo_path + "radial_profiles_einasto.txt", names=True)
    elif halo_profile_model == "hernquist":
        with open(numcosmo_path + "config_hernquist_benchmarks.json", "r") as fin:
            testcase = json.load(fin)
        numcosmo_profile = np.genfromtxt(
            numcosmo_path + "radial_profiles_hernquist.txt", names=True
        )
    else:
        # defaults to nfw profile
        with open(numcosmo_path + "config.json", "r") as fin:
            testcase = json.load(fin)
        numcosmo_profile = np.genfromtxt(numcosmo_path + "radial_profiles.txt", names=True)

    # Physical Constants
    CLMM_SIGMAC_PCST = compute_sigmac_physical_constant(
        clc.CLIGHT_KMS.value, clc.GNEWT.value, clc.SOLAR_MASS.value, clc.PC_TO_METER.value
    )
    testcase_SIGMAC_PCST = compute_sigmac_physical_constant(
        testcase["lightspeed[km/s]"],
        testcase["G[m3/km.s2]"],
        testcase["Msun[kg]"],
        testcase["pc_to_m"],
    )

    # Cosmology
    cosmo = theo.Cosmology(
        H0=testcase["cosmo_H0"], Omega_dm0=testcase["cosmo_Odm0"], Omega_b0=testcase["cosmo_Ob0"]
    )

    # Sets of parameters to be used by multiple functions
    RHO_PARAMS = {
        "r3d": np.array(numcosmo_profile["r3d"]),
        "mdelta": testcase["cluster_mass"],
        "cdelta": testcase["cluster_concentration"],
        "z_cl": testcase["z_cluster"],
        "halo_profile_model": testcase["density_profile_parametrization"],
    }
    SIGMA_PARAMS = {
        "r_proj": np.array(numcosmo_profile["r3d"]),
        "mdelta": testcase["cluster_mass"],
        "cdelta": testcase["cluster_concentration"],
        "z_cl": testcase["z_cluster"],
        "delta_mdef": testcase["mass_Delta"],
        "halo_profile_model": testcase["density_profile_parametrization"],
    }
    GAMMA_PARAMS = {
        "r_proj": np.array(numcosmo_profile["r3d"]),
        "mdelta": testcase["cluster_mass"],
        "cdelta": testcase["cluster_concentration"],
        "z_cluster": testcase["z_cluster"],
        "z_src": testcase["z_src"],
        "delta_mdef": testcase["mass_Delta"],
        "halo_profile_model": testcase["density_profile_parametrization"],
        "z_src_info": "discrete",
    }

    return {
        "TEST_CASE": testcase,
        "z_src": testcase["z_src"],
        "cosmo": cosmo,
        "cosmo_pars": {k.replace("cosmo_", ""): v for k, v in testcase.items() if "cosmo_" in k},
        "RHO_PARAMS": RHO_PARAMS,
        "SIGMA_PARAMS": SIGMA_PARAMS,
        "GAMMA_PARAMS": GAMMA_PARAMS,
        "numcosmo_profiles": numcosmo_profile,
        "TEST_CASE_SIGMAC_PCST": testcase_SIGMAC_PCST,
        "CLMM_SIGMAC_PCST": CLMM_SIGMAC_PCST,
    }


# --------------------------------------------------------------------------


def test_physical_constants(modeling_data):
    """Test physical values of physical_constants

    Notes
    -----
        The precision set for these tests put in here right now are somewhat arbitrary,
        has to be improved to values provided by CCL
    """
    cfg = load_validation_config()
    assert_allclose(cfg["TEST_CASE"]["lightspeed[km/s]"], clc.CLIGHT_KMS.value, 1e-3)
    assert_allclose(cfg["TEST_CASE"]["G[m3/km.s2]"], clc.GNEWT.value, 1e-3)
    assert_allclose(cfg["TEST_CASE"]["pc_to_m"], clc.PC_TO_METER.value, 1e-6)
    assert_allclose(cfg["TEST_CASE"]["Msun[kg]"], clc.SOLAR_MASS.value, 1e-2)
    assert_allclose(cfg["TEST_CASE_SIGMAC_PCST"], cfg["CLMM_SIGMAC_PCST"], 1e-2)


def test_cclify_astropy_cosmo(modeling_data):
    """Unit tests for theo.cllify_astropy_cosmo"""
    # Make some base objects
    truth = {"H0": 70.0, "Om0": 0.3, "Ob0": 0.05}
    apycosmo_flcdm = FlatLambdaCDM(**truth)
    apycosmo_lcdm = LambdaCDM(Ode0=1.0 - truth["Om0"], **truth)
    cclcosmo = {
        "Omega_c": truth["Om0"] - truth["Ob0"],
        "Omega_b": truth["Ob0"],
        "h": truth["H0"] / 100.0,
        "H0": truth["H0"],
    }

    # Test for exception if missing baryon density (everything else required)
    # missbaryons = FlatLambdaCDM(H0=truth['H0'], Om0=truth['Om0'])
    # assert_raises(KeyError, theo.cclify_astropy_cosmo, missbaryons)

    # Test output if we pass FlatLambdaCDM and LambdaCDM objects
    # assert_equal(theo.cclify_astropy_cosmo(apycosmo_flcdm), cclcosmo)
    # assert_equal(theo.cclify_astropy_cosmo(apycosmo_lcdm), cclcosmo)

    # Test output if we pass a CCL object (a dict right now)
    # assert_equal(theo.cclify_astropy_cosmo(cclcosmo), cclcosmo)

    # Test for exception if anything else is passed in
    # assert_raises(TypeError, theo.cclify_astropy_cosmo, 70.)
    # assert_raises(TypeError, theo.cclify_astropy_cosmo, [70., 0.3, 0.25, 0.05])


def test_astropyify_ccl_cosmo(modeling_data):
    """Unit tests for astropyify_ccl_cosmo"""
    # Make a bse object
    truth = {"H0": 70.0, "Om0": 0.3, "Ob0": 0.05}
    apycosmo_flcdm = FlatLambdaCDM(**truth)
    apycosmo_lcdm = LambdaCDM(Ode0=1.0 - truth["Om0"], **truth)
    cclcosmo = {
        "Omega_c": truth["Om0"] - truth["Ob0"],
        "Omega_b": truth["Ob0"],
        "h": truth["H0"] / 100.0,
        "H0": truth["H0"],
    }

    # Test output if we pass FlatLambdaCDM and LambdaCDM objects
    # assert_equal(theo.astropyify_ccl_cosmo(apycosmo_flcdm), apycosmo_flcdm)
    # assert_equal(theo.astropyify_ccl_cosmo(apycosmo_lcdm), apycosmo_lcdm)

    # Test output if we pass a CCL object, compare the dicts
    # assert_equal(theo.cclify_astropy_cosmo(theo.astropyify_ccl_cosmo(cclcosmo)),
    #             theo.cclify_astropy_cosmo(apycosmo_lcdm))

    # Test for exception if anything else is passed in
    # assert_raises(TypeError, theo.astropyify_ccl_cosmo, 70.)
    # assert_raises(TypeError, theo.astropyify_ccl_cosmo, [70., 0.3, 0.25, 0.05])


def test_compute_reduced_shear(modeling_data):
    """Unit tests for compute_reduced_shear_from_convergence"""
    # Make some base objects
    shear = [0.5, 0.75, 1.25, 0.0]
    convergence = [0.75, -0.2, 0.0, 2.3]
    truth = [2.0, 0.625, 1.25, 0.0]

    # Test for exception if shear and convergence are not the same length
    assert_raises(
        ValueError, theo.compute_reduced_shear_from_convergence, shear[:3], convergence[:2]
    )
    assert_raises(
        ValueError, theo.compute_reduced_shear_from_convergence, shear[:2], convergence[:3]
    )

    # Check output including: float, list, ndarray
    assert_allclose(
        theo.compute_reduced_shear_from_convergence(shear[0], convergence[0]), truth[0], **TOLERANCE
    )
    assert_allclose(
        theo.compute_reduced_shear_from_convergence(shear, convergence), truth, **TOLERANCE
    )
    assert_allclose(
        theo.compute_reduced_shear_from_convergence(np.array(shear), np.array(convergence)),
        np.array(truth),
        **TOLERANCE,
    )


def helper_profiles(func):
    """A helper function to repeat a set of unit tests on several functions
    that expect the same inputs.

    Tests the following functions: compute_3d_density, compute_surface_density,
                                   compute_excess_surface_density

    Tests that the functions:
    1. Throw an error if an invalid profile model is passed
    2. Test each default parameter to ensure that the defaults are not changed.
    """
    # Make some base objects
    r3d = np.logspace(-2, 2, 100)
    mdelta = 1.0e15
    cdelta = 4.0
    z_cl = 0.2
    cclcosmo = theo.Cosmology(Omega_dm0=0.25, Omega_b0=0.05)

    # Fail vals
    assert_raises(ValueError, func, r3d, 0, cdelta, z_cl, cclcosmo, 200)
    assert_raises(ValueError, func, r3d, mdelta, 0, z_cl, cclcosmo, 200)
    # r<0
    assert_raises(ValueError, func, -1, mdelta, cdelta, z_cl, cclcosmo, 200)
    # r=0
    assert_raises(ValueError, func, 0, mdelta, cdelta, z_cl, cclcosmo, 200)
    # other profiles models are passed
    assert_raises(ValueError, func, r3d, mdelta, cdelta, z_cl, cclcosmo, 200, "bleh")

    # Test defaults

    defaulttruth = func(r3d, mdelta, cdelta, z_cl, cclcosmo)
    assert_allclose(
        func(r3d, mdelta, cdelta, z_cl, cclcosmo, delta_mdef=200), defaulttruth, **TOLERANCE
    )
    assert_allclose(
        func(r3d, mdelta, cdelta, z_cl, cclcosmo, halo_profile_model="nfw"),
        defaulttruth,
        **TOLERANCE,
    )
    assert_allclose(
        func(r3d, mdelta, cdelta, z_cl, cclcosmo, massdef="mean"), defaulttruth, **TOLERANCE
    )
    # Test case fix
    assert_allclose(
        func(r3d, mdelta, cdelta, z_cl, cclcosmo, halo_profile_model="NFW"),
        defaulttruth,
        **TOLERANCE,
    )
    assert_allclose(
        func(r3d, mdelta, cdelta, z_cl, cclcosmo, massdef="MEAN"), defaulttruth, **TOLERANCE
    )


def test_profiles(modeling_data, profile_init):
    """Tests for profile functions, compute_3d_density, compute_surface_density,
    and compute_excess_surface_density"""

    # Validation tests
    # NumCosmo makes different choices for constants (Msun). We make this conversion
    # by passing the ratio of SOLAR_MASS in kg from numcosmo and CLMM
    cfg = load_validation_config(halo_profile_model=profile_init)
    cosmo = cfg["cosmo"]

    if (profile_init == "nfw" or theo.be_nick in ["nc", "ccl"]) and modeling_data["nick"] not in [
        "notabackend",
        "testnotabackend",
    ]:
        helper_profiles(theo.compute_3d_density)
        helper_profiles(theo.compute_surface_density)
        helper_profiles(theo.compute_mean_surface_density)
        helper_profiles(theo.compute_excess_surface_density)

        if profile_init == "nfw":
            reltol = modeling_data["theory_reltol"]
        else:
            reltol = modeling_data["theory_reltol_num"]

        # Object Oriented tests
        mod = theo.Modeling()
        mod.set_cosmo(cosmo)
        mod.set_halo_density_profile(halo_profile_model=cfg["SIGMA_PARAMS"]["halo_profile_model"])
        mod.set_concentration(cfg["SIGMA_PARAMS"]["cdelta"])
        mod.set_mass(cfg["SIGMA_PARAMS"]["mdelta"])
        assert_allclose(mod.cdelta, cfg["SIGMA_PARAMS"]["cdelta"], 1e-14)
        assert_allclose(mod.mdelta, cfg["SIGMA_PARAMS"]["mdelta"], 1e-14)
        # Need to set the alpha value for the NC backend to the one used for the benchmarks,
        # which is the CCL default value
        if profile_init == "einasto":
            alpha_ein = cfg["TEST_CASE"]["alpha_einasto"]
            mod.set_einasto_alpha(alpha_ein)

        if profile_init != "einasto":
            alpha_ein = None

        assert_allclose(
            mod.eval_3d_density(cfg["RHO_PARAMS"]["r3d"], cfg["RHO_PARAMS"]["z_cl"], verbose=True),
            cfg["numcosmo_profiles"]["rho"],
            reltol,
        )
        assert_allclose(
            mod.eval_surface_density(
                cfg["SIGMA_PARAMS"]["r_proj"], cfg["SIGMA_PARAMS"]["z_cl"], verbose=True
            ),
            cfg["numcosmo_profiles"]["Sigma"],
            reltol,
        )
        assert_allclose(
            mod.eval_mean_surface_density(
                cfg["SIGMA_PARAMS"]["r_proj"], cfg["SIGMA_PARAMS"]["z_cl"], verbose=True
            ),
            cfg["numcosmo_profiles"]["Sigma"] + cfg["numcosmo_profiles"]["DeltaSigma"],
            reltol,
        )
        assert_allclose(
            mod.eval_excess_surface_density(
                cfg["SIGMA_PARAMS"]["r_proj"], cfg["SIGMA_PARAMS"]["z_cl"], verbose=True
            ),
            cfg["numcosmo_profiles"]["DeltaSigma"],
            reltol,
        )
        # Test miscentering
        if mod.backend == "nc":
            assert_allclose(
                mod.eval_surface_density(
                    cfg["SIGMA_PARAMS"]["r_proj"],
                    cfg["SIGMA_PARAMS"]["z_cl"],
                    r_mis=0.1,
                    verbose=True,
                )[-40:],
                cfg["numcosmo_profiles"]["Sigma"][-40:],
                2.5e-2,
            )
            assert_allclose(
                mod.eval_surface_density(
                    cfg["SIGMA_PARAMS"]["r_proj"],
                    cfg["SIGMA_PARAMS"]["z_cl"],
                    r_mis=0.1,
                    verbose=True,
                    mis_from_backend=True,
                )[-40:],
                cfg["numcosmo_profiles"]["Sigma"][-40:],
                2.5e-2,
            )
            assert_allclose(
                mod.eval_surface_density(
                    cfg["SIGMA_PARAMS"]["r_proj"][-40],
                    cfg["SIGMA_PARAMS"]["z_cl"],
                    r_mis=0.1,
                    verbose=True,
                    mis_from_backend=True,
                ),
                cfg["numcosmo_profiles"]["Sigma"][-40],
                2.5e-2,
            )
            assert_allclose(
                mod.eval_mean_surface_density(
                    cfg["SIGMA_PARAMS"]["r_proj"],
                    cfg["SIGMA_PARAMS"]["z_cl"],
                    r_mis=0.1,
                    verbose=True,
                )[-40:],
                (cfg["numcosmo_profiles"]["Sigma"] + cfg["numcosmo_profiles"]["DeltaSigma"])[-40:],
                8.5e-3,
            )
            assert_allclose(
                mod.eval_mean_surface_density(
                    cfg["SIGMA_PARAMS"]["r_proj"],
                    cfg["SIGMA_PARAMS"]["z_cl"],
                    r_mis=0.1,
                    verbose=True,
                    mis_from_backend=True,
                )[-40:],
                (cfg["numcosmo_profiles"]["Sigma"] + cfg["numcosmo_profiles"]["DeltaSigma"])[-40:],
                8.5e-3,
            )
            assert_allclose(
                mod.eval_mean_surface_density(
                    cfg["SIGMA_PARAMS"]["r_proj"][-40],
                    cfg["SIGMA_PARAMS"]["z_cl"],
                    r_mis=0.1,
                    verbose=True,
                    mis_from_backend=True,
                ),
                (cfg["numcosmo_profiles"]["Sigma"] + cfg["numcosmo_profiles"]["DeltaSigma"])[-40],
                8.5e-3,
            )
            assert_allclose(
                mod.eval_excess_surface_density(
                    cfg["SIGMA_PARAMS"]["r_proj"],
                    cfg["SIGMA_PARAMS"]["z_cl"],
                    r_mis=0.1,
                    verbose=True,
                )[-40:],
                cfg["numcosmo_profiles"]["DeltaSigma"][-40:],
                2.5e-2,
            )
            assert_allclose(
                mod.eval_excess_surface_density(
                    cfg["SIGMA_PARAMS"]["r_proj"],
                    cfg["SIGMA_PARAMS"]["z_cl"],
                    r_mis=0.1,
                    verbose=True,
                    mis_from_backend=False,
                )[-40:],
                cfg["numcosmo_profiles"]["DeltaSigma"][-40:],
                2.5e-2,
            )
        assert_equal(theo.miscentering.integrand_surface_density_nfw(0, 0.3, 0, 0.3), 1.0 / 3.0)
        assert_equal(
            theo.miscentering.integrand_surface_density_hernquist(0, 0.3, 0, 0.3), 4.0 / 15.0
        )
        if mod.backend == "ct":
            assert_raises(
                ValueError, mod.eval_excess_surface_density, 1e-12, cfg["SIGMA_PARAMS"]["z_cl"]
            )

        # Functional interface tests
        # alpha_ein is None unless testing Einasto with the NC and CCL backend
        assert_allclose(
            theo.compute_3d_density(
                cosmo=cosmo, **cfg["RHO_PARAMS"], alpha_ein=alpha_ein, verbose=True
            ),
            cfg["numcosmo_profiles"]["rho"],
            reltol,
        )
        assert_allclose(
            theo.compute_surface_density(
                cosmo=cosmo, **cfg["SIGMA_PARAMS"], alpha_ein=alpha_ein, verbose=True
            ),
            cfg["numcosmo_profiles"]["Sigma"],
            reltol,
        )
        assert_allclose(
            theo.compute_mean_surface_density(
                cosmo=cosmo, **cfg["SIGMA_PARAMS"], alpha_ein=alpha_ein, verbose=True
            ),
            cfg["numcosmo_profiles"]["Sigma"] + cfg["numcosmo_profiles"]["DeltaSigma"],
            reltol,
        )
        assert_allclose(
            theo.compute_excess_surface_density(
                cosmo=cosmo, **cfg["SIGMA_PARAMS"], alpha_ein=alpha_ein, verbose=True
            ),
            cfg["numcosmo_profiles"]["DeltaSigma"],
            reltol,
        )

        # Test miscentering
        if mod.backend == "nc":
            assert_allclose(
                theo.compute_surface_density(
                    cosmo=cosmo, **cfg["SIGMA_PARAMS"], alpha_ein=alpha_ein, verbose=True, r_mis=0.1
                )[-40:],
                cfg["numcosmo_profiles"]["Sigma"][-40:],
                2.5e-2,
            )
            assert_allclose(
                theo.compute_surface_density(
                    cosmo=cosmo,
                    **cfg["SIGMA_PARAMS"],
                    alpha_ein=alpha_ein,
                    verbose=True,
                    r_mis=0.1,
                    mis_from_backend=True,
                )[-40:],
                cfg["numcosmo_profiles"]["Sigma"][-40:],
                2.5e-2,
            )
            assert_allclose(
                theo.compute_mean_surface_density(
                    cosmo=cosmo, **cfg["SIGMA_PARAMS"], alpha_ein=alpha_ein, verbose=True, r_mis=0.1
                )[-40:],
                (cfg["numcosmo_profiles"]["Sigma"] + cfg["numcosmo_profiles"]["DeltaSigma"])[-40:],
                8.5e-3,
            )
            assert_allclose(
                theo.compute_mean_surface_density(
                    cosmo=cosmo,
                    **cfg["SIGMA_PARAMS"],
                    alpha_ein=alpha_ein,
                    verbose=True,
                    r_mis=0.1,
                    mis_from_backend=True,
                )[-40:],
                (cfg["numcosmo_profiles"]["Sigma"] + cfg["numcosmo_profiles"]["DeltaSigma"])[-40:],
                8.5e-3,
            )
            assert_allclose(
                theo.compute_excess_surface_density(
                    cosmo=cosmo, **cfg["SIGMA_PARAMS"], alpha_ein=alpha_ein, verbose=True, r_mis=0.1
                )[-40:],
                cfg["numcosmo_profiles"]["DeltaSigma"][-40:],
                2.5e-2,
            )
            assert_allclose(
                theo.compute_excess_surface_density(
                    cosmo=cosmo,
                    **cfg["SIGMA_PARAMS"],
                    alpha_ein=alpha_ein,
                    verbose=True,
                    r_mis=0.1,
                    mis_from_backend=True,
                )[-40:],
                cfg["numcosmo_profiles"]["DeltaSigma"][-40:],
                2.5e-2,
            )

        # Test use_projected_quad
        if mod.backend == "ccl" and profile_init == "einasto":
            if hasattr(mod.hdpm, "projected_quad"):
                mod.set_projected_quad(True)
                assert_allclose(
                    mod.eval_surface_density(
                        cfg["SIGMA_PARAMS"]["r_proj"], cfg["SIGMA_PARAMS"]["z_cl"], verbose=True
                    ),
                    cfg["numcosmo_profiles"]["Sigma"],
                    reltol * 1e-1,
                )
                assert_allclose(
                    theo.compute_surface_density(
                        cosmo=cosmo,
                        **cfg["SIGMA_PARAMS"],
                        alpha_ein=alpha_ein,
                        verbose=True,
                        use_projected_quad=True,
                    ),
                    cfg["numcosmo_profiles"]["Sigma"],
                    reltol * 1e-1,
                )

                delattr(mod.hdpm, "projected_quad")
                assert_raises(NotImplementedError, mod.set_projected_quad, True)


def test_2halo_term(modeling_data):
    cfg = load_validation_config()
    cosmo = cfg["cosmo"]

    # Object Oriented tests
    mod = theo.Modeling()
    mod.set_cosmo(cosmo)

    if mod.backend not in ["ccl", "nc"]:
        assert_raises(
            NotImplementedError, mod.eval_surface_density_2h, 1.0, cfg["SIGMA_PARAMS"]["z_cl"]
        )
        assert_raises(
            NotImplementedError,
            mod.eval_excess_surface_density_2h,
            1.0,
            cfg["SIGMA_PARAMS"]["z_cl"],
        )
    else:
        # Just checking that it runs and returns array of the right length
        # To be updated with proper comparison to benchmark when available
        assert_equal(
            len(
                mod.eval_surface_density_2h(
                    cfg["SIGMA_PARAMS"]["r_proj"], cfg["SIGMA_PARAMS"]["z_cl"]
                )
            ),
            len(cfg["SIGMA_PARAMS"]["r_proj"]),
        )
        assert_equal(
            len(
                mod.eval_excess_surface_density_2h(
                    cfg["SIGMA_PARAMS"]["r_proj"], cfg["SIGMA_PARAMS"]["z_cl"]
                )
            ),
            len(cfg["SIGMA_PARAMS"]["r_proj"]),
        )

        # Checks that OO-oriented and functional interface give the same results
        assert_allclose(
            theo.compute_excess_surface_density_2h(
                cfg["SIGMA_PARAMS"]["r_proj"], cfg["SIGMA_PARAMS"]["z_cl"], cosmo
            ),
            mod.eval_excess_surface_density_2h(
                cfg["SIGMA_PARAMS"]["r_proj"], cfg["SIGMA_PARAMS"]["z_cl"]
            ),
            1.0e-10,
        )

        assert_allclose(
            theo.compute_surface_density_2h(
                cfg["SIGMA_PARAMS"]["r_proj"], cfg["SIGMA_PARAMS"]["z_cl"], cosmo
            ),
            mod.eval_surface_density_2h(cfg["SIGMA_PARAMS"]["r_proj"], cfg["SIGMA_PARAMS"]["z_cl"]),
            1.0e-10,
        )


def helper_physics_functions(func, additional_kwargs={}):
    """A helper function to repeat a set of unit tests on several functions
    that expect the same inputs.

    Tests the following functions: compute_tangential_shear, compute_convergence,
                                   compute_reduced_tangential_shear,
                                   compute_magnification, compute_magnification_bias

    Tests that the functions:
    1. Test each default parameter to ensure that the defaults are not changed.
    2. Test that exceptions are thrown for unsupported zsource models and profiles
    """
    # Make some base objects
    kwargs = {
        "r_proj": np.logspace(-2, 2, 100),
        "mdelta": 1.0e15,
        "cdelta": 4.0,
        "z_cluster": 0.2,
        "z_src": 0.45,
        "cosmo": theo.Cosmology(Omega_dm0=0.25, Omega_b0=0.05, H0=70.0),
    }
    kwargs.update(additional_kwargs)

    # Test defaults
    defaulttruth = func(**kwargs, delta_mdef=200, halo_profile_model="nfw", z_src_info="discrete")
    assert_allclose(func(**kwargs, delta_mdef=200), defaulttruth, **TOLERANCE)
    assert_allclose(func(**kwargs, halo_profile_model="nfw"), defaulttruth, **TOLERANCE)
    assert_allclose(func(**kwargs, z_src_info="discrete"), defaulttruth, **TOLERANCE)

    # Test for exception on unsupported z_src_info and halo profiles
    assert_raises(ValueError, func, **kwargs, halo_profile_model="blah")
    assert_raises(ValueError, func, **kwargs, massdef="blah")
    assert_raises(ValueError, func, **kwargs, z_src_info="blah")


def test_shear_convergence_unittests(modeling_data, profile_init):
    """Unit and validation tests for the shear and convergence calculations"""
    helper_physics_functions(theo.compute_tangential_shear)
    helper_physics_functions(theo.compute_convergence)
    helper_physics_functions(theo.compute_reduced_tangential_shear)
    helper_physics_functions(theo.compute_magnification)
    helper_physics_functions(theo.compute_magnification_bias, {"alpha": 1.0})

    # Validation Tests -------------------------
    # NumCosmo makes different choices for constants (Msun). We make this conversion
    # by passing the ratio of SOLAR_MASS in kg from numcosmo and CLMM
    cfg = load_validation_config(halo_profile_model=profile_init)
    cosmo = cfg["cosmo"]

    if profile_init == "nfw" or modeling_data["nick"] in ["nc", "ccl"]:
        if profile_init == "nfw":
            reltol = modeling_data["theory_reltol"]
        else:
            reltol = modeling_data["theory_reltol_num"]

        # Chech error is raised if too small radius
        assert_raises(
            ValueError, theo.compute_tangential_shear, 1.0e-12, 1.0e15, 4, 0.2, 0.45, cosmo
        )

        if profile_init == "einasto":
            cfg["GAMMA_PARAMS"]["alpha_ein"] = cfg["TEST_CASE"]["alpha_einasto"]

        # Validate tangential shear - discrete case
        gammat = theo.compute_tangential_shear(cosmo=cosmo, **cfg["GAMMA_PARAMS"])
        assert_allclose(gammat, cfg["numcosmo_profiles"]["gammat"], reltol)

        # Validate convergence - discrete case
        kappa = theo.compute_convergence(cosmo=cosmo, **cfg["GAMMA_PARAMS"])
        assert_allclose(kappa, cfg["numcosmo_profiles"]["kappa"], reltol)

        # Validate reduced tangential shear - discrete case
        gt = theo.compute_reduced_tangential_shear(cosmo=cosmo, **cfg["GAMMA_PARAMS"])
        assert_allclose(gt, gammat / (1.0 - kappa), 1.0e-10)
        assert_allclose(gt, cfg["numcosmo_profiles"]["gt"], 1.0e2 * reltol)

        # Validate magnification - discrete case
        mu = theo.compute_magnification(cosmo=cosmo, **cfg["GAMMA_PARAMS"])
        assert_allclose(mu, 1.0 / ((1 - kappa) ** 2 - abs(gammat) ** 2), 1.0e-10)
        assert_allclose(mu, cfg["numcosmo_profiles"]["mu"], 1.0e2 * reltol)

        # Validate magnification bias - discrete case
        alpha = 3.78
        mu_bias = theo.compute_magnification_bias(cosmo=cosmo, **cfg["GAMMA_PARAMS"], alpha=alpha)
        assert_allclose(
            mu_bias, (1.0 / ((1 - kappa) ** 2 - abs(gammat) ** 2)) ** (alpha - 1), 1.0e-10
        )
        assert_allclose(mu_bias, cfg["numcosmo_profiles"]["mu"] ** (alpha - 1), 1.0e3 * reltol)

        cfg_inf = load_validation_config()

        # compute some values
        cfg_inf["GAMMA_PARAMS"]["z_src"] = 1000.0
        beta_s_mean = compute_beta_s_mean_from_distribution(
            cfg_inf["GAMMA_PARAMS"]["z_cluster"], cfg_inf["GAMMA_PARAMS"]["z_src"], cosmo
        )
        beta_s_square_mean = compute_beta_s_square_mean_from_distribution(
            cfg_inf["GAMMA_PARAMS"]["z_cluster"], cfg_inf["GAMMA_PARAMS"]["z_src"], cosmo
        )

        gammat_inf = theo.compute_tangential_shear(cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"])
        kappa_inf = theo.compute_convergence(cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"])

        # test z_src = zdist.chang2013 distribution
        cfg_inf["GAMMA_PARAMS"]["z_src"] = zdist.chang2013
        cfg_inf["GAMMA_PARAMS"]["z_src_info"] = "distribution"

        # store original values
        r_proj = cfg_inf["GAMMA_PARAMS"]["r_proj"]
        # use only 5 largest radii
        cfg_inf["GAMMA_PARAMS"]["r_proj"] = cfg_inf["GAMMA_PARAMS"]["r_proj"][-5:]
        # calculate some true values
        gt = theo.compute_reduced_tangential_shear(cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"])
        mu = theo.compute_magnification(cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"])
        mu_bias = theo.compute_magnification_bias(
            cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"], alpha=alpha
        )
        cfg_inf["GAMMA_PARAMS"]["r_proj"] = r_proj
        # tangential shear and convergence cannot use z_src_info="distribution"
        # reduced tangential shear

        # test errors and also prepare for the next round of tests
        # test ValueError from unsupported approx
        assert_raises(
            ValueError,
            theo.compute_reduced_tangential_shear,
            cosmo=cosmo,
            **cfg_inf["GAMMA_PARAMS"],
            approx="notvalid",
        )
        assert_raises(
            ValueError,
            theo.compute_magnification,
            cosmo=cosmo,
            **cfg_inf["GAMMA_PARAMS"],
            approx="notvalid",
        )
        assert_raises(
            ValueError,
            theo.compute_magnification_bias,
            cosmo=cosmo,
            **cfg_inf["GAMMA_PARAMS"],
            alpha=alpha,
            approx="notvalid",
        )
        # test KeyError from invalid key in integ_kwargs
        assert_raises(
            KeyError,
            theo.compute_reduced_tangential_shear,
            cosmo=cosmo,
            **cfg_inf["GAMMA_PARAMS"],
            integ_kwargs={"notavalidkey": 0.0},
        )
        assert_raises(
            KeyError,
            theo.compute_magnification,
            cosmo=cosmo,
            **cfg_inf["GAMMA_PARAMS"],
            integ_kwargs={"notavalidkey": 0.0},
        )
        assert_raises(
            KeyError,
            theo.compute_magnification_bias,
            cosmo=cosmo,
            **cfg_inf["GAMMA_PARAMS"],
            alpha=alpha,
            integ_kwargs={"notavalidkey": 0.0},
        )
        # test ValueError from unsupported z_src_info
        cfg_inf["GAMMA_PARAMS"]["z_src_info"] = "notvalid"
        assert_raises(
            ValueError, theo.compute_tangential_shear, cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"]
        )
        assert_raises(ValueError, theo.compute_convergence, cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"])
        assert_raises(
            ValueError,
            theo.compute_reduced_tangential_shear,
            cosmo=cosmo,
            **cfg_inf["GAMMA_PARAMS"],
            approx="order1",
        )
        assert_raises(
            ValueError,
            theo.compute_magnification,
            cosmo=cosmo,
            **cfg_inf["GAMMA_PARAMS"],
            approx="order1",
        )
        assert_raises(
            ValueError,
            theo.compute_magnification_bias,
            cosmo=cosmo,
            **cfg_inf["GAMMA_PARAMS"],
            alpha=2,
            approx="order1",
        )

        # test z_src_info = 'beta'
        beta_s_mean, beta_s_square_mean = 0.9, 0.6
        cfg_inf["GAMMA_PARAMS"]["z_src"] = (beta_s_mean, beta_s_square_mean)
        cfg_inf["GAMMA_PARAMS"]["z_src_info"] = "beta"
        # tangential shear
        assert_allclose(
            theo.compute_tangential_shear(cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"]),
            beta_s_mean * gammat_inf,
            1.0e-10,
        )

        # convergence
        assert_allclose(
            theo.compute_convergence(cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"]),
            beta_s_mean * kappa_inf,
            1.0e-10,
        )

        # reduced tangential shear
        cfg_inf["GAMMA_PARAMS"]["approx"] = "order1"
        assert_allclose(
            theo.compute_reduced_tangential_shear(cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"]),
            beta_s_mean * gammat_inf / (1.0 - beta_s_mean * kappa_inf),
            1.0e-10,
        )
        cfg_inf["GAMMA_PARAMS"]["approx"] = "order2"
        assert_allclose(
            theo.compute_reduced_tangential_shear(cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"]),
            (
                1.0
                + (beta_s_square_mean / (beta_s_mean * beta_s_mean) - 1.0) * beta_s_mean * kappa_inf
            )
            * (beta_s_mean * gammat_inf / (1.0 - beta_s_mean * kappa_inf)),
            1.0e-10,
        )

        # magnification
        cfg_inf["GAMMA_PARAMS"]["approx"] = "order1"
        assert_allclose(
            theo.compute_magnification(cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"]),
            1 + 2 * beta_s_mean * kappa_inf,
            1.0e-10,
        )
        cfg_inf["GAMMA_PARAMS"]["approx"] = "order2"
        assert_allclose(
            theo.compute_magnification(cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"]),
            1
            + 2 * beta_s_mean * kappa_inf
            + beta_s_square_mean * gammat_inf**2
            + 3 * beta_s_square_mean * kappa_inf**2,
            1.0e-10,
        )

        # magnification bias
        cfg_inf["GAMMA_PARAMS"]["approx"] = "order1"
        assert_allclose(
            theo.compute_magnification_bias(cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"], alpha=alpha),
            1 + (alpha - 1) * (2 * beta_s_mean * kappa_inf),
            1.0e-10,
        )
        cfg_inf["GAMMA_PARAMS"]["approx"] = "order2"
        assert_allclose(
            theo.compute_magnification_bias(cosmo=cosmo, **cfg_inf["GAMMA_PARAMS"], alpha=alpha),
            1
            + (alpha - 1) * (2 * beta_s_mean * kappa_inf + beta_s_square_mean * gammat_inf**2)
            + (2 * alpha - 1) * (alpha - 1) * beta_s_square_mean * kappa_inf**2,
            1.0e-10,
        )

        # Check that shear, reduced shear and convergence return zero
        # and magnification and magnification bias return one
        # if source is in front of the cluster

        # First, check for a array of radius and single source z
        radius = np.logspace(-2, 2, 10)
        z_cluster = 0.3
        z_src = 0.2

        assert_allclose(
            theo.compute_convergence(
                radius,
                mdelta=1.0e15,
                cdelta=4.0,
                z_cluster=z_cluster,
                z_src=z_src,
                cosmo=cosmo,
            ),
            np.zeros(len(radius)),
            1.0e-10,
        )
        assert_allclose(
            theo.compute_tangential_shear(
                radius,
                mdelta=1.0e15,
                cdelta=4.0,
                z_cluster=z_cluster,
                z_src=z_src,
                cosmo=cosmo,
            ),
            np.zeros(len(radius)),
            1.0e-10,
        )
        assert_allclose(
            theo.compute_reduced_tangential_shear(
                radius,
                mdelta=1.0e15,
                cdelta=4.0,
                z_cluster=z_cluster,
                z_src=z_src,
                cosmo=cosmo,
            ),
            np.zeros(len(radius)),
            1.0e-10,
        )
        assert_allclose(
            theo.compute_magnification(
                radius,
                mdelta=1.0e15,
                cdelta=4.0,
                z_cluster=z_cluster,
                z_src=z_src,
                cosmo=cosmo,
            ),
            np.ones(len(radius)),
            1.0e-10,
        )
        assert_allclose(
            theo.compute_magnification_bias(
                radius,
                alpha=-1,
                mdelta=1.0e15,
                cdelta=4.0,
                z_cluster=z_cluster,
                z_src=z_src,
                cosmo=cosmo,
            ),
            np.ones(len(radius)),
            1.0e-10,
        )

        # Second, check a single radius and array of source z
        radius = 1.0
        z_src = [0.25, 0.1, 0.14, 0.02]
        assert_allclose(
            theo.compute_convergence(
                radius,
                mdelta=1.0e15,
                cdelta=4.0,
                z_cluster=z_cluster,
                z_src=z_src,
                cosmo=cosmo,
            ),
            np.zeros(len(z_src)),
            1.0e-10,
        )
        assert_allclose(
            theo.compute_tangential_shear(
                radius,
                mdelta=1.0e15,
                cdelta=4.0,
                z_cluster=z_cluster,
                z_src=z_src,
                cosmo=cosmo,
            ),
            np.zeros(len(z_src)),
            1.0e-10,
        )
        assert_allclose(
            theo.compute_reduced_tangential_shear(
                radius,
                mdelta=1.0e15,
                cdelta=4.0,
                z_cluster=z_cluster,
                z_src=z_src,
                cosmo=cosmo,
            ),
            np.zeros(len(z_src)),
            1.0e-10,
        )
        assert_allclose(
            theo.compute_magnification(
                radius,
                mdelta=1.0e15,
                cdelta=4.0,
                z_cluster=z_cluster,
                z_src=z_src,
                cosmo=cosmo,
            ),
            np.ones(len(z_src)),
            1.0e-10,
        )
        assert_allclose(
            theo.compute_magnification_bias(
                radius,
                alpha=-1,
                mdelta=1.0e15,
                cdelta=4.0,
                z_cluster=z_cluster,
                z_src=z_src,
                cosmo=cosmo,
            ),
            np.ones(len(z_src)),
            1.0e-10,
        )

        # Object Oriented tests
        mod = theo.Modeling()
        mod.set_cosmo(cosmo)
        mod.set_halo_density_profile(halo_profile_model=cfg["GAMMA_PARAMS"]["halo_profile_model"])
        mod.set_concentration(cfg["GAMMA_PARAMS"]["cdelta"])
        mod.set_mass(cfg["GAMMA_PARAMS"]["mdelta"])

        if profile_init == "einasto":
            alpha_ein = cfg["TEST_CASE"]["alpha_einasto"]

            mod.set_einasto_alpha(0.25)
            assert_allclose(mod.get_einasto_alpha(), 0.25, reltol)

            # test default value
            mod.set_einasto_alpha(None)
            if theo.be_nick == "ccl":
                assert_allclose(
                    mod.get_einasto_alpha(cfg["GAMMA_PARAMS"]["z_cluster"]), alpha_ein, reltol
                )
            if theo.be_nick == "nc":
                assert_allclose(mod.get_einasto_alpha(), 0.25, reltol)

            mod.set_einasto_alpha(alpha_ein)

        if profile_init != "einasto":
            assert_raises(ValueError, mod.get_einasto_alpha)
            assert_raises(NotImplementedError, mod.set_einasto_alpha, 0.25)

        profile_pars = [
            cfg["GAMMA_PARAMS"]["r_proj"],
            cfg["GAMMA_PARAMS"]["z_cluster"],
            cfg["GAMMA_PARAMS"]["z_src"],
        ]
        # Validate tangential shear
        gammat = mod.eval_tangential_shear(*profile_pars)
        assert_allclose(gammat, cfg["numcosmo_profiles"]["gammat"], reltol)

        # Validate convergence
        kappa = mod.eval_convergence(*profile_pars)
        assert_allclose(kappa, cfg["numcosmo_profiles"]["kappa"], reltol)

        # Validate reduced tangential shear
        assert_allclose(
            mod.eval_reduced_tangential_shear(*profile_pars), gammat / (1.0 - kappa), 1.0e-10
        )
        assert_allclose(
            mod.eval_reduced_tangential_shear(*profile_pars),
            cfg["numcosmo_profiles"]["gt"],
            1.0e2 * reltol,
        )

        # Validate magnification
        assert_allclose(
            mod.eval_magnification(*profile_pars),
            1.0 / ((1 - kappa) ** 2 - abs(gammat) ** 2),
            1.0e-10,
        )
        assert_allclose(
            mod.eval_magnification(*profile_pars), cfg["numcosmo_profiles"]["mu"], 1.0e2 * reltol
        )

        # Validate magnification bias
        alpha = -1.78
        assert_allclose(
            mod.eval_magnification_bias(*profile_pars, alpha=alpha),
            1.0 / ((1 - kappa) ** 2 - abs(gammat) ** 2) ** (alpha - 1),
            1.0e-10,
        )
        assert_allclose(
            mod.eval_magnification_bias(*profile_pars, alpha=alpha),
            cfg["numcosmo_profiles"]["mu"] ** (alpha - 1),
            1.0e3 * reltol,
        )

        beta_s_mean = 0.6
        beta_s_square_mean = 0.4
        source_redshift_inf = 1000.0
        gammat_inf = mod.eval_tangential_shear(
            profile_pars[0], profile_pars[1], source_redshift_inf
        )
        kappa_inf = mod.eval_convergence(profile_pars[0], profile_pars[1], source_redshift_inf)

        # Validate reduced tangential shear
        assert_allclose(
            mod.eval_reduced_tangential_shear(
                *profile_pars[:2], (beta_s_mean, beta_s_square_mean), "beta", "order1"
            ),
            beta_s_mean * gammat_inf / (1.0 - beta_s_mean * kappa_inf),
            1.0e-10,
        )
        assert_allclose(
            mod.eval_reduced_tangential_shear(
                *profile_pars[:2], (beta_s_mean, beta_s_square_mean), "beta", "order2"
            ),
            (
                1.0
                + (beta_s_square_mean / (beta_s_mean * beta_s_mean) - 1.0) * beta_s_mean * kappa_inf
            )
            * (beta_s_mean * gammat_inf / (1.0 - beta_s_mean * kappa_inf)),
            1.0e-10,
        )

        # Validate magnification
        alpha = -1.78
        assert_allclose(
            mod.eval_magnification(
                *profile_pars[:2], (beta_s_mean, beta_s_square_mean), "beta", "order1"
            ),
            1 + 2 * beta_s_mean * kappa_inf,
            1.0e-10,
        )
        assert_allclose(
            mod.eval_magnification(
                *profile_pars[:2], (beta_s_mean, beta_s_square_mean), "beta", "order2"
            ),
            1
            + 2 * beta_s_mean * kappa_inf
            + 3 * beta_s_square_mean * kappa_inf**2
            + beta_s_square_mean * gammat_inf**2,
            1.0e-10,
        )

        # Validate magnification bias
        assert_allclose(
            mod.eval_magnification_bias(
                *profile_pars[:2], (beta_s_mean, beta_s_square_mean), alpha, "beta", "order1"
            ),
            1 + (alpha - 1) * (2 * beta_s_mean * kappa_inf),
            1.0e-10,
        )
        assert_allclose(
            mod.eval_magnification_bias(
                *profile_pars[:2], (beta_s_mean, beta_s_square_mean), alpha, "beta", "order2"
            ),
            1
            + (alpha - 1) * (2 * beta_s_mean * kappa_inf)
            + (alpha - 1) * (beta_s_square_mean * gammat_inf**2)
            + (2 * alpha - 1) * (alpha - 1) * beta_s_square_mean * kappa_inf**2,
            1.0e-10,
        )

        # Check that shear, reduced shear and convergence return zero
        # and magnification and magnification_bias return one
        # if source is in front of the cluster

        # First, check for a array of radius and single source z
        radius = np.logspace(-2, 2, 10)
        z_cluster = 0.3
        z_src = 0.2

        assert_allclose(
            mod.eval_convergence(radius, z_cluster, z_src), np.zeros(len(radius)), 1.0e-10
        )
        assert_allclose(
            mod.eval_tangential_shear(radius, z_cluster, z_src), np.zeros(len(radius)), 1.0e-10
        )
        assert_allclose(
            mod.eval_reduced_tangential_shear(radius, z_cluster, z_src),
            np.zeros(len(radius)),
            1.0e-10,
        )
        assert_allclose(
            mod.eval_magnification(radius, z_cluster, z_src), np.ones(len(radius)), 1.0e-10
        )
        assert_allclose(
            mod.eval_magnification_bias(radius, z_cluster, z_src, alpha),
            np.ones(len(radius)),
            1.0e-10,
        )

        # Second, check a single radius and array of source z
        radius = 1.0
        z_src = [0.25, 0.1, 0.14, 0.02]

        assert_allclose(
            mod.eval_convergence(radius, z_cluster, z_src), np.zeros(len(z_src)), 1.0e-10
        )
        assert_allclose(
            mod.eval_tangential_shear(radius, z_cluster, z_src), np.zeros(len(z_src)), 1.0e-10
        )
        assert_allclose(
            mod.eval_reduced_tangential_shear(radius, z_cluster, z_src),
            np.zeros(len(z_src)),
            1.0e-10,
        )
        assert_allclose(
            mod.eval_magnification(radius, z_cluster, z_src), np.ones(len(z_src)), 1.0e-10
        )
        assert_allclose(
            mod.eval_magnification_bias(radius, z_cluster, z_src, alpha),
            np.ones(len(z_src)),
            1.0e-10,
        )


def test_compute_magnification_bias(modeling_data):
    """Unit tests for compute_magnification_bias_from_magnification"""
    # Make some base objects
    magnification = [1.0, 1.0, 1.001, 0.76]
    alpha = [1.0, -2.7, 5.0]
    truth = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.99630868, 2.76051244],
        [1.0, 1.0, 1.004006, 0.33362176],
    ]

    # Check output including: float, list, ndarray
    assert_allclose(
        theo.compute_magnification_bias_from_magnification(magnification[0], alpha[0]),
        truth[0][0],
        **TOLERANCE,
    )
    assert_allclose(
        theo.compute_magnification_bias_from_magnification(magnification, alpha), truth, **TOLERANCE
    )
    assert_allclose(
        theo.compute_magnification_bias_from_magnification(
            np.array(magnification), np.array(alpha)
        ),
        np.array(truth),
        **TOLERANCE,
    )


def test_mass_conversion(modeling_data, profile_init):
    """Unit tests for HaloProfile objects' instantiation"""
    if profile_init == "nfw" or modeling_data["nick"] in ["nc", "ccl"]:
        reltol = modeling_data["theory_reltol"]

        ### Loads values precomputed by numcosmo for comparison
        numcosmo_path = "tests/data/numcosmo/"
        with open(numcosmo_path + "config.json", "r") as fin:
            testcase = json.load(fin)
        cosmo = theo.Cosmology(
            H0=testcase["cosmo_H0"],
            Omega_dm0=testcase["cosmo_Odm0"],
            Omega_b0=testcase["cosmo_Ob0"],
        )
        # Config
        halo_profile_model = profile_init
        mdelta = testcase["cluster_mass"]
        cdelta = testcase["cluster_concentration"]
        z_cl = testcase["z_cluster"]
        delta_mdef = 200
        massdef = "mean"
        # Start tests
        profile = theo.Modeling()
        profile.set_cosmo(cosmo)
        profile.set_halo_density_profile(
            halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef
        )
        profile.set_concentration(cdelta)
        profile.set_mass(mdelta)
        if halo_profile_model == "einasto":
            profile.set_einasto_alpha(0.3)

        assert_allclose(
            profile.eval_mass_in_radius(profile.eval_rdelta(z_cl), z_cl, True), mdelta, 1e-15
        )

        if halo_profile_model == "nfw":
            assert_allclose(profile.eval_rdelta(z_cl), 1.5548751530053142, reltol)
            assert_allclose(profile.eval_mass_in_radius(1.0, z_cl), 683427961195829.4, reltol)

        assert_raises(ValueError, profile.convert_mass_concentration, z_cl, massdef="blu")
        assert_raises(
            ValueError, profile.convert_mass_concentration, z_cl, halo_profile_model="bla"
        )

        truth = {
            "nfw": {"mdelta": 617693839984902.6, "cdelta": 2.3143737357611425},
            "einasto": {"mdelta": 654444421625520.1, "cdelta": 2.3593914002446486},
        }
        if halo_profile_model != "hernquist":
            mdelta2, cdelta2 = profile.convert_mass_concentration(
                z_cl, massdef="critical", delta_mdef=500, verbose=True
            )
            assert_allclose(mdelta2, truth[halo_profile_model]["mdelta"], reltol)
            assert_allclose(cdelta2, truth[halo_profile_model]["cdelta"], reltol)
        # catch error in generic.compute_profile_mass_in_radius('einasto', alpha=None)
        if halo_profile_model == "einasto":
            profile._get_einasto_alpha = lambda z_cl: None
            assert_raises(ValueError, profile.convert_mass_concentration, z_cl)


def test_delta_mdef_virial(modeling_data):
    if modeling_data["nick"] in ["nc", "ccl"]:
        cfg = load_validation_config()
        cosmo = cfg['cosmo']
        mod = theo.Modeling(massdef="virial")
        mod.set_cosmo(cosmo)
        assert_equal(mod._get_delta_mdef(0.1), 111)
