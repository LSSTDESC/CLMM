# pylint: disable=no-member, protected-access
""" Tests for utils.py """
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
from scipy.integrate import quad
import clmm
import clmm.utils as utils
from clmm.utils import (
    compute_radial_averages,
    make_bins,
    convert_shapes_to_epsilon,
    arguments_consistency,
    validate_argument,
    DiffArray,
    redshift_distributions as zdist,
)


TOLERANCE = {"rtol": 1.0e-6, "atol": 0}


def test_compute_nfw_boost():
    """Test the nfw model for boost factor"""
    # Test data
    rvals = np.arange(1, 11).tolist()
    rscale = 1000

    boost_factors = utils.compute_nfw_boost(rvals, rscale)

    test_boost_factors = np.array(
        [
            1.66009126,
            1.59077917,
            1.55023667,
            1.52147373,
            1.4991658,
            1.48094117,
            1.46553467,
            1.4521911,
            1.44042332,
            1.42989872,
        ]
    )

    #  Test model
    assert_allclose(boost_factors, test_boost_factors)


def test_compute_powerlaw_boost():
    """Test the powerlaw model for boost factor"""
    # Test data
    rvals = np.arange(1, 11).tolist()  # Cannot contain 0 due to reciprocal term
    rscale = 1000

    boost_factors = utils.compute_powerlaw_boost(rvals, rscale)

    test_boost_factors = np.array(
        [101.0, 51.0, 34.33333333, 26.0, 21.0, 17.66666667, 15.28571429, 13.5, 12.11111111, 11.0]
    )

    # Test model
    assert_allclose(boost_factors, test_boost_factors)


def test_correct_with_boost_values():
    """ """
    # Make test data
    rvals = np.arange(1, 11)
    sigma_vals = (2 ** np.arange(10)).tolist()

    test_unit_boost_factors = np.ones(rvals.shape).tolist()

    corrected_sigma = utils.correct_with_boost_values(sigma_vals, test_unit_boost_factors)
    assert_allclose(sigma_vals, corrected_sigma)


def test_correct_with_boost_model():
    """ """
    # Make test data
    rvals = np.arange(1, 11).tolist()
    sigma_vals = (2 ** np.arange(10)).tolist()
    boost_rscale = 1000

    for boost_model in utils.boost_models.keys():
        # Check for no nans or inf with positive-definite rvals and sigma vals
        assert np.all(
            np.isfinite(
                utils.correct_with_boost_model(rvals, sigma_vals, boost_model, boost_rscale)
            )
        )

    # Test requesting unsupported boost model
    assert_raises(KeyError, utils.correct_with_boost_model, rvals, sigma_vals, "glue", boost_rscale)


def test_compute_radial_averages():
    """Tests compute_radial_averages, a function that computes several binned statistics"""
    # Make some test data
    binvals = np.array([2.0, 3.0, 6.0, 8.0, 4.0, 9.0])
    xbins1 = [0.0, 10.0]
    xbins2 = [0.0, 5.0, 10.0]

    # Test requesting an unsupported error model
    assert_raises(
        ValueError, compute_radial_averages, binvals, binvals, [0.0, 10.0], error_model="glue"
    )

    # Check the default error model
    assert_allclose(
        compute_radial_averages(binvals, binvals, xbins1)[:4],
        [[np.mean(binvals)], [np.mean(binvals)], [np.std(binvals) / np.sqrt(len(binvals))], [6]],
        **TOLERANCE
    )
    # Test weights
    # Normalized
    assert_allclose(
        compute_radial_averages([1, 1], [2, 3], [1, 2], weights=[0.5, 0.5])[:3],
        ([1], [2.5], [1 / np.sqrt(8)]),
        **TOLERANCE
    )
    # Not normalized
    assert_allclose(
        compute_radial_averages([1, 1], [2, 3], [1, 2], weights=[5, 5])[:3],
        ([1], [2.5], [1 / np.sqrt(8)]),
        **TOLERANCE
    )
    # Values outside bins
    assert_allclose(
        compute_radial_averages([1, 1, 3], [2, 3, 1000], [1, 2], weights=[0.5, 0.5, 100])[:3],
        ([1], [2.5], [1 / np.sqrt(8)]),
        **TOLERANCE
    )
    # Weighted values == Repeated values (std only)
    assert_allclose(
        compute_radial_averages([1, 1], [2, 3], [1, 2], weights=[1, 2], error_model="std")[:3],
        compute_radial_averages([1, 1, 1], [2, 3, 3], [1, 2], error_model="std")[:3],
        **TOLERANCE
    )
    # Zero yerr
    assert_allclose(
        compute_radial_averages([1, 1], [2, 3], [1, 2], weights=[0.5, 0.5], yerr=[0, 0])[:3],
        ([1], [2.5], [1 / np.sqrt(8)]),
        **TOLERANCE
    )
    # With yerr
    assert_allclose(
        compute_radial_averages([1, 1], [2, 3], [1, 2], weights=[0.5, 0.5], yerr=[1, 1])[:3],
        ([1], [2.5], [np.sqrt(5 / 8)]),
        **TOLERANCE
    )

    # Test 3 objects in one bin with various error models
    assert_allclose(
        compute_radial_averages(binvals, binvals, xbins1, error_model="ste")[:4],
        [[np.mean(binvals)], [np.mean(binvals)], [np.std(binvals) / np.sqrt(len(binvals))], [6]],
        **TOLERANCE
    )
    assert_allclose(
        compute_radial_averages(binvals, binvals, xbins1, error_model="std")[:4],
        [[np.mean(binvals)], [np.mean(binvals)], [np.std(binvals)], [6]],
        **TOLERANCE
    )

    # Repeat test with different error_model case
    assert_allclose(
        compute_radial_averages(binvals, binvals, xbins1, error_model="STE")[:4],
        [[np.mean(binvals)], [np.mean(binvals)], [np.std(binvals) / np.sqrt(len(binvals))], [6]],
        **TOLERANCE
    )
    assert_allclose(
        compute_radial_averages(binvals, binvals, xbins1, error_model="STD")[:4],
        [[np.mean(binvals)], [np.mean(binvals)], [np.std(binvals)], [6]],
        **TOLERANCE
    )

    # A slightly more complicated case with two bins
    inbin1 = binvals[(binvals > xbins2[0]) & (binvals < xbins2[1])]
    inbin2 = binvals[(binvals > xbins2[1]) & (binvals < xbins2[2])]
    assert_allclose(
        compute_radial_averages(binvals, binvals, xbins2, error_model="ste")[:4],
        [
            [np.mean(inbin1), np.mean(inbin2)],
            [np.mean(inbin1), np.mean(inbin2)],
            [np.std(inbin1) / np.sqrt(len(inbin1)), np.std(inbin2) / np.sqrt(len(inbin2))],
            [3, 3],
        ],
        **TOLERANCE
    )
    assert_allclose(
        compute_radial_averages(binvals, binvals, xbins2, error_model="std")[:4],
        [
            [np.mean(inbin1), np.mean(inbin2)],
            [np.mean(inbin1), np.mean(inbin2)],
            [np.std(inbin1), np.std(inbin2)],
            [3, 3],
        ],
        **TOLERANCE
    )

    # Test a much larger, random sample with unevenly spaced bins
    binvals = np.loadtxt("tests/data/radial_average_test_array.txt")
    xbins2 = [0.0, 3.33, 6.66, 10.0]
    inbin1 = binvals[(binvals > xbins2[0]) & (binvals < xbins2[1])]
    inbin2 = binvals[(binvals > xbins2[1]) & (binvals < xbins2[2])]
    inbin3 = binvals[(binvals > xbins2[2]) & (binvals < xbins2[3])]
    assert_allclose(
        compute_radial_averages(binvals, binvals, xbins2, error_model="ste")[:4],
        [
            [np.mean(inbin1), np.mean(inbin2), np.mean(inbin3)],
            [np.mean(inbin1), np.mean(inbin2), np.mean(inbin3)],
            [
                np.std(inbin1) / np.sqrt(len(inbin1)),
                np.std(inbin2) / np.sqrt(len(inbin2)),
                np.std(inbin3) / np.sqrt(len(inbin3)),
            ],
            [inbin1.size, inbin2.size, inbin3.size],
        ],
        **TOLERANCE
    )
    assert_allclose(
        compute_radial_averages(binvals, binvals, xbins2, error_model="std")[:4],
        [
            [np.mean(inbin1), np.mean(inbin2), np.mean(inbin3)],
            [np.mean(inbin1), np.mean(inbin2), np.mean(inbin3)],
            [np.std(inbin1), np.std(inbin2), np.std(inbin3)],
            [inbin1.size, inbin2.size, inbin3.size],
        ],
        **TOLERANCE
    )


def test_make_bins():
    """Test the make_bins function. Right now this function is pretty simplistic and the
    tests are pretty circular. As more functionality is added here the tests will
    become more substantial.
    """
    # Test various combinations of rmin and rmax with default values
    assert_allclose(make_bins(0.0, 10.0), np.linspace(0.0, 10.0, 11), **TOLERANCE)
    assert_raises(ValueError, make_bins, 0.0, -10.0)
    assert_raises(ValueError, make_bins, -10.0, 10.0)
    assert_raises(ValueError, make_bins, -10.0, -5.0)
    assert_raises(ValueError, make_bins, 10.0, 0.0)

    # Test various nbins
    assert_allclose(make_bins(0.0, 10.0, nbins=3), np.linspace(0.0, 10.0, 4), **TOLERANCE)
    assert_allclose(make_bins(0.0, 10.0, nbins=13), np.linspace(0.0, 10.0, 14), **TOLERANCE)
    assert_raises(ValueError, make_bins, 0.0, 10.0, -10)
    assert_raises(ValueError, make_bins, 0.0, 10.0, 0)

    # Test default method
    assert_allclose(
        make_bins(0.0, 10.0, nbins=10),
        make_bins(0.0, 10.0, nbins=10, method="evenwidth"),
        **TOLERANCE
    )

    # Test the different binning methods
    assert_allclose(
        make_bins(0.0, 10.0, nbins=10, method="evenwidth"), np.linspace(0.0, 10.0, 11), **TOLERANCE
    )
    assert_allclose(
        make_bins(1.0, 10.0, nbins=10, method="evenlog10width"),
        np.logspace(np.log10(1.0), np.log10(10.0), 11),
        **TOLERANCE
    )

    # Repeat test with different error_model case
    assert_allclose(
        make_bins(0.0, 10.0, nbins=10, method="EVENWIDTH"), np.linspace(0.0, 10.0, 11), **TOLERANCE
    )
    assert_allclose(
        make_bins(1.0, 10.0, nbins=10, method="EVENLOG10WIDTH"),
        np.logspace(np.log10(1.0), np.log10(10.0), 11),
        **TOLERANCE
    )

    # Test equaloccupation method. It needs a source_seps array, so create one
    test_array = np.sqrt(
        np.random.uniform(-10, 10, 1361) ** 2 + np.random.uniform(-10, 10, 1361) ** 2
    )
    test_bins = make_bins(1.0, 10.0, nbins=10, method="equaloccupation", source_seps=test_array)
    # Check that all bins have roughly equal occupation.
    # Assert needs atol=2, because len(source_seps)/nbins may not be an integer,
    # and for some random arrays atol=1 is not enough.
    assert_allclose(np.diff(np.histogram(test_array, bins=test_bins)[0]), np.zeros(9), atol=2)
    test_bins = make_bins(0.51396, 6.78, nbins=23, method="equaloccupation", source_seps=test_array)
    assert_allclose(np.diff(np.histogram(test_array, bins=test_bins)[0]), np.zeros(22), atol=2)
    assert_equal(
        make_bins(0, 15, nbins=23, method="equaloccupation", source_seps=test_array),
        make_bins(None, None, nbins=23, method="equaloccupation", source_seps=test_array),
    )
    assert_raises(ValueError, make_bins, 0, 10, 10, "equaloccupation", None)
    assert_raises(ValueError, make_bins, 0, 10, 10, "undefinedmethod")


def test_convert_units():
    """Test the wrapper function to convert units. Corner cases should be tested in the
    individual functions. This function should test one case for all supported conversions
    and the error handling.
    """
    # Make an astropy cosmology object for testing
    # cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
    cosmo = clmm.Cosmology(H0=70.0, Omega_dm0=0.3 - 0.045, Omega_b0=0.045)

    # Test that each unit is supported
    utils.convert_units(1.0, "radians", "degrees")
    utils.convert_units(1.0, "arcmin", "arcsec")
    utils.convert_units(1.0, "Mpc", "kpc")
    utils.convert_units(1.0, "Mpc", "kpc")

    # Error checking
    assert_raises(ValueError, utils.convert_units, 1.0, "radians", "CRAZY")
    assert_raises(ValueError, utils.convert_units, 1.0, "CRAZY", "radians")
    assert_raises(TypeError, utils.convert_units, 1.0, "arcsec", "Mpc")
    assert_raises(TypeError, utils.convert_units, 1.0, "arcsec", "Mpc", None, cosmo)
    assert_raises(TypeError, utils.convert_units, 1.0, "arcsec", "Mpc", 0.5, None)
    assert_raises(ValueError, utils.convert_units, 1.0, "arcsec", "Mpc", -0.5, cosmo)

    # Test cases to make sure angular -> angular is fitting together
    assert_allclose(utils.convert_units(np.pi, "radians", "degrees"), 180.0, **TOLERANCE)
    assert_allclose(utils.convert_units(180.0, "degrees", "radians"), np.pi, **TOLERANCE)
    assert_allclose(utils.convert_units(1.0, "degrees", "arcmin"), 60.0, **TOLERANCE)
    assert_allclose(utils.convert_units(1.0, "degrees", "arcsec"), 3600.0, **TOLERANCE)

    # Test cases to make sure physical -> physical is fitting together
    assert_allclose(utils.convert_units(1.0, "Mpc", "kpc"), 1.0e3, **TOLERANCE)
    assert_allclose(utils.convert_units(1000.0, "kpc", "Mpc"), 1.0, **TOLERANCE)
    assert_allclose(utils.convert_units(1.0, "Mpc", "pc"), 1.0e6, **TOLERANCE)

    # Test conversion from angular to physical
    # Using astropy, circular now but this will be fine since we are going to be
    # swapping to CCL soon and then its kosher
    r_arcmin, redshift = 20.0, 0.5
    d_a = cosmo.eval_da(redshift) * 1.0e3  # kpc
    truth = r_arcmin * (1.0 / 60.0) * (np.pi / 180.0) * d_a
    assert_allclose(
        utils.convert_units(r_arcmin, "arcmin", "kpc", redshift, cosmo), truth, **TOLERANCE
    )

    # Test conversion both ways between angular and physical units
    # Using astropy, circular now but this will be fine since we are going to be
    # swapping to CCL soon and then its kosher
    r_kpc, redshift = 20.0, 0.5
    #    d_a = cosmo.angular_diameter_distance(redshift).to('kpc').value
    d_a = cosmo.eval_da(redshift) * 1.0e3  # kpc
    truth = r_kpc * (1.0 / d_a) * (180.0 / np.pi) * 60.0
    assert_allclose(
        utils.convert_units(r_kpc, "kpc", "arcmin", redshift, cosmo), truth, **TOLERANCE
    )


def test_build_ellipticities():
    """test build ellipticities"""

    # second moments are floats
    q11 = 0.5
    q22 = 0.3
    q12 = 0.02

    assert_allclose(
        utils.build_ellipticities(q11, q22, q12),
        (0.25, 0.05, 0.12710007580505459, 0.025420015161010917),
        **TOLERANCE
    )

    # second moments are numpy array
    q11 = np.array([0.5, 0.3])
    q22 = np.array([0.8, 0.2])
    q12 = np.array([0.01, 0.01])

    assert_allclose(
        utils.build_ellipticities(q11, q22, q12),
        (
            [-0.23076923, 0.2],
            [0.01538462, 0.04],
            [-0.11697033, 0.10106221],
            [0.00779802, 0.02021244],
        ),
        **TOLERANCE
    )


def test_shape_conversion():
    """Test the helper function that convert user defined shapes into
    epsilon ellipticities or reduced shear. Both can be used for the galcat in
    the GalaxyCluster object"""

    # Number of random ellipticities to check
    niter = 25

    # Create random second moments and from that random ellipticities
    q11, q22 = np.random.randint(0, 20, (2, niter))
    # Q11 seperate to avoid a whole bunch of nans
    q12 = np.random.uniform(-1, 1, niter) * np.sqrt(q11 * q22)
    chi1, chi2, ellips1, ellips2 = utils.build_ellipticities(q11, q22, q12)

    # Test conversion from 'chi' to epsilon
    ellips1_2, ellips2_2 = convert_shapes_to_epsilon(chi1, chi2, shape_definition="chi")
    assert_allclose(ellips1, ellips1_2, **TOLERANCE)
    assert_allclose(ellips2, ellips2_2, **TOLERANCE)

    # Test that 'epsilon' just returns the same values
    ellips1_2, ellips2_2 = convert_shapes_to_epsilon(ellips1, ellips2, shape_definition="epsilon")
    assert_allclose(ellips1, ellips1_2, **TOLERANCE)
    assert_allclose(ellips2, ellips2_2, **TOLERANCE)

    # Test that 'reduced_shear' just returns the same values
    ellips1_2, ellips2_2 = convert_shapes_to_epsilon(
        ellips1, ellips2, shape_definition="reduced_shear"
    )
    assert_allclose(ellips1, ellips1_2, **TOLERANCE)
    assert_allclose(ellips2, ellips2_2, **TOLERANCE)

    # Test that 'shear' just returns the right values for reduced shear
    ellips1_2, ellips2_2 = convert_shapes_to_epsilon(
        ellips1, ellips2, shape_definition="shear", kappa=0.2
    )
    assert_allclose(ellips1 / 0.8, ellips1_2, **TOLERANCE)
    assert_allclose(ellips2 / 0.8, ellips2_2, **TOLERANCE)
    # Test known shape_definition
    assert_raises(
        TypeError, convert_shapes_to_epsilon, ellips1, ellips2, shape_definition="undefinedSD"
    )


def test_compute_lensed_ellipticities():
    """test compute lensed ellipticities"""

    # Validation test with floats
    es1 = 0
    es2 = 0
    gamma1 = 0.2
    gamma2 = 0.2
    kappa = 0.5
    assert_allclose(
        utils.compute_lensed_ellipticity(es1, es2, gamma1, gamma2, kappa), (0.4, 0.4), **TOLERANCE
    )

    # Validation test with array
    es1 = np.array([0, 0.5])
    es2 = np.array([0, 0.1])
    gamma1 = np.array([0.2, 0.0])
    gamma2 = np.array([0.2, 0.3])
    kappa = np.array([0.5, 0.2])

    assert_allclose(
        utils.compute_lensed_ellipticity(es1, es2, gamma1, gamma2, kappa),
        ([0.4, 0.38656171], [0.4, 0.52769188]),
        **TOLERANCE
    )


def test_arguments_consistency():
    """test arguments consistency"""
    assert_allclose(arguments_consistency([1, 2]), [1, 2], **TOLERANCE)
    assert_allclose(arguments_consistency([1, 2], names=["a", "b"]), [1, 2], **TOLERANCE)
    assert_allclose(arguments_consistency([1, 2], names="ab"), [1, 2], **TOLERANCE)
    assert_allclose(
        arguments_consistency([1, 2], names=["a", "b"], prefix="x"), [1, 2], **TOLERANCE
    )

    assert_allclose(arguments_consistency([[1], [2]]), [[1], [2]], **TOLERANCE)
    assert_allclose(arguments_consistency([[1], [2]], names=["a", "b"]), [[1], [2]], **TOLERANCE)
    assert_allclose(arguments_consistency([[1], [2]], names="ab"), [[1], [2]], **TOLERANCE)
    assert_allclose(
        arguments_consistency([[1], [2]], names=["a", "b"], prefix="x"), [[1], [2]], **TOLERANCE
    )
    # test error
    assert_raises(TypeError, arguments_consistency, [1, [1, 2]])
    assert_raises(TypeError, arguments_consistency, [[1], [1, 2]])
    assert_raises(TypeError, arguments_consistency, [1, 2], names=["a"])


def test_validate_argument():
    """test validate argument"""
    loc = {
        "float": 1.1,
        "int": 3,
        "str": "test",
        "int_array": [1, 2],
        "float_array": [1.1, 1.2],
        "float_str": "1.1",
        "none": None,
    }
    # Validate type
    for type_ in (int, float, "int_array", "float_array", (str, int)):
        assert validate_argument(loc, "int", type_) is None
    for type_ in (float, "float_array", (str, float)):
        assert validate_argument(loc, "float", type_) is None
    for type_ in ("int_array", "float_array", (str, "int_array")):
        assert validate_argument(loc, "int_array", type_) is None
    for type_ in ("float_array", (str, "float_array")):
        assert validate_argument(loc, "float_array", type_) is None
    for type_ in (str, ("float_array", str, float)):
        assert validate_argument(loc, "str", type_) is None
    assert validate_argument(loc, "none", float, none_ok=True) is None  # test none_ok

    for type_ in (bool, (bool, tuple)):
        for argname in loc:
            assert_raises(TypeError, validate_argument, loc, argname, type_)

    for type_ in (int, (str, "int_array")):
        assert_raises(TypeError, validate_argument, loc, "float", type_)

    for type_ in (int, float, (str, float)):
        for argname in ("int_array", "float_array"):
            assert_raises(TypeError, validate_argument, loc, argname, type_)

    for argname in ("float", "int", "int_array", "float_array", "float_str"):
        assert (
            validate_argument(
                loc, argname, ("float_array", str), argmin=0, argmax=4, eqmin=False, eqmax=False
            )
            is None
        )
        assert (
            validate_argument(
                loc, argname, ("float_array", str), argmin=0, argmax=4, eqmin=True, eqmax=True
            )
            is None
        )

    # Test shape
    for argname in ("int", "float", "str"):
        assert validate_argument(loc, argname, (str, "float_array"), shape=()) is None
    for argname in ("int_array", "float_array"):
        assert validate_argument(loc, argname, "float_array", shape=(2,)) is None
    assert_raises(ValueError, validate_argument, loc, "int", int, shape=np.shape(loc["int_array"]))

    assert_raises(TypeError, validate_argument, loc, "str", ("float_array", str), argmin=0)

    for argname in ("float", "float_array", "float_str"):
        assert_raises(ValueError, validate_argument, loc, argname, ("float_array", str), argmin=1.1)
        assert validate_argument(loc, argname, ("float_array", str), argmin=1.1, eqmin=True) is None
        assert_raises(ValueError, validate_argument, loc, argname, ("float_array", str), argmax=1.1)

    assert (
        validate_argument(loc, "float_array", ("float_array", str), argmax=1.2, eqmax=True) is None
    )


def test_diff_array():
    """test validate argument"""
    # Validate diffs
    assert DiffArray([1, 2]) == DiffArray([1, 2])
    assert DiffArray([1, 2]) == DiffArray(np.array([1, 2]))
    assert DiffArray([1, 2]) != DiffArray(np.array([2, 2]))
    assert DiffArray([1, 2]) != DiffArray(np.array([1, 2, 3]))
    assert DiffArray([1, 2]) != None
    # Validate prints
    arr = DiffArray([1, 2])
    assert str(arr.value) == arr.__repr__()
    arr = DiffArray(range(10))
    assert str(arr.value) != arr.__repr__()


def test_beta_functions(modeling_data):
    z_cl = 1.0
    z_src = [2.4, 2.1]
    shape_weights = [4.6, 6.4]
    z_inf = 1000.0
    zmax = 15.0
    nsteps = 1000
    zmin = z_cl + 0.1
    z_int = np.linspace(zmin, zmax, nsteps)
    cosmo = clmm.Cosmology(H0=70.0, Omega_dm0=0.27 - 0.045, Omega_b0=0.045, Omega_k0=0.0)
    beta_test = [
        np.heaviside(z_s - z_cl, 0) * cosmo.eval_da_z1z2(z_cl, z_s) / cosmo.eval_da(z_s)
        for z_s in z_src
    ]
    beta_s_test = utils.compute_beta(z_src, z_cl, cosmo) / utils.compute_beta(z_inf, z_cl, cosmo)

    assert_allclose(utils.compute_beta(z_src, z_cl, cosmo), beta_test, **TOLERANCE)
    assert_allclose(utils.compute_beta_s(z_src, z_cl, z_inf, cosmo), beta_s_test, **TOLERANCE)

    # beta mean from distributions

    for model in (None, zdist.chang2013, zdist.desc_srd):
        # None defaults to chang2013 for compute_beta* functions

        if model is None:
            model = zdist.chang2013

        def integrand1(z_i, z_inf=z_inf, z_cl=z_cl, cosmo=cosmo):
            return utils.compute_beta_s(z_i, z_cl, z_inf, cosmo) * model(z_i)

        def integrand2(z_i, z_inf=z_inf, z_cl=z_cl, cosmo=cosmo):
            return utils.compute_beta_s(z_i, z_cl, z_inf, cosmo) ** 2 * model(z_i)

        assert_allclose(
            utils.compute_beta_s_mean_from_distribution(
                z_cl, z_inf, cosmo, zmax, z_distrib_func=model
            ),
            quad(integrand1, zmin, zmax)[0] / quad(model, zmin, zmax)[0],
            **TOLERANCE
        )
        assert_allclose(
            utils.compute_beta_s_square_mean_from_distribution(
                z_cl, z_inf, cosmo, zmax, z_distrib_func=model
            ),
            quad(integrand2, zmin, zmax)[0] / quad(model, zmin, zmax)[0],
            **TOLERANCE
        )

    # beta mean from weights

    assert_allclose(
        utils.compute_beta_s_mean_from_weights(z_src, z_cl, z_inf, cosmo, shape_weights),
        np.sum(
            shape_weights * utils.compute_beta_s(z_src, z_cl, z_inf, cosmo) / np.sum(shape_weights)
        ),
        **TOLERANCE
    )
    assert_allclose(
        utils.compute_beta_s_square_mean_from_weights(z_src, z_cl, z_inf, cosmo, shape_weights),
        np.sum(
            shape_weights
            * utils.compute_beta_s(z_src, z_cl, z_inf, cosmo) ** 2
            / np.sum(shape_weights)
        ),
        **TOLERANCE
    )

    no_weights = [1, 1]
    assert_allclose(
        utils.compute_beta_s_mean_from_weights(z_src, z_cl, z_inf, cosmo, None),
        np.sum(no_weights * utils.compute_beta_s(z_src, z_cl, z_inf, cosmo) / np.sum(no_weights)),
        **TOLERANCE
    )
    assert_allclose(
        utils.compute_beta_s_square_mean_from_weights(z_src, z_cl, z_inf, cosmo, None),
        np.sum(
            no_weights * utils.compute_beta_s(z_src, z_cl, z_inf, cosmo) ** 2 / np.sum(no_weights)
        ),
        **TOLERANCE
    )
