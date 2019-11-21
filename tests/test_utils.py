# pylint: disable=no-member, protected-access
""" Tests for utils.py """
import numpy as np
from numpy.testing import assert_raises, assert_allclose
from astropy.cosmology import FlatLambdaCDM

import clmm.utils as utils
from clmm.utils import compute_radial_averages, make_bins


TOLERANCE = {'rtol': 1.0e-6, 'atol': 0}


def test_compute_radial_averages():
    """ Tests compute_radial_averages, a function that computes several binned statistics """
    # Make some test data
    binvals = np.array([2., 3., 6., 8., 4., 9.])
    xbins1 = [0., 10.]
    xbins2 = [0., 5., 10.]

    # Test requesting an unsupported error model
    assert_raises(ValueError, compute_radial_averages, binvals, binvals, [0., 10.], 'glue')

    # Check the default error model
    assert_allclose(compute_radial_averages(binvals, binvals, xbins1),
                    [[np.mean(binvals)], [np.mean(binvals)], [np.std(binvals)/len(binvals)]],
                    **TOLERANCE)

    # Test 3 objects in one bin with various error models
    assert_allclose(compute_radial_averages(binvals, binvals, xbins1, error_model='std/n'),
                    [[np.mean(binvals)], [np.mean(binvals)], [np.std(binvals)/len(binvals)]],
                    **TOLERANCE)
    assert_allclose(compute_radial_averages(binvals, binvals, xbins1, error_model='std'),
                    [[np.mean(binvals)], [np.mean(binvals)], [np.std(binvals)]], **TOLERANCE)

    # A slightly more complicated case with two bins
    inbin1 = binvals[(binvals > xbins2[0]) & (binvals < xbins2[1])]
    inbin2 = binvals[(binvals > xbins2[1]) & (binvals < xbins2[2])]
    assert_allclose(compute_radial_averages(binvals, binvals, xbins2, error_model='std/n'),
                    [[np.mean(inbin1), np.mean(inbin2)], [np.mean(inbin1), np.mean(inbin2)],
                     [np.std(inbin1)/len(inbin1), np.std(inbin2)/len(inbin2)]], **TOLERANCE)
    assert_allclose(compute_radial_averages(binvals, binvals, xbins2, error_model='std'),
                    [[np.mean(inbin1), np.mean(inbin2)], [np.mean(inbin1), np.mean(inbin2)],
                     [np.std(inbin1), np.std(inbin2)]], **TOLERANCE)

    # Test a much larger, random sample with unevenly spaced bins
    binvals = np.loadtxt('tests/data/radial_average_test_array.txt')
    xbins2 = [0.0, 3.33, 6.66, 10.0]
    inbin1 = binvals[(binvals > xbins2[0]) & (binvals < xbins2[1])]
    inbin2 = binvals[(binvals > xbins2[1]) & (binvals < xbins2[2])]
    inbin3 = binvals[(binvals > xbins2[2]) & (binvals < xbins2[3])]
    assert_allclose(compute_radial_averages(binvals, binvals, xbins2, error_model='std/n'),
                    [[np.mean(inbin1), np.mean(inbin2), np.mean(inbin3)],
                     [np.mean(inbin1), np.mean(inbin2), np.mean(inbin3)],
                     [np.std(inbin1)/len(inbin1), np.std(inbin2)/len(inbin2),
                      np.std(inbin3)/len(inbin3)]], **TOLERANCE)
    assert_allclose(compute_radial_averages(binvals, binvals, xbins2, error_model='std'),
                    [[np.mean(inbin1), np.mean(inbin2), np.mean(inbin3)],
                     [np.mean(inbin1), np.mean(inbin2), np.mean(inbin3)],
                     [np.std(inbin1), np.std(inbin2), np.std(inbin3)]], **TOLERANCE)



def test_make_bins():
    """ Test the make_bins function. Right now this function is pretty simplistic and the
    tests are pretty circular. As more functionality is added here the tests will
    become more substantial.
    """
    # Test various combinations of rmin and rmax with default values
    assert_allclose(make_bins(0.0, 10.), np.linspace(0.0, 10., 11), **TOLERANCE)
    assert_raises(ValueError, make_bins, 0.0, -10.)
    assert_raises(ValueError, make_bins, -10., 10.)
    assert_raises(ValueError, make_bins, -10., -5.)
    assert_raises(ValueError, make_bins, 10., 0.0)

    # Test various nbins
    assert_allclose(make_bins(0.0, 10., nbins=3), np.linspace(0.0, 10., 4), **TOLERANCE)
    assert_allclose(make_bins(0.0, 10., nbins=13), np.linspace(0.0, 10., 14), **TOLERANCE)
    assert_raises(ValueError, make_bins, 0.0, 10., -10)
    assert_raises(ValueError, make_bins, 0.0, 10., 0)

    # Test default method
    assert_allclose(make_bins(0.0, 10., nbins=10),
                    make_bins(0.0, 10., nbins=10, method='evenwidth'),
                    **TOLERANCE)

    # Test the different binning methods
    assert_allclose(make_bins(0.0, 10., nbins=10, method='evenwidth'),
                    np.linspace(0.0, 10., 11), **TOLERANCE)
    assert_allclose(make_bins(1.0, 10., nbins=10, method='evenlog10width'),
                    np.logspace(np.log10(1.0), np.log10(10.), 11), **TOLERANCE)


def _rad_to_mpc_helper(dist, redshift, cosmo, do_inverse):
    """ Helper function to clean up test_convert_rad_to_mpc. Truth is computed using
    astropy so this test is very circular. Once we swap to CCL very soon this will be
    a good source of truth. """
    d_a = cosmo.angular_diameter_distance(redshift).to('Mpc').value
    if do_inverse:
        truth = dist / d_a
    else:
        truth = dist * d_a
    assert_allclose(utils._convert_rad_to_mpc(dist, redshift, cosmo, do_inverse),
                    truth, **TOLERANCE)


def test_convert_rad_to_mpc():
    """ Test conversion between physical and angular units and vice-versa. """
    # Set some default values if I want them
    redshift = 0.25
    cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

    # Test the default behavior
    assert_allclose(utils._convert_rad_to_mpc(0.33, redshift, cosmo),
                    utils._convert_rad_to_mpc(0.33, redshift, cosmo, False), **TOLERANCE)

    # Test basic conversions each way
    _rad_to_mpc_helper(0.003, redshift, cosmo, do_inverse=False)
    _rad_to_mpc_helper(1.0, redshift, cosmo, do_inverse=True)

    # Convert back and forth and make sure I get the same answer
    midtest = utils._convert_rad_to_mpc(0.003, redshift, cosmo)
    assert_allclose(utils._convert_rad_to_mpc(midtest, redshift, cosmo, do_inverse=True),
                    0.003, **TOLERANCE)

    # Test some different redshifts
    for onez in [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]:
        _rad_to_mpc_helper(0.33, onez, cosmo, do_inverse=False)
        _rad_to_mpc_helper(1.0, onez, cosmo, do_inverse=True)

    # Test some different H0
    for oneh0 in [30., 50., 67.3, 74.7, 100.]:
        _rad_to_mpc_helper(0.33, 0.5, FlatLambdaCDM(H0=oneh0, Om0=0.3), do_inverse=False)
        _rad_to_mpc_helper(1.0, 0.5, FlatLambdaCDM(H0=oneh0, Om0=0.3), do_inverse=True)

    # Test some different Omega_M
    for oneomm in [0.1, 0.3, 0.5, 1.0]:
        _rad_to_mpc_helper(0.33, 0.5, FlatLambdaCDM(H0=70., Om0=oneomm), do_inverse=False)
        _rad_to_mpc_helper(1.0, 0.5, FlatLambdaCDM(H0=70., Om0=oneomm), do_inverse=True)


def test_convert_units():
    """ Test the wrapper function to convert units. Corner cases should be tested in the
    individual functions. This function should test one case for all supported conversions
    and the error handling.
    """
    # Make an astropy cosmology object for testing
    cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

    # Test that each unit is supported
    utils.convert_units(1.0, 'radians', 'degrees')
    utils.convert_units(1.0, 'arcmin', 'arcsec')
    utils.convert_units(1.0, 'Mpc', 'kpc')
    utils.convert_units(1.0, 'Mpc', 'kpc')

    # Error checking
    assert_raises(ValueError, utils.convert_units, 1.0, 'radians', 'CRAZY')
    assert_raises(ValueError, utils.convert_units, 1.0, 'CRAZY', 'radians')
    assert_raises(TypeError, utils.convert_units, 1.0, 'arcsec', 'Mpc')
    assert_raises(TypeError, utils.convert_units, 1.0, 'arcsec', 'Mpc', None, cosmo)
    assert_raises(TypeError, utils.convert_units, 1.0, 'arcsec', 'Mpc', 0.5, None)

    # Test cases to make sure angular -> angular is fitting together
    assert_allclose(utils.convert_units(np.pi, 'radians', 'degrees'), 180., **TOLERANCE)
    assert_allclose(utils.convert_units(180.0, 'degrees', 'radians'), np.pi, **TOLERANCE)
    assert_allclose(utils.convert_units(1.0, 'degrees', 'arcmin'), 60., **TOLERANCE)
    assert_allclose(utils.convert_units(1.0, 'degrees', 'arcsec'), 3600., **TOLERANCE)

    # Test cases to make sure physical -> physical is fitting together
    assert_allclose(utils.convert_units(1.0, 'Mpc', 'kpc'), 1.0e3, **TOLERANCE)
    assert_allclose(utils.convert_units(1000., 'kpc', 'Mpc'), 1.0, **TOLERANCE)
    assert_allclose(utils.convert_units(1.0, 'Mpc', 'pc'), 1.0e6, **TOLERANCE)

    # Test conversion from angular to physical
    # Using astropy, circular now but this will be fine since we are going to be
    # swapping to CCL soon and then its kosher
    r_arcmin, redshift = 20.0, 0.5
    d_a = cosmo.angular_diameter_distance(redshift).to('kpc').value
    truth = r_arcmin * (1.0 / 60.0) * (np.pi / 180.0) * d_a
    assert_allclose(utils.convert_units(r_arcmin, 'arcmin', 'kpc', redshift, cosmo),
                    truth, **TOLERANCE)

    # Test conversion both ways between angular and physical units
    # Using astropy, circular now but this will be fine since we are going to be
    # swapping to CCL soon and then its kosher
    r_kpc, redshift = 20.0, 0.5
    d_a = cosmo.angular_diameter_distance(redshift).to('kpc').value
    truth = r_kpc * (1.0 / d_a) * (180. / np.pi) * 60.
    assert_allclose(utils.convert_units(r_kpc, 'kpc', 'arcmin', redshift, cosmo),
                    truth, **TOLERANCE)
