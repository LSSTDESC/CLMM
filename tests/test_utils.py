""" Tests for utils.py """
from astropy.cosmology import FlatLambdaCDM
from numpy import testing
import numpy as np
from numpy.testing import assert_raises, assert_allclose
from astropy.table import Table
import astropy.units as u
import os
import pytest

import clmm
import clmm.utils as utils
from clmm.utils import compute_radial_averages

tolerance = {'rtol': 1.0e-6, 'atol': 0}


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
            [[np.mean(binvals)], [np.mean(binvals)], [np.std(binvals)/len(binvals)]], **tolerance)

    # Test 3 objects in one bin with various error models
    assert_allclose(compute_radial_averages(binvals, binvals, xbins1, error_model='std/n'),
            [[np.mean(binvals)], [np.mean(binvals)], [np.std(binvals)/len(binvals)]], **tolerance)
    assert_allclose(compute_radial_averages(binvals, binvals, xbins1, error_model='std'),
            [[np.mean(binvals)], [np.mean(binvals)], [np.std(binvals)]], **tolerance)

    # A slightly more complicated case with two bins
    inbin1 = binvals[(binvals > xbins2[0]) & (binvals < xbins2[1])]
    inbin2 = binvals[(binvals > xbins2[1]) & (binvals < xbins2[2])]
    assert_allclose(compute_radial_averages(binvals, binvals, xbins2, error_model='std/n'),
            [[np.mean(inbin1), np.mean(inbin2)], [np.mean(inbin1), np.mean(inbin2)],
             [np.std(inbin1)/len(inbin1), np.std(inbin2)/len(inbin2)]], **tolerance)
    assert_allclose(compute_radial_averages(binvals, binvals, xbins2, error_model='std'),
            [[np.mean(inbin1), np.mean(inbin2)], [np.mean(inbin1), np.mean(inbin2)],
             [np.std(inbin1), np.std(inbin2)]], **tolerance)

    # Test a much larger, random sample with unevenly spaced bins
    binvals = 10.0 * np.random.random(100)
    xbins2 = [0.0, 10.0 * np.random.random(), 10.0]
    inbin1 = binvals[(binvals > xbins2[0]) & (binvals < xbins2[1])]
    inbin2 = binvals[(binvals > xbins2[1]) & (binvals < xbins2[2])]
    assert_allclose(compute_radial_averages(binvals, binvals, xbins2, error_model='std/n'),
            [[np.mean(inbin1), np.mean(inbin2)], [np.mean(inbin1), np.mean(inbin2)],
             [np.std(inbin1)/len(inbin1), np.std(inbin2)/len(inbin2)]], **tolerance)
    assert_allclose(compute_radial_averages(binvals, binvals, xbins2, error_model='std'),
            [[np.mean(inbin1), np.mean(inbin2)], [np.mean(inbin1), np.mean(inbin2)],
             [np.std(inbin1), np.std(inbin2)]], **tolerance)



def test_make_bins():
    """ Matts comments for whoever addresses Issue 164.
    - These tests came 100% commented out. I have no idea why.
    - Lets just rewrite tests for this function, its pretty simple func
    - For each number below, a new option is passed. After testing the option, just pass it
      fixed to the default value.
    1. Pass just rmin and rmax, everything else default, checkout output
       A) rmin positive, rmax positive
       B) rmin positive, rmax negative
       C) rmin negative, rmax positive
       D) rmin negative, rmax negative
       E) rmin > rmax but both positive
       Note: It should break if either is negative I think?
       Note: From here you can assume 0 < rmin < rmax
    2. Pass rmin, rmax, n_bins, everything else default
       Note: We already tested rmin, rmax so just chose a reasonable value for each
       A) n_bins=-10
       B) n_bins=0
       C) n_bins=1
       D) n_bins=13 (just something larger than the default value)
    3. Pass rmin, rmax, n_bins=10, log10_bins, everything else default
       Note: We already tested rmin, rmax so just chose a reasonable value for each
       Note: We already tests n_bins so just manually set to default, n_bins=10
       A) log10_bins=True
       B) log10_bins=False
    4. Pass rmin, rmax, n_bins=10, log10_bins=False to test method keyword
       A) Use default
       B) set method='equal'
       Note: We want to test every line of code. If another method was added and after
             checking which method, we use an if statement to split on log10_bins, we
             would also want to check that method for log10_bins True and False
    """

    pass
#     testing.assert_equal(len( utils.make_bins(1,10,9,False)),10 )
#     testing.assert_allclose( utils.make_bins(1,10,9,False) , np.arange(1.,11.) )
#     testing.assert_allclose( utils.make_bins(1,10000,4,True) ,10.**(np.arange(5)) )
#     
#     testing.assert_raises(TypeError, utils.make_bins, rmin='glue', rmax=10, n_bins=9, log_bins=False)
#     testing.assert_raises(TypeError, utils.make_bins, rmin=1, rmax='glue', n_bins=9, log_bins=False)
#     testing.assert_raises(TypeError, utils.make_bins, rmin=1, rmax=10, n_bins='glue', log_bins=False)
#     testing.assert_raises(TypeError, utils.make_bins, rmin=1, rmax=10, n_bins=9, log_bins='glue')
#
#     testing.assert_raises(ValueError, utils.make_bins, rmin=1, rmax=10, n_bins=-4, log_bins=False)
#     testing.assert_raises(ValueError, utils.make_bins, rmin=1, rmax=-10, n_bins=9, log_bins=False)
#     testing.assert_raises(ValueError, utils.make_bins, rmin=1, rmax=10, n_bins=0, log_bins=False)
#     testing.assert_raises(TypeError, utils.make_bins, rmin=1, rmax=10, n_bins=9.9, log_bins=False)



def _rad_to_mpc_helper(dist, redshift, cosmo, do_inverse):
    """ Helper function to clean up test_convert_rad_to_mpc. Truth is computed using
    astropy so this test is very circular. Once we swap to CCL very soon this will be
    a good source of truth. """
    d_a = cosmo.angular_diameter_distance(redshift).to('Mpc').value
    print(d_a)
    if do_inverse: truth = dist / d_a
    else: truth = dist * d_a
    assert_allclose(utils._convert_rad_to_mpc(dist, redshift, cosmo, do_inverse),
                    truth, **tolerance)


def test_convert_rad_to_mpc():
    """ Test conversion between physical and angular units and vice-versa. """
    # Set some default values if I want them
    redshift = 0.25
    cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

    # Test the default behavior
    assert_allclose(utils._convert_rad_to_mpc(0.33, redshift, cosmo),
                    utils._convert_rad_to_mpc(0.33, redshift, cosmo, False), **tolerance)
                        
    # Test basic conversions each way
    _rad_to_mpc_helper(0.003, redshift, cosmo, do_inverse=False)
    _rad_to_mpc_helper(1.0, redshift, cosmo, do_inverse=True)

    # Convert back and forth and make sure I get the same answer
    midtest = utils._convert_rad_to_mpc(0.003, redshift, cosmo)
    assert_allclose(utils._convert_rad_to_mpc(midtest, redshift, cosmo, do_inverse=True),
                    0.003, **tolerance)

    # Test some different redshifts
    for z_ in [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]:
        _rad_to_mpc_helper(0.33, z_, cosmo, do_inverse=False)
        _rad_to_mpc_helper(1.0, z_, cosmo, do_inverse=True)

    # Test some different H0
    for H0_ in [30., 50., 67.3, 74.7, 100.]:
        _rad_to_mpc_helper(0.33, 0.5, FlatLambdaCDM(H0=H0_, Om0=0.3), do_inverse=False)
        _rad_to_mpc_helper(1.0, 0.5, FlatLambdaCDM(H0=H0_, Om0=0.3), do_inverse=True)

    # Test some different Omega_M
    for Om0_ in [0.1, 0.3, 0.5, 1.0]:
        _rad_to_mpc_helper(0.33, 0.5, FlatLambdaCDM(H0=70., Om0=Om0_), do_inverse=False)
        _rad_to_mpc_helper(1.0, 0.5, FlatLambdaCDM(H0=70., Om0=Om0_), do_inverse=True)


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
    assert_allclose(utils.convert_units(np.pi, 'radians', 'degrees'), 180., **tolerance)
    assert_allclose(utils.convert_units(180.0, 'degrees', 'radians'), np.pi, **tolerance)
    assert_allclose(utils.convert_units(1.0, 'degrees', 'arcmin'), 60., **tolerance)
    assert_allclose(utils.convert_units(1.0, 'degrees', 'arcsec'), 3600., **tolerance)

    # Test cases to make sure physical -> physical is fitting together
    assert_allclose(utils.convert_units(1.0, 'Mpc', 'kpc'), 1.0e3, **tolerance)
    assert_allclose(utils.convert_units(1000., 'kpc', 'Mpc'), 1.0, **tolerance)
    assert_allclose(utils.convert_units(1.0, 'Mpc', 'pc'), 1.0e6, **tolerance)

    # Test conversion from angular to physical
    # Using astropy, circular now but this will be fine since we are going to be
    # swapping to CCL soon and then its kosher
    r_arcmin, redshift = 20.0, 0.5
    d_a = cosmo.angular_diameter_distance(redshift).to('kpc').value
    truth = r_arcmin * (1.0 / 60.0) * (np.pi / 180.0) * d_a
    assert_allclose(utils.convert_units(r_arcmin, 'arcmin', 'kpc', redshift, cosmo),
                    truth, **tolerance)

    # Test conversion both ways between angular and physical units
    # Using astropy, circular now but this will be fine since we are going to be
    # swapping to CCL soon and then its kosher
    r_kpc, redshift = 20.0, 0.5
    d_a = cosmo.angular_diameter_distance(redshift).to('kpc').value
    truth = r_kpc * (1.0 / d_a) * (180. / np.pi) * 60.
    assert_allclose(utils.convert_units(r_kpc, 'kpc', 'arcmin', redshift, cosmo),
                    truth, **tolerance)
