"""Tests for polaraveraging.py"""
import clmm
import clmm.utils as utils
from numpy import testing
import numpy as np
from astropy.table import Table
import astropy.units as u
import os
import pytest

rtol = 1.e-6


# Commented tests should be made to work or removed if nonsensical
# Ensure full coverage of all corner cases etc
def test_compute_radial_averages():
    """ Matt Comments for whoever addresses Issue 164.
    - Most of this is probably meh.
    - Lets write a test to
    1. Given a couple of points all in the same bin, compute on paper the answer and check
       A) Correct answer with no error_model (so, the default)
       B) Correct answer with error_model='std'
       C) Correct answer with error_model='std/n'
    2. Repeat step 1 for a couple of points split in between two bins.
       A) first 3 obj in bin1, second 3 in bin2 for example
       B) bin1, bin2, bin1, bin2, etc... Just shuffle the order
    3. Error checking like what is done is fine. BUT if we decide to do it we NEED to be complete
    """
    #testing input types
    testing.assert_raises(TypeError, utils._compute_radial_averages, radius="glue", g=10,
                          bins=[np.arange(1.,16.)])
    testing.assert_raises(TypeError, utils._compute_radial_averages, radius=np.arange(1.,10.),
                          g="glue", bins=[np.arange(1.,16.)])  
    testing.assert_raises(TypeError, utils._compute_radial_averages, radius=np.arange(1.,10.),
                          g=np.arange(1.,10.), bins='glue') 

    #want radius and g to have same number of entries
    testing.assert_raises(TypeError, utils._compute_radial_averages, radius=np.arange(1.,10.),
                          g=np.arange(1.,7.), bins=[np.arange(1.,16.)])

    #want binning to encompass entire radial range
    # testing.assert_raises(UserWarning, utils._compute_radial_averages, radius=np.arange(1.,10.),
    #                       g=np.arange(1.,10.), bins=[1,6,7])
    # testing.assert_raises(UserWarning, utils._compute_radial_averages, radius=np.arange(1.,6.),
    #                       g=np.arange(1.,6.), bins=[5,6,7])

    # Test that that we get the expected outputs
    bins = [0.5, 1.0]
    dists = np.hstack([.7*np.ones(5), .8*np.ones(5)])
    vals = np.arange(1, 11, 1)
    rtest, ytest, yerr_std = utils._compute_radial_averages(dists, vals, bins, error_model='std')
    _, _, yerr_stdn = utils._compute_radial_averages(dists, vals, bins, error_model='std/n')
    testing.assert_allclose(rtest, np.mean(dists), rtol)
    testing.assert_allclose(ytest, np.mean(vals), rtol)
    testing.assert_allclose(yerr_std, np.std(vals), rtol)
    testing.assert_allclose(yerr_stdn, np.std(vals)/len(vals), rtol)


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


def angular_conversion_helper(dist1, unit1_string, unit1_astropy):
    """ Matts comments for whoever addresses Issue 164.
    - The the conversion tests should aim to test every permutation of supported units
    - This function takes a distance, a string describing the unit (that is passe to
      our function) and an astropy unit that is used to compute the truth.
    - This function isn't run as a test since it doesn't begin with `test`. Cool, eh!
    """
    testing.assert_allclose(utils._convert_angular_units(dist1, unit1_string, 'degrees'),
                            (dist1 * unit1_astropy).to(u.deg).value,
                            rtol)
    testing.assert_allclose(utils._convert_angular_units(dist1, unit1_string, 'radians'),
                            (dist1 * unit1_astropy).to(u.rad).value,
                            rtol)
    testing.assert_allclose(utils._convert_angular_units(dist1, unit1_string, 'arcmin'),
                            (dist1 * unit1_astropy).to(u.arcmin).value,
                            rtol)
    testing.assert_allclose(utils._convert_angular_units(dist1, unit1_string, 'arcsec'),
                            (dist1 * unit1_astropy).to(u.arcsec).value,
                            rtol)



def test_convert_angular_units():
    """ Matts comments for whoever addresses Issue 164.
    - This is sufficiently tested to me.
    - For a handful of angles, I test every permutation of the conversion
    - If you think of anything else, do not hesistate to add it! More test good
    """
    # Test conversion from degrees
    dist1_deg = np.array([0.0, 3.0, 15.0, 45.0, 90.0, 180.0])
    angular_conversion_helper(dist1_deg, 'degrees', u.deg)
    
    # Test conversion from radians
    dist1_rad = np.pi * np.array([0.0, 0.1, 0.333, 0.5, 0.8, 1.0])
    angular_conversion_helper(dist1_rad, 'radians', u.rad)

    # Test conversion from arcmin
    dist1_am = dist1_deg * 60.
    angular_conversion_helper(dist1_am, 'arcmin', u.arcmin)

    # Test conversion from arcsec
    dist1_as = dist1_am * 60.
    angular_conversion_helper(dist1_as, 'arcsec', u.arcsec)


def test_convert_physical_units():
    """ Matts comments for whoever addresses Issue 164.
    - This should be very similar to test_convert_angular_units above but for physical units
    - Write a helper function like the angular case
    1. Helper function should convert input to (all options in func)
       A) pc
       B) kpc
       C) Mpc
    2. Pass a list of distances into helper and assert
       A) pass in pc
       B) pass in kpc
       C) pass in Mpc
    """
    # Test each conversion between physical units
    pass


def test_convert_between_rad_mpc():
    """ Matts comments for whoever addresses Issue 164.
    - This converts radians to Mpc and vice versa
    1. Convert rad to Mpc
    2. Convert Mpc to rad
    3. Convert rad to Mpc to rad (should be the same)
    4. Test rad to Mpc
       Note: we know the conversions generally work, to test corner cases lets just go one way
       A) z = 0.0, flat LCDM OmM=0.3, H0=70
       B) z = 0.5, flat LCDM OmM=0.3, H0=70
    5. Test some cosmos, (set z=0.5 to make it a little more interesting)
       A) Support only flat??? idk
       B) H0 = 50, 70, 100
       C) OmM = 0.1, 0.25, 0.4, 1.0
    """
    pass


def test_convert_units():
    """ Matts comments for whoever addresses Issue 164.
    - Since we have tested all the parts, we just need to test that it all fits together here
    1. What if we pass in unsupported units?
    2. angular->physics (or vice versa) without a redshift? without a cosmology?
    3. One case of angular -> angular units
    4. One case of physical -> physical units
    5. One case of angular -> physical
    6. One case of physical -> angular
    """
    pass
    # Test that we get the appropriate errors when we pass in unsupported units

    # Test that we get the appropriate errors when no redshift or cosmology 
    # when physical units are involved

    # Test angular to angular example

    # Test physical to physical example

    # Test angular to physical example

    # Test physical to angular example



def test_theta_units_conversion():
    """ Matts comments for whoever addresses Issue 164.
    - This function is what is still used in the code so I left the tests
    - Once all of the above tests are written and the code is refactored, this can be scrapped.
    """
    # tests for invaid input
    testing.assert_raises(ValueError, utils._theta_units_conversion, np.pi, 'radians', 'crazy units')
    testing.assert_raises(ValueError, utils._theta_units_conversion, np.pi, 'crazy units', 'radians')
    testing.assert_raises(ValueError, utils._theta_units_conversion, np.pi, 'radians', 'Mpc')
    testing.assert_raises(ValueError, utils._theta_units_conversion, np.pi, 'radians', 'Mpc', 0.5)

    # tests for angular conversion
    testing.assert_equal(utils._theta_units_conversion(np.pi, 'radians', 'radians'), np.pi)
    testing.assert_almost_equal(utils._theta_units_conversion(np.pi, 'radians', 'deg'), 180.)
    testing.assert_almost_equal(utils._theta_units_conversion(np.pi, 'radians', 'arcmin'), 180.*60)
    testing.assert_almost_equal(utils._theta_units_conversion(np.pi, 'radians', 'arcsec'), 180.*60*60)
    testing.assert_almost_equal(utils._theta_units_conversion(180., 'deg', 'radians'), np.pi)


