"""Tests for polaraveraging.py"""
import clmm
import clmm.utils as utils
from numpy import testing
import numpy as np
from astropy.table import Table
import os
import pytest

rtol = 1.e-6

# def test_make_bins():
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


def test_compute_radial_averages():
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
    rtest, ytest, yerr_std, countstest = utils._compute_radial_averages(dists, vals, bins,
                                                                     error_model='std')
    _, _, yerr_stdn, _= utils._compute_radial_averages(dists, vals, bins, error_model='std/n')
    testing.assert_allclose(rtest, np.mean(dists), rtol)
    testing.assert_allclose(ytest, np.mean(vals), rtol)
    testing.assert_allclose(yerr_std, np.std(vals), rtol)
    testing.assert_allclose(countstest, len(vals), rtol)
    testing.assert_allclose(yerr_stdn, np.std(vals)/countstest, rtol)


def test_theta_units_conversion():
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

    # Test for radians to kpc

    # Test for radians to Mpc

    # THIS WILL NOT SUPPORT PHYSICAL UNITS TO ANGULAR UNITS!!!


