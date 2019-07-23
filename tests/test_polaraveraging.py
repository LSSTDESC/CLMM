"""
Tests for polaraveraging.py
"""
from numpy import testing
#import clmm
from astropy.table import Table
import os
import polaraveraging
from polaraveraging import *

def test_make_bins():
    ## do something

    testing.assert_equal(len( polaraveraging._make_bins(1,10,9,False)),10 )
    testing.assert_allclose( polaraveraging._make_bins(1,10,9,False) , np.arange(1.,11.) )
    testing.assert_allclose( polaraveraging._make_bins(1,10000,4,True) ,10.**(np.arange(5)) )
    
    testing.assert_raises(TypeError, polaraveraging._make_bins, rmin='glue', rmax=10, n_bins=9, log_bins=False)
    testing.assert_raises(TypeError, polaraveraging._make_bins, rmin=1, rmax='glue', n_bins=9, log_bins=False)
    testing.assert_raises(TypeError, polaraveraging._make_bins, rmin=1, rmax=10, n_bins='glue', log_bins=False)
    testing.assert_raises(TypeError, polaraveraging._make_bins, rmin=1, rmax=10, n_bins=9, log_bins='glue')

    testing.assert_raises(ValueError, polaraveraging._make_bins, rmin=1, rmax=10, n_bins=-4, log_bins=False)
    testing.assert_raises(ValueError, polaraveraging._make_bins, rmin=1, rmax=-10, n_bins=9, log_bins=False)
    testing.assert_raises(ValueError, polaraveraging._make_bins, rmin=1, rmax=10, n_bins=0, log_bins=False)
    testing.assert_raises(TypeError, polaraveraging._make_bins, rmin=1, rmax=10, n_bins=9.9, log_bins=False)

    
    


if __name__ == "__main__":
    test_make_bins()
