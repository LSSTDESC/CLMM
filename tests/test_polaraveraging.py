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

    testing.assert_raises(TypeError, polaraveraging._make_bins, rmin='glue', rmax=10, n_bins=9, log_bins=False)
    testing.assert_raises(TypeError, polaraveraging._make_bins, rmin=1, rmax='glue', n_bins=9, log_bins=False)
    testing.assert_raises(TypeError, polaraveraging._make_bins, rmin=1, rmax=10, n_bins='glue', log_bins=False)
#    testing.assert_raises(TypeError, polaraveraging._make_bins, rmin=1, rmax=10, n_bins=9, log_bins='glue')

    testing.assert_raises(ValueError, polaraveraging._make_bins, rmin=1, rmax=10, n_bins=-4, log_bins=False)
    testing.assert_raises(ValueError, polaraveraging._make_bins, rmin=1, rmax=-10, n_bins=9, log_bins=False)
  #  testing.assert_raises(ValueError, polaraveraging._make_bins, rmin=1, rmax=10, n_bins=0, log_bins=False)
   # testing.assert_raises(ValueError, polaraveraging._make_bins, rmin=1, rmax=10, n_bins=9, log_bins=4)

    
    


if __name__ == "__main__":
    test_make_bins()
