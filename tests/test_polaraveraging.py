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

    
    


def test_compute_g_x():
    data = np.array([[0.01, 0.02, 0.01], # g1
                     [0.01, 0.02, 0.03], # g2
                     [1.0, 2.0, 0.5]]) # phi

    # Test that function works for scalar and vector input
    testing.assert(isinstance(float, clmm._compute_g_x(*(data[:,0]))))
    testing.assert(isinstance(np.array, clmm._compute_g_x(*data)))
    testing.assert_equal(3, len(clmm._compute_g_x(*data)))
    testing.assert_equal(clmm._compute_g_x(*(data[:,0])), clmm._compute_g_x(*data)[0])
    
    # test same length array input
    testing.assert_raises(ValueError, clmm._compute_g_x, data[0,0], data[1], data[2])
    testing.assert_raises(ValueError, clmm._compute_g_x, data[0], data[1,0], data[2])
    testing.assert_raises(ValueError, clmm._compute_g_x, data[0], data[1], data[2,0])
    testing.assert_raises(ValueError, clmm._compute_g_x, data[0,0], data[1,0], data[2])
    testing.assert_raises(ValueError, clmm._compute_g_x, data[0], data[1,0], data[2,0])
    testing.assert_raises(ValueError, clmm._compute_g_x, data[0,0], data[1], data[2,0])
    
    # test for phi=0 (reasonable values)
    # phi=0, phi=90
    # g1, g2 = 0
    # phi <0, >180, also =
    

if __name__ == "__main__":
    test_make_bins()
