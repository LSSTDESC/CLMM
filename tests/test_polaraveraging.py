"""
Tests for polaraveraging.py
"""
from numpy import testing
import clmm
from astropy.table import Table
import os

def test_theta_g():
    ## do something


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
    test_theta_g()
