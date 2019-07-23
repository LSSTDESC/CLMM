"""
Tests for polaraveraging.py
"""
from numpy import testing
import clmm
from astropy.table import Table
import os

def test_theta_g():
    ## do something
    pass





def test_compute_theta_phi():
    ra_l, dec_l = 161., 65.
    ra_s, dec_s = 163., 53.

    # Test type on inputs
    testing.assert_raises(TypeError, clmm._compute_theta_phi, str(ra_l), dec_l, ra_s, dec_s, 'flat')
    testing.assert_raises(TypeError, clmm._compute_theta_phi, ra_l, str(dec_l), ra_s, dec_s, 'flat')
    testing.assert_raises(TypeError, clmm._compute_theta_phi, ra_l, dec_l, str(ra_s), dec_s, 'flat')
    testing.assert_raises(TypeError, clmm._compute_theta_phi, ra_l, dec_l, ra_s, str(dec_s), 'flat')

    # Test domains on inputs
    testing.assert_raises(ValueError, clmm._compute_theta_phi, ra_l, dec_l, ra_s, dec_s, 'phat')
    testing.assert_raises(ValueError, clmm._compute_theta_phi, ra_l, dec_l, ra_s, dec_s, 'flat')
    testing.assert_raises(ValueError, clmm._compute_theta_phi, ra_l, dec_l, ra_s, dec_s, 'flat')
    testing.assert_raises(ValueError, clmm._compute_theta_phi, ra_l, dec_l, ra_s, dec_s, 'flat')
    testing.assert_raises(ValueError, clmm._compute_theta_phi, ra_l, dec_l, ra_s, dec_s, 'flat')

    # Test outputs for some realistic values, including one with and without a default option

    # Test outputs for array_like source positions

    # Test outputs for edge cases 
    # ra/dec=0
    # l/s at same ra or dec
    # l/s at same ra AND dec
    # ra1, ra2 = .1 and 359.9
    # ra1, ra2 = 0, 180.1
    # ra1, ra2 = -180, 180
    # dec1, dec2 = 90, -90



if __name__ == "__main__":
    test_theta_g()
