"""
Tests for polaraveraging.py
"""
import clmm.polaraveraging as pa
from numpy import testing
import numpy as np
from astropy.table import Table
import os

def test_theta_g():
    ## do something
    pass





def test_compute_theta_phi():
    ra_l, dec_l = 161., 65.
    ra_s, dec_s = np.array([-355., 355.]), np.array([-85., 85.])
    rtol=1.e-7

    # Test type on inputs - In retrospect, I don't think these tests really matter much as
    # the math will crash on its own. They alll passed without adding any checks
    # testing.assert_raises(TypeError, pa._compute_theta_phi, str(ra_l), dec_l, ra_s, dec_s, 'flat')
    # testing.assert_raises(TypeError, pa._compute_theta_phi, ra_l, str(dec_l), ra_s, dec_s, 'flat')
    # testing.assert_raises(TypeError, pa._compute_theta_phi, ra_l, dec_l, str(ra_s), dec_s, 'flat')
    # testing.assert_raises(TypeError, pa._compute_theta_phi, ra_l, dec_l, ra_s, str(dec_s), 'flat')

    # Test domains on inputs
    testing.assert_raises(ValueError, pa._compute_theta_phi, ra_l, dec_l, ra_s, dec_s, 'phat')
    testing.assert_raises(ValueError, pa._compute_theta_phi, -365., dec_l, ra_s, dec_s, 'flat')
    testing.assert_raises(ValueError, pa._compute_theta_phi, 365., dec_l, ra_s, dec_s, 'flat')
    testing.assert_raises(ValueError, pa._compute_theta_phi, ra_l, 95., ra_s, dec_s, 'flat')
    testing.assert_raises(ValueError, pa._compute_theta_phi, ra_l, -95., ra_s, dec_s, 'flat')
    testing.assert_raises(ValueError, pa._compute_theta_phi, ra_l, dec_l, ra_s-10., dec_s, 'flat')
    testing.assert_raises(ValueError, pa._compute_theta_phi, ra_l, dec_l, ra_s+10., dec_s, 'flat')
    testing.assert_raises(ValueError, pa._compute_theta_phi, ra_l, dec_l, ra_s, dec_s-10., 'flat')
    testing.assert_raises(ValueError, pa._compute_theta_phi, ra_l, dec_l, ra_s, dec_s+10., 'flat')

    # Test outputs for reasonable values with flat sky
    ra_l, dec_l = 161.32, 51.49
    ra_s, dec_s = np.array([161.29, 161.34]), np.array([51.45, 51.55])
    thetas, phis = pa._compute_theta_phi(ra_l, dec_l, ra_s, dec_s, 'flat')
    testing.assert_allclose(thetas, np.array([0.00077050407583119666, 0.00106951489719733675]), rtol=rtol,
                            err_msg="Reasonable values with flat sky not matching to precision for theta")
    testing.assert_allclose(phis, np.array([-1.13390499136495481736, 1.77544123918164542530]), rtol=rtol,
                            err_msg="Reasonable values with flat sky not matching to precision for phi")

    # Test that flat sky remains the default
    thetas2, phis2 = pa._compute_theta_phi(ra_l, dec_l, ra_s, dec_s)
    testing.assert_allclose(thetas2, thetas, rtol=rtol, err_msg="Theta for default sky value not in agreement for precision")
    testing.assert_allclose(phis2, phis, rtol=rtol, err_msg="Phi for default sky value not in agreement for precision")

    # Test outputs for reasonable values with curved sky - Not implemented in the code!
    # theta, phi = pa._compute_theta_phi(ral, decl, ras, decs, 'curved')
    # testing.assert_allclose(theta, desired, rtol=rtol, err_msg="Reasonable values with curved sky not matching to precision for theta")
    # testing.assert_allclose(phi, desired, rtol=rtol, err_msg="Reasonable values with curved sky not matching to precision for phi")

    # Test outputs for edge cases 
    # ra/dec=0
    testing.assert_allclose(pa._compute_theta_phi(0.0, 0.0, ra_s, dec_s),
                            [[2.95479482616592248334, 2.95615695795537858359], [2.83280558128919901506, 2.83233281390148761147]],
                            rtol, err_msg="Failure when RA_lens=DEC_lens=0.0")
    # l/s at same ra or dec
    testing.assert_allclose(pa._compute_theta_phi(ra_l, dec_l, np.array([161.32, 161.34]), dec_s),
                            [[0.00069813170079771690, 0.00106951489719733675], [-1.57079632679489655800, 1.77544123918164542530]],
                            rtol, err_msg="Failure when lens and a source share an RA")
    testing.assert_allclose(pa._compute_theta_phi(ra_l, dec_l, ra_s, np.array([51.49, 51.55])),
                            [[0.00032601941539388962, 0.00106951489719733675], [0.00000000000000000000, 1.77544123918164542530]],
                            rtol, err_msg="Failure when lens and a source share a DEC")
    # l/s at same ra AND dec
    testing.assert_allclose(pa._compute_theta_phi(ra_l, dec_l, np.array([161.32, 161.34]), np.array([51.49, 51.55])),
                            [[0.00000000000000000000, 0.00106951489719733675], [0.00000000000000000000, 1.77544123918164542530]],
                            rtol, err_msg="Failure when lens and a source share an RA and a DEC")
    # ra1, ra2 = .1 and 359.9
    # testing.assert_allclose(pa._compute_theta_phi(), desired, rtol, err_msg="")
    # ra1, ra2 = 0, 180.1
    # testing.assert_allclose(pa._compute_theta_phi(), desired, rtol, err_msg="")
    # ra1, ra2 = -180, 180
    # testing.assert_allclose(pa._compute_theta_phi(), desired, rtol, err_msg="")
    # dec1, dec2 = 90, -90
    # testing.assert_allclose(pa._compute_theta_phi(), desired, rtol, err_msg="")



# if __name__ == "__main__":
#     test_theta_g()
