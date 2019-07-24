"""
Tests for polaraveraging.py
"""
import clmm
import clmm.polaraveraging as pa
from numpy import testing
import numpy as np
from astropy.table import Table
import os

def test_make_bins():
    ## do something
    testing.assert_equal(len( pa.make_bins(1,10,9,False)),10 )
    testing.assert_allclose( pa.make_bins(1,10,9,False) , np.arange(1.,11.) )
    testing.assert_allclose( pa.make_bins(1,10000,4,True) ,10.**(np.arange(5)) )
    
    testing.assert_raises(TypeError, pa.make_bins, rmin='glue', rmax=10, n_bins=9, log_bins=False)
    testing.assert_raises(TypeError, pa.make_bins, rmin=1, rmax='glue', n_bins=9, log_bins=False)
    testing.assert_raises(TypeError, pa.make_bins, rmin=1, rmax=10, n_bins='glue', log_bins=False)
    testing.assert_raises(TypeError, pa.make_bins, rmin=1, rmax=10, n_bins=9, log_bins='glue')

    testing.assert_raises(ValueError, pa.make_bins, rmin=1, rmax=10, n_bins=-4, log_bins=False)
    testing.assert_raises(ValueError, pa.make_bins, rmin=1, rmax=-10, n_bins=9, log_bins=False)
    testing.assert_raises(ValueError, pa.make_bins, rmin=1, rmax=10, n_bins=0, log_bins=False)
    testing.assert_raises(TypeError, pa.make_bins, rmin=1, rmax=10, n_bins=9.9, log_bins=False)

def test_compute_g_x():
    pass
    data = np.array([[0.01, 0.02, 0.01], # g1
                     [0.01, 0.02, 0.03], # g2
                     [0.1, 1.2, 3.]]) # phi [rad]

    # test that function works for scalar and vector input
    assert(isinstance(pa._compute_g_x(*data[:,0]), float))
    assert(isinstance(pa._compute_g_x(*data), np.ndarray))
    testing.assert_equal(3, len(pa._compute_g_x(*data)))
    testing.assert_equal(pa._compute_g_x(*(data[:,0])), pa._compute_g_x(*data)[0])
    
    # test same length array input
    testing.assert_raises(ValueError, pa._compute_g_x, data[0,0], data[1], data[2])
    testing.assert_raises(ValueError, pa._compute_g_x, data[0], data[1,0], data[2])
    testing.assert_raises(ValueError, pa._compute_g_x, data[0], data[1], data[2,0])
    testing.assert_raises(ValueError, pa._compute_g_x, data[0,0], data[1,0], data[2])
    testing.assert_raises(ValueError, pa._compute_g_x, data[0], data[1,0], data[2,0])
    testing.assert_raises(ValueError, pa._compute_g_x, data[0,0], data[1], data[2,0])
    
    # test for input range
    testing.assert_raises(ValueError, pa._compute_g_x, 0.1, 0.1, -3.15)
    testing.assert_raises(ValueError, pa._compute_g_x, 0.1, 0.1, 2.*np.pi+0.1)
    testing.assert_almost_equal(pa._compute_g_x(0.1, 0.1, 0.), pa._compute_g_x(0.1, 0.1, np.pi))

    
    # test for reasonable values
    testing.assert_almost_equal(pa._compute_g_x(100., 0., 0.), 0.0)
    testing.assert_almost_equal(pa._compute_g_x(0., 100., np.pi/4.), 0.0)
    testing.assert_almost_equal(pa._compute_g_x(0., 0., 0.3), 0.)

def test_compute_g_t():
    data = np.array([[0.01, 0.02, 0.01], # g1
                     [0.01, 0.02, 0.03], # g2
                     [0.1, 1.2, 3.]]) # phi [rad]

    # test that function works for scalar and vector input
#    testing.assert(isinstance(float, pa._compute_g_t(*(data[:,0]))))
#    testing.assert(isinstance(np.array, pa._compute_g_t(*data)))
    testing.assert_equal(3, len(pa._compute_g_t(*data)))
    testing.assert_equal(pa._compute_g_t(*(data[:,0])), pa._compute_g_t(*data)[0])
    
    # test same length array input
    testing.assert_raises(ValueError, pa._compute_g_t, data[0,0], data[1], data[2])
    testing.assert_raises(ValueError, pa._compute_g_t, data[0], data[1,0], data[2])
    testing.assert_raises(ValueError, pa._compute_g_t, data[0], data[1], data[2,0])
    testing.assert_raises(ValueError, pa._compute_g_t, data[0,0], data[1,0], data[2])
    testing.assert_raises(ValueError, pa._compute_g_t, data[0], data[1,0], data[2,0])
    testing.assert_raises(ValueError, pa._compute_g_t, data[0,0], data[1], data[2,0])
    
    # test for input range
    testing.assert_raises(ValueError, pa._compute_g_t, 0.1, 0.1, -3.15)
    testing.assert_raises(ValueError, pa._compute_g_t, 0.1, 0.1, 2.*np.pi+0.1)
    testing.assert_almost_equal(pa._compute_g_t(0.1, 0.1, 0.), pa._compute_g_t(0.1, 0.1, np.pi))
    
    # test for reasonable values
    testing.assert_almost_equal(pa._compute_g_t(100., 0., 0.), -100.0)
    testing.assert_almost_equal(pa._compute_g_t(0., 100., np.pi/4.), -100.0)
    testing.assert_almost_equal(pa._compute_g_t(0., 0., 0.3), 0.)

    
def test_compute_radial_averages():

    #testing input types
    testing.assert_raises(TypeError, pa._compute_radial_averages, radius="glue", g=10, bins=[np.arange(1.,16.)])
    testing.assert_raises(TypeError, pa._compute_radial_averages, radius=np.arange(1.,10.), g="glue", bins=[np.arange(1.,16.)])  
    testing.assert_raises(TypeError, pa._compute_radial_averages, radius=np.arange(1.,10.), g=np.arange(1.,10.), bins='glue') 

    #want radius and g to have same number of entries
    testing.assert_raises(TypeError, pa._compute_radial_averages, radius=np.arange(1.,10.), g=np.arange(1.,7.), bins=[np.arange(1.,16.)])

    #want binning to encompass entire radial range
    testing.assert_raises(ValueError, pa._compute_radial_averages, radius=np.arange(1.,10.), g=np.arange(1.,10.), bins=[1,6,7])
    testing.assert_raises(ValueError, pa._compute_radial_averages, radius=np.arange(1.,6.), g=np.arange(1.,6.), bins=[5,6,7]) 

    

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
    testing.assert_raises(ValueError, pa._compute_theta_phi, ra_l, dec_l, np.array([161.32, 161.34]), np.array([51.49, 51.55]))
    
    #testing.assert_allclose(pa._compute_theta_phi(ra_l, dec_l, np.array([161.32, 161.34]), np.array([51.49, 51.55])),
    #                        [[0.00000000000000000000, 0.00106951489719733675], [0.00000000000000000000, 1.77544123918164542530]],
    #                        rtol, err_msg="Failure when lens and a source share an RA and a DEC")
    
    # ra1, ra2 = .1 and 359.9
    # testing.assert_allclose(pa._compute_theta_phi(), desired, rtol, err_msg="")
    # ra1, ra2 = 0, 180.1
    # testing.assert_allclose(pa._compute_theta_phi(), desired, rtol, err_msg="")
    # ra1, ra2 = -180, 180
    # testing.assert_allclose(pa._compute_theta_phi(), desired, rtol, err_msg="")
    # dec1, dec2 = 90, -90
    # testing.assert_allclose(pa._compute_theta_phi(), desired, rtol, err_msg="")

def test_compute_shear():
    # the numbers on this test must be revise, output values were extracted from the first run of the code
    g1, g2 = np.ones(2), np.ones(2)
    rtol = 1e-6
    ra_l, dec_l = 161.32, 51.49
    ra_s, dec_s = np.array([161.29, 161.34]), np.array([51.45, 51.55])
    testing.assert_allclose(
        pa._compute_shear(ra_l, dec_l, ra_s, dec_s, g1, g2, sky='flat'),
        np.array([[0.00077050407583119666, 0.00106951489719733675],
            [ 1.408693e+00,  1.315362e+00],
            [ 1.248319e-01, -5.194458e-01]]
            ),
        rtol=rtol)

def test_gc_wrapper():
    # the numbers on this test must be revise, output values were extracted from the first run of the code
    ra_in = np.array([161.29, 161.34])
    dec_in =  np.array([51.45, 51.55])
    theta_out = [0.00077050407583119666, 0.00106951489719733675]
    gt_out = [ 1.408693e+00,  1.315362e+00]
    gx_out = [ 1.248319e-01, -5.194458e-01]
    gc = clmm.GalaxyCluster(unique_id='blah', 
        ra=161.32, dec=51.49, z=0.5,
        galcat=Table([
            np.append(ra_in, ra_in),
            np.append(dec_in, dec_in),
            np.ones(4), np.ones(4)],
            names=('ra', 'dec', 'e1', 'e2'))
        )
    #test compute_shear
    gc.compute_shear()
    testing.assert_allclose(
        [gc.galcat[c] for c in ('theta', 'gt', 'gx')],
        [np.append(theta_out, theta_out),
            np.append(gt_out, gt_out),
            np.append(gx_out, gx_out)],
        rtol=1e-6)
    #test make_shear_profile
    gc.make_shear_profile('rad', bins=[0, .001, .002])
    testing.assert_allclose(
        [gc.profile[c] for c in ('radius', 'gt', 'gx')],
        [theta_out, gt_out, gx_out],
        rtol=1e-6)

if __name__ == "__main__":
    #test_make_bins()
    #test_compute_g_x()
    #test_compute_g_t()
    test_compute_radial_averages()
    test_compute_shear()
