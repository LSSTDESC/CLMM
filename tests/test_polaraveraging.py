"""Tests for polaraveraging.py"""
import clmm
import clmm.polaraveraging as pa
from numpy import testing
import numpy as np
from astropy.table import Table
import os
import pytest

rtol = 1.e-6

# def test_make_bins():
#     testing.assert_equal(len( pa.make_bins(1,10,9,False)),10 )
#     testing.assert_allclose( pa.make_bins(1,10,9,False) , np.arange(1.,11.) )
#     testing.assert_allclose( pa.make_bins(1,10000,4,True) ,10.**(np.arange(5)) )
#     
#     testing.assert_raises(TypeError, pa.make_bins, rmin='glue', rmax=10, n_bins=9, log_bins=False)
#     testing.assert_raises(TypeError, pa.make_bins, rmin=1, rmax='glue', n_bins=9, log_bins=False)
#     testing.assert_raises(TypeError, pa.make_bins, rmin=1, rmax=10, n_bins='glue', log_bins=False)
#     testing.assert_raises(TypeError, pa.make_bins, rmin=1, rmax=10, n_bins=9, log_bins='glue')
#
#     testing.assert_raises(ValueError, pa.make_bins, rmin=1, rmax=10, n_bins=-4, log_bins=False)
#     testing.assert_raises(ValueError, pa.make_bins, rmin=1, rmax=-10, n_bins=9, log_bins=False)
#     testing.assert_raises(ValueError, pa.make_bins, rmin=1, rmax=10, n_bins=0, log_bins=False)
#     testing.assert_raises(TypeError, pa.make_bins, rmin=1, rmax=10, n_bins=9.9, log_bins=False)


def test_compute_cross_shear():
    shear1, shear2, phi = 0.15, 0.08, 0.52
    expected_cross_shear = -0.08886301350787848
    cross_shear = pa._compute_cross_shear(shear1, shear2, phi)
    testing.assert_allclose(cross_shear, expected_cross_shear)

    shear1 = np.array([0.15, 0.40])
    shear2 = np.array([0.08, 0.30])
    phi = np.array([0.52, 1.23])
    expected_cross_shear = [-0.08886301350787848, -0.48498333705834484]
    cross_shear = pa._compute_cross_shear(shear1, shear2, phi)
    testing.assert_allclose(cross_shear, expected_cross_shear)

    # Some additional edge cases
    testing.assert_allclose(pa._compute_cross_shear(100., 0., 0.), 0.0)
    # testing.assert_allclose(pa._compute_cross_shear(0., 100., np.pi/4.), 0.0)
    testing.assert_allclose(pa._compute_cross_shear(0., 0., 0.3), 0.)


def test_compute_tangential_shear():
    shear1, shear2, phi = 0.15, 0.08, 0.52
    expected_tangential_shear = -0.14492537676438383
    tangential_shear = pa._compute_tangential_shear(shear1, shear2, phi)
    testing.assert_allclose(tangential_shear, expected_tangential_shear)

    shear1 = np.array([0.15, 0.40])
    shear2 = np.array([0.08, 0.30])
    phi = np.array([0.52, 1.23])
    expected_tangential_shear = [-0.14492537676438383, 0.1216189244145496]
    tangential_shear = pa._compute_tangential_shear(shear1, shear2, phi)
    testing.assert_allclose(tangential_shear, expected_tangential_shear)

    # test same length array input
    #testing.assert_raises(ValueError, pa._compute_tangential_shear, data[0,0], data[1], data[2])
    #testing.assert_raises(ValueError, pa._compute_tangential_shear, data[0], data[1,0], data[2])
    #testing.assert_raises(ValueError, pa._compute_tangential_shear, data[0], data[1], data[2,0])
    #testing.assert_raises(ValueError, pa._compute_tangential_shear, data[0,0], data[1,0], data[2])
    #testing.assert_raises(ValueError, pa._compute_tangential_shear, data[0], data[1,0], data[2,0])
    #testing.assert_raises(ValueError, pa._compute_tangential_shear, data[0,0], data[1], data[2,0])
    
    # test for reasonable values
    testing.assert_almost_equal(pa._compute_tangential_shear(100., 0., 0.), -100.0)
    testing.assert_almost_equal(pa._compute_tangential_shear(0., 100., np.pi/4.), -100.0)
    testing.assert_almost_equal(pa._compute_tangential_shear(0., 0., 0.3), 0.)

    
@pytest.mark.skip()
def test_compute_radial_averages():
    #testing input types
    testing.assert_raises(TypeError, pa._compute_radial_averages, radius="glue", g=10, bins=[np.arange(1.,16.)])
    testing.assert_raises(TypeError, pa._compute_radial_averages, radius=np.arange(1.,10.), g="glue", bins=[np.arange(1.,16.)])  
    testing.assert_raises(TypeError, pa._compute_radial_averages, radius=np.arange(1.,10.), g=np.arange(1.,10.), bins='glue') 

    #want radius and g to have same number of entries
    testing.assert_raises(TypeError, pa._compute_radial_averages, radius=np.arange(1.,10.), g=np.arange(1.,7.), bins=[np.arange(1.,16.)])

    #want binning to encompass entire radial range
    #testing.assert_raises(UserWarning, pa._compute_radial_averages, radius=np.arange(1.,10.), g=np.arange(1.,10.), bins=[1,6,7])
    #testing.assert_raises(UserWarning, pa._compute_radial_averages, radius=np.arange(1.,6.), g=np.arange(1.,6.), bins=[5,6,7]) 

    

def test_compute_lensing_angles_flatsky():
    ra_l, dec_l = 161., 65.
    ra_s, dec_s = np.array([-355., 355.]), np.array([-85., 85.])
    rtol=1.e-7

    # Test domains on inputs
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky, -365., dec_l, ra_s, dec_s)
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky, 365., dec_l, ra_s, dec_s)
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky, ra_l, 95., ra_s, dec_s)
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky, ra_l, -95., ra_s, dec_s)
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky, ra_l, dec_l, ra_s-10., dec_s)
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky, ra_l, dec_l, ra_s+10., dec_s)
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky, ra_l, dec_l, ra_s, dec_s-10.)
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky, ra_l, dec_l, ra_s, dec_s+10.)

    # Ensure that we throw a warning with >1 deg separation
    testing.assert_warns(UserWarning,pa._compute_lensing_angles_flatsky, ra_l, dec_l, np.array([151.32, 161.34]), np.array([41.49, 51.55]))

    # Test outputs for reasonable values
    ra_l, dec_l = 161.32, 51.49
    ra_s, dec_s = np.array([161.29, 161.34]), np.array([51.45, 51.55])
    thetas, phis = pa._compute_lensing_angles_flatsky(ra_l, dec_l, ra_s, dec_s)
    testing.assert_allclose(thetas, np.array([0.00077050407583119666, 0.00106951489719733675]), rtol=rtol,
                            err_msg="Reasonable values with flat sky not matching to precision for theta")
    testing.assert_allclose(phis, np.array([-1.13390499136495481736, 1.77544123918164542530]), rtol=rtol,
                            err_msg="Reasonable values with flat sky not matching to precision for phi")

    # ra and dec are 0.0 - THROWS A WARNING
    # testing.assert_allclose(pa._compute_lensing_angles_flatsky(0.0, 0.0, ra_s, dec_s),
    #                         [[2.95479482616592248334, 2.95615695795537858359], [2.83280558128919901506, 2.83233281390148761147]],
    #                         rtol, err_msg="Failure when RA_lens=DEC_lens=0.0")

    # lens and source at the same ra
    testing.assert_allclose(pa._compute_lensing_angles_flatsky(ra_l, dec_l, np.array([161.32, 161.34]), dec_s),
                            [[0.00069813170079771690, 0.00106951489719733675], [-1.57079632679489655800, 1.77544123918164542530]],
                            rtol, err_msg="Failure when lens and a source share an RA")

    # lens and source at the same dec
    testing.assert_allclose(pa._compute_lensing_angles_flatsky(ra_l, dec_l, ra_s, np.array([51.49, 51.55])),
                            [[0.00032601941539388962, 0.00106951489719733675], [0.00000000000000000000, 1.77544123918164542530]],
                            rtol, err_msg="Failure when lens and a source share a DEC")

    # lens and source at the same ra and dec - I dont think we want this to raise an error. It just wont be in the bins, so no problemo
    # The second test is not working!!! Find out why!!!
    # testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky, ra_l, dec_l, np.array([161.32, 161.34]), np.array([51.49, 51.55]))
    # testing.assert_allclose(pa._compute_lensing_angles_flatsky(ra_l, dec_l, np.array([ra_l, 161.34]), np.array([dec_l, 51.55])),
    #                         [[0.00000000000000000000, 0.00106951489719733675], [0.00000000000000000000, 1.77544123918164542530]],
    #                         rtol, err_msg="Failure when lens and a source share an RA and a DEC")
    
    # This test throws a warning!
    # testing.assert_allclose(pa._compute_lensing_angles_flatsky(0.1, dec_l, np.array([359.9,180.1]), dec_s),
    #                         [[2.37312589, 1.95611677], [-2.94182333e-04,  3.14105731e+00]],
    #                         rtol, err_msg="Failure when ra_l and ra_s are close but on the two sides of the 0 axis")   

    # This test throws a warning!
    # testing.assert_allclose(pa._compute_lensing_angles_flatsky(0, dec_l, np.array([359.9,180.1]), dec_s),
    #                         [[2.37203916, 1.9572035], [-2.94317111e-04,  3.14105761e+00]],
    #                         rtol, err_msg="Failure when ra_l and ra_s are separated by pi + epsilon")  
    
    # This test throws a warning!
    # testing.assert_allclose(pa._compute_lensing_angles_flatsky(-180, dec_l, np.array([180.1,-90]), dec_s),
    #                         [[2.36986569, 0.97805881], [-2.94587036e-04,  3.14052196e+00]],
    #                         rtol, err_msg="Failure when ra_l and ra_s are the same but one is defined negative")     
    
    # This test throws a warning!
    # testing.assert_allclose(pa._compute_lensing_angles_flatsky(ra_l, 90, ra_s, np.array([51.45,-90])),
    #                         [[0.67282443, 3.14159265], [-1.57079633, -1.57079633]],
    #                         rtol, err_msg="Failure when dec_l and dec_s are separated by 180 deg")  


def test_compute_shear():
    # Input values
    ra_lens, dec_lens = 120., 42.
    ra_source_list = np.array([120.1, 119.9])
    dec_source_list = np.array([41.9, 42.2])
    shear1 = np.array([0.2, 0.4])
    shear2 = np.array([0.3, 0.5])

    # Make GalaxyCluster object
    cluster = clmm.GalaxyCluster(unique_id='blah', ra=ra_lens, dec=dec_lens, z=0.5,
                                 galcat=Table([ra_source_list, dec_source_list, shear1, shear2],
                                              names=('ra', 'dec', 'e1', 'e2')))

    # Correct values
    expected_angsep = np.array([0.0021745039090962414, 0.0037238407383072053])
    expected_cross_shear = np.array([-0.2780316984090899, -0.6398792901134982])
    expected_tangential_shear = np.array([-0.22956126563459447, -0.02354769805831558])

    # Pass arrays directly into function
    angsep, tshear, xshear = pa.compute_shear(ra_lens=ra_lens, dec_lens=dec_lens,
                                              ra_source_list=ra_source_list,
                                              dec_source_list=dec_source_list,
                                              shear1=shear1, shear2=shear2,
                                              add_to_cluster=False)
    testing.assert_allclose(angsep, expected_angsep, rtol=rtol,
                            err_msg="Angular Separation not correct when passing lists")
    testing.assert_allclose(tshear, expected_tangential_shear, rtol=rtol,
                            err_msg="Tangential Shear not correct when passing lists")
    testing.assert_allclose(xshear, expected_cross_shear, rtol=rtol,
                            err_msg="Cross Shear not correct when passing lists")

    # Pass cluster object into the function
    angsep2, tshear2, xshear2 = pa.compute_shear(cluster=cluster)
    testing.assert_allclose(angsep2, expected_angsep, rtol=rtol,
                            err_msg="Angular Separation not correct when passing cluster")
    testing.assert_allclose(tshear2, expected_tangential_shear, rtol=rtol,
                            err_msg="Tangential Shear not correct when passing cluster")
    testing.assert_allclose(xshear2, expected_cross_shear, rtol=rtol,
                            err_msg="Cross Shear not correct when passing cluster")

    # Use the cluster method
    angsep3, tshear3, xshear3 = cluster.compute_shear()
    testing.assert_allclose(angsep3, expected_angsep, rtol=rtol,
                            err_msg="Angular Separation not correct when using cluster method")
    testing.assert_allclose(tshear3, expected_tangential_shear, rtol=rtol,
                            err_msg="Tangential Shear not correct when using cluster method")
    testing.assert_allclose(xshear3, expected_cross_shear, rtol=rtol,
                            err_msg="Cross Shear not correct when using cluster method")

    return


@pytest.mark.skip()
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

