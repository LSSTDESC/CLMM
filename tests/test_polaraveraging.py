"""Tests for polaraveraging.py"""
import clmm
import clmm.polaraveraging as pa
from numpy import testing
import numpy as np
from astropy.table import Table
import os
import pytest


rtol = 1.e-6
atol = 1.e-7


def test_compute_cross_shear():
    # Edge case tests
    testing.assert_allclose(pa._compute_cross_shear(100., 0., 0.), 0.0,atol=atol)
    testing.assert_allclose(pa._compute_cross_shear(100., 0., np.pi/2), 0.0,atol=atol)
    testing.assert_allclose(pa._compute_cross_shear(0., 100., 0.), -100.0,atol=atol)
    testing.assert_allclose(pa._compute_cross_shear(0., 100., np.pi/2), 100.0,atol=atol)
    testing.assert_allclose(pa._compute_cross_shear(0., 100., np.pi/4.), 0.0,atol=atol)
    testing.assert_allclose(pa._compute_cross_shear(0., 0., 0.3), 0.,atol=atol)


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

    # test for reasonable values
    testing.assert_almost_equal(pa._compute_tangential_shear(100., 0., 0.), -100.0)
    testing.assert_almost_equal(pa._compute_tangential_shear(0., 100., np.pi/4.), -100.0)
    testing.assert_almost_equal(pa._compute_tangential_shear(0., 0., 0.3), 0.)

    
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

    # lens and source at the same ra
    testing.assert_allclose(pa._compute_lensing_angles_flatsky(ra_l, dec_l, np.array([161.32, 161.34]), dec_s),
                            [[0.00069813170079771690, 0.00106951489719733675], [-1.57079632679489655800, 1.77544123918164542530]],
                            rtol, err_msg="Failure when lens and a source share an RA")

    # lens and source at the same dec
    testing.assert_allclose(pa._compute_lensing_angles_flatsky(ra_l, dec_l, ra_s, np.array([51.49, 51.55])),
                            [[0.00032601941539388962, 0.00106951489719733675], [0.00000000000000000000, 1.77544123918164542530]],
                            rtol, err_msg="Failure when lens and a source share a DEC")



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
    expected_cross_shear = np.array([0.2780316984090899, 0.6398792901134982])
    expected_tangential_shear = np.array([-0.22956126563459447, -0.02354769805831558])

    # Pass arrays directly into function
    angsep, tshear, xshear = pa.compute_shear(ra_lens=ra_lens, dec_lens=dec_lens,
                                              ra_source_list=ra_source_list,
                                              dec_source_list=dec_source_list,
                                              shear1=shear1, shear2=shear2,
                                              add_to_cluster=False)
    testing.assert_allclose(angsep, expected_angsep, rtol=rtol, atol=atol,
                            err_msg="Angular Separation not correct when passing lists")
    testing.assert_allclose(tshear, expected_tangential_shear, rtol=rtol, atol=atol,
                            err_msg="Tangential Shear not correct when passing lists")
    testing.assert_allclose(xshear, expected_cross_shear, rtol=rtol, atol=atol,
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


def test_make_shear_profiles():
    # Set up a cluster object and compute cross and tangential shears
    ra_lens, dec_lens = 120., 42.
    ra_source_list = np.array([120.1, 119.9, 119.9])
    dec_source_list = np.array([41.9, 42.2, 42.2])
    shear1 = np.array([0.2, 0.4, 0.4])
    shear2 = np.array([0.3, 0.5, 0.5])
    cluster = clmm.GalaxyCluster(unique_id='blah', ra=ra_lens, dec=dec_lens, z=0.5,
                                 galcat=Table([ra_source_list, dec_source_list, shear1, shear2],
                                              names=('ra', 'dec', 'e1', 'e2')))
    angsep, tshear, xshear = pa.compute_shear(cluster=cluster, add_to_cluster=True)

    # Test the outputs of compute_shear just to be save
    expected_angsep = np.array([0.0021745039090962414, 0.0037238407383072053, 0.0037238407383072053])
    expected_cross_shear = np.array([0.2780316984090899, 0.6398792901134982, 0.6398792901134982])
    expected_tan_shear = np.array([-0.22956126563459447, -0.02354769805831558, -0.02354769805831558])
    testing.assert_allclose(angsep, expected_angsep, rtol=rtol,
                            err_msg="Angular Separation not correct when testing shear profiles")
    testing.assert_allclose(tshear, expected_tan_shear, rtol=rtol,
                            err_msg="Tangential Shear not correct when testing shear profiles")
    testing.assert_allclose(xshear, expected_cross_shear, rtol=rtol,
                            err_msg="Cross Shear not correct when testing shear profiles")

    # Make the shear profile and check it
    bins_radians = np.array([0.002, 0.003, 0.004])
    profile = pa.make_shear_profile(cluster, 'radians', 'radians', bins=bins_radians)
    testing.assert_allclose(profile['radius_min'], bins_radians[:-1], rtol=rtol,
                            err_msg="Minimum radius in bin not expected.")
    testing.assert_allclose(profile['radius'], [0.0021745039090962414, 0.0037238407383072053], rtol=rtol,
                            err_msg="Mean radius in bin not expected.")
    testing.assert_allclose(profile['radius_max'], bins_radians[1:], rtol=rtol,
                            err_msg="Maximum radius in bin not expected.")
    testing.assert_allclose(profile['gt'], expected_tan_shear[:-1], rtol=rtol,
                            err_msg="Tangential shear in bin not expected")
    testing.assert_allclose(profile['gx'], expected_cross_shear[:-1], rtol=rtol,
                            err_msg="Cross shear in bin not expected")

    # Repeat the same tests when we call make_shear_profile through the GalaxyCluster method
    profile2 = cluster.make_shear_profile('radians', 'radians', bins=bins_radians)
    testing.assert_allclose(profile2['radius_min'], [0.002, 0.003], rtol=rtol, atol=atol,
                            err_msg="Minimum radius in bin not expected.")
    testing.assert_allclose(profile2['radius'], [0.0021745039090962414, 0.0037238407383072053], rtol=rtol, atol=atol,
                            err_msg="Mean radius in bin not expected.")
    testing.assert_allclose(profile2['radius_max'], [0.003, 0.004], rtol=rtol,atol=atol,
                            err_msg="Maximum radius in bin not expected.")
    testing.assert_allclose(profile2['gt'], [-0.22956126563459447, -0.02354769805831558], rtol=rtol,atol=atol,
                            err_msg="Tangential shear in bin not expected")
    testing.assert_allclose(profile2['gx'], expected_cross_shear[:-1], rtol=rtol, atol=atol,
                            err_msg="Cross shear in bin not expected")


