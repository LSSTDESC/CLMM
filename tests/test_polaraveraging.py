"""Tests for polaraveraging.py"""
import numpy as np
from numpy import testing

import clmm
from clmm import GCData
import clmm.polaraveraging as pa
from astropy.cosmology import FlatLambdaCDM

TOLERANCE = {'atol':1.e-7, 'rtol':1.e-7}

def test_compute_cross_shear():
    shear1, shear2, phi = 0.15, 0.08, 0.52
    expected_cross_shear = 0.08886301350787848
    cross_shear = pa._compute_cross_shear(shear1, shear2, phi)
    testing.assert_allclose(cross_shear, expected_cross_shear)

    shear1 = np.array([0.15, 0.40])
    shear2 = np.array([0.08, 0.30])
    phi = np.array([0.52, 1.23])
    expected_cross_shear = [0.08886301350787848, 0.48498333705834484]
    cross_shear = pa._compute_cross_shear(shear1, shear2, phi)
    testing.assert_allclose(cross_shear, expected_cross_shear)

    # Edge case tests
    testing.assert_allclose(pa._compute_cross_shear(100., 0., 0.), 0.0,
                            **TOLERANCE)
    testing.assert_allclose(pa._compute_cross_shear(100., 0., np.pi/2), 0.0,
                            **TOLERANCE)
    testing.assert_allclose(pa._compute_cross_shear(0., 100., 0.), -100.0,
                            **TOLERANCE)
    testing.assert_allclose(pa._compute_cross_shear(0., 100., np.pi/2), 100.0,
                            **TOLERANCE)
    testing.assert_allclose(pa._compute_cross_shear(0., 100., np.pi/4.), 0.0,
                            **TOLERANCE)
    testing.assert_allclose(pa._compute_cross_shear(0., 0., 0.3), 0.,
                            **TOLERANCE)


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

    # Test domains on inputs
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky,
                          -365., dec_l, ra_s, dec_s)
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky,
                          365., dec_l, ra_s, dec_s)
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky,
                          ra_l, 95., ra_s, dec_s)
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky,
                          ra_l, -95., ra_s, dec_s)
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky,
                          ra_l, dec_l, ra_s-10., dec_s)
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky,
                          ra_l, dec_l, ra_s+10., dec_s)
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky,
                          ra_l, dec_l, ra_s, dec_s-10.)
    testing.assert_raises(ValueError, pa._compute_lensing_angles_flatsky,
                          ra_l, dec_l, ra_s, dec_s+10.)

    # Ensure that we throw a warning with >1 deg separation
    testing.assert_warns(UserWarning, pa._compute_lensing_angles_flatsky,
                         ra_l, dec_l, np.array([151.32, 161.34]), np.array([41.49, 51.55]))

    # Test outputs for reasonable values
    ra_l, dec_l = 161.32, 51.49
    ra_s, dec_s = np.array([161.29, 161.34]), np.array([51.45, 51.55])
    thetas, phis = pa._compute_lensing_angles_flatsky(ra_l, dec_l, ra_s, dec_s)

    testing.assert_allclose(thetas, np.array([0.00077050407583119666, 0.00106951489719733675]),
                            **TOLERANCE,
                            err_msg="Reasonable values with flat sky not matching to precision for theta")

    testing.assert_allclose(phis, np.array([-1.13390499136495481736, 1.77544123918164542530]),
                            **TOLERANCE,
                            err_msg="Reasonable values with flat sky not matching to precision for phi")

    # lens and source at the same ra
    testing.assert_allclose(pa._compute_lensing_angles_flatsky(ra_l, dec_l, np.array([161.32, 161.34]), dec_s),
                            [[0.00069813170079771690, 0.00106951489719733675], [-1.57079632679489655800, 1.77544123918164542530]],
                            **TOLERANCE, err_msg="Failure when lens and a source share an RA")

    # lens and source at the same dec
    testing.assert_allclose(pa._compute_lensing_angles_flatsky(ra_l, dec_l, ra_s, np.array([51.49, 51.55])),
                            [[0.00032601941539388962, 0.00106951489719733675], [0.00000000000000000000, 1.77544123918164542530]],
                            **TOLERANCE, err_msg="Failure when lens and a source share a DEC")

    # lens and source at the same ra and dec
    testing.assert_allclose(pa._compute_lensing_angles_flatsky(ra_l, dec_l, np.array([ra_l, 161.34]), np.array([dec_l, 51.55])),
                            [[0.00000000000000000000, 0.00106951489719733675], [0.00000000000000000000, 1.77544123918164542530]],
                            TOLERANCE['rtol'], err_msg="Failure when lens and a source share an RA and a DEC")

    # angles over the branch cut between 0 and 360
    testing.assert_allclose(pa._compute_lensing_angles_flatsky(0.1, dec_l, np.array([359.9, 359.5]), dec_s),
                            [[0.0022828333888309108, 0.006603944760273219], [-0.31079754672938664, 0.15924369771830643]],
                            TOLERANCE['rtol'], err_msg="Failure when ra_l and ra_s are close but on the opposite sides of the 0 axis")

    # angles over the branch cut between 0 and 360
    testing.assert_allclose(pa._compute_lensing_angles_flatsky(-180, dec_l, np.array([180.1, 179.7]), dec_s),
                            [[0.0012916551296819666, 0.003424250083245557], [-2.570568636904587, 0.31079754672944354]],
                            TOLERANCE['rtol'], err_msg="Failure when ra_l and ra_s are the same but one is defined negative")

def test_compute_tangential_and_cross_components():
    # Input values
    ra_lens, dec_lens, z_lens = 120., 42., 0.5
    ra_source = np.array([120.1, 119.9])
    dec_source = np.array([41.9, 42.2])
    z_source = np.array([1.,2.])
    
    shear1 = np.array([0.2, 0.4])
    shear2 = np.array([0.3, 0.5])

    # Make GalaxyCluster object
    cluster = clmm.GalaxyCluster(unique_id='blah', ra=ra_lens, dec=dec_lens, z=z_lens,
                                 galcat=GCData([ra_source, dec_source, shear1, shear2],
                                              names=('ra', 'dec', 'e1', 'e2')))

    # Correct values
    expected_angsep = np.array([0.0021745039090962414, 0.0037238407383072053])
    expected_cross_shear = np.array([0.2780316984090899, 0.6398792901134982])
    expected_tangential_shear = np.array([-0.22956126563459447, -0.02354769805831558])
    
    # DeltaSigma expected values for FlatLambdaCDM(H0=70., Om0=0.3, Ob0=0.025)    
    expected_cross_DS = np.array([1224.3326297393244, 1899.6061989365176])
    expected_tangential_DS = np.array([-1010.889584349285, -69.9059242788237])
    
    # Pass arrays directly into function
    angsep, tshear, xshear = pa.compute_tangential_and_cross_components(ra_lens=ra_lens, dec_lens=dec_lens,
                                              ra_source=ra_source,
                                              dec_source=dec_source,
                                              shear1=shear1, shear2=shear2,
                                              add_to_cluster=False)
    testing.assert_allclose(angsep, expected_angsep, **TOLERANCE,
                            err_msg="Angular Separation not correct when passing lists")
    testing.assert_allclose(tshear, expected_tangential_shear, **TOLERANCE,
                            err_msg="Tangential Shear not correct when passing lists")
    testing.assert_allclose(xshear, expected_cross_shear, **TOLERANCE,
                            err_msg="Cross Shear not correct when passing lists")

    # Pass cluster object into the function
    angsep2, tshear2, xshear2 = pa.compute_tangential_and_cross_components(cluster=cluster)
    testing.assert_allclose(angsep2, expected_angsep, **TOLERANCE,
                            err_msg="Angular Separation not correct when passing cluster")
    testing.assert_allclose(tshear2, expected_tangential_shear, **TOLERANCE,
                            err_msg="Tangential Shear not correct when passing cluster")
    testing.assert_allclose(xshear2, expected_cross_shear, **TOLERANCE,
                            err_msg="Cross Shear not correct when passing cluster")

    # Use the cluster method
    angsep3, tshear3, xshear3 = cluster.compute_tangential_and_cross_components()
    testing.assert_allclose(angsep3, expected_angsep, **TOLERANCE, 
                            err_msg="Angular Separation not correct when using cluster method")
    testing.assert_allclose(tshear3, expected_tangential_shear, **TOLERANCE, 
                            err_msg="Tangential Shear not correct when using cluster method")
    testing.assert_allclose(xshear3, expected_cross_shear, **TOLERANCE, 
                            err_msg="Cross Shear not correct when using cluster method")
    
    
    # Check behaviour for the deltasigma option.
    # cluster object missing source redshift, and function call missing cosmology
    testing.assert_raises(TypeError, cluster.compute_tangential_and_cross_components, is_deltasigma=True)

    # cluster object OK but function call missing cosmology
    cluster = clmm.GalaxyCluster(unique_id='blah', ra=ra_lens, dec=dec_lens, z=z_lens,
                                 galcat=GCData([ra_source, dec_source, shear1, shear2, z_source],
                                               names=('ra', 'dec', 'e1', 'e2','z')))
    testing.assert_raises(TypeError, cluster.compute_tangential_and_cross_components, is_deltasigma=True)
    
    # check values for DeltaSigma
    cosmo = FlatLambdaCDM(H0=70., Om0=0.3, Ob0=0.025)
    angsep_DS, tDS, xDS = cluster.compute_tangential_and_cross_components(cosmo=cosmo, is_deltasigma=True)
    testing.assert_allclose(angsep_DS, expected_angsep, **TOLERANCE, 
                            err_msg="Angular Separation not correct when using cluster method")
    testing.assert_allclose(tDS, expected_tangential_DS, **TOLERANCE, 
                            err_msg="Tangential Shear not correct when using cluster method")
    testing.assert_allclose(xDS, expected_cross_DS, **TOLERANCE, 
                            err_msg="Cross Shear not correct when using cluster method")
   
def test_make_binned_profiles():
    # Set up a cluster object and compute cross and tangential shears
    ra_lens, dec_lens, z_lens = 120., 42., 0.5
    ra_source = np.array([120.1, 119.9, 119.9])
    dec_source = np.array([41.9, 42.2, 42.2])
    id_source = np.array([1, 2, 3])
    shear1 = np.array([0.2, 0.4, 0.4])
    shear2 = np.array([0.3, 0.5, 0.5])
    z_sources = np.ones(3)
    angsep_units, bin_units = 'radians', 'radians'
    cluster = clmm.GalaxyCluster(unique_id='blah', ra=ra_lens, dec=dec_lens, z=z_lens,
                                 galcat=GCData([ra_source, dec_source,
                                               shear1, shear2, z_sources, id_source],
                                              names=('ra', 'dec', 'e1', 'e2', 'z', 'id')))

    # Test error of missing redshift
    cluster_noz = clmm.GalaxyCluster(unique_id='blah', ra=ra_lens, dec=dec_lens, z=z_lens,
                                     galcat=GCData([ra_source, dec_source,
                                                   shear1, shear2],
                                                  names=('ra', 'dec', 'e1', 'e2')))
    cluster_noz.compute_tangential_and_cross_components()
    testing.assert_raises(TypeError, pa.make_binned_profile, cluster_noz, angsep_units, bin_units)

    # Test error of missing shear
    testing.assert_raises(TypeError, pa.make_binned_profile, cluster, angsep_units, bin_units)

    angsep, tshear, xshear = pa.compute_tangential_and_cross_components(cluster=cluster, add_to_cluster=True)
    # Test the outputs of compute_tangential_and_cross_components just to be safe
    expected_angsep = np.array([0.0021745039090962414, 0.0037238407383072053, 0.0037238407383072053])
    expected_cross_shear = np.array([0.2780316984090899, 0.6398792901134982, 0.6398792901134982])
    expected_tan_shear = np.array([-0.22956126563459447, -0.02354769805831558, -0.02354769805831558])
    testing.assert_allclose(angsep, expected_angsep, **TOLERANCE, 
                            err_msg="Angular Separation not correct when testing shear profiles")
    testing.assert_allclose(tshear, expected_tan_shear, **TOLERANCE, 
                            err_msg="Tangential Shear not correct when testing shear profiles")
    testing.assert_allclose(xshear, expected_cross_shear, **TOLERANCE, 
                            err_msg="Cross Shear not correct when testing shear profiles")

    # Tests passing int as bins arg makes the correct bins
    bins = 2
    vec_bins = clmm.utils.make_bins(np.min(cluster.galcat['theta']),
                                    np.max(cluster.galcat['theta']), bins)
    testing.assert_array_equal(pa.make_binned_profile(cluster, angsep_units, bin_units, bins=bins),
                               pa.make_binned_profile(cluster, angsep_units, bin_units, bins=vec_bins))
    # Make the shear profile and check it
    bins_radians = np.array([0.002, 0.003, 0.004])
    expected_radius = [0.0021745039090962414, 0.0037238407383072053]
    # remember that include_empty_bins=False excludes all bins with N>=1
    profile = pa.make_binned_profile(cluster, angsep_units, bin_units, bins=bins_radians,
                                    include_empty_bins=False)
    testing.assert_allclose(profile['radius_min'], bins_radians[1],  **TOLERANCE,
                            err_msg="Minimum radius in bin not expected.")
    testing.assert_allclose(profile['radius'], expected_radius[1], **TOLERANCE,
                            err_msg="Mean radius in bin not expected.")
    testing.assert_allclose(profile['radius_max'], bins_radians[2], **TOLERANCE,
                            err_msg="Maximum radius in bin not expected.")
    testing.assert_allclose(profile['gt'], expected_tan_shear[1], **TOLERANCE,
                            err_msg="Tangential shear in bin not expected")
    testing.assert_allclose(profile['gx'], expected_cross_shear[1], **TOLERANCE,
                            err_msg="Cross shear in bin not expected")
    testing.assert_array_equal(profile['n_src'], [2])

    # Test metadata
    testing.assert_array_equal(profile.meta['bin_units'], bin_units)
    testing.assert_array_equal(profile.meta['cosmo'], None)

    # Repeat the same tests when we call make_binned_profile through the GalaxyCluster method
    profile2 = cluster.make_binned_profile(
        angsep_units, bin_units, bins=bins_radians, include_empty_bins=False)
    testing.assert_allclose(profile2['radius_min'], bins_radians[1], **TOLERANCE,
                            err_msg="Minimum radius in bin not expected.")
    testing.assert_allclose(profile2['radius'], expected_radius[1], **TOLERANCE,
                            err_msg="Mean radius in bin not expected.")
    testing.assert_allclose(profile2['radius_max'], bins_radians[2], **TOLERANCE,
                            err_msg="Maximum radius in bin not expected.")
    testing.assert_allclose(profile2['gt'], expected_tan_shear[1], **TOLERANCE,
                            err_msg="Tangential shear in bin not expected")
    testing.assert_allclose(profile2['gx'], expected_cross_shear[1], **TOLERANCE,
                            err_msg="Cross shear in bin not expected")
    testing.assert_array_equal(profile['n_src'], [2])

    # including empty bins
    profile3 = pa.make_binned_profile(
        cluster, angsep_units, bin_units, bins=bins_radians, include_empty_bins=True)
    testing.assert_allclose(profile3['radius_min'], bins_radians[:-1],  **TOLERANCE,
                            err_msg="Minimum radius in bin not expected.")
    testing.assert_allclose(profile3['radius'], expected_radius, **TOLERANCE,
                            err_msg="Mean radius in bin not expected.")
    testing.assert_allclose(profile3['radius_max'], bins_radians[1:], **TOLERANCE,
                            err_msg="Maximum radius in bin not expected.")
    testing.assert_allclose(profile3['gt'], expected_tan_shear[:-1], **TOLERANCE,
                            err_msg="Tangential shear in bin not expected")
    testing.assert_allclose(profile3['gx'], expected_cross_shear[:-1], **TOLERANCE,
                            err_msg="Cross shear in bin not expected")
    testing.assert_array_equal(profile3['n_src'], [1,2])

    # Repeat the same tests when we call make_binned_profile through the GalaxyCluster method
    profile4 = cluster.make_binned_profile(
        angsep_units, bin_units, bins=bins_radians, include_empty_bins=True)
    testing.assert_allclose(profile4['radius_min'], bins_radians[:-1], **TOLERANCE,
                            err_msg="Minimum radius in bin not expected.")
    testing.assert_allclose(profile4['radius'], expected_radius,
                            err_msg="Mean radius in bin not expected.")
    testing.assert_allclose(profile4['radius_max'], bins_radians[1:], **TOLERANCE,
                            err_msg="Maximum radius in bin not expected.")
    testing.assert_allclose(profile4['gt'], expected_tan_shear[:-1], **TOLERANCE,
                            err_msg="Tangential shear in bin not expected")
    testing.assert_allclose(profile4['gx'], expected_cross_shear[:-1], **TOLERANCE,
                            err_msg="Cross shear in bin not expected")
    testing.assert_array_equal(profile4['n_src'], [1,2])

    # Repeat the same tests but also asking for list of galaxy IDs in each bin
    cluster_noid = clmm.GalaxyCluster(unique_id='blah', ra=ra_lens, dec=dec_lens, z=z_lens,
                                 galcat=GCData([ra_source, dec_source,
                                               shear1, shear2, z_sources],
                                              names=('ra', 'dec', 'e1', 'e2', 'z')))
    cluster_noid.compute_tangential_and_cross_components()
    testing.assert_raises(TypeError, pa.make_binned_profile, cluster_noid, angsep_units, bin_units, gal_ids_in_bins=True)
   
    profile5 = cluster.make_binned_profile(
        angsep_units, bin_units, bins=bins_radians, include_empty_bins=True, gal_ids_in_bins=True)
    testing.assert_allclose(profile5['radius_min'], bins_radians[:-1], **TOLERANCE,
                            err_msg="Minimum radius in bin not expected.")
    testing.assert_allclose(profile5['radius'], expected_radius,
                            err_msg="Mean radius in bin not expected.")
    testing.assert_allclose(profile5['radius_max'], bins_radians[1:], **TOLERANCE,
                            err_msg="Maximum radius in bin not expected.")
    testing.assert_allclose(profile5['gt'], expected_tan_shear[:-1], **TOLERANCE,
                            err_msg="Tangential shear in bin not expected")
    testing.assert_allclose(profile5['gx'], expected_cross_shear[:-1], **TOLERANCE,
                            err_msg="Cross shear in bin not expected")
    testing.assert_array_equal(profile5['n_src'], [1,2])
    testing.assert_array_equal(profile5['gal_id'], [[1],[2,3]])

