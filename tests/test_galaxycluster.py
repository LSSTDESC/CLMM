"""
Tests for datatype and galaxycluster
"""
import os
import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_allclose
import clmm
from clmm import GCData
from clmm import Cosmology
from scipy.stats import multivariate_normal

TOLERANCE = {'rtol': 1.e-7, 'atol': 1.e-7}

def test_initialization():
    """test initialization"""
    testdict1 = {'unique_id': '1', 'ra': 161.3,
                 'dec': 34., 'z': 0.3, 'galcat': GCData()}
    cl1 = clmm.GalaxyCluster(**testdict1)

    assert_equal(testdict1['unique_id'], cl1.unique_id)
    assert_equal(testdict1['ra'], cl1.ra)
    assert_equal(testdict1['dec'], cl1.dec)
    assert_equal(testdict1['z'], cl1.z)
    assert isinstance(cl1.galcat, GCData)


def test_integrity():  # Converge on name
    """test integrity"""  # Converge on name
    # Ensure we have all necessary values to make a GalaxyCluster
    assert_raises(TypeError, clmm.GalaxyCluster, ra=161.3,
                  dec=34., z=0.3, galcat=GCData())
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1,
                  dec=34., z=0.3, galcat=GCData())
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1,
                  ra=161.3, z=0.3, galcat=GCData())
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1,
                  ra=161.3, dec=34., galcat=GCData())

    # Test that we get errors when we pass in values outside of the domains
    assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1,
                  ra=-360.3, dec=34., z=0.3, galcat=GCData())
    assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1,
                  ra=360.3, dec=34., z=0.3, galcat=GCData())
    assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1,
                  ra=161.3, dec=95., z=0.3, galcat=GCData())
    assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1,
                  ra=161.3, dec=-95., z=0.3, galcat=GCData())
    assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1,
                  ra=161.3, dec=34., z=-0.3, galcat=GCData())

    # Test that inputs are the correct type
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=None,
                  ra=161.3, dec=34., z=0.3, galcat=GCData())
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1,
                  ra=161.3, dec=34., z=0.3, galcat=1)
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1,
                  ra=161.3, dec=34., z=0.3, galcat=[])
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1,
                  ra=None, dec=34., z=0.3, galcat=GCData())
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1,
                  ra=161.3, dec=None, z=0.3, galcat=GCData())
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1,
                  ra=161.3, dec=34., z=None, galcat=GCData())

    # Test that id can support numbers and strings
    assert isinstance(clmm.GalaxyCluster(unique_id=1, ra=161.3,
                      dec=34., z=0.3, galcat=GCData()).unique_id, str)
    assert isinstance(clmm.GalaxyCluster(
        unique_id='1', ra=161.3, dec=34., z=0.3, galcat=GCData()).unique_id, str)

    # Test that ra/dec/z can be converted from int/str to float if needed
    assert clmm.GalaxyCluster('1', '161.', '55.', '.3', GCData())
    assert clmm.GalaxyCluster('1', 161, 55, 1, GCData())


def test_save_load():
    """test save load"""
    cl1 = clmm.GalaxyCluster(unique_id='1', ra=161.3,
                             dec=34., z=0.3, galcat=GCData())
    cl1.save('testcluster.pkl')
    cl2 = clmm.GalaxyCluster.load('testcluster.pkl')
    os.system('rm testcluster.pkl')

    assert_equal(cl2.unique_id, cl1.unique_id)
    assert_equal(cl2.ra, cl1.ra)
    assert_equal(cl2.dec, cl1.dec)
    assert_equal(cl2.z, cl1.z)

    # remeber to add tests for the tables of the cluster

# def test_find_data():
#     """test find data"""
#     gc = GalaxyCluster('test_cluster', test_data)
#
#     tst.assert_equal([], gc.find_data(test_creator_diff, test_dict))
#
#     tst.assert_equal([test_data], gc.find_data(test_creator, test_dict))
#     tst.assert_equal([test_data], gc.find_data(test_creator, test_dict_sub))
#     tst.assert_equal([], gc.find_data(test_creator, test_dict_diff))
#
#     tst.assert_equal([test_data], gc.find_data(test_creator, test_dict, exact=True))
#     tst.assert_equal([], gc.find_data(test_creator, test_dict_sub, exact=True))
#     tst.assert_equal([], gc.find_data(test_creator, test_dict_diff, exact=True))


def test_print_gc():
    """test print gc"""
    # Cluster with empty galcat
    cluster = clmm.GalaxyCluster(
        unique_id='1', ra=161.3, dec=34., z=0.3, galcat=GCData())
    print(cluster)
    assert isinstance(cluster.__str__(), str)
    assert isinstance(cluster.__repr__(), str)
    assert isinstance(cluster._repr_html_(), str)
    # Cluster with galcat
    galcat = GCData([[120.1, 119.9, 119.9], [41.9, 42.2, 42.2],
                     [1, 1, 1], [1, 2, 3]],
                    names=('ra', 'dec', 'z', 'id'))
    cluster = clmm.GalaxyCluster(unique_id='1', ra=161.3,
                            dec=34., z=0.3, galcat=galcat)
    print(cluster)
    assert isinstance(cluster.__str__(), str)
    assert isinstance(cluster.__repr__(), str)


def test_integrity_of_lensfuncs():
    """test integrity of lensfuncs"""
    ra_source, dec_source = [120.1, 119.9, 119.9], [41.9, 42.2, 42.2]
    id_source, z_sources = [1, 2, 3], [1, 1, 1]
    galcat = GCData([ra_source, dec_source, z_sources, id_source],
                    names=('ra', 'dec', 'z', 'id'))
    galcat_noz = GCData([ra_source, dec_source, id_source],
                       names=('ra', 'dec', 'id'))
    cosmo = clmm.Cosmology(H0=70.0, Omega_dm0=0.275, Omega_b0=0.025)
    # Missing cosmo
    cluster = clmm.GalaxyCluster(unique_id='1', ra=161.3,
                            dec=34., z=0.3, galcat=galcat)
    assert_raises(TypeError, cluster.add_critical_surface_density, None)
    # Missing cl redshift
    cluster = clmm.GalaxyCluster(unique_id='1', ra=161.3,
                            dec=34., z=0.3, galcat=galcat)
    cluster.z = None
    assert_raises(TypeError, cluster.add_critical_surface_density, cosmo)
    # Missing galaxy redshift
    cluster = clmm.GalaxyCluster(unique_id='1', ra=161.3,
                            dec=34., z=0.3, galcat=galcat_noz)
    assert_raises(TypeError, cluster.add_critical_surface_density, cosmo)

def test_integrity_of_probfuncs():
    """test integrity of prob funcs"""
    ra_source, dec_source = [120.1, 119.9, 119.9], [41.9, 42.2, 42.2]
    id_source, z_sources = [1, 2, 3], [1, 1, 1]
    cluster = clmm.GalaxyCluster(
        unique_id='1', ra=161.3, dec=34., z=0.3,
        galcat=GCData([ra_source, dec_source, z_sources, id_source],
                      names=('ra', 'dec', 'z', 'id')))
    # true redshift
    cluster.compute_background_probability(use_photoz=False, p_background_name='p_bkg_true')
    expected = np.array([1., 1., 1.])
    assert_allclose(cluster.galcat['p_bkg_true'], expected, **TOLERANCE)

    #photoz + deltasigma
    assert_raises(TypeError, cluster.compute_background_probability, use_photoz=True)
    pzbin = np.linspace(.0001, 5, 100)
    cluster.galcat['pzbins'] = [pzbin for i in range(len(z_sources))]
    cluster.galcat['pzpdf'] = [multivariate_normal.pdf(pzbin, mean=z, cov=.01) for z in z_sources]
    cluster.compute_background_probability(use_photoz=True, p_background_name='p_bkg_pz')
    assert_allclose(cluster.galcat['p_bkg_pz'], expected, **TOLERANCE)

def test_integrity_of_weightfuncs():
    """test integrity of weight funcs"""
    cosmo = Cosmology(H0=71.0, Omega_dm0=0.265 - 0.0448, Omega_b0=0.0448, Omega_k0=0.0)
    z_lens = .1
    z_source = [.22, .35, 1.7]
    shape_component1 = np.array([.143, .063, -.171])
    shape_component2 = np.array([-.011, .012,-.250])
    shape_component1_err = np.array([.11, .01, .2])
    shape_component2_err = np.array([.14, .16, .21])
    p_background = np.array([1., 1., 1.])
    cluster = clmm.GalaxyCluster(
        unique_id='1', ra=161.3, dec=34., z=z_lens,
        galcat=GCData(
            [shape_component1, shape_component2,
             shape_component1_err, shape_component2_err, z_source],
            names=('e1', 'e2', 'e1_err', 'e2_err', 'z')))
    #true redshift + deltasigma
    cluster.compute_galaxy_weights(cosmo=cosmo, use_shape_noise=False,
                                   is_deltasigma=True)
    expected = np.array([4.58644320e-31, 9.68145632e-31, 5.07260777e-31])
    assert_allclose(cluster.galcat['w_ls']*1e20, expected*1e20,**TOLERANCE)

    #photoz + deltasigma
    pzbin = np.linspace(.0001, 5, 100)
    pzbins = np.zeros([len(z_source), len(pzbin)])
    pzpdf = pzbins
    pzbin = np.linspace(.0001, 5, 100)
    cluster.galcat['pzbins'] = [pzbin for i in range(len(z_source))]
    cluster.galcat['pzpdf'] = [multivariate_normal.pdf(pzbin, mean=z, cov=.3) for z in z_source]
    cluster.compute_galaxy_weights(cosmo=cosmo, use_shape_noise=False, use_photoz=True,
                                   is_deltasigma=True)
    expected = np.array([9.07709345e-33, 1.28167582e-32, 4.16870389e-32])
    assert_allclose(cluster.galcat['w_ls']*1e20, expected*1e20,**TOLERANCE)

    # test with noise
    cluster.compute_galaxy_weights(cosmo=cosmo, use_shape_noise=True, use_photoz=True,
                                   use_shape_error=True, is_deltasigma=True)

    expected = np.array([9.07709345e-33, 1.28167582e-32, 4.16870389e-32])
    assert_allclose(cluster.galcat['w_ls']*1e20, expected*1e20,**TOLERANCE)

def test_plot_profiles():
    """test plot profiles"""
    # Input values
    ra_lens, dec_lens, z_lens = 120., 42., 0.5
    ra_source = [120.1, 119.9]
    dec_source = [41.9, 42.2]
    z_source = [1., 2.]
    shear1 = [0.2, 0.4]
    shear2 = [0.3, 0.5]
    # Set up radial values
    bins_radians = [0.002, 0.003, 0.004]
    bin_units = 'radians'
    # create cluster
    cluster = clmm.GalaxyCluster(unique_id='test', ra=ra_lens, dec=dec_lens, z=z_lens,
                                 galcat=GCData([ra_source, dec_source, shear1, shear2, z_source],
                                               names=('ra', 'dec', 'e1', 'e2', 'z')))
    cluster.compute_tangential_and_cross_components()
    cluster.make_radial_profile(
        bin_units, bins=bins_radians, include_empty_bins=True)
    # missing profile name
    assert_raises(ValueError, cluster.plot_profiles,
                  table_name='made_up_table')
    # missing shear component
    assert_raises(ValueError, cluster.plot_profiles,
                  cross_component='made_up_component')
    # check basic plot is working
    cluster.plot_profiles()
    # check it passes missing a component error
    cluster.plot_profiles(cross_component_error='made_up_component')
