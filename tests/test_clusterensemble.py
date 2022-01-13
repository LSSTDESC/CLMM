"""
tests for clusterensemble.py
"""
import os
from numpy.testing import assert_raises, assert_equal
import clmm
import numpy as np
from clmm import clusterensemble
from clmm import Cosmology

def test_len():
    """test lenght of clusterensemble attributes"""
    cosmo = Cosmology(H0=70, Omega_dm0=0.262, Omega_b0=0.049)
    #create galaxycluster object
    ra_lens, dec_lens, z_lens = 120., 42., 0.5
    ra_source = [120.1, 119.9]
    dec_source = [41.9, 42.2]
    theta_source = [0.0025, 0.015]
    z_source = [1., 2.]
    shear1 = [0.2, 0.4]
    shear2 = [0.3, 0.5]
    w_ls = [1.e-30, 1.e-31]
    # Set up radial values
    bins_radians = np.logspace(np.log10(.001), np.log10(.02), 10)
    bin_units = 'radians'
    names = ('ra', 'dec', 'theta', 'w_ls', 'e1', 'e2', 'z')
    # create cluster
    cluster = clmm.GalaxyCluster(unique_id='test', ra=ra_lens, dec=dec_lens, z=z_lens,
                                 galcat=clmm.GCData([ra_source, dec_source, theta_source, w_ls,
                                                shear1, shear2, z_source],
                                               names=names))
    cluster.compute_tangential_and_cross_components()
    bins = bins_radians
    gc_list = [cluster]
    ce = clusterensemble.ClusterEnsemble('cluster_ensemble', gc_list, tan_component_in='et',
    cross_component_in='ex', weights_in = 'w_ls', bins=bins, bin_units='radians', cosmo=cosmo)
    #test the lenght of the clusterensemble data attribute
    assert_equal(ce.__len__(), 1)
    ce._add_values([cluster], tan_component_in='et',cross_component_in='ex', 
                   weights_in = 'w_ls', bins=bins, bin_units='radians', cosmo=cosmo)
    #test the lenght of the clusterensemble data attribute (after doubling the number of individual cluster)
    assert_equal(ce.__len__(), 2)
    #test if the len of averaged profile has the lenght of binning axis
    assert_equal(len(ce.data['W_l'][0]), len(bins_radians)-1)