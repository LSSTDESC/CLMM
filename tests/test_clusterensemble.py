"""
tests for clusterensemble.py
"""
import os
from numpy.testing import assert_raises, assert_equal, assert_allclose
import clmm
import numpy as np
from clmm import clusterensemble
from clmm import Cosmology
from clmm import GCData
from clmm.support import mock_data as mock
import matplotlib.pyplot as plt
import clmm.dataops as da

TOLERANCE = {'rtol': 5.0e-4, 'atol': 1.e-4}

def test_cluster_ensemble():
    """test clusterensemble attributes"""
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
    
    #test without kwargs, args
    ce = clusterensemble.ClusterEnsemble('cluster_ensemble', gc_list, tan_component_in='et',
    cross_component_in='ex', weights_in = 'w_ls', bins=bins, bin_units='radians', cosmo=cosmo)
    
    #test the lenght of the clusterensemble data attribute
    assert_equal(ce.__len__(), 1)
    
    #test the lenght of the clusterensemble data attribute (after doubling the number of individual cluster)
    ce._add_values([cluster], tan_component_in='et',cross_component_in='ex', 
                   weights_in = 'w_ls', bins=bins, bin_units='radians', cosmo=cosmo)
    assert_equal(ce.__len__(), 2)
    #test if the len of averaged profile has the lenght of binning axis
    assert_equal(len(ce.data['W_l'][0]), len(bins_radians)-1)
    assert_equal(ce.__getitem__('gt'), ce.data['gt'])
    
def test_covariance():
    """test the shapes of covariance matrix with different methods"""
    cosmo = Cosmology(H0=70, Omega_dm0=0.262, Omega_b0=0.049)
    dz = cosmo.eval_da_z1z2(0., 1.)
    n_catalogs = 3
    ngals = 1000
    cluster_ra = [50, 100, 200] #from 0 to 360 deg
    sindec = [-.1, 0, .1]
    cluster_dec = np.arcsin(sindec)*180/np.pi #from -90 to 90 deg
    gclist = []
    Rmin, Rmax = .3, 5 #Mpc
    thetamin, thetamax = Rmin/dz, Rmax/dz # radians
    phi = np.pi
    for i in range(n_catalogs):
        #generate random catalog
        e1, e2 = np.random.randn(ngals)*0.001, np.random.randn(ngals)*0.001
        et, ex = da._compute_tangential_shear(e1, e2, phi), da._compute_cross_shear(e1, e2, phi)
        z_gal = np.random.random(ngals)*(3 - 1.1) + 1.1
        id_gal = np.arange(ngals)
        theta_gal = np.linspace(0,1,ngals)*(thetamax - thetamin) + thetamin
        w_ls = np.zeros(ngals) + 1.
        data = {'theta':theta_gal, 'z':z_gal, 'id':id_gal, 'e1':e1, 'e2':e2, 'et':et, 'ex':ex, 'w_ls':w_ls}
        cl = clmm.GalaxyCluster('mock_cluster', cluster_ra[i], cluster_dec[i], 1., GCData(data))
        gclist.append(cl)
    ensemble_id = 1
    names = ['id', 'ra', 'dec', 'z', 'radius', 'gt', 'gx', 'W_l']
    bins = np.logspace(np.log10(0.3),np.log10(5), 10)
    
    #test without args, kwargs
    ce = clusterensemble.ClusterEnsemble(ensemble_id, gclist)
    assert_raises(KeyError, ce.make_stacked_radial_profile)
    
    #test with args, kwargs
    ce = clusterensemble.ClusterEnsemble(ensemble_id, gclist, tan_component_in='et',
    cross_component_in='ex', weights_in = 'w_ls', bins=bins, bin_units='Mpc', cosmo=cosmo)
    ce.make_stacked_radial_profile()
    
    #comparing brut force calculation for cross and tangential component
    gt_individual, gx_individual = ce.data['gt'], ce.data['gx']
    Wl_individual = ce.data['W_l']
    gt_stack = np.average(gt_individual, weights=Wl_individual, axis = 0)
    gx_stack = np.average(gx_individual, weights=Wl_individual, axis = 0)
    gt_stack_method = ce.stacked_data['gt']
    gx_stack_method = ce.stacked_data['gx']
    assert_equal(gt_stack,gt_stack_method)
    assert_equal(gx_stack,gx_stack_method)
    ce.compute_sample_covariance()
    ce.compute_bootstrap_covariance(n_bootstrap=3)
    ce.compute_jackknife_covariance(n_side=2)
    
    #cross vs tangential covariances within a method -> shapes
    assert_equal(ce.sample_tangential_covariance.shape,ce.sample_cross_covariance.shape )
    assert_equal(ce.bootstrap_tangential_covariance.shape,ce.bootstrap_cross_covariance.shape )
    assert_equal(ce.jackknife_tangential_covariance.shape,ce.jackknife_cross_covariance.shape )
    
    #comparing covariance estimation methods -> shapes
    assert_equal(ce.sample_tangential_covariance.shape,ce.bootstrap_tangential_covariance.shape )
    assert_equal(ce.bootstrap_tangential_covariance.shape,ce.jackknife_tangential_covariance.shape )
    assert_equal(ce.jackknife_tangential_covariance.shape,ce.sample_tangential_covariance.shape )
    
    #comparing brut force calculation for sample variance
    std_gt_stack = np.std(gt_individual, axis = 0)/np.sqrt(n_catalogs-1)
    assert_allclose(ce.sample_tangential_covariance.diagonal()**.5, std_gt_stack, 1e-6)