"""
tests for clusterensemble.py
"""
import os
from numpy.testing import assert_raises, assert_equal, assert_allclose, assert_array_equal
import clmm
import numpy as np
from clmm import ClusterEnsemble
from clmm import Cosmology
from clmm import GCData
from clmm.support import mock_data as mock
import matplotlib.pyplot as plt
import clmm.dataops as da

TOLERANCE = {"rtol": 5.0e-4, "atol": 1.0e-4}


def test_cluster_ensemble():
    """test clusterensemble attributes"""
    cosmo = Cosmology(H0=70, Omega_dm0=0.262, Omega_b0=0.049)
    # create galaxycluster object
    ra_lens, dec_lens, z_lens = 120.0, 42.0, 0.5
    ra_source = [120.1, 119.9]
    dec_source = [41.9, 42.2]
    theta_source = [0.0025, 0.015]
    z_src = [1.0, 2.0]
    shear1 = [0.2, 0.4]
    shear2 = [0.3, 0.5]
    w_ls = [1.0e-30, 1.0e-31]
    # Set up radial values
    bins_radians = np.logspace(np.log10(0.001), np.log10(0.02), 10)
    bin_units = "radians"
    names = ("ra", "dec", "theta", "w_ls", "e1", "e2", "z")

    galcat = clmm.GCData(
        [ra_source, dec_source, theta_source, w_ls, shear1, shear2, z_src], names=names
    )
    # create cluster
    cluster = clmm.GalaxyCluster(
        unique_id="test", ra=ra_lens, dec=dec_lens, z=z_lens, galcat=galcat
    )
    cluster_quad = clmm.GalaxyCluster(
        unique_id="test", ra=ra_lens, dec=dec_lens, z=z_lens, galcat=galcat, include_quadrupole=True
    )
    cluster.compute_tangential_and_cross_components()
    cluster_quad.compute_tangential_and_cross_components(
        phi_major=0.0,
        info_mem=[np.array([119.9, 120.1]), np.array([41.9, 42.1]), np.array([1.0, 1.0])],
    )
    bins = bins_radians
    gc_list = [cluster]
    gc_list_quad = [cluster_quad]
    # check empty cluster list
    ce_empty = ClusterEnsemble(
        "cluster_ensemble",
        tan_component_in="et",
        cross_component_in="ex",
        weights_in="w_ls",
        bins=bins,
        bin_units="radians",
        cosmo=cosmo,
    )
    ce_empty_quad = ClusterEnsemble(
        "cluster_ensemble",
        tan_component_in="et",
        cross_component_in="ex",
        quad_4theta_component_in="e4theta",
        quad_const_component_in="econst",
        weights_in="w_ls",
        bins=bins,
        bin_units="radians",
        cosmo=cosmo,
        include_quadrupole=True,
    )
    assert_raises(ValueError, ce_empty.make_stacked_radial_profile)
    assert_raises(ValueError, ce_empty_quad.make_stacked_radial_profile)
    ce_empty.make_individual_radial_profile(
        cluster,
        tan_component_in="et",
        cross_component_in="ex",
        weights_in="w_ls",
        bins=bins,
        bin_units="radians",
        cosmo=cosmo,
    )
    ce_empty_quad.make_individual_radial_profile(
        cluster_quad,
        tan_component_in="et",
        cross_component_in="ex",
        quad_4theta_component_in="e4theta",
        quad_const_component_in="econst",
        weights_in="w_ls",
        bins=bins,
        bin_units="radians",
        cosmo=cosmo,
    )

    # check bad id
    assert_raises(TypeError, ClusterEnsemble, 1.3, gc_list)
    assert_raises(TypeError, ClusterEnsemble, 1.3, gc_list_quad)

    # test without kwargs, args
    ce = ClusterEnsemble(
        "cluster_ensemble",
        gc_list,
        tan_component_in="et",
        cross_component_in="ex",
        weights_in="w_ls",
        bins=bins,
        bin_units="radians",
        cosmo=cosmo,
    )
    ce_quad = ClusterEnsemble(
        "cluster_ensemble",
        gc_list_quad,
        tan_component_in="et",
        cross_component_in="ex",
        quad_4theta_component_in="e4theta",
        quad_const_component_in="econst",
        weights_in="w_ls",
        bins=bins,
        bin_units="radians",
        cosmo=cosmo,
        include_quadrupole=True,
    )

    # test the lenght of the clusterensemble data attribute
    assert_equal(ce.__len__(), 1)
    assert_equal(ce_quad.__len__(), 1)

    # test the lenght of the clusterensemble data attribute (after doubling the number of individual cluster)
    ce._add_values(
        [cluster],
        tan_component_in="et",
        cross_component_in="ex",
        weights_in="w_ls",
        bins=bins,
        bin_units="radians",
        cosmo=cosmo,
    )
    ce_quad._add_values(
        [cluster_quad],
        tan_component_in="et",
        cross_component_in="ex",
        quad_4theta_component_in="e4theta",
        quad_const_component_in="econst",
        weights_in="w_ls",
        bins=bins,
        bin_units="radians",
        cosmo=cosmo,
    )
    assert_equal(ce.__len__(), 2)
    assert_equal(ce_quad.__len__(), 2)
    # test if the len of averaged profile has the lenght of binning axis
    assert_equal(len(ce.data["W_l"][0]), len(bins_radians) - 1)
    assert_equal(ce.__getitem__("gt"), ce.data["gt"])
    assert_equal(len(ce_quad.data["W_l"][0]), len(bins_radians) - 1)
    assert_equal(ce_quad.__getitem__("gconst"), ce_quad.data["gconst"])


def test_covariance():
    """test the shapes of covariance matrix with different methods"""
    cosmo = Cosmology(H0=70, Omega_dm0=0.262, Omega_b0=0.049)
    dz = cosmo.eval_da_z1z2(0.0, 1.0)
    n_catalogs = 3
    ngals = 1000
    cluster_ra = [50, 100, 200]  # from 0 to 360 deg
    sindec = [-0.1, 0, 0.1]
    cluster_dec = np.arcsin(sindec) * 180 / np.pi  # from -90 to 90 deg
    gclist = []
    gclist_quad = []
    Rmin, Rmax = 0.3, 5  # Mpc
    thetamin, thetamax = Rmin / dz, Rmax / dz  # radians
    phi = np.pi
    bins = np.logspace(np.log10(0.3), np.log10(5), 10)
    ce_empty = ClusterEnsemble(
        "2",
        tan_component_in="et",
        cross_component_in="ex",
        weights_in="w_ls",
        bins=bins,
        bin_units="radians",
        cosmo=cosmo,
    )
    ce_empty_quad = ClusterEnsemble(
        "2",
        tan_component_in="et",
        cross_component_in="ex",
        quad_4theta_component_in="e4theta",
        quad_const_component_in="econst",
        weights_in="w_ls",
        bins=bins,
        bin_units="radians",
        cosmo=cosmo,
        include_quadrupole=True,
    )

    # check empty cluster list
    assert_raises(ValueError, ce_empty.compute_sample_covariance)
    assert_raises(ValueError, ce_empty.compute_bootstrap_covariance)
    assert_raises(ValueError, ce_empty.compute_jackknife_covariance)
    assert_raises(ValueError, ce_empty_quad.compute_sample_covariance)
    assert_raises(ValueError, ce_empty_quad.compute_bootstrap_covariance)
    assert_raises(ValueError, ce_empty_quad.compute_jackknife_covariance)

    for i in range(n_catalogs):
        # generate random catalog
        e1, e2 = np.random.randn(ngals) * 0.001, np.random.randn(ngals) * 0.001
        et, ex = da._compute_tangential_shear(e1, e2, phi), da._compute_cross_shear(e1, e2, phi)
        eft = da._compute_4theta_shear(e1, e2, phi)
        ecn = e1
        z_gal = np.random.random(ngals) * (3 - 1.1) + 1.1
        id_gal = np.arange(ngals)
        theta_gal = np.linspace(0, 1, ngals) * (thetamax - thetamin) + thetamin
        w_ls = np.zeros(ngals) + 1.0
        data = {
            "theta": theta_gal,
            "z": z_gal,
            "id": id_gal,
            "e1": e1,
            "e2": e2,
            "et": et,
            "ex": ex,
            "e4theta": eft,
            "econst": ecn,
            "w_ls": w_ls,
        }
        cl = clmm.GalaxyCluster("mock_cluster", cluster_ra[i], cluster_dec[i], 1.0, GCData(data))
        cl_quad = clmm.GalaxyCluster(
            "mock cluster",
            cluster_ra[i],
            cluster_dec[i],
            1.0,
            GCData(data),
            include_quadrupole=True,
        )
        gclist.append(cl)
        gclist_quad.append(cl_quad)
        ce_empty.make_individual_radial_profile(
            galaxycluster=cl,
            tan_component_in="et",
            cross_component_in="ex",
            weights_in="w_ls",
            bins=bins,
            bin_units="Mpc",
            cosmo=cosmo,
        )
        ce_empty_quad.make_individual_radial_profile(
            galaxycluster=cl_quad,
            tan_component_in="et",
            cross_component_in="ex",
            quad_4theta_component_in="e4theta",
            quad_const_component_in="econst",
            weights_in="w_ls",
            bins=bins,
            bin_units="Mpc",
            cosmo=cosmo,
        )

    ensemble_id = 1
    names = ["id", "ra", "dec", "z", "radius", "gt", "gx", "g4theta", "gconst", "W_l"]

    # test without args, kwargs
    ce = ClusterEnsemble(ensemble_id)
    ce_quad = ClusterEnsemble(ensemble_id, include_quadrupole=True)
    assert_raises(ValueError, ce.make_stacked_radial_profile)
    assert_raises(ValueError, ce_quad.make_stacked_radial_profile)

    # test with args, kwargs
    ce = ClusterEnsemble(
        ensemble_id,
        gclist,
        tan_component_in="et",
        cross_component_in="ex",
        weights_in="w_ls",
        bins=bins,
        bin_units="Mpc",
        cosmo=cosmo,
    )
    ce_quad = ClusterEnsemble(
        ensemble_id,
        gclist_quad,
        tan_component_in="et",
        cross_component_in="ex",
        quad_4theta_component_in="e4theta",
        quad_const_component_in="econst",
        weights_in="w_ls",
        bins=bins,
        bin_units="Mpc",
        cosmo=cosmo,
        include_quadrupole=True,
    )

    ce.make_stacked_radial_profile()
    ce_quad.make_stacked_radial_profile()

    assert_raises(ValueError, ce.make_individual_radial_profile, gclist[0], bin_units="radians")
    assert_raises(
        ValueError, ce_quad.make_individual_radial_profile, gclist_quad[0], bin_units="radians"
    )

    # test if te list object matches the calculation from the object with manually added clusters
    ce_empty.make_stacked_radial_profile()
    ce_empty_quad.make_stacked_radial_profile()
    assert_array_equal(ce_empty.stacked_data, ce.stacked_data)
    assert_array_equal(ce_empty_quad.stacked_data, ce_quad.stacked_data)

    # comparing brut force calculation for cross and tangential component
    gt_individual, gx_individual = ce.data["gt"], ce.data["gx"]
    gft_individual, gcn_individual = ce_quad.data["g4theta"], ce_quad.data["gconst"]
    Wl_individual = ce.data["W_l"]
    gt_stack = np.average(gt_individual, weights=Wl_individual, axis=0)
    gx_stack = np.average(gx_individual, weights=Wl_individual, axis=0)
    gft_stack = np.average(gft_individual, weights=Wl_individual, axis=0)
    gcn_stack = np.average(gcn_individual, weights=Wl_individual, axis=0)
    gt_stack_method = ce.stacked_data["gt"]
    gx_stack_method = ce.stacked_data["gx"]
    gft_stack_method = ce_quad.stacked_data["g4theta"]
    gcn_stack_method = ce_quad.stacked_data["gconst"]
    assert_equal(gt_stack, gt_stack_method)
    assert_equal(gx_stack, gx_stack_method)
    assert_equal(gft_stack, gft_stack_method)
    assert_equal(gcn_stack, gcn_stack_method)
    ce.compute_sample_covariance()
    ce_quad.compute_sample_covariance()
    ce.compute_bootstrap_covariance(n_bootstrap=3)
    ce_quad.compute_bootstrap_covariance(n_bootstrap=3)
    ce.compute_jackknife_covariance(n_side=2)
    ce_quad.compute_jackknife_covariance(n_side=2)

    # cross vs tangential covariances within a method -> shapes
    assert_equal(ce.cov["tan_sc"].shape, ce.cov["cross_sc"].shape)
    assert_equal(ce.cov["tan_bs"].shape, ce.cov["cross_bs"].shape)
    assert_equal(ce.cov["tan_jk"].shape, ce.cov["cross_jk"].shape)
    # 4theta vs constant covariances within a method -> shapes
    assert_equal(ce_quad.cov["quad_4theta_sc"].shape, ce_quad.cov["quad_const_sc"].shape)
    assert_equal(ce_quad.cov["quad_4theta_bs"].shape, ce_quad.cov["quad_const_bs"].shape)
    assert_equal(ce_quad.cov["quad_4theta_jk"].shape, ce_quad.cov["quad_const_jk"].shape)

    # comparing covariance estimation methods -> shapes
    assert_equal(ce.cov["tan_sc"].shape, ce.cov["tan_bs"].shape)
    assert_equal(ce.cov["tan_bs"].shape, ce.cov["tan_jk"].shape)
    assert_equal(ce.cov["tan_jk"].shape, ce.cov["tan_sc"].shape)
    # comparing covariance estimation methods -> shapes
    assert_equal(ce_quad.cov["quad_4theta_sc"].shape, ce_quad.cov["quad_4theta_bs"].shape)
    assert_equal(ce_quad.cov["quad_4theta_bs"].shape, ce_quad.cov["quad_4theta_jk"].shape)
    assert_equal(ce_quad.cov["quad_4theta_jk"].shape, ce_quad.cov["quad_4theta_sc"].shape)
    
    # comparing brute force calculation for sample variance
    std_gt_stack = np.std(gt_individual, axis=0) / np.sqrt(n_catalogs - 1)
    assert_allclose(ce.cov["tan_sc"].diagonal() ** 0.5, std_gt_stack, 1e-6)
    std_gft_stack = np.std(gft_individual, axis=0) / np.sqrt(n_catalogs - 1)
    assert_allclose(ce_quad.cov["quad_4theta_sc"].diagonal() ** 0.5, std_gft_stack, 1e-6)

    # test save/load
    ce.save("ce.test.pkl")
    ce_quad.save("ce_quad.test.pkl")

    ce2 = ClusterEnsemble.load("ce.test.pkl")
    ce_quad2 = ClusterEnsemble.load("ce_quad.test.pkl")
    os.system("rm ce.test.pkl")
    os.system("rm ce_quad.test.pkl")

    assert_array_equal(ce.stacked_data, ce2.stacked_data)
    assert_array_equal(ce_quad.stacked_data, ce_quad2.stacked_data)
    assert_equal(ce.cov["tan_sc"].shape, ce2.cov["tan_sc"].shape)
    assert_equal(ce.cov["tan_bs"].shape, ce2.cov["tan_bs"].shape)
    assert_equal(ce.cov["tan_jk"].shape, ce2.cov["tan_jk"].shape)
    assert_equal(ce.cov["cross_sc"].shape, ce2.cov["cross_sc"].shape)
    assert_equal(ce.cov["cross_bs"].shape, ce2.cov["cross_bs"].shape)
    assert_equal(ce.cov["cross_jk"].shape, ce2.cov["cross_jk"].shape)
    assert_equal(ce_quad.cov["quad_4theta_sc"].shape, ce_quad2.cov["quad_4theta_sc"].shape)
    assert_equal(ce_quad.cov["quad_4theta_bs"].shape, ce_quad2.cov["quad_4theta_bs"].shape)
    assert_equal(ce_quad.cov["quad_4theta_jk"].shape, ce_quad2.cov["quad_4theta_jk"].shape)
    assert_equal(ce_quad.cov["quad_const_sc"].shape, ce_quad2.cov["quad_const_sc"].shape)
    assert_equal(ce_quad.cov["quad_const_bs"].shape, ce_quad2.cov["quad_const_bs"].shape)
    assert_equal(ce_quad.cov["quad_const_jk"].shape, ce_quad2.cov["quad_const_jk"].shape)
