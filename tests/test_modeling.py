"""
Tests for modeling
"""

import astropy
from astropy import cosmology
from numpy import testing as tst
import numpy as np
import clmm
# from clmm import modeling as pp
# from modeling import *

import pickle as pkl
example_case = pkl.load( open( "../support/example_case.p", "rb" ) )

#density_profile_parametrization = 'nfw'
density_profile_parametrization = example_case['density_profile_parametrization']
#mass_Delta = 200
mass_Delta = example_case['mass_Delta']
#cluster_mass = 1.e15
cluster_mass = example_case['cluster_mass']
#cluster_concentration = 4
cluster_concentration = example_case['cluster_concentration']
z_max = example_case['z_max']
#z_cluster = 1.
z_cluster = example_case['z_cluster']
z_source = example_case['z_source']

# cosmo_ccl = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)
#cosmo_apy= astropy.cosmology.core.FlatLambdaCDM(H0=70, Om0=0.27, Ob0=0.045)
cosmo_apy = example_case['astropy_cosmology_object']
cosmo_ccl = clmm.cclify_astropy_cosmo(cosmo_apy)

def test_cosmo_type():
    tst.assert_equal(type(cosmo_apy), astropy.cosmology.core.FlatLambdaCDM)
    tst.assert_equal(type(cosmo_ccl), dict)
    tst.assert_equal(cosmo_ccl['Omega_c'] + cosmo_ccl['Omega_b'], cosmo_apy.Odm0 + cosmo_apy.Ob0)

#r3d = np.logspace(-2, 2, 100)
r3d = example_case['r3d']
r3d_one = r3d[-1]

rho = clmm.get_3d_density(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, z_cl=z_cluster, cosmo=cosmo_ccl)
rho_one = clmm.get_3d_density(r3d_one, mdelta=cluster_mass, cdelta=cluster_concentration, z_cl=z_cluster, cosmo=cosmo_ccl)

def test_rho():
    # consistency test
    tst.assert_equal(rho[-1], rho_one)
    # physical value test
    tst.assert_allclose(example_case['nc_rho'], rho, 1e-8)

Sigma = clmm.predict_surface_density(r3d, cluster_mass, cluster_concentration, z_cl=z_cluster, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw')
Sigma_one = clmm.predict_surface_density(r3d_one, cluster_mass, cluster_concentration, z_cl=z_cluster, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw')

def test_Sigma():
    assert(np.all(Sigma > 0.))
    tst.assert_equal(Sigma[-1], Sigma_one)

DeltaSigma = clmm.predict_excess_surface_density(r3d, cluster_mass, cluster_concentration, z_cl=z_cluster, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw')
DeltaSigma_one = clmm.predict_excess_surface_density(r3d_one, cluster_mass, cluster_concentration, z_cl=z_cluster, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw')

def test_DeltaSigma():
    tst.assert_equal(DeltaSigma[-1], DeltaSigma_one)
    assert(np.all(DeltaSigma > 0.))
    assert(DeltaSigma_one > 0.)

Sigmac = clmm.get_critical_surface_density(cosmo_ccl, z_cluster=1.0, z_source=2.0)

# def test_Sigmac():
    # not sure what to put here yet

gammat = clmm.predict_tangential_shear(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=1.0, z_source=2.0, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw', z_src_model='single_plane')
gammat_one = clmm.predict_tangential_shear(r3d_one, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=1.0, z_source=2.0, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw', z_src_model='single_plane')

def test_gammat():
    tst.assert_equal(gammat[-1], gammat_one)
    tst.assert_equal(gammat, DeltaSigma / Sigmac)
    tst.assert_equal(gammat_one, DeltaSigma_one / Sigmac)

kappa = clmm.predict_convergence(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=1.0, z_source=2.0, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw', z_src_model='single_plane')
kappa_one = clmm.predict_convergence(r3d_one, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=1.0, z_source=2.0, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw', z_src_model='single_plane')

def test_kappa():
    tst.assert_equal(kappa[-1], kappa_one)
    assert(kappa_one > 0.)
    assert(np.all(kappa > 0.))

gt = clmm.predict_reduced_tangential_shear(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=1.0, z_source=2.0, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw', z_src_model='single_plane')
gt_one = clmm.predict_reduced_tangential_shear(r3d_one, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=1.0, z_source=2.0, cosmo=cosmo_ccl, delta_mdef=200, halo_profile_model='nfw', z_src_model='single_plane')

def test_gt():
    tst.assert_equal(gt[-1], gt_one)
    tst.assert_equal(gt, gammat / (1. - kappa))
    tst.assert_equal(gt_one, gammat_one / (1. - kappa_one))

# others: test that inputs are as expected, values from demos
# positive values from sigma onwards
