"""
Tests for modeling
"""

import astropy
from astropy import cosmology, constants
from numpy import testing as tst
import numpy as np
import clmm

import pickle as pkl
import os
code_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
example_case = pkl.load( open( "%s/examples/support/example_case.p"%code_path, "rb" ) )

# Account for values of constants in different versions of astropy
##c2_over_G = constants.c.to(units.pc/units.s).value**2/constants.G.to(units.pc**3/units.M_sun/units.s**2).value
##corr_factor = example_case['c2_over_G']/c2_over_G
corr_factor = constants.G.to(units.pc**3/units.M_sun/units.s**2).value/example_case['G']

# Cluster parameters
density_profile_parametrization = example_case['density_profile_parametrization']
mass_Delta = example_case['mass_Delta']/corr_factor
cluster_mass = example_case['cluster_mass']
cluster_concentration = example_case['cluster_concentration']
z_max = example_case['z_max']
z_cluster = example_case['z_cluster']
z_source = example_case['z_source']

# Cosmology
#cosmo_apy= astropy.cosmology.core.FlatLambdaCDM(H0=70, Om0=0.27, Ob0=0.045)
cosmo_apy = example_case['astropy_cosmology_object']
cosmo_ccl = clmm.cclify_astropy_cosmo(cosmo_apy)

# vectors
r3d = example_case['r3d']
r3d_one = r3d[-1]

def test_cosmo_type():
    # consistency test
    tst.assert_equal(type(cosmo_apy), astropy.cosmology.core.FlatLambdaCDM)
    tst.assert_equal(type(cosmo_ccl), dict)
    tst.assert_equal(cosmo_ccl['Omega_c'] + cosmo_ccl['Omega_b'], cosmo_apy.Odm0 + cosmo_apy.Ob0)


rho_params = {'mdelta':cluster_mass, 'cdelta':cluster_concentration, 'z_cl':z_cluster, 'cosmo':cosmo_ccl}
rho = clmm.get_3d_density(r3d, **rho_params)*corr_factor
rho_one = clmm.get_3d_density(r3d_one, **rho_params)*corr_factor

def test_rho():
    # consistency test
    tst.assert_equal(rho[-1], rho_one)
    # physical value test
    tst.assert_allclose(example_case['nc_rho'], rho, 1e-11)


Sigma_params = {'mdelta':cluster_mass, 'cdelta':cluster_concentration, 'z_cl':z_cluster, 'cosmo':cosmo_ccl, 'delta_mdef':mass_Delta, 'halo_profile_model':density_profile_parametrization}
Sigma = clmm.predict_surface_density(r3d, **Sigma_params)*corr_factor
Sigma_one = clmm.predict_surface_density(r3d_one, **Sigma_params)*corr_factor


def test_Sigma():
    # consistency test
    assert(np.all(Sigma > 0.))
    tst.assert_equal(Sigma[-1], Sigma_one)
    # physical value test
    tst.assert_allclose(example_case['nc_Sigma'], Sigma*constants_corr, 1e-9)

DeltaSigma = clmm.predict_excess_surface_density(r3d, **Sigma_params)*corr_factor
DeltaSigma_one = clmm.predict_excess_surface_density(r3d_one, **Sigma_params)*corr_factor

def test_DeltaSigma():
    # consistency test
    tst.assert_equal(DeltaSigma[-1], DeltaSigma_one)
    assert(np.all(DeltaSigma > 0.))
    assert(DeltaSigma_one > 0.)
    # physical value test
    tst.assert_allclose(example_case['nc_DeltaSigma'], DeltaSigma, 1e-8)

Sigmac = clmm.get_critical_surface_density(cosmo_ccl, z_cluster=z_cluster, z_source=z_source)*corr_factor

def test_Sigmac():
    # physical value test
    tst.assert_allclose(example_case['nc_Sigmac'], Sigmac, 1e-8)

gamma_params = {'mdelta':cluster_mass, 'cdelta':cluster_concentration, 'z_cluster':z_cluster, 'z_source':z_source, 'cosmo':cosmo_ccl, 'delta_mdef':mass_Delta, 'halo_profile_model':density_profile_parametrization, 'z_src_model':'single_plane'}
gammat = clmm.predict_tangential_shear(r3d, **gamma_params)*corr_factor
gammat_one = clmm.predict_tangential_shear(r3d_one, **gamma_params)*corr_factor

def test_gammat():
    # consistency test
    tst.assert_equal(gammat[-1], gammat_one)
    tst.assert_equal(gammat, DeltaSigma / Sigmac)
    tst.assert_equal(gammat_one, DeltaSigma_one / Sigmac)
    # physical value test
    tst.assert_allclose(example_case['nc_gammat'], gammat, 1e-8)

kappa = clmm.predict_convergence(r3d, **gamma_params)*corr_factor
kappa_one = clmm.predict_convergence(r3d_one, **gamma_params)*corr_factor

def test_kappa():
    # consistency test
    tst.assert_equal(kappa[-1], kappa_one)
    assert(kappa_one > 0.)
    assert(np.all(kappa > 0.))
    # physical value test
    tst.assert_allclose(example_case['nc_kappa'], kappa, 1e-8)

gt = clmm.predict_reduced_tangential_shear(r3d, **gamma_params)
gt_one = clmm.predict_reduced_tangential_shear(r3d_one, **gamma_params)

def test_gt():
    # consistency test
    tst.assert_equal(gt[-1], gt_one)
    tst.assert_equal(gt, gammat / (cor_factor - kappa))
    tst.assert_equal(gt_one, gammat_one / (cor_factor - kappa_one))
    # physical value test
    tst.assert_allclose(example_case['nc_gt'], gt, 1e-6)

# others: test that inputs are as expected, values from demos
# positive values from sigma onwards
if __name__=='__main__':
    test_cosmo_type()
    test_rho()
    test_Sigma()
    test_Sigmac()
    test_DeltaSigma()
    test_gammat()
    test_kappa()
    test_gt()
