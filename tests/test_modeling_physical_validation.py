"""
Tests for modeling
"""

import astropy
from astropy import cosmology, constants, units
from numpy import testing as tst
import numpy as np
import clmm
from clmm.constants import Constants as clmmconst

# Read test case
import os
code_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
import ast
with open('%s/tests/physical_values_nc/float_vals.txt'%code_path, 'r') as f:
    test_case = ast.literal_eval(f.read())
f.close()
nc_prof = np.genfromtxt('%s/tests/physical_values_nc/numcosmo_profiles.txt'%code_path, names=True)
nc_dist = np.genfromtxt('%s/tests/physical_values_nc/numcosmo_angular_diameter_distance.txt'%code_path, names=True)

r3d = np.array(nc_prof['r3d'])

# Physical Constants
G_physconst_correction = test_case['G[m3/km.s2]']/clmmconst.GNEWT.value
clmm_sigmac_physical_constant = (clmmconst.CLIGHT_KMS.value * 1000. / clmmconst.PC_TO_METER.value)**2/(clmmconst.GNEWT.value * clmmconst.SOLAR_MASS.value / clmmconst.PC_TO_METER.value**3)
test_case_sigmac_physical_constant = (test_case['lightspeed[km/s]']*1000./test_case['pc_to_m'])**2/(test_case['G[m3/km.s2]']*test_case['Msun[kg]']/test_case['pc_to_m']**3)
sigmac_physconst_correction = test_case_sigmac_physical_constant/clmm_sigmac_physical_constant

# Cosmology
#cosmo_apy= astropy.cosmology.core.FlatLambdaCDM(H0=70, Om0=0.27, Ob0=0.045)
cosmo_apy= astropy.cosmology.core.FlatLambdaCDM(H0=test_case['cosmo_H0'],
        Om0=test_case['cosmo_Om0'], Ob0=test_case['cosmo_Ob0'])
cosmo_ccl = clmm.cclify_astropy_cosmo(cosmo_apy)

# Sets of parameters to be used by multiple functions
rho_params = {
        'mdelta':test_case['cluster_mass'],
        'cdelta':test_case['cluster_concentration'],
        'z_cl':test_case['z_cluster'],
        'cosmo':cosmo_ccl,
        }
Sigma_params = {
        'mdelta':test_case['cluster_mass'],
        'cdelta':test_case['cluster_concentration'],
        'z_cl':test_case['z_cluster'],
        'cosmo':cosmo_ccl,
        'delta_mdef':test_case['mass_Delta'],
        'halo_profile_model':test_case['density_profile_parametrization'],
        }
gamma_params = {
        'mdelta':test_case['cluster_mass'],
        'cdelta':test_case['cluster_concentration'],
        'z_cluster':test_case['z_cluster'],
        'z_source':test_case['z_source'],
        'cosmo':cosmo_ccl,
        'delta_mdef':test_case['mass_Delta'],
        'halo_profile_model':test_case['density_profile_parametrization'],
        'z_src_model':'single_plane',
        }


def test_cosmo_type():
    # consistency test
    tst.assert_equal(type(cosmo_apy), astropy.cosmology.core.FlatLambdaCDM)
    tst.assert_equal(type(cosmo_ccl), dict)
    tst.assert_equal(cosmo_ccl['Omega_c'] + cosmo_ccl['Omega_b'], cosmo_apy.Odm0 + cosmo_apy.Ob0)

def test_physical_constants():
    tst.assert_allclose(test_case['lightspeed[km/s]'], clmmconst.CLIGHT_KMS.value, 1e-3)
    tst.assert_allclose(test_case['G[m3/km.s2]'], clmmconst.GNEWT.value, 1e-3)
    tst.assert_allclose(test_case['pc_to_m'], clmmconst.PC_TO_METER.value, 1e-6)
    tst.assert_allclose(test_case['Msun[kg]'], clmmconst.SOLAR_MASS.value, 1e-2)
    tst.assert_allclose(test_case_sigmac_physical_constant, clmm_sigmac_physical_constant, 1e-2)

def test_rho():
    rho = clmm.get_3d_density(r3d, **rho_params)
    rho_one = clmm.get_3d_density(r3d[-1], **rho_params)
    # consistency test
    tst.assert_equal(rho[-1], rho_one)
    # physical value test
    tst.assert_allclose(nc_prof['rho'], rho*G_physconst_correction, 1e-11)

def test_Sigma():
    Sigma = clmm.predict_surface_density(r3d, **Sigma_params)
    Sigma_one = clmm.predict_surface_density(r3d[-1], **Sigma_params)
    # consistency test
    assert(np.all(Sigma > 0.))
    tst.assert_equal(Sigma[-1], Sigma_one)
    # physical value test
    tst.assert_allclose(nc_prof['Sigma'], Sigma*G_physconst_correction, 1e-9)

def test_DeltaSigma():
    DeltaSigma = clmm.predict_excess_surface_density(r3d, **Sigma_params)
    DeltaSigma_one = clmm.predict_excess_surface_density(r3d[-1], **Sigma_params)
    # consistency test
    tst.assert_equal(DeltaSigma[-1], DeltaSigma_one)
    assert(np.all(DeltaSigma > 0.))
    assert(DeltaSigma_one > 0.)
    # physical value test
    tst.assert_allclose(nc_prof['DeltaSigma'], DeltaSigma*G_physconst_correction, 1e-8)

def test_modeling_get_a_from_z():
    aexp_cluster = clmm.modeling._get_a_from_z(test_case['z_cluster'])
    aexp_source = clmm.modeling._get_a_from_z(test_case['z_source'])    
    tst.assert_allclose(test_case['aexp_cluster'], aexp_cluster, 1e-10)
    tst.assert_allclose(test_case['aexp_source'], aexp_source, 1e-10)

def test_get_angular_diameter_distance_a():
    dl = clmm.modeling.get_angular_diameter_distance_a(cosmo_ccl,
        test_case['aexp_cluster'])
    ds = clmm.modeling.get_angular_diameter_distance_a(cosmo_ccl,
        test_case['aexp_source'])
    dsl = clmm.modeling.get_angular_diameter_distance_a(cosmo_ccl,
        test_case['aexp_source'], test_case['aexp_cluster'])
    tst.assert_allclose(test_case['dl'], dl, 1e-10)
    tst.assert_allclose(test_case['ds'], ds, 1e-10)
    tst.assert_allclose(test_case['dsl'], dsl, 1e-10)

def test_Sigmac():
    # final test
    Sigmac = clmm.get_critical_surface_density(cosmo_ccl,
        z_cluster=test_case['z_cluster'],
        z_source=test_case['z_source'])
    # physical value test
    tst.assert_allclose(test_case['nc_Sigmac'], Sigmac*sigmac_physconst_correction, 1e-8)

def test_gammat():
    DeltaSigma = clmm.predict_excess_surface_density(r3d, **Sigma_params)
    DeltaSigma_one = clmm.predict_excess_surface_density(r3d[-1], **Sigma_params)
    Sigmac = clmm.get_critical_surface_density(cosmo_ccl, z_cluster=test_case['z_cluster'], z_source=test_case['z_source'])
    gammat = clmm.predict_tangential_shear(r3d, **gamma_params)
    gammat_one = clmm.predict_tangential_shear(r3d[-1], **gamma_params)
    # consistency test
    tst.assert_equal(gammat[-1], gammat_one)
    tst.assert_equal(gammat, DeltaSigma / Sigmac)
    tst.assert_equal(gammat_one, DeltaSigma_one / Sigmac)
    # physical value test
    tst.assert_allclose(nc_prof['gammat'], gammat/sigmac_physconst_correction, 1e-8)

def test_kappa():
    kappa = clmm.predict_convergence(r3d, **gamma_params)
    kappa_one = clmm.predict_convergence(r3d[-1], **gamma_params)
    # consistency test
    tst.assert_equal(kappa[-1], kappa_one)
    assert(kappa_one > 0.)
    assert(np.all(kappa > 0.))
    # physical value test
    tst.assert_allclose(nc_prof['kappa'], kappa/sigmac_physconst_correction, 1e-8)

def test_gt():
    gammat = clmm.predict_tangential_shear(r3d, **gamma_params)
    gammat_one = clmm.predict_tangential_shear(r3d[-1], **gamma_params)
    kappa = clmm.predict_convergence(r3d, **gamma_params)
    kappa_one = clmm.predict_convergence(r3d[-1], **gamma_params)
    gt = clmm.predict_reduced_tangential_shear(r3d, **gamma_params)
    gt_one = clmm.predict_reduced_tangential_shear(r3d[-1], **gamma_params)
    # consistency test
    tst.assert_equal(gt[-1], gt_one)
    tst.assert_equal(gt, gammat / (1. - kappa))
    tst.assert_equal(gt_one, gammat_one / (1. - kappa_one))
    # physical value test
    tst.assert_allclose(nc_prof['gt'], gammat/(sigmac_physconst_correction-kappa), 1e-6)

# others: test that inputs are as expected, values from demos
# positive values from sigma onwards
if __name__=='__main__':
    test_cosmo_type()
    test_physical_constants()
    test_rho()
    test_Sigma()
    test_modeling_get_a_from_z()
    test_get_angular_diameter_distance_a()
    test_Sigmac()
    test_DeltaSigma()
    test_gammat()
    test_kappa()
    test_gt()
