"""
Tests for modeling
"""

import os
import ast
import astropy
from numpy import testing as tst
import numpy as np
import clmm
from clmm.constants import Constants as clmmconst

# Read test case
CODE_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
VAL_FILES_PATH = '%s/tests/physical_values_nc'%CODE_PATH
with open('%s/float_vals.txt'%VAL_FILES_PATH, 'r') as f:
    TEST_CASE = ast.literal_eval(f.read())
f.close()
NC_PROF = np.genfromtxt('%s/numcosmo_profiles.txt'%VAL_FILES_PATH,
                        names=True)
NC_DIST = np.genfromtxt('%s/numcosmo_angular_diameter_distance.txt'%
                        VAL_FILES_PATH, names=True)

R3D = np.array(NC_PROF['r3d'])

# Physical Constants
G_PHYSCONST_CORRECTION = TEST_CASE['G[m3/km.s2]']/clmmconst.GNEWT.value
CLMM_SIGMAC_PHYSICAL_CONSTANT = (clmmconst.CLIGHT_KMS.value*1000./clmmconst.PC_TO_METER.value)**2/\
    (clmmconst.GNEWT.value*clmmconst.SOLAR_MASS.value/clmmconst.PC_TO_METER.value**3)
TEST_CASE_SIGMAC_PHYSICAL_CONSTANT = (TEST_CASE['lightspeed[km/s]']*1000./TEST_CASE['pc_to_m'])**2/\
    (TEST_CASE['G[m3/km.s2]']*TEST_CASE['Msun[kg]']/TEST_CASE['pc_to_m']**3)
SIGMAC_PHYSCONST_CORRECTION = TEST_CASE_SIGMAC_PHYSICAL_CONSTANT/CLMM_SIGMAC_PHYSICAL_CONSTANT

# Cosmology
#COSMO_APY = astropy.cosmology.core.FlatLambdaCDM(H0=70, Om0=0.27, Ob0=0.045)
COSMO_APY = astropy.cosmology.core.FlatLambdaCDM(H0=TEST_CASE['cosmo_H0'],
                                                 Om0=TEST_CASE['cosmo_Om0'],
                                                 Ob0=TEST_CASE['cosmo_Ob0'])
COSMO_CCL = clmm.cclify_astropy_cosmo(COSMO_APY)

# Sets of parameters to be used by multiple functions
RHO_PARAMS = {
    'mdelta':TEST_CASE['cluster_mass'],
    'cdelta':TEST_CASE['cluster_concentration'],
    'z_cl':TEST_CASE['z_cluster'],
    'cosmo':COSMO_CCL,
    }
SIGMA_PARAMS = {
    'mdelta':TEST_CASE['cluster_mass'],
    'cdelta':TEST_CASE['cluster_concentration'],
    'z_cl':TEST_CASE['z_cluster'],
    'cosmo':COSMO_CCL,
    'delta_mdef':TEST_CASE['mass_Delta'],
    'halo_profile_model':TEST_CASE['density_profile_parametrization'],
    }
GAMMA_PARAMS = {
    'mdelta':TEST_CASE['cluster_mass'],
    'cdelta':TEST_CASE['cluster_concentration'],
    'z_cluster':TEST_CASE['z_cluster'],
    'z_source':TEST_CASE['z_source'],
    'cosmo':COSMO_CCL,
    'delta_mdef':TEST_CASE['mass_Delta'],
    'halo_profile_model':TEST_CASE['density_profile_parametrization'],
    'z_src_model':'single_plane',
    }

def test_physical_constants():
    '''
    Test physical values of physical_constants

    Notes
    -----
        The precision set for these tests are somewhat arbitrary
        has to be better defined at some point
    '''
    tst.assert_allclose(TEST_CASE['lightspeed[km/s]'], clmmconst.CLIGHT_KMS.value, 1e-3)
    tst.assert_allclose(TEST_CASE['G[m3/km.s2]'], clmmconst.GNEWT.value, 1e-3)
    tst.assert_allclose(TEST_CASE['pc_to_m'], clmmconst.PC_TO_METER.value, 1e-6)
    tst.assert_allclose(TEST_CASE['Msun[kg]'], clmmconst.SOLAR_MASS.value, 1e-2)
    tst.assert_allclose(TEST_CASE_SIGMAC_PHYSICAL_CONSTANT, CLMM_SIGMAC_PHYSICAL_CONSTANT, 1e-2)

def test_rho():
    '''
    Test physical values of rho
    '''
    rho = clmm.get_3d_density(R3D, **RHO_PARAMS)
    tst.assert_allclose(NC_PROF['rho'], rho*G_PHYSCONST_CORRECTION, 1e-11)

def test_sigma():
    '''
    Test physical values of sigma
    '''
    tst.assert_allclose(NC_PROF['Sigma'], G_PHYSCONST_CORRECTION*\
                        clmm.predict_surface_density(R3D, **SIGMA_PARAMS), 1e-9)

def test_delta_sigma():
    '''
    Test physical values of delta_sigma
    '''
    tst.assert_allclose(NC_PROF['DeltaSigma'], G_PHYSCONST_CORRECTION*\
                        clmm.predict_excess_surface_density(R3D, **SIGMA_PARAMS), 1e-8)

def test_get_da():
    '''
    Test physical values of Da
    '''
    dl_clmm = clmm.modeling.get_angular_diameter_distance_a(COSMO_CCL,
                                                            TEST_CASE['aexp_cluster'])
    ds_clmm = clmm.modeling.get_angular_diameter_distance_a(COSMO_CCL,
                                                            TEST_CASE['aexp_source'])
    dsl_clmm = clmm.modeling.get_angular_diameter_distance_a(COSMO_CCL,
                                                             TEST_CASE['aexp_source'],
                                                             TEST_CASE['aexp_cluster'])
    tst.assert_allclose(TEST_CASE['dl'], dl_clmm, 1e-10)
    tst.assert_allclose(TEST_CASE['ds'], ds_clmm, 1e-10)
    tst.assert_allclose(TEST_CASE['dsl'], dsl_clmm, 1e-10)

def test_sigmac():
    '''
    Test physical values of Sigmac
    '''
    tst.assert_allclose(TEST_CASE['nc_Sigmac'], SIGMAC_PHYSCONST_CORRECTION*\
                        clmm.get_critical_surface_density(COSMO_CCL,
                                                          z_cluster=TEST_CASE['z_cluster'],
                                                          z_source=TEST_CASE['z_source']),
                        1e-8)

def test_gammat():
    '''
    Test physical values of gammat
    '''
    gammat = clmm.predict_tangential_shear(R3D, **GAMMA_PARAMS)
    tst.assert_allclose(NC_PROF['gammat'], gammat/SIGMAC_PHYSCONST_CORRECTION, 1e-8)

def test_kappa():
    '''
    Test physical values of kappa
    '''
    kappa = clmm.predict_convergence(R3D, **GAMMA_PARAMS)
    tst.assert_allclose(NC_PROF['kappa'], kappa/SIGMAC_PHYSCONST_CORRECTION, 1e-8)

def test_gt():
    '''
    Test physical values of gt
    '''
    gammat = clmm.predict_tangential_shear(R3D, **GAMMA_PARAMS)
    kappa = clmm.predict_convergence(R3D, **GAMMA_PARAMS)
    tst.assert_allclose(NC_PROF['gt'], gammat/(SIGMAC_PHYSCONST_CORRECTION-kappa), 1e-6)

# others: test that inputs are as expected, values from demos
# positive values from sigma onwards
if __name__ == '__main__':
    test_physical_constants()
    test_rho()
    test_sigma()
    test_get_da()
    test_sigmac()
    test_delta_sigma()
    test_gammat()
    test_kappa()
    test_gt()
