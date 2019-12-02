"""
Tests for modeling
"""

import os
import ast
import astropy
from numpy import testing as tst
import numpy as np
from clmm import modeling
from clmm.constants import Constants as clmmconst

def load_validation_config():
    """
    Loads values precomputed by numcosmo for comparison
    """

    class cf():
        VAL_FILES_PATH = 'tests/data/numcosmo/'
        with open('tests/data/numcosmo/config.txt', 'r') as f:
            TEST_CASE = ast.literal_eval(f.read())
        f.close()
        NC_PROF = np.genfromtxt('tests/data/numcosmo/radial_profiles.txt', names=True)
        NC_DIST = np.genfromtxt('tests/data/numcosmo/angular_diameter_distance.txt', names=True)
        
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
        COSMO_CCL = modeling.cclify_astropy_cosmo(COSMO_APY)

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

    return cf


def test_physical_constants():
    '''
    Test physical values of physical_constants

    Notes
    -----
        The precision set for these tests are somewhat arbitrary
        has to be better defined at some point
    '''
    cf = load_validation_config()
    tst.assert_allclose(cf.TEST_CASE['lightspeed[km/s]'], clmmconst.CLIGHT_KMS.value, 1e-3)
    tst.assert_allclose(cf.TEST_CASE['G[m3/km.s2]'], clmmconst.GNEWT.value, 1e-3)
    tst.assert_allclose(cf.TEST_CASE['pc_to_m'], clmmconst.PC_TO_METER.value, 1e-6)
    tst.assert_allclose(cf.TEST_CASE['Msun[kg]'], clmmconst.SOLAR_MASS.value, 1e-2)
    tst.assert_allclose(cf.TEST_CASE_SIGMAC_PHYSICAL_CONSTANT, cf.CLMM_SIGMAC_PHYSICAL_CONSTANT, 1e-2)

def test_rho():
    '''
    Test physical values of rho
    '''
    cf = load_validation_config()
    rho = modeling.get_3d_density(cf.R3D, **cf.RHO_PARAMS)
    tst.assert_allclose(cf.NC_PROF['rho'], rho*cf.G_PHYSCONST_CORRECTION, 1e-11)

def test_sigma():
    '''
    Test physical values of sigma
    '''
    cf = load_validation_config()
    tst.assert_allclose(cf.NC_PROF['Sigma'], cf.G_PHYSCONST_CORRECTION*\
                        modeling.predict_surface_density(cf.R3D, **cf.SIGMA_PARAMS), 1e-9)

def test_delta_sigma():
    '''
    Test physical values of delta_sigma
    '''
    cf = load_validation_config()
    tst.assert_allclose(cf.NC_PROF['DeltaSigma'], cf.G_PHYSCONST_CORRECTION*\
                        modeling.predict_excess_surface_density(cf.R3D, **cf.SIGMA_PARAMS), 1e-8)

def test_get_da():
    '''
    Test physical values of Da
    '''
    cf = load_validation_config()
    dl_clmm = modeling.get_angular_diameter_distance_a(cf.COSMO_CCL,
                                                            cf.TEST_CASE['aexp_cluster'])
    ds_clmm = modeling.get_angular_diameter_distance_a(cf.COSMO_CCL,
                                                            cf.TEST_CASE['aexp_source'])
    dsl_clmm = modeling.get_angular_diameter_distance_a(cf.COSMO_CCL,
                                                             cf.TEST_CASE['aexp_source'],
                                                             cf.TEST_CASE['aexp_cluster'])
    tst.assert_allclose(cf.TEST_CASE['dl'], dl_clmm, 1e-10)
    tst.assert_allclose(cf.TEST_CASE['ds'], ds_clmm, 1e-10)
    tst.assert_allclose(cf.TEST_CASE['dsl'], dsl_clmm, 1e-10)

def test_sigmac():
    '''
    Test physical values of Sigmac
    '''
    cf = load_validation_config()
    tst.assert_allclose(cf.TEST_CASE['nc_Sigmac'], cf.SIGMAC_PHYSCONST_CORRECTION*\
                        modeling.get_critical_surface_density(cf.COSMO_CCL,
                                                          z_cluster=cf.TEST_CASE['z_cluster'],
                                                          z_source=cf.TEST_CASE['z_source']),
                        1e-8)

def test_gammat():
    '''
    Test physical values of gammat
    '''
    cf = load_validation_config()
    gammat = modeling.predict_tangential_shear(cf.R3D, **cf.GAMMA_PARAMS)
    tst.assert_allclose(cf.NC_PROF['gammat'], gammat/cf.SIGMAC_PHYSCONST_CORRECTION, 1e-8)

def test_kappa():
    '''
    Test physical values of kappa
    '''
    cf = load_validation_config()
    kappa = modeling.predict_convergence(cf.R3D, **cf.GAMMA_PARAMS)
    tst.assert_allclose(cf.NC_PROF['kappa'], kappa/cf.SIGMAC_PHYSCONST_CORRECTION, 1e-8)

def test_gt():
    '''
    Test physical values of gt
    '''
    cf = load_validation_config()
    gammat = modeling.predict_tangential_shear(cf.R3D, **cf.GAMMA_PARAMS)
    kappa = modeling.predict_convergence(cf.R3D, **cf.GAMMA_PARAMS)
    tst.assert_allclose(cf.NC_PROF['gt'], gammat/(cf.SIGMAC_PHYSCONST_CORRECTION-kappa), 1e-6)

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
