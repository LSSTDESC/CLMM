"""
Tests for modeling
"""

import ast
import astropy
from numpy import testing as tst
import numpy as np
from clmm.modeling import cclify_astropy_cosmo, get_3d_density, predict_surface_density,\
    predict_excess_surface_density, get_angular_diameter_distance_a,\
    get_critical_surface_density, predict_tangential_shear, predict_convergence
from clmm.constants import Constants as clc

def compute_sigmac_physical_constant(lightspeed, gnewt, msun, pc_to_m):
    """
    Computes physical constant used to in Sigma_crit

    Parameters
    ----------
    lightspeed,: float
        Lightspeed in km/s
    gnewt: float
        Gravitational constant in m^3/(km s^2)
    msun: float
        Solar mass in kg
    pc_to_m: float
        Value of 1 parsec in meters

    Returns
    -------
    float
        lightspeed^2/G[Msun/pc]
    """
    return (lightspeed*1000./pc_to_m)**2/(gnewt*msun/pc_to_m**3)

def load_validation_config():
    """
    Loads values precomputed by numcosmo for comparison
    """

    class Config:
        """
        Object to pass loaded values
        """
        VAL_FILES_PATH = 'tests/data/numcosmo/'
        with open('tests/data/numcosmo/config.txt', 'r') as f:
            TEST_CASE = ast.literal_eval(f.read())
        f.close()
        NC_PROF = np.genfromtxt('tests/data/numcosmo/radial_profiles.txt', names=True)
        NC_DIST = np.genfromtxt('tests/data/numcosmo/angular_diameter_distance.txt', names=True)

        r3d = np.array(NC_PROF['r3d'])
        # Physical Constants
        G_PHYSCONST_CORRECTION = TEST_CASE['G[m3/km.s2]']/clc.GNEWT.value

        CLMM_SIGMAC_PCST = compute_sigmac_physical_constant(
            clc.CLIGHT_KMS.value, clc.GNEWT.value,
            clc.SOLAR_MASS.value, clc.PC_TO_METER.value)
        TEST_CASE_SIGMAC_PCST = compute_sigmac_physical_constant(
            TEST_CASE['lightspeed[km/s]'], TEST_CASE['G[m3/km.s2]'],
            TEST_CASE['Msun[kg]'], TEST_CASE['pc_to_m'])
        SIGMAC_PHYSCONST_CORRECTION = TEST_CASE_SIGMAC_PCST/CLMM_SIGMAC_PCST

        # Cosmology
        #cosmo_apy = astropy.cosmology.core.FlatLambdaCDM(H0=70, Om0=0.27, Ob0=0.045)
        cosmo_apy = astropy.cosmology.core.FlatLambdaCDM(H0=TEST_CASE['cosmo_H0'],
                                                         Om0=TEST_CASE['cosmo_Om0'],
                                                         Ob0=TEST_CASE['cosmo_Ob0'])
        cosmo_ccl = cclify_astropy_cosmo(cosmo_apy)

        # Sets of parameters to be used by multiple functions
        RHO_PARAMS = {
            'mdelta':TEST_CASE['cluster_mass'],
            'cdelta':TEST_CASE['cluster_concentration'],
            'z_cl':TEST_CASE['z_cluster'],
            'cosmo':cosmo_ccl,
            }
        SIGMA_PARAMS = {
            'mdelta':TEST_CASE['cluster_mass'],
            'cdelta':TEST_CASE['cluster_concentration'],
            'z_cl':TEST_CASE['z_cluster'],
            'cosmo':cosmo_ccl,
            'delta_mdef':TEST_CASE['mass_Delta'],
            'halo_profile_model':TEST_CASE['density_profile_parametrization'],
            }
        GAMMA_PARAMS = {
            'mdelta':TEST_CASE['cluster_mass'],
            'cdelta':TEST_CASE['cluster_concentration'],
            'z_cluster':TEST_CASE['z_cluster'],
            'z_source':TEST_CASE['z_source'],
            'cosmo':cosmo_ccl,
            'delta_mdef':TEST_CASE['mass_Delta'],
            'halo_profile_model':TEST_CASE['density_profile_parametrization'],
            'z_src_model':'single_plane',
            }

    return Config


def test_physical_constants():
    '''
    Test physical values of physical_constants

    Notes
    -----
        The precision set for these tests put in here right now are somewhat arbitrary,
        has to be improved to values provided by CCL
    '''
    cf = load_validation_config()
    tst.assert_allclose(cf.TEST_CASE['lightspeed[km/s]'], clc.CLIGHT_KMS.value, 1e-3)
    tst.assert_allclose(cf.TEST_CASE['G[m3/km.s2]'], clc.GNEWT.value, 1e-3)
    tst.assert_allclose(cf.TEST_CASE['pc_to_m'], clc.PC_TO_METER.value, 1e-6)
    tst.assert_allclose(cf.TEST_CASE['Msun[kg]'], clc.SOLAR_MASS.value, 1e-2)
    tst.assert_allclose(cf.TEST_CASE_SIGMAC_PCST, cf.CLMM_SIGMAC_PCST, 1e-2)

def test_rho():
    '''
    Test physical values of rho
    '''
    cf = load_validation_config()
    rho = get_3d_density(cf.r3d, **cf.RHO_PARAMS)
    tst.assert_allclose(cf.NC_PROF['rho'], rho*cf.G_PHYSCONST_CORRECTION, 1e-11)

def test_sigma():
    '''
    Test physical values of sigma
    '''
    cf = load_validation_config()
    tst.assert_allclose(cf.NC_PROF['Sigma'], cf.G_PHYSCONST_CORRECTION*\
                        predict_surface_density(cf.r3d, **cf.SIGMA_PARAMS), 1e-9)

def test_delta_sigma():
    '''
    Test physical values of delta_sigma
    '''
    cf = load_validation_config()
    tst.assert_allclose(cf.NC_PROF['DeltaSigma'], cf.G_PHYSCONST_CORRECTION*\
                        predict_excess_surface_density(cf.r3d, **cf.SIGMA_PARAMS), 1e-8)

def test_get_da():
    '''
    Test physical values of Da
    '''
    cf = load_validation_config()
    tst.assert_allclose(cf.TEST_CASE['dl'],
                        get_angular_diameter_distance_a(cf.cosmo_ccl,
                                                        cf.TEST_CASE['aexp_cluster']),
                        1e-10)
    tst.assert_allclose(cf.TEST_CASE['ds'],
                        get_angular_diameter_distance_a(cf.cosmo_ccl,
                                                        cf.TEST_CASE['aexp_source']),
                        1e-10)
    tst.assert_allclose(cf.TEST_CASE['dsl'],
                        get_angular_diameter_distance_a(cf.cosmo_ccl,
                                                        cf.TEST_CASE['aexp_source'],
                                                        cf.TEST_CASE['aexp_cluster']),
                        1e-10)

def test_sigmac():
    '''
    Test physical values of Sigmac
    '''
    cf = load_validation_config()
    tst.assert_allclose(cf.TEST_CASE['nc_Sigmac'], cf.SIGMAC_PHYSCONST_CORRECTION*\
                        get_critical_surface_density(
                            cf.cosmo_ccl,
                            z_cluster=cf.TEST_CASE['z_cluster'],
                            z_source=cf.TEST_CASE['z_source']),
                        1e-8)

def test_gammat():
    '''
    Test physical values of gammat
    '''
    cf = load_validation_config()
    gammat = predict_tangential_shear(cf.r3d, **cf.GAMMA_PARAMS)
    tst.assert_allclose(cf.NC_PROF['gammat'], gammat/cf.SIGMAC_PHYSCONST_CORRECTION, 1e-8)

def test_kappa():
    '''
    Test physical values of kappa
    '''
    cf = load_validation_config()
    kappa = predict_convergence(cf.r3d, **cf.GAMMA_PARAMS)
    tst.assert_allclose(cf.NC_PROF['kappa'], kappa/cf.SIGMAC_PHYSCONST_CORRECTION, 1e-8)

def test_gt():
    '''
    Test physical values of gt
    '''
    cf = load_validation_config()
    gammat = predict_tangential_shear(cf.r3d, **cf.GAMMA_PARAMS)
    kappa = predict_convergence(cf.r3d, **cf.GAMMA_PARAMS)
    tst.assert_allclose(cf.NC_PROF['gt'], gammat/(cf.SIGMAC_PHYSCONST_CORRECTION-kappa), 1e-6)

# others: test that inputs are as expected, values from demos
# positive values from sigma onwards
