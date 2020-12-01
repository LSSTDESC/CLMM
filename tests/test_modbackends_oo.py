"""Tests for modeling.py"""
import json
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal

from .test_modeling import load_validation_config

# Backends for testing, comment ones you do not want to test for,
# but remember to NOT commit this change
backends = [
    'cluster_toolkit',
    'numcosmo',
    'ccl'
]

# ----------- Config ---------------
mod_backends = []
if 'ccl' in backends:
    from clmm.modbackend import ccl as modccl
    mod_backends.append(modccl)
if 'cluster_toolkit' in backends:
    from clmm.modbackend import cluster_toolkit as modct
    mod_backends.append(modct)
if 'numccosmo' in backends:
    from clmm.modbackend import numcosmo as modnc
    mod_backends.append(modnc)

TOLERANCE = {'rtol': 1.0e-6, 'atol': 1.0e-6}
# ----------- Testing ---------------

def test_profiles(modeling_data):
    """ Tests for profile functions, get_3d_density, predict_surface_density,
    and predict_excess_surface_density """

    # Validation tests
    # NumCosmo makes different choices for constants (Msun). We make this conversion
    # by passing the ratio of SOLAR_MASS in kg from numcosmo and CLMM
    cfg = load_validation_config()
    # Object Oriented tests
    cosmo_pars = {
        'H0': cfg['cosmo_pars']['H0'],
        'Omega_dm0': cfg['cosmo_pars']['Om0']-cfg['cosmo_pars']['Ob0'],
        'Omega_b0': cfg['cosmo_pars']['Ob0']}
    #for mod in (modccl, modct, modnc):
    #    for cosmod in (modccl, modct, modnc):
    for mod in mod_backends:
        for cosmod in mod_backends:
            m = mod.Modeling()
            m.set_cosmo(cosmod.Cosmology(**cosmo_pars))
            m.set_halo_density_profile(halo_profile_model=cfg['SIGMA_PARAMS']['halo_profile_model'])
            m.set_concentration(cfg['SIGMA_PARAMS']['cdelta'])
            m.set_mass(cfg['SIGMA_PARAMS']['mdelta'])

            assert_allclose(m.eval_density(cfg['SIGMA_PARAMS']['r_proj'], cfg['SIGMA_PARAMS']['z_cl']),
                            cfg['numcosmo_profiles']['rho'], **TOLERANCE)
            assert_allclose(m.eval_sigma(cfg['SIGMA_PARAMS']['r_proj'], cfg['SIGMA_PARAMS']['z_cl']),
                            cfg['numcosmo_profiles']['Sigma'], **TOLERANCE)
            assert_allclose(m.eval_sigma_excess(cfg['SIGMA_PARAMS']['r_proj'], cfg['SIGMA_PARAMS']['z_cl']),
                            cfg['numcosmo_profiles']['DeltaSigma'], **TOLERANCE)

def test_get_critical_surface_density(modeling_data):
    """ Validation test for critical surface density """
    cfg = load_validation_config()
    cosmo_pars = {
        'H0': cfg['cosmo_pars']['H0'],
        'Omega_dm0': cfg['cosmo_pars']['Om0']-cfg['cosmo_pars']['Ob0'],
        'Omega_b0': cfg['cosmo_pars']['Ob0']}
    for mod in mod_backends:
        for cosmod in mod_backends:
            m = mod.Modeling()
            m.set_cosmo(cosmod.Cosmology(**cosmo_pars))
            assert_allclose(m.eval_sigma_crit(cfg['TEST_CASE']['z_cluster'],
                                              cfg['TEST_CASE']['z_source']),
                        cfg['TEST_CASE']['nc_Sigmac'], 1.2e-8)
            # Check behaviour when sources are in front of the lens
            z_cluster = 0.3
            z_source = 0.2
            assert_allclose(m.eval_sigma_crit(z_cluster, z_source),
                        np.inf, 1.0e-10)
            z_source = [0.2,0.12,0.25]
            assert_allclose(m.eval_sigma_crit(z_cluster, z_source),
                        [np.inf,np.inf, np.inf], 1.0e-10)

def test_shear_convergence_unittests(modeling_data):
    """ Unit and validation tests for the shear and convergence calculations """
    cfg = load_validation_config()
    cosmo_pars = {
        'H0': cfg['cosmo_pars']['H0'],
        'Omega_dm0': cfg['cosmo_pars']['Om0']-cfg['cosmo_pars']['Ob0'],
        'Omega_b0': cfg['cosmo_pars']['Ob0']}
    for mod in mod_backends:
        for cosmod in mod_backends:
            m = mod.Modeling()
            m.set_cosmo(cosmod.Cosmology(**cosmo_pars))
            m.set_halo_density_profile(halo_profile_model=cfg['GAMMA_PARAMS']['halo_profile_model'])
            m.set_concentration(cfg['GAMMA_PARAMS']['cdelta'])
            m.set_mass(cfg['GAMMA_PARAMS']['mdelta'])
            # First compute SigmaCrit to correct cosmology changes
            sigma_c = m.eval_sigma_crit(cfg['GAMMA_PARAMS']['z_cluster'], cfg['GAMMA_PARAMS']['z_source'])

            # Compute sigma_c in the new cosmology and get a correction factor
            sigma_c_undo = m.eval_sigma_crit(cfg['GAMMA_PARAMS']['z_cluster'], cfg['GAMMA_PARAMS']['z_source'])
            sigmac_corr = (sigma_c_undo/sigma_c)

            # Validate tangential shear
            profile_pars = (cfg['GAMMA_PARAMS']['r_proj'], cfg['GAMMA_PARAMS']['z_cluster'],
                            cfg['GAMMA_PARAMS']['z_source'])
            gammat = m.eval_shear(*profile_pars)
            assert_allclose(gammat*sigmac_corr, cfg['numcosmo_profiles']['gammat'], 1.0e-8)

            # Validate convergence
            kappa = m.eval_convergence(*profile_pars)
            assert_allclose(kappa*sigmac_corr, cfg['numcosmo_profiles']['kappa'], 1.0e-8)

            # Validate reduced tangential shear
            assert_allclose(m.eval_reduced_shear(*profile_pars),
                            gammat/(1.0-kappa), 1.0e-10)
            assert_allclose(gammat*sigmac_corr/(1.-(kappa*sigmac_corr)), cfg['numcosmo_profiles']['gt'], 1.0e-6)

            # Validate magnification
            assert_allclose(m.eval_magnification(*profile_pars),
                            1./((1-kappa)**2-abs(gammat)**2), 1.0e-10)
            assert_allclose(1./((1-kappa)**2-abs(gammat)**2), cfg['numcosmo_profiles']['mu'], 4.0e-7)

            # Check that shear, reduced shear and convergence return zero and magnification returns one if source is in front of the cluster
            # First, check for a array of radius and single source z
            r = np.logspace(-2,2,10)
            z_cluster = 0.3
            z_source = 0.2

            assert_allclose(m.eval_convergence(r, z_cluster, z_source), np.zeros(len(r)), 1.0e-10)
            assert_allclose(m.eval_shear(r, z_cluster, z_source), np.zeros(len(r)), 1.0e-10)
            assert_allclose(m.eval_reduced_shear(r, z_cluster, z_source), np.zeros(len(r)), 1.0e-10)
            assert_allclose(m.eval_magnification(r, z_cluster, z_source), np.ones(len(r)), 1.0e-10)

            # Second, check a single radius and array of source z
            r = 1.
            z_source = [0.25, 0.1, 0.14, 0.02]

            assert_allclose(m.eval_convergence(r, z_cluster, z_source), np.zeros(len(z_source)), 1.0e-10)
            assert_allclose(m.eval_shear(r, z_cluster, z_source), np.zeros(len(z_source)), 1.0e-10)
            assert_allclose(m.eval_reduced_shear(r, z_cluster, z_source), np.zeros(len(z_source)), 1.0e-10)
            assert_allclose(m.eval_magnification(r, z_cluster, z_source), np.ones(len(z_source)), 1.0e-10)
