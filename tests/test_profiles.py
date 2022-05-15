"""Tests for modeling.py"""
import json
import numpy as np
from numpy.testing import assert_raises, assert_allclose
import clmm.profiles as pf
import clmm.theory as theo

def load_validation_config():
    """ Loads values precomputed by numcosmo for comparison """
    numcosmo_path = 'tests/data/numcosmo/'
    with open(numcosmo_path+'config.json', 'r') as fin:
        testcase = json.load(fin)
    # Cosmology
    cosmo = theo.Cosmology(H0=testcase['cosmo_H0'], Omega_dm0=testcase['cosmo_Odm0'],
                           Omega_b0=testcase['cosmo_Ob0'])

    return cosmo, testcase

def test_instantiate(modeling_data):
    """ Unit tests for HaloProfile objects' instantiation """
    reltol = modeling_data['theory_reltol']

    cosmo, testcase = load_validation_config()

    mdelta = testcase['cluster_mass']
    cdelta = testcase['cluster_concentration']
    z_cl = testcase['z_cluster']

    profile = pf.HaloProfileNFW(mdelta=mdelta, cdelta=cdelta, z_cl=z_cl, cosmo=cosmo)

    model = testcase['density_profile_parametrization']
    profile.set_halo_density_profile(halo_profile_model=model, delta_mdef=200)

    assert_raises(ValueError, profile.set_halo_density_profile, massdef='blu')
    assert_raises(ValueError, profile.set_halo_density_profile, halo_profile_model='bla')
    assert_raises(NotImplementedError, profile.set_einasto_alpha, alpha=0.3)
    assert_raises(NotImplementedError, profile.get_einasto_alpha)

    assert_allclose(profile.rdelta(), cdelta*profile.rscale(), reltol)
    assert_allclose(profile.M(profile.rdelta()), mdelta, reltol)
    assert_allclose(profile.rdelta(), 1.5548751530053142, reltol)
    assert_allclose(profile.rscale(), 0.38871878825132855, reltol)
    assert_allclose(profile.M(1.), 683427961195829.4, reltol)
    profile2 = profile.to_def('critical', 500)
    assert_raises(ValueError, profile.to_def, massdef2='blu', delta_mdef2=500)

    her = pf.HaloProfileHernquist(mdelta=mdelta, cdelta=cdelta, z_cl=z_cl, cosmo=cosmo)
    assert_allclose(her.M(her.rdelta()), mdelta, 1e-12)

def test_einasto(modeling_data):
    """ Basic checks for the Einasto profile """

    cosmo, testcase = load_validation_config()

    mdelta = testcase['cluster_mass']
    cdelta = testcase['cluster_concentration']
    z_cl = testcase['z_cluster']

    ein = pf.HaloProfileEinasto(mdelta=mdelta, cdelta=cdelta, z_cl=z_cl, cosmo=cosmo, alpha=0.3)

    assert_allclose(ein.get_einasto_alpha(), 0.3, 1e-18)

    ein.to_def('critical', 500)
