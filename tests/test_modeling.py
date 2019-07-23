"""
Tests for modeling
"""

import astropy
from numpy import testing as tst

from clmm import modeling as pp

density_profile_parametrization = 'nfw'
mass_Delta = 200
cluster_mass = 1.e15
cluster_concentration = 4

astropy_cosmology_object = astropy.cosmology.FlatLambdaCDM(H0=70, Om0=0.27, Ob0=0.045)
cosmo_ccl = pp._cclify_astropy_cosmo(astropy_cosmology_object)

def test_cosmo_type(cosmo_apy):
    assert(type(cosmo_apy) == astropy.cosmology.FlatLambdaCDM)
    assert(type(cosmo_ccl) == dict)
    assert(cosmo_ccl['Omega_c'] + cosmo_ccl['Omega_b'] == cosmo_apy.Odm_0 + cosmo_apy.Ob_0)

r3d = np.logspace(-2, 2, 100)
rho = pp.get_3d_density(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, cosmo=cosmo_ccl)

# def test_set_omega_m(cosmo):
#     # check that Om_b, Om_c exist
#     # use numpy asserts (numpy.testing) when possible
#     pass
#

# others: test that inputs are as expected, values from demos
# points vs arrays
# deltasigma/sigmacrit = gammat
# gt = gammat/(1-kappa)
# positive values from sigmacrit onwards

# AIM: I'm removing these hardcoded things from the notebook.
# Define CCL cosmology object
# cosmo_ccl = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)

# Select density profile and profile parametrization options

mass_lims = (1.e12, 1.e16)
