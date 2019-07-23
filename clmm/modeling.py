"""@file.py modeling.py
Functions for theoretical models.  Default is NFW.
"""

import astropy
from astropy import constants, cosmology, units
import cluster_toolkit as ct
import numpy as np

def _cclify_astropy_cosmo(apy_cosmo) :
    '''
    Generates a ccl-looking cosmology object (with all values needed for modeling) from an astropy cosmology object.

    Parameters
    ----------
    apy_cosmo : astropy.cosmology.core.FlatLambdaCDM or pyccl.core.Cosmology
        astropy or CCL cosmology object

    Returns
    -------
    ccl_cosmo : dictionary
        modified astropy cosmology object

    Notes
    -----
    Need to replace:
    `import pyccl as ccl
    cosmo_ccl = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)`
    with
    `from astropy.cosmology import FlatLambdaCDM
    astropy_cosmology_object = FlatLambdaCDM(H0=70, Om0=0.27, Ob0=0.045)
    cosmo_ccl = create_ccl_cosmo_object_from_astropy(astropy_cosmology_object)``
    '''
    if type(apy_cosmo) == astropy.cosmology.core.FlatLambdaCDM:
        ccl_cosmo = {'Omega_c': apy_cosmo.Om0,
                 'Omega_b': apy_cosmo.Ob0,
                 'h': apy_cosmo.h,
                 'H0': apy_cosmo.H0.value}
    else:
        ccl_cosmo = apy_cosmo
    return ccl_cosmo

def _get_a_from_z(z):
    '''
    Convert redshift to scale factor

    Parameters
    ----------
    z : array-like, float
        redshift

    Returns
    -------
    a : array-like, float
        scale factor
    '''
    a = 1. / (1. + z)
    return a

def _get_z_from_a(a):
    '''
    Convert scale factor to redshift

    Parameters
    ----------
    a : array-like, float
        scale factor

    Returns
    -------
    z : array-like, float
        redshift
    '''
    z = 1. / a - 1.
    return z

