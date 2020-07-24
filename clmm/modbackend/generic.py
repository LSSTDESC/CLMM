# Functions to model halo profiles

import numpy as np
from astropy import units
from astropy.cosmology import LambdaCDM
from ..constants import Constants as const
import warnings

__all__ = ['astropyify_ccl_cosmo', 'cclify_astropy_cosmo', '_get_a_from_z', 
           '_get_z_from_a', 'get_reduced_shear_from_convergence']

# functions that are general to all backends


def astropyify_ccl_cosmo(cosmoin):
    """ Given a CCL cosmology object, create an astropy cosmology object

    Parameters
    ----------
    cosmoin : astropy.cosmology.FlatLambdaCDM or pyccl.core.Cosmology
        astropy or CCL cosmology object

    Returns
    -------
    cosmodict : astropy.cosmology.LambdaCDM
        Astropy cosmology object

    Notes
    -----
    Need to replace:
    `import pyccl as ccl
    cosmo_ccl = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)`
    with
    `from astropy.cosmology import FlatLambdaCDM
    astropy_cosmology_object = FlatLambdaCDM(H0=70, Om0=0.27, Ob0=0.045)
    cosmo_ccl = cclify_astropy_cosmo(astropy_cosmology_object)``
    """
    if isinstance(cosmoin, LambdaCDM):
        return cosmoin
    if isinstance(cosmoin, dict):
        omega_m = cosmoin['Omega_b'] + cosmoin['Omega_c']
        return LambdaCDM(H0=cosmoin['H0'], Om0=omega_m, Ob0=cosmoin['Omega_b'], Ode0=1.0-omega_m)
    raise TypeError("Only astropy LambdaCDM objects or dicts can be converted to astropy.")


def cclify_astropy_cosmo(cosmoin):
    """ Given an astropy.cosmology object, creates a CCL-like dictionary
    of the relevant model parameters.

    Parameters
    ----------
    cosmoin : astropy.cosmology.core.FlatLambdaCDM or pyccl.core.Cosmology
        astropy or CCL cosmology object

    Returns
    -------
    cosmodict : dictionary
        modified astropy cosmology object

    Notes
    -----
    Need to replace:
    `import pyccl as ccl
    cosmo_ccl = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)`
    with
    `from astropy.cosmology import FlatLambdaCDM
    astropy_cosmology_object = FlatLambdaCDM(H0=70, Om0=0.27, Ob0=0.045)
    cosmo_ccl = cclify_astropy_cosmo(astropy_cosmology_object)``
    """
    if isinstance(cosmoin, dict):
        return cosmoin
    if isinstance(cosmoin, LambdaCDM):
        if cosmoin.Ob0 is None:
            raise KeyError("Cosmology object must have a defined baryon density.")
        return {'Omega_c': cosmoin.Odm0, 'Omega_b': cosmoin.Ob0,
                'h': cosmoin.h, 'H0': cosmoin.H0.value}
    raise TypeError("Only astropy LambdaCDM objects or dicts can be made CCL-like.")


def get_reduced_shear_from_convergence(shear, convergence):
    """ Calculates reduced shear from shear and convergence

    Parameters
    ----------
    shear : array_like
        Shear
    convergence : array_like
        Convergence

    Returns:
    reduced_shear : array_like
        Reduced shear
    """
    shear, convergence = np.array(shear), np.array(convergence)
    reduced_shear = shear / (1. - convergence)
    return reduced_shear


def _get_a_from_z(redshift):
    """ Convert redshift to scale factor

    Parameters
    ----------
    redshift : array_like
        Redshift

    Returns
    -------
    scale_factor : array_like
        Scale factor
    """
    redshift = np.array(redshift)
    if np.any(redshift < 0.0):
        raise ValueError(f"Cannot convert negative redshift to scale factor")
    return 1. / (1. + redshift)


def _get_z_from_a(scale_factor):
    """ Convert scale factor to redshift

    Parameters
    ----------
    scale_factor : array_like
        Scale factor

    Returns
    -------
    redshift : array_like
        Redshift
    """
    scale_factor = np.array(scale_factor)
    if np.any(scale_factor > 1.0):
        raise ValueError(f"Cannot convert invalid scale factor a > 1 to redshift")
    return 1. / scale_factor - 1.
