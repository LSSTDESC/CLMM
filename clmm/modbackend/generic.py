# Functions to model halo profiles

import numpy as np
from astropy import units
from astropy.cosmology import LambdaCDM
from ..constants import Constants as const
import warnings

__all__ = ['_get_a_from_z', '_get_z_from_a', 'get_reduced_shear_from_convergence']

# functions that are general to all backends

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
