# Functions to model halo profiles

import numpy as np
from astropy import units
from astropy.cosmology import LambdaCDM
from ..constants import Constants as const
import warnings

__all__ = ['get_reduced_shear_from_convergence']

# functions that are general to all backends

def get_reduced_shear_from_convergence(shear, convergence):
    """ Calculates reduced shear from shear and convergence

    Parameters
    ----------
    shear : array_like
        Shear
    convergence : array_like
        Convergence

    Returns
    -------
    reduced_shear : array_like
        Reduced shear
    """
    shear, convergence = np.array(shear), np.array(convergence)
    reduced_shear = shear/(1.-convergence)
    return reduced_shear


