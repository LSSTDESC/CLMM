# Functions to model halo profiles

import numpy as np

__all__ = ['compute_reduced_shear_from_convergence']

# functions that are general to all backends


def compute_reduced_shear_from_convergence(shear, convergence):
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
    reduced_shear = np.array(shear)/(1.-np.array(convergence))
    return reduced_shear


