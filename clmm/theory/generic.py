"""@file generic.py
Model independent theory functions
"""
# Functions to model halo profiles

import numpy as np

__all__ = ['compute_reduced_shear_from_convergence',
           'compute_magnification_bias_from_magnification']

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
    array_like
            Reduced shear
    """
    reduced_shear = np.array(shear)/(1.-np.array(convergence))
    return reduced_shear


def compute_magnification_bias_from_magnification(magnification, alpha):
    r""" Calculates magnification_bias from magnification and alpha parameter as :
    
    .. math::
        \mu^{\alpha - 1}
        
    The alpha parameter depends on the source sample and is computed as the slope of the 
    cummulative numer counts at a given magnitude:
    
    .. math::
        \alpha \equiv \alpha(f) = - \frac{\mathrm d}{\mathrm d \log{f}} \log{n_0(>f)}

    or,
    
    .. math::
        \alpha \equiv \alpha(m) = 2.5 \frac{\mathrm d}{\mathrm d m} \log{n_0(<m)}
            
    see e.g.  Bartelmann & Schneider 2001; Umetsu 2020

    Parameters
    ----------
    magnification : array_like
            Magnification
    alpha : array like
            Source cummulative number density slope

    Returns
    -------
    compute_magnification_bias : array_like
            magnification bias
    """
    magnification_bias_from_magnification = np.array(
        magnification)**(np.array([alpha]).T - 1)
    return magnification_bias_from_magnification
