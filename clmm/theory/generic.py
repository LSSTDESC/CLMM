"""@file generic.py
Model independent theory functions
"""
# Functions to model halo profiles

import warnings
import numpy as np

__all__ = ['compute_reduced_shear_from_convergence',
           'compute_magnification_bias_from_magnification']

# functions that are general to all backends


def compute_reduced_shear_from_convergence(shear, convergence):
    """ Calculates reduced shear from shear and convergence

    Parameters
    ----------
    shear : array_like, float
            Shear
    convergence : array_like, float
            Convergence

    Returns
    -------
    g : array_like, float
            Reduced shear
    """
    reduced_shear = np.array(shear)/(1.-np.array(convergence))
    return reduced_shear


def compute_magnification_bias_from_magnification(magnification, alpha):
    r""" Computes magnification bias from magnification :math:`\mu` and slope parameter 
    :math:`\alpha` as :
    
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
    magnification bias : array_like
            magnification bias
    """
    if np.any(np.array(magnification) < 0):
        warnings.warn('Magnification is negative for certain radii, \
                      returning nan for magnification bias in this case.')
    magnification_bias_from_magnification = np.array(
        magnification)**(np.array([alpha]).T - 1)
    return magnification_bias_from_magnification
