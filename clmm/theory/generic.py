"""@file generic.py
Model independent theory functions
"""
# Functions to model halo profiles

import warnings
import numpy as np
from scipy.optimize import fsolve
from scipy.special import gamma, gammainc

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
    return np.array(magnification)**(np.array([alpha]).T - 1)
def compute_rdelta(mdelta, redshift, cosmo, massdef='mean', delta_mdef=200):
    # Check massdef
    if massdef=='mean':
        rho = cosmo.get_rho_m
    elif massdef in ('critical', 'virial'):
        rho = cosmo.get_rho_c
    else:
        raise ValueError(f'massdef(={massdef}) must be mean, critical or virial')
    return ((3.*mdelta)/(4.*np.pi*delta_mdef*rho(redshift)))**(1./3.)
def compute_profile_mass_in_radius(r3d, redshift, cosmo, mdelta, cdelta,
                                   massdef='mean', delta_mdef=200,
                                   halo_profile_model='nfw', alpha=None):
    rdelta = compute_rdelta(mdelta, redshift, cosmo, massdef, delta_mdef)
    # Check halo_profile_model
    if halo_profile_model=='nfw':
        prof_integ = lambda c: np.log(1. + c) - c/(1. + c)
    elif halo_profile_model=='einasto':
        if einasto_alpha is None:
            raise ValueError('alpha must be provided when Einasto profile is selected!')
        prof_integ = lambda c: gamma(3./alpha)*gammainc(3./alpha, 2./alpha*c**alpha)
    elif halo_profile_model=='hernquist':
        prof_integ = lambda c: (c/(1. + c))**2.
    else:
        raise ValueError(f'halo_profile_model=(={halo_profile_model=}) must be '
                         'nfw, einasto, or hernquist!')
    x = np.array(r3d)/(rdelta/cdelta)
    return mdelta*prof_integ(x)/prof_integ(cdelta)
def convert_profile_mass_concentration(
        mdelta, cdelta, redshift, cosmo, massdef, delta_mdef, halo_profile_model,
        massdef2, delta_mdef2, halo_profile_model2, alpha=None, alpha2=None):
    """
    Parameters
    ----------
    massdef2: float
        Background density definition to convert to (`critical`, `mean`)
    delta_mdef2: str
        Overdensity scale to convert to

    Returns
    -------
    HaloProfile:
        HaloProfile object
    """
    rdelta = compute_rdelta(mdelta, redshift, cosmo, massdef, delta_mdef)
    # Eq. to solve
    def f(params):
        mdelta2, cdelta2 = params
        rdelta2 = compute_rdelta(mdelta2, redshift, cosmo, massdef2, delta_mdef2)
        mdelta2_rad1 = compute_profile_mass_in_radius(
            rdelta, redshift, cosmo, mdelta2, cdelta2,
            massdef2, delta_mdef2, halo_profile_model2, alpha2)
        mdelta1_rad2 = compute_profile_mass_in_radius(
            rdelta2, redshift, cosmo, mdelta, cdelta,
            massdef, delta_mdef, halo_profile_model, alpha)
        return mdelta-mdelta2_rad1, mdelta2-mdelta1_rad2
    # Interate 2 times:
    mdelta2, cdelta2 = fsolve(func=f, x0=[mdelta, cdelta])
    mdelta2, cdelta2 = fsolve(func=f, x0=[mdelta2, cdelta2])

    return mdelta2, cdelta2
