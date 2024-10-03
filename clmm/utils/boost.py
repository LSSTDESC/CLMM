"""General utility functions that are used in multiple modules"""

import numpy as np


def compute_nfw_boost(rvals, rscale, boost0=0.1):
    r"""Computes the boost factor at radii rvals using a parametric form 
    following that of the analytical NFW excess surface density.

    Setting :math:`x = R/R_s` 

    .. math::
        B(x) = 1 + B_0\frac{1-F(x)}{(x)^2 -1}

    where

    .. math::
        F(x) = \begin{cases} \frac{\arctan \sqrt{x^2 -1 }}{\sqrt{x^2 -1 }} & \text{if $x>1$}, \\
        \text{1} & \text{if x = 1}, \\
        \frac{\text{arctanh} \sqrt{1-x^2 }}{\sqrt{1- x^2 }} & \text{if $x<1$}.
        \end{cases}


    Parameters
    ----------
    rvals : array_like
        Radii
    rscale : float
        Scale radius in same units as rvals
    boost0: float, optional
        Boost factor normalisation

    Returns
    -------
    array
        Boost factor at each value of rvals
    """

    r_norm = np.array(rvals) / rscale

    def _calc_finternal(r_norm):
        radicand = r_norm**2 - 1

        finternal = (
            -1j
            * np.log(
                (1 + np.lib.scimath.sqrt(radicand) * 1j) / (1 - np.lib.scimath.sqrt(radicand) * 1j)
            )
            / (2 * np.lib.scimath.sqrt(radicand))
        )

        return np.nan_to_num(finternal, copy=False, nan=1.0).real

    return 1.0 + boost0 * (1 - _calc_finternal(r_norm)) / (r_norm**2 - 1)


def compute_powerlaw_boost(rvals, rscale, boost0=0.1, alpha=-1):
    r"""Computes the boost factor at radii rvals using a power-law parametric form

    .. math::
        B(R) = 1 + B_0 \left(\frac{R}{R_s}\right)^\alpha

    Parameters
    ----------
    rvals : array_like
        Radii
    rscale : float
        Scale radius in same units as rvals
    boost0 : float, optional
        Boost factor normalisation
    alpha : float, optional
        Exponent for the power-law parametrisation. NB: -1.0 (as in Melchior+16)

    Returns
    -------
    array
        Boost factor at each value of rvals
    """

    r_norm = np.array(rvals) / rscale

    return 1.0 + boost0 * (r_norm) ** alpha


boost_models = {
    "nfw_boost": compute_nfw_boost,
    "powerlaw_boost": compute_powerlaw_boost,
}


def correct_with_boost_values(profile_vals, boost_factors):
    """Given a list of profile (shear or DeltaSigma) values and boost values, compute corrected profile

    Parameters
    ----------
    profile_vals : array_like
        Uncorrected profile values
    boost_factors : array_like
        Boost values, pre-computed

    Returns
    -------
    profile_vals_corrected : numpy.ndarray
        Corrected radial profile
    """

    profile_vals_corrected = np.array(profile_vals) / np.array(boost_factors)
    return profile_vals_corrected


def correct_with_boost_model(rvals, profile_vals, boost_model, boost_rscale, **boost_model_kw):
    """Given a boost model and sigma profile, compute corrected sigma

    Parameters
    ----------
    rvals : array_like
        Radii
    profile_vals : array_like
        Uncorrected profile values
    boost_model : str, optional
        Boost model to use for correcting sigma

            * 'nfw_boost' - NFW profile model (Default)
            * 'powerlaw_boost' - Powerlaw profile

    Returns
    -------
    profile_vals_corrected : numpy.ndarray
        Correted radial profile
    """
    boost_model_func = boost_models[boost_model]
    boost_factors = boost_model_func(rvals, boost_rscale, **boost_model_kw)

    profile_vals_corrected = np.array(profile_vals) / boost_factors
    return profile_vals_corrected


boost_models = {
    "nfw_boost": compute_nfw_boost,
    "powerlaw_boost": compute_powerlaw_boost,
}
