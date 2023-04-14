"""General utility functions that are used in multiple modules"""
import numpy as np


def compute_nfw_boost(rvals, rscale=1000, boost0=0.1):
    """Given a list of `rvals`, and optional `rscale` and `boost0`, return the corresponding
    boost factor at each rval

    Parameters
    ----------
    rvals : array_like
        Radii
    rscale : float, optional
        scale radius for NFW in same units as rvals (default 2000 kpc)
    boost0 : float, optional
        Boost factor at each value of rvals

    Returns
    -------
    array
        Boost factor
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


def compute_powerlaw_boost(rvals, rscale=1000, boost0=0.1, alpha=-1.0):
    """Given a list of `rvals`, and optional `rscale` and `boost0`, and `alpha`,
    return the corresponding boost factor at each `rval`

    Parameters
    ----------
    rvals : array_like
        Radii
    rscale : float, optional
        Scale radius for NFW in same units as rvals (default 2000 kpc)
    boost0 : float, optional
        Boost factor at each value of rvals
    alpha : float, optional
        Exponent from Melchior+16. Default: -1.0

    Returns
    -------
    array
        Boost factor
    """

    r_norm = np.array(rvals) / rscale

    return 1.0 + boost0 * (r_norm) ** alpha


boost_models = {
    "nfw_boost": compute_nfw_boost,
    "powerlaw_boost": compute_powerlaw_boost,
}


def correct_sigma_with_boost_values(sigma_vals, boost_factors):
    """Given a list of boost values and sigma profile, compute corrected sigma

    Parameters
    ----------
    sigma_vals : array_like
        uncorrected sigma with cluster member dilution
    boost_factors : array_like
        Boost values pre-computed

    Returns
    -------
    sigma_corrected : numpy.ndarray
        correted radial profile
    """

    sigma_corrected = np.array(sigma_vals) / np.array(boost_factors)
    return sigma_corrected


def correct_sigma_with_boost_model(rvals, sigma_vals, boost_model="nfw_boost", **boost_model_kw):
    """Given a boost model and sigma profile, compute corrected sigma

    Parameters
    ----------
    rvals : array_like
        radii
    sigma_vals : array_like
        uncorrected sigma with cluster member dilution
    boost_model : str, optional
        Boost model to use for correcting sigma

            * 'nfw_boost' - NFW profile model (Default)
            * 'powerlaw_boost' - Powerlaw profile

    Returns
    -------
    sigma_corrected : numpy.ndarray
        correted radial profile
    """
    boost_model_func = boost_models[boost_model]
    boost_factors = boost_model_func(rvals, **boost_model_kw)

    sigma_corrected = np.array(sigma_vals) / boost_factors
    return sigma_corrected


boost_models = {
    "nfw_boost": compute_nfw_boost,
    "powerlaw_boost": compute_powerlaw_boost,
}
