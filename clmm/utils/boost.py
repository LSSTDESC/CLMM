"""General utility functions that are used in multiple modules"""
import numpy as np


def compute_nfw_boost(rvals, rscale, boost_norm):
    """Given a list of `rvals`, and optional `rscale` and `boost0`, returns the corresponding
    boost factor at each rval

    Parameters
    ----------
    rvals : array_like
        Radii
    rscale : float, optional
        scale radius in same units as rvals
    boost_norm : float, optional
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


def compute_powerlaw_boost(rvals, rscale, boost_norm, slope):
    """Given a list of `rvals`, and optional `rscale` and `boost0`, and `alpha`,
    return the corresponding boost factor at each `rval`

    Parameters
    ----------
    rvals : array_like
        Radii
    rscale : float
        Scale radius in same units as rvals
    boost_norm : float
        Boost factor normalisation
    slope : float
        Exponent for the power-law parametrisation. NB: Melchior+16 uses -1.0

    Returns
    -------
    array
        Boost factor at each value of rvals
    """

    r_norm = np.array(rvals) / rscale

    return 1.0 + boost0 * (r_norm) ** slope


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
