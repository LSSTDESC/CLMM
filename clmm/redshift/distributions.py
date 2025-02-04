"""Redshift distribution functions"""
import numpy as np
from scipy.special import gamma, gammainc


def _functional_form(redshift, alpha, beta, redshift0, is_cdf=False):
    r"""
    A private function that returns the functionnal form of the redshift distribution used in
    Chang et al (2013):

    .. math::
       P(z) = z^{\\alpha}\times\exp^{\left(-\frac{z}{z0}^\beta\right)}

    Parameters
    ----------
    redshift : float
        Galaxy redshift
    alpha, beta, z0 : floats
        Parameters describing the function
    is_cdf : bool
        If True, returns cumulative function.

    Returns
    -------
    The value of the function at z
    """
    if is_cdf:
        return (
            redshift0 ** (alpha + 1)
            * gammainc((alpha + 1) / beta, (redshift / redshift0) ** beta)
            / beta
            * gamma((alpha + 1) / beta)
        )
    return (redshift**alpha) * np.exp(-((redshift / redshift0) ** beta))


def chang2013(redshift, is_cdf=False):
    """
    Chang et al (2013) unnormalized galaxy redshift distribution function, with the fiducial
    set of parameters.

    Parameters
    ----------
    redshift : float
        Galaxy redshift
    is_cdf : bool
        If True, returns cumulative distribution function.

    Returns
    -------
    The value of the distribution at z
    """
    alpha, beta, redshift0 = 1.24, 1.01, 0.51
    return _functional_form(redshift, alpha, beta, redshift0, is_cdf)


def desc_srd(redshift, is_cdf=False):
    """
    Unnormalized galaxy redshift distribution function used in
    the LSST/DESC Science Requirement Document (arxiv:1809.01669).

    Parameters
    ----------
    redshift : float
        Galaxy redshift
    is_cdf : bool
        If True, returns cumulative distribution function.

    Returns
    -------
    The value of the distribution at z
    """
    alpha, beta, redshift0 = 2.0, 0.9, 0.28
    return _functional_form(redshift, alpha, beta, redshift0, is_cdf)
