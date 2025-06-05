"""@file miscentering.py
Model functions with miscentering
"""
# Functions to model halo profiles

import numpy as np
from scipy.integrate import quad, dblquad, tplquad


def integrand_surface_density_nfw(theta, r_proj, r_mis, r_s):
    r"""Computes integrand for surface mass density with the NFW profile.

    Parameters
    ----------
    theta : float
        Angle of polar coordinates of the miscentering direction.
    r_proj : float
        Projected radial position from the cluster center in :math:`M\!pc`.
    r_mis : float
        Projected miscenter distance in :math:`M\!pc`.
    r_s : float
        Scale radius

    Returns
    -------
    float
        2D projected density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    r_norm = np.sqrt(r_proj**2.0 + r_mis**2.0 - 2.0 * r_proj * r_mis * np.cos(theta)) / r_s

    r2m1 = r_norm**2.0 - 1.0
    if r_norm < 1:
        sqrt_r2m1 = np.sqrt(-r2m1)
        res = np.arcsinh(sqrt_r2m1 / r_norm) / (-r2m1) ** (3.0 / 2.0) + 1.0 / r2m1
    elif r_norm > 1:
        sqrt_r2m1 = np.sqrt(r2m1)
        res = -np.arcsin(sqrt_r2m1 / r_norm) / (r2m1) ** (3.0 / 2.0) + 1.0 / r2m1
    else:
        res = 1.0 / 3.0
    return res


def integrand_surface_density_einasto(r_par, theta, r_proj, r_mis, r_s, alpha_ein):
    r"""Computes integrand for surface mass density with the Einasto profile.

    Parameters
    ----------
    r_par : float
        Parallel radial position from the cluster center in :math:`M\!pc`.
    theta : float
        Angle of polar coordinates of the miscentering direction.
    r_proj : array_like
        Projected radial position from the cluster center in :math:`M\!pc`.
    r_mis : float
        Projected miscenter distance in :math:`M\!pc`.
    r_s : float
        Scale radius
    alpha_ein : float
        Einasto slope

    Returns
    -------
    float
        2D projected density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    # Projected surface mass density element for numerical integration
    r_norm = (
        np.sqrt(r_par**2.0 + r_proj**2.0 + r_mis**2.0 - 2.0 * r_proj * r_mis * np.cos(theta))
        / r_s
    )

    return np.exp(-2.0 * (r_norm**alpha_ein - 1.0) / alpha_ein)


def integrand_surface_density_hernquist(theta, r_proj, r_mis, r_s):
    r"""Computes integrand for surface mass density with the Hernquist profile.

    Parameters
    ----------
    theta : float
        Angle of polar coordinates of the miscentering direction.
    r_proj : float
        Projected radial position from the cluster center in :math:`M\!pc`.
    r_mis : float
        Projected miscenter distance in :math:`M\!pc`.
    r_s : float
        Scale radius

    Returns
    -------
    float
        2D projected density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    r_norm = np.sqrt(r_proj**2.0 + r_mis**2.0 - 2.0 * r_proj * r_mis * np.cos(theta)) / r_s

    r2m1 = r_norm**2.0 - 1.0
    if r_norm < 1:
        res = -3 / r2m1**2 + (r2m1 + 3) * np.arcsinh(np.sqrt(-r2m1) / r_norm) / (-r2m1) ** 2.5
    elif r_norm > 1:
        res = -3 / r2m1**2 + (r2m1 + 3) * np.arcsin(np.sqrt(r2m1) / r_norm) / (r2m1) ** 2.5
    else:
        res = 4.0 / 15.0
    return res


def integrand_mean_surface_density_nfw(theta, r_proj, r_mis, r_s):
    r"""Computes integrand for mean surface mass density with the NFW profile.

    Parameters
    ----------
    theta : float
        Angle of polar coordinates of the miscentering direction.
    r_proj : float
        Projected radial position from the cluster center in :math:`M\!pc`.
    r_mis : float
        Projected miscenter distance in :math:`M\!pc`.
    r_s : float
        Scale radius

    Returns
    -------
    float
        Mean surface density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    return r_proj * integrand_surface_density_nfw(theta, r_proj, r_mis, r_s)


def integrand_mean_surface_density_einasto(r_par, theta, r_proj, r_mis, r_s, alpha_ein):
    r"""Computes integrand for mean surface mass density with the Einasto profile.

    Parameters
    ----------
    r_par : float
        Parallel radial position from the cluster center in :math:`M\!pc`.
    theta : float
        Angle of polar coordinates of the miscentering direction.
    r_proj : float
        Projected radial position from the cluster center in :math:`M\!pc`.
    r_mis : float
        Projected miscenter distance in :math:`M\!pc`.
    r_s : float
        Scale radius
    alpha_ein : float
        Einasto slope

    Returns
    -------
    float
        Mean surface density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    return r_proj * integrand_surface_density_einasto(r_par, theta, r_proj, r_mis, r_s, alpha_ein)


def integrand_mean_surface_density_hernquist(theta, r_proj, r_mis, r_s):
    r"""Computes integrand for mean surface mass density with the Hernquist profile.

    Parameters
    ----------
    theta : float
        Angle of polar coordinates of the miscentering direction.
    r_proj : float
        Projected radial position from the cluster center in :math:`M\!pc`.
    r_mis : float
        Projected miscenter distance in :math:`M\!pc`.
    r_s : float
        Scale radius

    Returns
    -------
    float
        Mean surface density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    return r_proj * integrand_surface_density_hernquist(theta, r_proj, r_mis, r_s)


def integrate_azimuthially_miscentered_surface_density(
    r_proj, r_mis, integrand, norm, aux_args, extra_integral
):
    r"""Integrates azimuthally the miscentered surface mass density kernel.

    Parameters
    ----------
    r_proj : float, array_like
        Projected radial position from the cluster center in :math:`M\!pc`.
    r_mis : float
        Projected miscenter distance in :math:`M\!pc`.
    integrand : function
        Function to be integrated
    norm : float
        Normalization value for integral
    aux_args : list
        Auxiliary arguments used in the integral
    extra_integral : bool
        Additional dimension for the integral

    Returns
    -------
    float, numpy.ndarray
        2D projected density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    args_list = [(r, r_mis, *aux_args) for r in np.atleast_1d(r_proj)]
    if extra_integral:
        res = [
            dblquad(integrand, 0.0, np.pi, 0, np.inf, args=args, epsrel=1e-6)[0]
            for args in args_list
        ]
    else:
        res = [quad(integrand, 0.0, np.pi, args=args, epsrel=1e-6)[0] for args in args_list]

    res = np.array(res) * norm / np.pi

    if not np.iterable(r_proj):
        return res[0]
    return res


def integrate_azimuthially_miscentered_mean_surface_density(
    r_proj, r_mis, integrand, norm, aux_args, extra_integral
):
    r"""Integrates azimuthally the miscentered mean surface mass density kernel.

    Parameters
    ----------
    r_proj : float, array_like
        Projected radial position from the cluster center in :math:`M\!pc`.
    r_mis : float
        Projected miscenter distance in :math:`M\!pc`.
    integrand : function
        Function to be integrated
    norm : float
        Normalization value for integral
    aux_args : list
        Auxiliary arguments used in the integral
    extra_integral : bool
        Additional dimension for the integral

    Returns
    -------
    float, numpy.ndarray
        Mean surface density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    r_proj_use = np.atleast_1d(r_proj)
    r_lower = np.zeros_like(r_proj_use)
    r_lower[1:] = r_proj_use[:-1]
    args = (r_mis, *aux_args)

    if extra_integral:
        res = [
            tplquad(integrand, r_low, r_high, 0, np.pi, 0, np.inf, args=args, epsrel=1e-6)[0]
            for r_low, r_high in zip(r_lower, r_proj_use)
        ]
    else:
        res = [
            dblquad(integrand, r_low, r_high, 0, np.pi, args=args, epsrel=1e-6)[0]
            for r_low, r_high in zip(r_lower, r_proj_use)
        ]

    if not np.iterable(r_proj):
        return res[0] * norm * 2 / np.pi / r_proj**2
    return np.cumsum(res) * norm * 2 / np.pi / r_proj**2
