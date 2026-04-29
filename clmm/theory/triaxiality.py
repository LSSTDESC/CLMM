"""@file triaxialit.py
Model functions with triaxiality
"""

# pylint: disable=too-many-lines
import warnings

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


def excess_surface_density_mono_correction(surface_density_func, r_proj, z_cl, ell, n_grid=10000):
    """
    Compute the ellipticity correction for the monopole term.

    Parameters
    ----------
    surface_density_func: function
        Function that computes the surface density, must take (r_proj, z_cl) as input.
    r_proj: array
        Projected radial position from the cluster center in :math:`M\!pc`.
    z_cl: float
        Redshift of lens cluster
    ell: float
        ellipticity of halo defined by e = (1-q)/(1+q), q is the axis ratio.
        q=b/a (Ratio of major axis to the minor axis lengths)
    n_grid: int
        Grid steps for gradient calculations.

    Returns
    -------
    array
        The ellipticity correction for the monopole term.
    """

    grid = np.logspace(-3, np.log10(3 * np.max(r_proj)), n_grid)

    sigma0_grid = surface_density_func(grid, z_cl)
    sigma0 = surface_density_func(r_proj, z_cl)

    eta_grid = grid * np.gradient(np.log(sigma0_grid), grid)
    eta = InterpolatedUnivariateSpline(grid, eta_grid, k=5)(r_proj)

    deta_dlnr_grid = grid * np.gradient(eta_grid, grid)
    deta_dlnr = InterpolatedUnivariateSpline(grid, deta_dlnr_grid, k=5)(r_proj)

    sigma_correction_grid = sigma0_grid * (
        0.5 * ell**2 * (eta_grid + 0.5 * eta_grid**2 + 0.5 * deta_dlnr_grid)
    )
    sigma_correction = sigma0 * (0.5 * ell**2 * (eta + 0.5 * eta**2 + 0.5 * deta_dlnr))

    integral_vec = np.vectorize(
        InterpolatedUnivariateSpline(grid, grid * sigma_correction_grid, k=5).integral
    )
    integral = integral_vec(0, r_proj)

    return (2 / r_proj**2) * integral - sigma_correction


def excess_surface_density_quad_4theta(surface_density_func, r_proj, z_cl, ell, n_grid=10000):
    """
    Compute the 4theta component of the quadrupole term.

    Parameters
    ----------
    surface_density_func: function
        Function that computes the surface density, must take (r_proj, z_cl) as input.
    r_proj: array
        Projected radial position from the cluster center in :math:`M\!pc`.
    z_cl: float
        Redshift of lens cluster
    ell: float
        ellipticity of halo defined by e = (1-q)/(1+q), q is the axis ratio.
        q=b/a (Ratio of major axis to the minor axis lengths)
    n_grid: int
        Grid steps for gradient calculations.

    Returns
    -------
    array
        The 4theta component of the quadrupole term.
    """

    grid = np.logspace(-3, np.log10(3 * np.max(r_proj)), n_grid)

    sigma0_grid = surface_density_func(grid, z_cl)
    sigma0 = surface_density_func(r_proj, z_cl)

    eta_grid = grid * np.gradient(np.log(sigma0_grid), grid)
    eta = InterpolatedUnivariateSpline(grid, eta_grid, k=5)(r_proj)

    integral_vec = np.vectorize(
        InterpolatedUnivariateSpline(grid, grid**3 * sigma0_grid * eta_grid, k=5).integral
    )
    integral = 3 / r_proj**4 * integral_vec(0, r_proj)

    return 0.5 * ell * (2 * integral - sigma0 * eta)


def excess_surface_density_quad_const(surface_density_func, r_proj, z_cl, ell, n_grid=10000):
    """
    Compute the constant component of the quadrupole term.

    Parameters
    ----------
    surface_density_func: function
        Function that computes the surface density, must take (r_proj, z_cl) as input.
    r_proj: array
        Projected radial position from the cluster center in :math:`M\!pc`.
    z_cl: float
        Redshift of lens cluster
    ell: float
        ellipticity of halo defined by e = (1-q)/(1+q), q is the axis ratio.
        q=b/a (Ratio of major axis to the minor axis lengths)
    n_grid: int
        Grid steps for gradient calculations.

    Returns
    -------
    array
        The constant component of the quadrupole term.
    """

    grid = np.logspace(-3, np.log10(3 * np.max(r_proj)), n_grid)

    sigma0_grid = surface_density_func(grid, z_cl)
    sigma0 = surface_density_func(r_proj, z_cl)

    eta_grid = grid * np.gradient(np.log(sigma0_grid), grid)
    eta = InterpolatedUnivariateSpline(grid, eta_grid, k=5)(r_proj)

    integral_vec = np.vectorize(
        InterpolatedUnivariateSpline(grid, sigma0_grid * eta_grid / grid, k=5).integral
    )
    integral = integral_vec(r_proj, np.inf)

    return 0.5 * ell * (2 * integral - sigma0 * eta)
