"""@file triaxialit.py
Model functions with triaxiality
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


def _d_dlnr(r_grid, f_grid):
    """Derivate of function with respect to ln(R)

    Parameters
    ----------
    r_grid: np.array
        Radius value in a grid
    f_grid: np.array
        Corresponding value of the function in the grid

    Returns
    -------
    np.array
        Derivative of function with respect to ln(R).
    """
    return r_grid * np.gradient(f_grid, r_grid)


def _value_from_grid(r_grid, f_grid, r_value):
    """Get value of tabulated function at given value

    Parameters
    ----------
    r_grid: np.array
        Radius value in a grid
    f_grid: np.array
        Corresponding value of the function in the grid
    r_value: float
        Value where to evaluate function

    Returns
    -------
    float
        Value of function at r_value.
    """
    return InterpolatedUnivariateSpline(r_grid, f_grid, k=5)(r_value)


def _integrate_grid(r_grid, f_grid, r_limits):
    """Integrate tabulated function.

    Parameters
    ----------
    r_grid: np.array
        Radius value in a grid
    f_grid: np.array
        Corresponding value of the function in the grid
    r_limits: tuple
        Limits of integration.

    Returns
    -------
    float
        Integrated value of function.
    """
    return np.vectorize(InterpolatedUnivariateSpline(r_grid, f_grid, k=5).integral)(*r_limits)


def _sigma0_correction(ell, sigma_sph, eta, deta_dlnr):
    """
    Zero-th order correction for ellipticity.

    Parameters
    ----------
    ell: float
        ellipticity of halo defined by e = (1-q)/(1+q), q is the axis ratio.
    sigma_sph: array
        Surface density assuming the halo is spherical.
    eta: array
        Derivative of sigma_sph with radius: dln(sigma_sph)/dln(R)
    deta_dlnr: array
        Derivative of eta with radius: dln(eta)/dln(R)

    Returns
    -------
    float, np.array
        Zero-th order correction for ellipticity.
    """
    return sigma_sph * (0.5 * ell**2 * (eta + 0.5 * eta**2 + 0.5 * deta_dlnr))


def excess_surface_density_mono_correction(surface_density_func, r_proj, z_cl, ell, n_grid=10000):
    r"""
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
    # Compute grid for integration and interpolation
    r_grid = np.logspace(-3, np.log10(3 * np.max(r_proj)), n_grid)
    sigma_sph_grid = surface_density_func(r_grid, z_cl)
    eta_grid = _d_dlnr(r_grid, np.log(sigma_sph_grid))
    deta_dlnr_grid = _d_dlnr(r_grid, eta_grid)

    # Compute equation terms
    integral_term = _integrate_grid(
        r_grid,
        r_grid * _sigma0_correction(ell, sigma_sph_grid, eta_grid, deta_dlnr_grid),
        (0, r_proj),
    )
    sigma_correction = _sigma0_correction(
        ell,
        sigma_sph=surface_density_func(r_proj, z_cl),
        eta=_value_from_grid(r_grid, eta_grid, r_proj),
        deta_dlnr=_value_from_grid(r_grid, deta_dlnr_grid, r_proj),
    )

    return (2 * integral_term / r_proj**2) - sigma_correction


def excess_surface_density_quad_4theta(surface_density_func, r_proj, z_cl, ell, n_grid=10000):
    r"""
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
    # Compute grid for integration and interpolation
    r_grid = np.logspace(-3, np.log10(3 * np.max(r_proj)), n_grid)
    sigma_sph_grid = surface_density_func(r_grid, z_cl)
    eta_grid = _d_dlnr(r_grid, np.log(sigma_sph_grid))

    # Compute equation terms
    integral_term = _integrate_grid(r_grid, r_grid**3 * sigma_sph_grid * eta_grid, (0, r_proj))
    sigma_sph = surface_density_func(r_proj, z_cl)
    eta = _value_from_grid(r_grid, eta_grid, r_proj)

    return ell * (3 * integral_term / r_proj**4 - 0.5 * sigma_sph * eta)


def excess_surface_density_quad_const(surface_density_func, r_proj, z_cl, ell, n_grid=10000):
    r"""
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
    # Compute grid for integration and interpolation
    r_grid = np.logspace(-3, np.log10(3 * np.max(r_proj)), n_grid)
    sigma_sph_grid = surface_density_func(r_grid, z_cl)
    eta_grid = _d_dlnr(r_grid, np.log(sigma_sph_grid))

    # Compute equation terms
    integral_term = _integrate_grid(r_grid, sigma_sph_grid * eta_grid / r_grid, (r_proj, np.inf))
    sigma_sph = surface_density_func(r_proj, z_cl)
    eta = _value_from_grid(r_grid, eta_grid, r_proj)

    return ell * (integral_term - 0.5 * sigma_sph * eta)
