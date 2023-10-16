"""General utility functions that are used in multiple modules"""
import numpy as np
from scipy.integrate import quad

from ..redshift import distributions as zdist


def compute_beta(z_src, z_cl, cosmo):
    r"""Geometric lensing efficicency

    .. math::
        \beta = max(0, D_{a,\ ls}/D_{a,\ s})

    Eq.2 in https://arxiv.org/pdf/1611.03866.pdf

    Parameters
    ----------
    z_src:  float
        Source galaxy redshift
    z_cl: float
        Galaxy cluster redshift
    cosmo: clmm.Cosmology
        CLMM Cosmology object

    Returns
    -------
    float
        Geometric lensing efficicency
    """
    beta = np.heaviside(z_src - z_cl, 0) * cosmo.eval_da_z1z2(z_cl, z_src) / cosmo.eval_da(z_src)
    return beta


def compute_beta_s(z_src, z_cl, z_inf, cosmo):
    r"""Geometric lensing efficicency ratio

    .. math::
        \beta_s = \beta(z_{src})/\beta(z_{inf})

    Parameters
    ----------
    z_src: float
        Source galaxy redshift
    z_cl: float
        Galaxy cluster redshift
    z_inf: float
        Redshift at infinity
    cosmo: clmm.Cosmology
        CLMM Cosmology object

    Returns
    -------
    float
        Geometric lensing efficicency ratio
    """
    beta_s = compute_beta(z_src, z_cl, cosmo) / compute_beta(z_inf, z_cl, cosmo)
    return beta_s


def compute_beta_s_func(z_src, z_cl, z_inf, cosmo, func, *args, **kwargs):
    r"""Geometric lensing efficicency ratio times a value of a function

    .. math::
        \beta_{s}\times \text{func} = \beta_s(z_{src}, z_{cl}, z_{inf})
        \times\text{func}(*args,\ **kwargs)

    Parameters
    ----------
    z_src: float
        Source galaxy redshift
    z_cl: float
        Galaxy cluster redshift
    z_inf: float
        Redshift at infinity
    cosmo: clmm.Cosmology
        CLMM Cosmology object
    func: callable
        A scalar function
    *args: positional arguments
        args to be passed to `func`
    **kwargs: keyword arguments
        kwargs to be passed to `func`

    Returns
    -------
    float
        Geometric lensing efficicency ratio
    """
    beta_s = compute_beta(z_src, z_cl, cosmo) / compute_beta(z_inf, z_cl, cosmo)
    beta_s_func = beta_s * func(*args, **kwargs)
    return beta_s_func


def compute_beta_mean(z_cl, cosmo, zmax=10.0, delta_z_cut=0.1, zmin=None, z_distrib_func=None):
    r"""Mean value of the geometric lensing efficicency

    .. math::
       \left<\beta\right> = \frac{\int_{z = z_{min}}^{z_{max}}\beta(z)N(z)}
       {\int_{z = z_{min}}^{z_{max}}N(z)}

    Parameters
    ----------
    z_cl: float
        Galaxy cluster redshift
    cosmo: clmm.Cosmology
        CLMM Cosmology object
    zmax: float, optional
        Maximum redshift to be set as the source of the galaxy when performing the sum.
        Default: 10
    delta_z_cut: float, optional
        Redshift interval to be summed with :math:`z_{cl}` to return :math:`z_{min}`.
        This feature is not used if :math:`z_{min}` is provided by the user. Default: 0.1
    zmin: float, None, optional
        Minimum redshift to be set as the source of the galaxy when performing the sum.
        Default: None
    z_distrib_func: one-parameter function, optional
        Redshift distribution function. Default is Chang et al (2013) distribution function.

    Returns
    -------
    float
        Mean value of the geometric lensing efficicency
    """
    if z_distrib_func is None:
        z_distrib_func = zdist.chang2013

    def integrand(z_i, z_cl=z_cl, cosmo=cosmo):
        return compute_beta(z_i, z_cl, cosmo) * z_distrib_func(z_i)

    if zmin is None:
        zmin = z_cl + delta_z_cut

    return quad(integrand, zmin, zmax)[0] / quad(z_distrib_func, zmin, zmax)[0]


def compute_beta_s_mean(
    z_cl, z_inf, cosmo, zmax=10.0, delta_z_cut=0.1, zmin=None, z_distrib_func=None
):
    r"""Mean value of the geometric lensing efficicency ratio

    .. math::
       \left<\beta_s\right> =\frac{\int_{z = z_{min}}^{z_{max}}\beta_s(z)N(z)}
       {\int_{z = z_{min}}^{z_{max}}N(z)}

    Parameters
    ----------
    z_cl: float
        Galaxy cluster redshift
    z_inf: float
        Redshift at infinity
    cosmo: clmm.Cosmology
        CLMM Cosmology object
    zmax: float
        Maximum redshift to be set as the source of the galaxy when performing the sum.
        Default: 10
    delta_z_cut: float, optional
        Redshift interval to be summed with :math:`z_{cl}` to return :math:`z_{min}`.
        This feature is not used if :math:`z_{min}` is provided by the user. Default: 0.1
    zmin: float, None, optional
        Minimum redshift to be set as the source of the galaxy when performing the sum.
        Default: None
    z_distrib_func: one-parameter function, optional
        Redshift distribution function. Default is Chang et al (2013) distribution function.

    Returns
    -------
    float
        Mean value of the geometric lensing efficicency ratio
    """
    if z_distrib_func is None:
        z_distrib_func = zdist.chang2013

    def integrand(z_i, z_cl=z_cl, z_inf=z_inf, cosmo=cosmo):
        return compute_beta_s(z_i, z_cl, z_inf, cosmo) * z_distrib_func(z_i)

    if zmin is None:
        zmin = z_cl + delta_z_cut

    return quad(integrand, zmin, zmax)[0] / quad(z_distrib_func, zmin, zmax)[0]


def compute_beta_s_square_mean(
    z_cl, z_inf, cosmo, zmax=10.0, delta_z_cut=0.1, zmin=None, z_distrib_func=None
):
    r"""Mean square value of the geometric lensing efficiency ratio

    .. math::
       \left<\beta_s^2\right> =\frac{\int_{z = z_{min}}^{z_{max}}\beta_s^2(z)N(z)}
       {\int_{z = z_{min}}^{z_{max}}N(z)}

    Parameters
    ----------
    z_cl: float
        Galaxy cluster redshift
    z_inf: float
        Redshift at infinity
    cosmo: clmm.Cosmology
        CLMM Cosmology object
    zmax: float
        Maximum redshift to be set as the source of the galaxy when performing the sum.
        Default: 10
    delta_z_cut: float, optional
        Redshift interval to be summed with :math:`z_{cl}` to return :math:`z_{min}`.
        This feature is not used if :math:`z_{min}` is provided by the user. Default: 0.1
    zmin: float, None, optional
        Minimum redshift to be set as the source of the galaxy when performing the sum.
        Default: None
    z_distrib_func: one-parameter function, optional
        Redshift distribution function. Default is Chang et al (2013) distribution function.

    Returns
    -------
    float
        Mean square value of the geometric lensing efficicency ratio.
    """
    if z_distrib_func is None:
        z_distrib_func = zdist.chang2013

    def integrand(z_i, z_cl=z_cl, z_inf=z_inf, cosmo=cosmo):
        return compute_beta_s(z_i, z_cl, z_inf, cosmo) ** 2 * z_distrib_func(z_i)

    if zmin is None:
        zmin = z_cl + delta_z_cut

    return quad(integrand, zmin, zmax)[0] / quad(z_distrib_func, zmin, zmax)[0]
