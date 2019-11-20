"""@file.py modeling.py
Functions for theoretical models.  Default is NFW.
"""
from astropy import constants, cosmology, units
import cluster_toolkit as ct
import numpy as np


def cclify_astropy_cosmo(apy_cosmo):
    """Generates a ccl-looking cosmology object (with all values needed for modeling) from
    an astropy cosmology object.

    Parameters
    ----------
    apy_cosmo : astropy.cosmology.core.FlatLambdaCDM or pyccl.core.Cosmology
        astropy or CCL cosmology object

    Returns
    -------
    ccl_cosmo : dictionary
        modified astropy cosmology object

    Notes
    -----
    Need to replace:
    `import pyccl as ccl
    cosmo_ccl = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)`
    with
    `from astropy.cosmology import FlatLambdaCDM
    astropy_cosmology_object = FlatLambdaCDM(H0=70, Om0=0.27, Ob0=0.045)
    cosmo_ccl = cclify_astropy_cosmo(astropy_cosmology_object)``
    """
    if isinstance(apy_cosmo, cosmology.core.FlatLambdaCDM):
        ccl_cosmo = {'Omega_c': apy_cosmo.Odm0,
                     'Omega_b': apy_cosmo.Ob0,
                     'h': apy_cosmo.h,
                     'H0': apy_cosmo.H0.value}
    else:
        ccl_cosmo = apy_cosmo
    return ccl_cosmo


def _get_a_from_z(redshift):
    """Convert redshift to scale factor

    Parameters
    ----------
    redshift : array_like, float
        redshift

    Returns
    -------
    scale_factor : array_like, float
        scale factor
    """
    scale_factor = 1. / (1. + redshift)
    return scale_factor


def _get_z_from_a(scale_factor):
    """Convert scale factor to redshift

    Parameters
    ----------
    scale_factor : array_like, float
        scale factor

    Returns
    -------
    redshift : array_like, float
        redshift
    """
    redshift = 1. / scale_factor - 1.
    return redshift

def _ct_omega_m_fix(omega_m, redshift):
    r"""
    Patch to fix `cluster_toolkit` z=0 limitation. It works by passing
    :math:`\Omega_m(z)` instead of :math:`\Omega_m(z=0)` to
    `cluster_toolkit` functions.

    Parameters
    ----------
    omega_m: float
        :math:`\Omega_m(z=0)`
    redshift: float
        Redshift of the conversion

    Returns
    -------
    float
        :math:`\Omega_m(z)=\Omega_m(z=0)(1+z)^3`
    """
    return omega_m*(1.0 + redshift)**3

def get_reduced_shear_from_convergence(gamma, k):
    """Calculates reduced shear from shear and convergence
    
    Parameters
    ----------
    gamma : array_like, float
        shear
    k : array_like, float
        convergence
    
    Returns:
    g : array_like, float
        reduced shear
    """
    g = gamma/(1-k)
    return g

def get_3d_density(r3d, mdelta, cdelta, z_cl, cosmo, delta_mdef=200, halo_profile_model='nfw'):
    r"""Retrieve the 3d density :math:`\rho(r)`.

    Profiles implemented so far are:

        `nfw`: :math:`\rho(r) = \frac{\rho_0}{\frac{c}{(r/R_{vir})}
        \left(1+\frac{c}{(r/R_{vir})}\right)^2}` [insert citation here]

    Parameters
    ----------
    r3d : array_like, float
        The radial positions in :math:`M\!pc/h`.
    mdelta : float
        Galaxy cluster mass in :math:`M_\odot/h`.
    cdelta : float
        Galaxy cluster NFW concentration.
    z_cl: float
        Redshift of the cluster
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    delta_mdef: int, optional
        Mass overdensity definition; defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization, with the following supported options:

            `nfw` (default)
    z_src_model : str, optional
        Source redshift model, with the following supported options:

            `single_plane` (default) - all sources at one redshift

            `known_z_src` - known individual source galaxy redshifts, e.g. discrete case

            `z_src_distribution` - known source redshift distribution, e.g. continuous case
            requiring integration.

    Returns
    -------
    rho : array_like, float
        3-dimensional mass density

    Notes
    -----
    Need to refactor later so we only require arguments that are necessary for all profiles
    and use another structure to take the arguments necessary for specific models
    """
    cosmo = cclify_astropy_cosmo(cosmo)
    omega_m_z = _ct_omega_m_fix(cosmo['Omega_c'] + cosmo['Omega_b'], z_cl)

    if halo_profile_model == 'nfw':
        rho = ct.density.rho_nfw_at_r(r3d, mdelta, cdelta, omega_m_z, delta=delta_mdef)
    else:
        raise ValueError("Profile models other than nfw not currently supported")
    return rho


def predict_surface_density(r_proj, mdelta, cdelta, z_cl, cosmo, delta_mdef=200,
                            halo_profile_model='nfw'):
    r"""Computes the surface mass density

    .. math::
        \Sigma(R) = \Omega_m \rho_{crit} \int^\infty_{-\infty} dz \Xi_{hm} (\sqrt{R^2+z^2}),

    where :math:`\Xi_{hm}` is the halo mass function.

    Parameters
    ----------
    r_proj : array_like
        The projected radial positions in :math:`M\!pc/h`.
    mdelta : float
        Galaxy cluster mass in :math:`M_\odot/h`.
    cdelta : float
        Galaxy cluster NFW concentration.
    z_cl: float
        Redshift of the cluster
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition; defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts e.g. discrete case
        `z_src_distribution` - known source redshift distribution e.g. continuous
        case requiring integration.

    Returns
    -------
    sigma : array_like, float
        Surface density, Sigma in units of [:math:`h M_\odot/pc^2`]

    Notes
    -----
    Need to refactory so we only require arguments that are necessary for all models and use
    another structure to take the arguments necessary for specific models.
    """
    cosmo = cclify_astropy_cosmo(cosmo)
    omega_m_z = _ct_omega_m_fix(cosmo['Omega_c'] + cosmo['Omega_b'], z_cl)

    if halo_profile_model == 'nfw':
        sigma = ct.deltasigma.Sigma_nfw_at_R(r_proj, mdelta, cdelta, omega_m_z, delta=delta_mdef)
    else:
        raise ValueError("Profile models other than nfw not currently supported")
    return sigma


def predict_excess_surface_density(r_proj, mdelta, cdelta, z_cl, cosmo, delta_mdef=200,
                                   halo_profile_model='nfw'):
    r"""Computes the excess surface density

    .. math::
        \Delta\Sigma(R) = \bar{\Sigma}(<R)-\Sigma(R),

    where

    .. math::
        \bar{\Sigma}(<R) = \frac{2}{R^2} \int^R_0 dR' R' \Sigma(R')

    Parameters
    ----------
    r_proj : array_like
        The projected radial positions in :math:`M\!pc/h`.
    mdelta : float
        Galaxy cluster mass in :math:`M_\odot/h`.
    cdelta : float
        Galaxy cluster NFW concentration.
    z_cl: float
        Redshift of the cluster
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition; defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts e.g. discrete case
        `z_src_distribution` - known source redshift distribution e.g. continuous
        case requiring integration.

    Returns
    -------
    deltasigma : array_like, float
        Excess surface density, DeltaSigma in units of [:math:`h M_\odot/pc^2`].
    """
    cosmo = cclify_astropy_cosmo(cosmo)
    omega_m_z = _ct_omega_m_fix(cosmo['Omega_c'] + cosmo['Omega_b'], z_cl)

    if halo_profile_model == 'nfw':
        sigma_r_proj = np.logspace(-3, 4, 1000)
        sigma = ct.deltasigma.Sigma_nfw_at_R(sigma_r_proj, mdelta, cdelta,
                                             omega_m_z, delta=delta_mdef)
        # ^ Note: Let's not use this naming convention when transfering ct to ccl....
        deltasigma = ct.deltasigma.DeltaSigma_at_R(r_proj, sigma_r_proj,
                                                   sigma, mdelta, cdelta,
                                                   omega_m_z, delta=delta_mdef)
    else:
        raise ValueError("Profile models other than nfw not currently supported")
    return deltasigma


def get_angular_diameter_distance_a(cosmo, scale_factor2, scale_factor1=1.):
    r"""This is a function to calculate d_LS (angular distance between lens
    and source) because CCL cannot yet do it.

    Temporarily using the astropy implementation.

    Parameters
    ----------
    cosmo : pyccl.core.Cosmology object
            CCL Cosmology object
    scale_factor2 : float
        smaller scale factor
    scale_factor1 : float, optional
        larger scale factor; defaults to 1.

    Returns
    -------
    d_a : float
        angular diameter distance in :math:`\mathrm{Mpc\ h}^{-1}`

    Notes
    -----
    This is definitely broken if other cosmological parameter specifications differ,
    so we'll have to revise this later. We need to switch angular_diameter_distance_z1z2
    to CCL equivalent angular distance once implemented
    """
    redshift1 = _get_z_from_a(scale_factor1)
    redshift2 = _get_z_from_a(scale_factor2)
    if isinstance(cosmo, cosmology.core.FlatLambdaCDM):
        ap_cosmo = cosmo
    else:
        omega_m = cosmo['Omega_b'] + cosmo['Omega_c']
        ap_cosmo = cosmology.core.FlatLambdaCDM(H0=cosmo['H0'], Om0=omega_m,
                                                Ob0=cosmo['Omega_b'])
    # astropy angular diameter distance in Mpc
    # need to return in pc/h
    return ap_cosmo.angular_diameter_distance_z1z2(redshift1, redshift2).to_value(units.pc)\
           * ap_cosmo.H0.value*.01


def get_critical_surface_density(cosmo, z_cluster, z_source):
    r"""Computes the critical surface density

    .. math::
        \Sigma_{crit} = \frac{c^2}{4\pi G} \frac{D_s}{D_LD_{LS}}

    Parameters
    ----------
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    z_cluster : float
        Galaxy cluster redshift
    z_source : array_like, float
        Background source galaxy redshift(s)

    Returns
    -------
    sigmacrit : float
        Cosmology-dependent critical surface density in units of :math:`h M_\odot/pc^2`

    Notes
    -----
    We will need :math:`\gamma_\infty` and :math:`\kappa_\infty` for alternative
    z_src_models using :math:`\beta_s`.
    """
    clight_pc_s = constants.c.to(units.pc/units.s).value
    gnewt_pc3_msun_s2 = constants.G.to(units.pc**3/units.M_sun/units.s**2).value

    aexp_cluster = _get_a_from_z(z_cluster)
    aexp_src = _get_a_from_z(z_source)

    d_l = get_angular_diameter_distance_a(cosmo, aexp_cluster)
    d_s = get_angular_diameter_distance_a(cosmo, aexp_src)
    d_ls = get_angular_diameter_distance_a(cosmo, aexp_src, aexp_cluster)

    sigmacrit = d_s / (d_l * d_ls) * clight_pc_s * clight_pc_s / (4.0 * np.pi * gnewt_pc3_msun_s2)
    return sigmacrit


def predict_tangential_shear(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, delta_mdef=200,
                             halo_profile_model='nfw', z_src_model='single_plane'):
    r"""Computes the tangential shear

    .. math::
        \gamma_t = \frac{\Delta\Sigma}{\Sigma_{crit}} = \frac{\bar{\Sigma}-\Sigma}{\Sigma_{crit}}

    or

    .. math::
        \gamma_t = \gamma_\infty \times \beta_s

    Parameters
    ----------
    r_proj : array_like
        The projected radial positions in :math:`M\!pc/h`.
    mdelta : float
        Galaxy cluster mass in :math:`M_\odot/h`.
    cdelta : float
        Galaxy cluster NFW concentration.
    z_cluster : float
        Galaxy cluster redshift
    z_source : array_like, float
        Background source galaxy redshift(s)
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts e.g. discrete case
        `z_src_distribution` - known source redshift distribution e.g. continuous
        case requiring integration.

    Returns
    -------
    gammat : array_like, float
        tangential shear

    Notes
    -----
    We will need :math:`\gamma_\infty` and :math:`\kappa_\infty` for alternative
    z_src_models using :math:`\beta_s`.
    Need to figure out if we want to raise exceptions rather than errors here?
    """
    delta_sigma = predict_excess_surface_density(r_proj, mdelta, cdelta, z_cluster, cosmo,
                                                 delta_mdef=delta_mdef,
                                                 halo_profile_model=halo_profile_model)

    if z_src_model == 'single_plane':
        sigma_c = get_critical_surface_density(cosmo, z_cluster, z_source)
        gammat = delta_sigma / sigma_c
    elif z_src_model == 'known_z_src': # Discrete case
        raise NotImplementedError('Need to implemnt Beta_s functionality, or average' +
                                  'delta_sigma/sigma_c gamma_t = Beta_s*gamma_inf')
    elif z_src_model == 'z_src_distribution': # Continuous ( from a distribution) case
        raise NotImplementedError('Need to implement Beta_s calculation from integrating' +
                                  'distribution of redshifts in each radial bin')
    else:
        raise ValueError("Unsupported z_src_model")
    return gammat


def predict_convergence(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, delta_mdef=200,
                        halo_profile_model='nfw', z_src_model='single_plane'):
    r"""Computes the mass convergence

    .. math::
        \kappa = \frac{\Sigma}{\Sigma_{crit}}

    or

    .. math::
        \kappa = \kappa_\infty \times \beta_s

    Parameters
    ----------
    r_proj : array_like
        The projected radial positions in :math:`M\!pc/h`.
    mdelta : float
        Galaxy cluster mass in :math:`M_\odot/h`.
    cdelta : float
        Galaxy cluster NFW concentration.
    z_cluster : float
        Galaxy cluster redshift
    z_source : array_like, float
        Background source galaxy redshift(s)
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts e.g. discrete case
        `z_src_distribution` - known source redshift distribution e.g. continuous
        case requiring integration.

    Returns
    -------
    kappa : array_like, float
        Mass convergence, kappa.

    Notes
    -----
    Need to figure out if we want to raise exceptions rather than errors here?
    """
    sigma = predict_surface_density(r_proj, mdelta, cdelta, z_cluster, cosmo,
                                    delta_mdef=delta_mdef, halo_profile_model=halo_profile_model)

    if z_src_model == 'single_plane':
        sigma_c = get_critical_surface_density(cosmo, z_cluster, z_source)
        kappa = sigma / sigma_c
    elif z_src_model == 'known_z_src': # Discrete case
        raise NotImplementedError('Need to implemnt Beta_s functionality, or average' +\
                                  'sigma/sigma_c kappa_t = Beta_s*kappa_inf')
    elif z_src_model == 'z_src_distribution': # Continuous ( from a distribution) case
        raise NotImplementedError('Need to implement Beta_s calculation from integrating' +\
                                  'distribution of redshifts in each radial bin')
    else:
        raise ValueError("Unsupported z_src_model")
    return kappa


def predict_reduced_tangential_shear(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo,
                                     delta_mdef=200, halo_profile_model='nfw',
                                     z_src_model='single_plane'):
    r"""Computes the reduced tangential shear :math:`g_t = \frac{\gamma_t}{1-\kappa}`.

    Parameters
    ----------
    r_proj : array_like
        The projected radial positions in :math:`M\!pc/h`.
    mdelta : float
        Galaxy cluster mass in :math:`M_\odot/h`.
    cdelta : float
        Galaxy cluster NFW concentration.
    z_cluster : float
        Galaxy cluster redshift
    z_source : array_like, float
        Background source galaxy redshift(s)
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous
        case requiring integration.

    Returns
    -------
    gt : array_like, float
        Reduced tangential shear

    Notes
    -----
    Need to figure out if we want to raise exceptions rather than errors here?
    """
    if z_src_model == 'single_plane':
        kappa = predict_convergence(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, delta_mdef,
                                    halo_profile_model,
                                    z_src_model)
        gamma_t = predict_tangential_shear(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo,
                                           delta_mdef, halo_profile_model, z_src_model)
        red_tangential_shear = gamma_t / (1 - kappa)
    elif z_src_model == 'known_z_src': # Discrete case
        raise NotImplementedError('Need to implemnt Beta_s functionality, or average' +
                                  'sigma/sigma_c kappa_t = Beta_s*kappa_inf')
    elif z_src_model == 'z_src_distribution': # Continuous ( from a distribution) case
        raise NotImplementedError('Need to implement Beta_s and Beta_s2 calculation from' +
                                  'integrating distribution of redshifts in each radial bin')
    else:
        raise ValueError("Unsupported z_src_model")
    return red_tangential_shear
