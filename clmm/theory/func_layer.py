# Thin functonal layer on top of the class implementation of CLMModeling .
# The functions expect a global instance of the actual CLMModeling named
# `gcm'.

import numpy as np
import warnings

from . import generic
from . generic import compute_reduced_shear_from_convergence

__all__ = generic.__all__+['compute_3d_density', 'compute_surface_density',
           'compute_excess_surface_density', 'compute_critical_surface_density',
           'compute_tangential_shear', 'compute_convergence',
           'compute_reduced_tangential_shear', 'compute_magnification']


def compute_3d_density(r3d, mdelta, cdelta, z_cl, cosmo, delta_mdef=200, halo_profile_model='nfw', massdef='mean'):
    r"""Retrieve the 3d density :math:`\rho(r)`.

    Profiles implemented so far are:

        `nfw`: :math:`\rho(r) = \frac{\rho_0}{\frac{c}{(r/R_{vir})}
        \left(1+\frac{c}{(r/R_{vir})}\right)^2}` [insert citation here]

    Parameters
    ----------
    r3d : array_like, float
        Radial position from the cluster center in :math:`M\!pc`.
    mdelta : float
        Galaxy cluster mass in :math:`M_\odot`.
    cdelta : float
        Galaxy cluster concentration
    z_cl: float
        Redshift of the cluster
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition; defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization, with the following supported options (letter case independent):
            `nfw` (default)
            `einasto` - valid in numcosmo and ccl (version>=TBA)
            `hernquist` - valid in numcosmo and ccl (version>=TBA)
    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):
            `mean` (default)
            `critical` - not in cluster_toolkit
            `virial` - not in cluster_toolkit

    Returns
    -------
    rho : array_like, float
        3-dimensional mass density in units of :math:`M_\odot\ Mpc^{-3}`

    Notes
    -----
    Need to refactor later so we only require arguments that are necessary for all profiles
    and use another structure to take the arguments necessary for specific models
    """

    gcm.set_cosmo(cosmo)
    gcm.set_halo_density_profile(halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
    gcm.set_concentration(cdelta)
    gcm.set_mass(mdelta)

    return gcm.eval_3d_density(r3d, z_cl)


def compute_surface_density(r_proj, mdelta, cdelta, z_cl, cosmo, delta_mdef=200,
                            halo_profile_model='nfw', massdef='mean'):
    r""" Computes the surface mass density

    .. math::
        \Sigma(R) = \Omega_m \rho_{crit} \int^\infty_{-\infty} dz \Xi_{hm} (\sqrt{R^2+z^2}),

    where :math:`\Xi_{hm}` is the halo-matter correlation function.

    Parameters
    ----------
    r_proj : array_like
        Projected radial position from the cluster center in :math:`M\!pc`.
    mdelta : float
        Galaxy cluster mass in :math:`M_\odot`.
    cdelta : float
        Galaxy cluster concentration
    z_cl: float
        Redshift of the cluster
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition; defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization, with the following supported options (letter case independent):
            `nfw` (default)
            `einasto` - valid in numcosmo and ccl (version>=TBA)
            `hernquist` - valid in numcosmo and ccl (version>=TBA)
    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):
            `mean` (default)
            `critical` - not in cluster_toolkit
            `virial` - not in cluster_toolkit

    Returns
    -------
    sigma : array_like, float
        2D projected surface density in units of :math:`M_\odot\ Mpc^{-2}`

    Notes
    -----
    Need to refactory so we only require arguments that are necessary for all models and use
    another structure to take the arguments necessary for specific models.
    """

    gcm.set_cosmo(cosmo)
    gcm.set_halo_density_profile(halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
    gcm.set_concentration(cdelta)
    gcm.set_mass(mdelta)

    return gcm.eval_surface_density(r_proj, z_cl)


def compute_excess_surface_density(r_proj, mdelta, cdelta, z_cl, cosmo, delta_mdef=200,
                                   halo_profile_model='nfw', massdef='mean'):
    r""" Computes the excess surface density

    .. math::
        \Delta\Sigma(R) = \bar{\Sigma}(<R)-\Sigma(R),

    where

    .. math::
        \bar{\Sigma}(<R) = \frac{2}{R^2} \int^R_0 dR' R' \Sigma(R')

    Parameters
    ----------
    r_proj : array_like
        Projected radial position from the cluster center in :math:`M\!pc`.
    mdelta : float
        Galaxy cluster mass in :math:`M_\odot`.
    cdelta : float
        Galaxy cluster concentration
    z_cl: float
        Redshift of the cluster
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition; defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization, with the following supported options (letter case independent):
            `nfw` (default)
            `einasto` - valid in numcosmo and ccl (version>=TBA)
            `hernquist` - valid in numcosmo and ccl (version>=TBA)
    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):
            `mean` (default)
            `critical` - not in cluster_toolkit
            `virial` - not in cluster_toolkit

    Returns
    -------
    deltasigma : array_like, float
        Excess surface density in units of :math:`M_\odot\ Mpc^{-2}`.
    """

    gcm.set_cosmo(cosmo)
    gcm.set_halo_density_profile(halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
    gcm.set_concentration(cdelta)
    gcm.set_mass(mdelta)

    return gcm.eval_excess_surface_density(r_proj, z_cl)


def compute_critical_surface_density(cosmo, z_cluster, z_source):
    r"""Computes the critical surface density

    .. math::
        \Sigma_{crit} = \frac{c^2}{4\pi G} \frac{D_s}{D_LD_{LS}}

    Parameters
    ----------
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    z_cluster : float
        Galaxy cluster redshift
    z_source : array_like, float
        Background source galaxy redshift(s)

    Returns
    -------
    sigma_c : float
        Cosmology-dependent critical surface density in units of :math:`M_\odot\ Mpc^{-2}`

    Notes
    -----
    We will need :math:`\gamma_\infty` and :math:`\kappa_\infty` for alternative
    z_src_models using :math:`\beta_s`.
    """

    gcm.set_cosmo(cosmo)
    return gcm.eval_critical_surface_density(z_cluster, z_source)


def compute_tangential_shear(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, delta_mdef=200,
                              halo_profile_model='nfw', massdef='mean', z_src_model='single_plane'):
    r"""Computes the tangential shear

    .. math::
        \gamma_t = \frac{\Delta\Sigma}{\Sigma_{crit}} = \frac{\bar{\Sigma}-\Sigma}{\Sigma_{crit}}

    or

    .. math::
        \gamma_t = \gamma_\infty \times \beta_s

    Parameters
    ----------
    r_proj : array_like, float
        The projected radial positions in :math:`M\!pc`.
    mdelta : float
        Galaxy cluster mass in :math:`M_\odot`.
    cdelta : float
        Galaxy cluster NFW concentration.
    z_cluster : float
        Galaxy cluster redshift
    z_source : array_like, float
        Background source galaxy redshift(s)
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization, with the following supported options (letter case independent):
            `nfw` (default)
            `einasto` - valid in numcosmo and ccl (version>=TBA)
            `hernquist` - valid in numcosmo and ccl (version>=TBA)
    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):
            `mean` (default)
            `critical` - not in cluster_toolkit
            `virial` - not in cluster_toolkit
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift (if `z_source`  is float) or known individual source galaxy redshifts (if `z_source` is an array). At the moment the latter case only works if `r_proj` is a float.

    Returns
    -------
    gammat : array_like, float
        Tangential shear

    Notes
    -----
    TODO: Implement `known_z_src` (known individual source galaxy redshifts e.g. discrete case) and `z_src_distribution` (known source redshift distribution e.g. continuous case requiring integration) options for `z_src_model`. We will need :math:`\gamma_\infty` and :math:`\kappa_\infty` for alternative z_src_models using :math:`\beta_s`.
    """

    if z_src_model == 'single_plane':

        gcm.set_cosmo(cosmo)
        gcm.set_halo_density_profile(halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
        gcm.set_concentration(cdelta)
        gcm.set_mass(mdelta)

        if np.min(r_proj) < 1.e-11:
            raise ValueError(f"Rmin = {np.min(r_proj):.2e} Mpc/h! This value is too small and may cause computational issues.")

        gammat = gcm.eval_tangential_shear(r_proj, z_cluster, z_source)
    else:
        raise ValueError("Unsupported z_src_model")

    return gammat


def compute_convergence(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, delta_mdef=200,
                        halo_profile_model='nfw', massdef='mean', z_src_model='single_plane'):
    r"""Computes the mass convergence

    .. math::
        \kappa = \frac{\Sigma}{\Sigma_{crit}}

    or

    .. math::
        \kappa = \kappa_\infty \times \beta_s

    Parameters
    ----------
    r_proj : array_like, float
        The projected radial positions in :math:`M\!pc`.
    mdelta : float
        Galaxy cluster mass in :math:`M_\odot`.
    cdelta : float
        Galaxy cluster NFW concentration.
    z_cluster : float
        Galaxy cluster redshift
    z_source : array_like, float
        Background source galaxy redshift(s)
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization, with the following supported options (letter case independent):
            `nfw` (default)
            `einasto` - valid in numcosmo and ccl (version>=TBA)
            `hernquist` - valid in numcosmo and ccl (version>=TBA)
    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):
            `mean` (default)
            `critical` - not in cluster_toolkit
            `virial` - not in cluster_toolkit
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift (if `z_source`  is float) or known individual source galaxy redshifts (if `z_source` is an array). At the moment the latter case only works if `r_proj` is a float.

    Returns
    -------
    kappa : array_like, float
        Mass convergence, kappa.

    Notes
    -----
    TODO: Implement `known_z_src` (known individual source galaxy redshifts e.g. discrete case) and `z_src_distribution` (known source redshift distribution e.g. continuous case requiring integration) options for `z_src_model`. We will need :math:`\gamma_\infty` and :math:`\kappa_\infty` for alternative z_src_models using :math:`\beta_s`.
    """

    sigma = compute_surface_density(r_proj, mdelta, cdelta, z_cluster, cosmo,
                                    delta_mdef=delta_mdef, halo_profile_model=halo_profile_model,
                                    massdef=massdef)

    if z_src_model == 'single_plane':

        gcm.set_cosmo(cosmo)
        gcm.set_halo_density_profile(halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
        gcm.set_concentration(cdelta)
        gcm.set_mass(mdelta)

        kappa = gcm.eval_convergence(r_proj, z_cluster, z_source)

    # elif z_src_model == 'known_z_src': # Discrete case
    #     raise NotImplementedError('Need to implemnt Beta_s functionality, or average'+\
    #                               'sigma/sigma_c kappa_t = Beta_s*kappa_inf')
    # elif z_src_model == 'z_src_distribution': # Continuous ( from a distribution) case
    #     raise NotImplementedError('Need to implement Beta_s calculation from integrating'+\
    #                               'distribution of redshifts in each radial bin')
    else:
        raise ValueError("Unsupported z_src_model")

    if np.any(np.array(z_source)<=z_cluster):
        warnings.warn(f'Some source redshifts are lower than the cluster redshift. kappa = 0 for those galaxies.')

    return kappa


def compute_reduced_tangential_shear(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo,
                                     delta_mdef=200, halo_profile_model='nfw', massdef='mean',
                                     z_src_model='single_plane'):
    r"""Computes the reduced tangential shear :math:`g_t = \frac{\gamma_t}{1-\kappa}`.

    Parameters
    ----------
    r_proj : array_like, float
        The projected radial positions in :math:`M\!pc`.
    mdelta : float
        Galaxy cluster mass in :math:`M_\odot`.
    cdelta : float
        Galaxy cluster NFW concentration.
    z_cluster : float
        Galaxy cluster redshift
    z_source : array_like, float
        Background source galaxy redshift(s)
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization, with the following supported options (letter case independent):
            `nfw` (default)
            `einasto` - valid in numcosmo and ccl (version>=TBA)
            `hernquist` - valid in numcosmo and ccl (version>=TBA)
    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):
            `mean` (default)
            `critical` - not in cluster_toolkit
            `virial` - not in cluster_toolkit
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift (if `z_source`  is float) or known individual source galaxy redshifts (if `z_source` is an array). At the moment the latter case only works if `r_proj` is a float.

    Returns
    -------
    gt : array_like, float
        Reduced tangential shear

    Notes
    -----
    TODO: Implement `known_z_src` (known individual source galaxy redshifts e.g. discrete case) and `z_src_distribution` (known source redshift distribution e.g. continuous case requiring integration) options for `z_src_model`. We will need :math:`\gamma_\infty` and :math:`\kappa_\infty` for alternative z_src_models using :math:`\beta_s`.
    """

    if z_src_model == 'single_plane':

        gcm.set_cosmo(cosmo)
        gcm.set_halo_density_profile(halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
        gcm.set_concentration(cdelta)
        gcm.set_mass(mdelta)

        red_tangential_shear = gcm.eval_reduced_tangential_shear(r_proj, z_cluster, z_source)

    # elif z_src_model == 'known_z_src': # Discrete case
    #     raise NotImplementedError('Need to implemnt Beta_s functionality, or average'+
    #                               'sigma/sigma_c kappa_t = Beta_s*kappa_inf')
    # elif z_src_model == 'z_src_distribution': # Continuous ( from a distribution) case
    #     raise NotImplementedError('Need to implement Beta_s and Beta_s2 calculation from'+
    #                               'integrating distribution of redshifts in each radial bin')
    else:
        raise ValueError("Unsupported z_src_model")

    if np.any(np.array(z_source)<=z_cluster):
        warnings.warn(f'Some source redshifts are lower than the cluster redshift. shear = 0 for those galaxies.')


    return red_tangential_shear


# The magnification is computed taking into account just the tangential shear. This is valid for
# spherically averaged profiles, e.g., NFW and Einasto (by construction the cross shear is zero).


def compute_magnification(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, delta_mdef=200,
                        halo_profile_model='nfw', massdef='mean', z_src_model='single_plane'):
    r"""Computes the magnification

    .. math::
        \mu = \frac{1}{(1-\kappa)^2-|\gamma_t|^2}



    Parameters
    ----------
    r_proj : array_like, float
        The projected radial positions in :math:`M\!pc`.
    mdelta : float
        Galaxy cluster mass in :math:`M_\odot`.
    cdelta : float
        Galaxy cluster NFW concentration.
    z_cluster : float
        Galaxy cluster redshift
    z_source : array_like, float
        Background source galaxy redshift(s)
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization, with the following supported options (letter case independent):
            `nfw` (default)
            `einasto` - valid in numcosmo and ccl (version>=TBA)
            `hernquist` - valid in numcosmo and ccl (version>=TBA)
    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):
            `mean` (default)
            `critical` - not in cluster_toolkit
            `virial` - not in cluster_toolkit
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift (if `z_source`  is float) or known individual source galaxy redshifts (if `z_source` is an array). At the moment the latter case only works if `r_proj` is a float.


    Returns
    -------
    mu : array_like, float
        magnification, mu.

    Notes
    -----
    TODO: Implement `known_z_src` (known individual source galaxy redshifts e.g. discrete case) and `z_src_distribution` (known source redshift distribution e.g. continuous case requiring integration) options for `z_src_model`. We will need :math:`\gamma_\infty` and :math:`\kappa_\infty` for alternative z_src_models using :math:`\beta_s`.
    """

    if z_src_model == 'single_plane':

        gcm.set_cosmo(cosmo)
        gcm.set_halo_density_profile(halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
        gcm.set_concentration(cdelta)
        gcm.set_mass(mdelta)

        mu = gcm.eval_magnification(r_proj, z_cluster, z_source)

    # elif z_src_model == 'known_z_src': # Discrete case
    #     raise NotImplementedError('Need to implemnt Beta_s functionality, or average'+\
    #                               'sigma/sigma_c kappa_t = Beta_s*kappa_inf')
    # elif z_src_model == 'z_src_distribution': # Continuous ( from a distribution) case
    #     raise NotImplementedError('Need to implement Beta_s calculation from integrating'+\
    #                               'distribution of redshifts in each radial bin')
    else:
        raise ValueError("Unsupported z_src_model")

    if np.any(np.array(z_source)<=z_cluster):
        warnings.warn(f'Some source redshifts are lower than the cluster redshift. mu = 1 for those galaxies.')

    return mu
