"""@file func_layer.py
Main functions to encapsule oo calls
"""
# Thin functonal layer on top of the class implementation of CLMModeling .
# The functions expect a global instance of the actual CLMModeling named
# `gcm'.

import warnings
import numpy as np

from . import generic
from . generic import (compute_reduced_shear_from_convergence,
                       compute_magnification_bias_from_magnification,
                       compute_rdelta, compute_profile_mass_in_radius,
                       convert_profile_mass_concentration)

__all__ = generic.__all__+['compute_3d_density', 'compute_surface_density',
                           'compute_excess_surface_density','compute_excess_surface_density_2h', 
                           'compute_surface_density_2h',
                           'compute_critical_surface_density',
                           'compute_tangential_shear', 'compute_convergence',
                           'compute_reduced_tangential_shear','compute_magnification',
                           'compute_magnification_bias']


def compute_3d_density(
        r3d, mdelta, cdelta, z_cl, cosmo, delta_mdef=200,
        halo_profile_model='nfw', massdef='mean', alpha_ein=None, verbose=False, validate_input=True):
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
        Profile model parameterization (letter case independent):

            * `nfw` (default);
            * `einasto` - valid in numcosmo only;
            * `hernquist` - valid in numcosmo only;

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * `mean` (default);
            * `critical`;
            * `virial`;

    alpha_ein : float, 'cosmo'(CCL backend only), optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        Set to 'cosmo' to use the cosmology-dependent :math:`\alpha` parameter in the CCL backend.
    verbose : boolean, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only available for the NC and CCL backends.
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.

    Returns
    -------
    rho : array_like, float
        3-dimensional mass density in units of :math:`M_\odot\ Mpc^{-3}`

    Notes
    -----
    Need to refactor later so we only require arguments that are necessary for all profiles
    and use another structure to take the arguments necessary for specific models
    """
    gcm.validate_input = validate_input
    gcm.set_cosmo(cosmo)
    gcm.set_halo_density_profile(
        halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
    gcm.set_concentration(cdelta)
    gcm.set_mass(mdelta)
    if alpha_ein is not None:
        gcm.set_einasto_alpha(alpha_ein)

    rho = gcm.eval_3d_density(r3d, z_cl, verbose=verbose)

    gcm.validate_input = True
    return rho


def compute_surface_density(r_proj, mdelta, cdelta, z_cl, cosmo, delta_mdef=200,
                            halo_profile_model='nfw', massdef='mean', alpha_ein=None, verbose=False, validate_input=True):
    r""" Computes the surface mass density

    .. math::
        \Sigma(R) = \int^\infty_{-\infty} dx\; \rho \left(\sqrt{R^2+x^2}\right),

    where :math:`\rho(r)` is the 3d density profile.

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
        Profile model parameterization (letter case independent):

            * `nfw` (default);
            * `einasto` - valid in numcosmo only;
            * `hernquist` - valid in numcosmo only;

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * `mean` (default);
            * `critical` ;
            * `virial` ;

    alpha_ein : float, 'cosmo'(CCL backend only), optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        Set to 'cosmo' to use the cosmology-dependent :math:`\alpha` parameter in the CCL backend.
    verbose : boolean, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only available for the NC and CCL backends.
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.


    Returns
    -------
    sigma : array_like, float
        2D projected surface density in units of :math:`M_\odot\ Mpc^{-2}`

    Notes
    -----
    Need to refactory so we only require arguments that are necessary for all models and use
    another structure to take the arguments necessary for specific models.
    """
    gcm.validate_input = validate_input
    gcm.set_cosmo(cosmo)
    gcm.set_halo_density_profile(
        halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
    gcm.set_concentration(cdelta)
    gcm.set_mass(mdelta)
    if alpha_ein is not None:
        gcm.set_einasto_alpha(alpha_ein)

    sigma = gcm.eval_surface_density(r_proj, z_cl, verbose=verbose)

    gcm.validate_input = True
    return sigma

def compute_excess_surface_density(r_proj, mdelta, cdelta, z_cl, cosmo, delta_mdef=200,
                                   halo_profile_model='nfw', massdef='mean', alpha_ein=None, verbose=False, validate_input=True):
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
        Profile model parameterization (letter case independent):

            * `nfw` (default);
            * `einasto` - valid in numcosmo only;
            * `hernquist` - valid in numcosmo only;

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * `mean` (default);
            * `critical` - not in cluster_toolkit;
            * `virial` - not in cluster_toolkit;

    alpha_ein : float, 'cosmo'(CCL backend only), optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        Set to 'cosmo' to use the cosmology-dependent :math:`\alpha` parameter in the CCL backend.
    verbose : boolean, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only available for the NC and CCL backends.
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.

    Returns
    -------
    deltasigma : array_like, float
        Excess surface density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    gcm.validate_input = validate_input
    gcm.set_cosmo(cosmo)
    gcm.set_halo_density_profile(
        halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
    gcm.set_concentration(cdelta)
    gcm.set_mass(mdelta)
    if alpha_ein is not None:
        gcm.set_einasto_alpha(alpha_ein)

    deltasigma = gcm.eval_excess_surface_density(r_proj, z_cl, verbose=verbose)

    gcm.validate_input = True
    return deltasigma

def compute_excess_surface_density_2h(r_proj, z_cl, cosmo, halobias=1., lsteps=500, validate_input=True):
    r""" Computes the 2-halo term excess surface density from eq.(13) of Oguri & Hamana (2011)

    .. math::
        \Delta\Sigma_{\rm 2h}(R) = \frac{\rho_m(z)b(M)}{(1 + z)^3D_A(z)^2} \int\frac{ldl}{(2\pi)} P_{\rm mm}(k_l, z)J_2(l\theta)

    where

    .. math::
        k_l = \frac{l}{D_A(z)(1 +z)}

    and :math:`b(M)` is the halo bias

    Parameters
    ----------
    r_proj : array_like
        Projected radial position from the cluster center in :math:`M\!pc`.
    z_cl: float
        Redshift of the cluster
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    halobias : float, optional
        Value of the halo bias
    lsteps : int, optional
        Steps for the numerical integration 
    validate_input: bool
        Validade each input argument

    Returns
    -------
    deltasigma_2h : array_like, float
        2-halo term excess surface density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    gcm.validate_input = validate_input
    gcm.set_cosmo(cosmo)

    deltasigma_2h = gcm.eval_excess_surface_density_2h(r_proj, z_cl, halobias=halobias, lsteps=lsteps)

    gcm.validate_input = True
    return deltasigma_2h

def compute_surface_density_2h(r_proj, z_cl, cosmo, halobias=1, lsteps=500, validate_input=True):
    r""" Computes the 2-halo term surface density from eq.(13) of Oguri & Hamana (2011)

    .. math::
        \Sigma_{\rm 2h}(R) = \frac{\rho_m(z)b(M)}{(1 + z)^3D_A(z)^2} \int\frac{ldl}{(2\pi)} P_{\rm mm}(k_l, z)J_0(l\theta)

    where

    .. math::
        k_l = \frac{l}{D_A(z)(1 +z)}

    and :math:`b(M)` is the halo bias

    Parameters
    ----------
    r_proj : array_like
        Projected radial position from the cluster center in :math:`M\!pc`.
    z_cl: float
        Redshift of the cluster
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    halobias : float, optional
        Value of the halo bias
    lsteps : int, optional
        Steps for the numerical integration 
    validate_input: bool
        Validade each input argument

    Returns
    -------
    sigma_2h : array_like, float
        2-halo term surface density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    gcm.validate_input = validate_input
    gcm.set_cosmo(cosmo)

    sigma_2h = gcm.eval_surface_density_2h(r_proj, z_cl, halobias = halobias, lsteps=lsteps)

    gcm.validate_input = True
    return sigma_2h

def compute_critical_surface_density(cosmo, z_cluster, z_source=None, use_pdz=False, pzbins=None, pzpdf=None, validate_input=True):
    r"""Computes either 

    - the critical surface density if `use_pdz=False`

    .. math::
        \Sigma_{\rm crit} = \frac{c^2}{4\pi G} \frac{D_s}{D_LD_{LS}}

    or

    - the 'effective critical surface density' if `use_pdz=True`
     
    .. math::
        \langle \Sigma_{\rm crit}^{-1}\rangle^{-1} = \left(\int \frac{1}{\Sigma_{\rm crit}(z)} p(z) dz\right)^{-1} 
    
    where :math:`p(z)` is the source photoz probability density function. 
    This comes from the maximum likelihood estimator for evaluating a :math:`\Delta\Sigma` profile.


    Parameters
    ----------
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    z_cluster : float
        Galaxy cluster redshift
    z_source : array_like, float
        Background source galaxy redshift(s)
    use_pdz : bool
        Flag to use the photoz pdf. If `False` (default), `sigma_c` is computed using the source redshift point estimates `z_source`. 
        If `True`, `sigma_c` is computed as 1/<1/Sigma_crit>, where the average is performed using the individual galaxy redshift pdf. 
        In that case, the `pzbins` and `pzpdf` should be specified. 
    pzbins : array-like
        Bins where the source redshift pdf is defined
    pzpdf : array-like
        Values of the source redshift pdf
    validate_input: bool
        Validade each input argument


    Returns
    -------
    sigma_c : array_like, float
        Cosmology-dependent (effective) critical surface density in units of :math:`M_\odot\ Mpc^{-2}`
    """

    gcm.validate_input = validate_input
    gcm.set_cosmo(cosmo)
    sigma_c = gcm.eval_critical_surface_density(z_cluster, z_src=z_source, use_pdz=use_pdz, pzbins=pzbins, pzpdf=pzpdf)

    gcm.validate_input = True
    return sigma_c

def compute_tangential_shear(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, delta_mdef=200,
                             halo_profile_model='nfw', massdef='mean', alpha_ein=None, z_src_model='single_plane', 
                             verbose=False, validate_input=True):
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
        Profile model parameterization (letter case independent):

            * `nfw` (default);
            * `einasto` - valid in numcosmo only;
            * `hernquist` - valid in numcosmo only;

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * `mean` (default);
            * `critical` - not in cluster_toolkit;
            * `virial` - not in cluster_toolkit;

    alpha_ein : float, 'cosmo'(CCL backend only), optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        Set to 'cosmo' to use the cosmology-dependent :math:`\alpha` parameter in the CCL backend.
    z_src_model : str, optional
        Source redshift model, with the following supported options:
            `single_plane` (default) - all sources at one redshift (if
            `z_source` is a float) or known individual source galaxy redshifts
            (if `z_source` is an array and `r_proj` is a float);
    verbose : bool, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and CCL backends. 
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.

    Returns
    -------
    gammat : array_like, float
        Tangential shear

    Notes
    -----
    TODO: Implement `known_z_src` (known individual source galaxy redshifts
    e.g. discrete case) and `z_src_distribution` (known source redshift
    distribution e.g. continuous case requiring integration) options for
    `z_src_model`. We will need :math:`\gamma_\infty` and :math:`\kappa_\infty`
    for alternative z_src_models using :math:`\beta_s`.
    """
    if z_src_model == 'single_plane':
        gcm.validate_input = validate_input
        gcm.set_cosmo(cosmo)
        gcm.set_halo_density_profile(
            halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
        gcm.set_concentration(cdelta)
        gcm.set_mass(mdelta)
        if alpha_ein is not None:
            gcm.set_einasto_alpha(alpha_ein)
        if np.min(r_proj) < 1.e-11:
            raise ValueError(
                f"Rmin = {np.min(r_proj):.2e} Mpc/h! This value is too small "
                "and may cause computational issues.")

        gammat = gcm.eval_tangential_shear(r_proj, z_cluster, z_source, verbose=verbose)
    else:
        raise ValueError("Unsupported z_src_model")

    gcm.validate_input = True
    return gammat


def compute_convergence(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, delta_mdef=200,
                        halo_profile_model='nfw', massdef='mean', alpha_ein=None, z_src_model='single_plane',
                        verbose=False, validate_input=True):
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
        Profile model parameterization (letter case independent):

            * `nfw` (default);
            * `einasto` - valid in numcosmo only;
            * `hernquist` - valid in numcosmo only;

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * `mean` (default);
            * `critical` - not in cluster_toolkit;
            * `virial` - not in cluster_toolkit;

    alpha_ein : float, 'cosmo'(CCL backend only), optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        Set to 'cosmo' to use the cosmology-dependent :math:`\alpha` parameter in the CCL backend.
    z_src_model : str, optional
        Source redshift model, with the following supported options:
            `single_plane` (default) - all sources at one redshift (if
            `z_source` is a float) or known individual source galaxy redshifts
            (if `z_source` is an array and `r_proj` is a float);
    verbose : bool, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and CCL backends. 
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.

    Returns
    -------
    kappa : array_like, float
        Mass convergence, kappa.

    Notes
    -----
    TODO: Implement `known_z_src` (known individual source galaxy redshifts
    e.g. discrete case) and `z_src_distribution` (known source redshift
    distribution e.g. continuous case requiring integration) options for
    `z_src_model`. We will need :math:`\gamma_\infty` and :math:`\kappa_\infty`
    for alternative z_src_models using :math:`\beta_s`.
    """

    if z_src_model == 'single_plane':

        gcm.validate_input = validate_input
        gcm.set_cosmo(cosmo)
        gcm.set_halo_density_profile(
            halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
        gcm.set_concentration(cdelta)
        gcm.set_mass(mdelta)
        if alpha_ein is not None:
            gcm.set_einasto_alpha(alpha_ein)

        kappa = gcm.eval_convergence(r_proj, z_cluster, z_source, verbose=verbose)

    # elif z_src_model == 'known_z_src': # Discrete case
    #     raise NotImplementedError('Need to implemnt Beta_s functionality, or average'+\
    #                               'sigma/sigma_c kappa_t = Beta_s*kappa_inf')
    # elif z_src_model == 'z_src_distribution': # Continuous ( from a distribution) case
    #     raise NotImplementedError('Need to implement Beta_s calculation from integrating'+\
    #                               'distribution of redshifts in each radial bin')
    else:
        raise ValueError("Unsupported z_src_model")

    if np.any(np.array(z_source) <= z_cluster):
        warnings.warn(
            'Some source redshifts are lower than the cluster redshift.'
            ' kappa = 0 for those galaxies.')

    gcm.validate_input = True
    return kappa


def compute_reduced_tangential_shear(
        r_proj, mdelta, cdelta, z_cluster, z_source, cosmo,
        delta_mdef=200, halo_profile_model='nfw', massdef='mean', alpha_ein=None,
        z_src_model='single_plane', beta_s_mean=None, beta_s_square_mean=None,
        z_distrib_func=None, validate_input=True, verbose=False):
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
        Profile model parameterization (letter case independent):

            * `nfw` (default);
            * `einasto` - valid in numcosmo only;
            * `hernquist` - valid in numcosmo only;

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * `mean` (default);
            * `critical` - not in cluster_toolkit;
            * `virial` - not in cluster_toolkit;

    alpha_ein : float, 'cosmo'(CCL backend only), optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        Set to 'cosmo' to use the cosmology-dependent :math:`\alpha` parameter in the CCL backend.
    z_src_model : str, optional
        Source redshift model, with the following supported options:

            * `single_plane` (default): all sources at one redshift (if `z_source` is a float) \
                or known individual source galaxy redshifts (if `z_source` is an array and \
                `r_proj` is a float);
            * `applegate14`: use the equation (6) in Weighing the Giants - III \
                (Applegate et al. 2014; https://arxiv.org/abs/1208.0605) to evaluate tangential reduced shear;
            * `schrabback18`: use the equation (12) in Cluster Mass Calibration at High Redshift \
                (Schrabback et al. 2017; https://arxiv.org/abs/1611.03866) to evaluate tangential reduced shear;

        z_distrib_func: one-parameter function
            Redshift distribution function. This function is used to compute the beta values if they are not provided. The default is the Chang et al (2013) distribution function.

        beta_s_mean: array_like, float, optional
            Lensing efficiency averaged over the galaxy redshift distribution. If not provided, it will be computed using the default redshift distribution or the one given by the user.

                .. math::
                    \langle \beta_s \rangle = \left\langle \frac{D_{LS}}{D_S}\frac{D_\infty}{D_{L,\infty}}\right\rangle

        beta_s_square_mean: array_like, float, optional
            Square of the lensing efficiency averaged over the galaxy redshift distribution. If not provided, it will be computed using the default redshift distribution or the one given by the user.

                .. math::
                    \langle \beta_s^2 \rangle = \left\langle \left(\frac{D_{LS}}{D_S}\frac{D_\infty}{D_{L,\infty}}\right)^2 \right\rangle
    
    verbose : bool, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and CCL backends. 
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.

                .. math::
                    \langle \beta_s^2 \rangle = \left\langle \left(\frac{D_{LS}}{D_S}\frac{D_\infty}{D_{L,\infty}}\right)^2 \right\rangle

    Returns
    -------
    gt : array_like, float
        Reduced tangential shear

    """
    gcm.validate_input = validate_input
    gcm.set_cosmo(cosmo)
    gcm.set_halo_density_profile(
        halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
    gcm.set_concentration(cdelta)
    gcm.set_mass(mdelta)
    if alpha_ein is not None:
        gcm.set_einasto_alpha(alpha_ein)

    red_tangential_shear = gcm.eval_reduced_tangential_shear(
        r_proj, z_cluster, z_source, z_src_model, beta_s_mean, beta_s_square_mean, z_distrib_func, verbose=verbose)

    gcm.validate_input = True
    return red_tangential_shear

# The magnification is computed taking into account just the tangential shear. This is valid for
# spherically averaged profiles, e.g., NFW and Einasto (by construction the cross shear is zero).


def compute_magnification(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, delta_mdef=200,
                          halo_profile_model='nfw', massdef='mean', alpha_ein=None, z_src_model='single_plane',
                          verbose=False, validate_input=True):
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
        Profile model parameterization (letter case independent):

            * `nfw` (default);
            * `einasto` - valid in numcosmo only;
            * `hernquist` - valid in numcosmo only;

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * `mean` (default);
            * `critical` - not in cluster_toolkit;
            * `virial` - not in cluster_toolkit;

    alpha_ein : float, 'cosmo'(CCL backend only), optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        Set to 'cosmo' to use the cosmology-dependent :math:`\alpha` parameter in the CCL backend.
    z_src_model : str, optional
        Source redshift model, with the following supported options:
            `single_plane` (default) - all sources at one redshift (if
            `z_source` is a float) or known individual source galaxy redshifts
            (if `z_source` is an array and `r_proj` is a float);
    verbose : bool, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and CCL backends. 
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.

    Returns
    -------
    magnification : array_like, float
        Magnification :math:`\mu`.

    Notes
    -----
    TODO: Implement `known_z_src` (known individual source galaxy redshifts e.g. discrete case) and
    `z_src_distribution` (known source redshift distribution e.g. continuous case requiring
    integration) options for `z_src_model`. We will need :math:`\gamma_\infty` and
    :math:`\kappa_\infty` for alternative z_src_models using :math:`\beta_s`.
    """

    if z_src_model == 'single_plane':

        gcm.validate_input = validate_input
        gcm.set_cosmo(cosmo)
        gcm.set_halo_density_profile(
            halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
        gcm.set_concentration(cdelta)
        gcm.set_mass(mdelta)
        if alpha_ein is not None:
            gcm.set_einasto_alpha(alpha_ein)

        magnification = gcm.eval_magnification(r_proj, z_cluster, z_source, verbose=verbose)

    # elif z_src_model == 'known_z_src': # Discrete case
    #     raise NotImplementedError('Need to implemnt Beta_s functionality, or average'+\
    #                               'sigma/sigma_c kappa_t = Beta_s*kappa_inf')
    # elif z_src_model == 'z_src_distribution': # Continuous ( from a distribution) case
    #     raise NotImplementedError('Need to implement Beta_s calculation from integrating'+\
    #                               'distribution of redshifts in each radial bin')
    else:
        raise ValueError("Unsupported z_src_model")

    if np.any(np.array(z_source) <= z_cluster):
        warnings.warn(
            'Some source redshifts are lower than the cluster redshift.'
            ' magnification = 1 for those galaxies.')

    gcm.validate_input = True
    return magnification



def compute_magnification_bias(r_proj, alpha, mdelta, cdelta, z_cluster, z_source, cosmo,
                               delta_mdef=200, halo_profile_model='nfw', massdef='mean',
                               z_src_model='single_plane', validate_input=True):

    r""" Computes magnification bias from magnification :math:`\mu` 
    and slope parameter :math:`\alpha` as :

    .. math::
        \mu^{\alpha - 1}.

    The alpha parameter depends on the source sample and is computed as the slope of the 
    cummulative numer counts at a given magnitude :

    .. math::
        \alpha \equiv \alpha(f) = - \frac{\mathrm{d}}{\mathrm{d}\log{f}} \log{n_0(>f)}

    or,

    .. math::
        \alpha \equiv \alpha(m) = 2.5 \frac{\mathrm d}{\mathrm d m} \log{n_0(<m)}

    see e.g.  Bartelmann & Schneider 2001; Umetsu 2020

    Parameters
    ----------
    r_proj : array_like, float
        The projected radial positions in :math:`M\!pc`.
    alpha : array like
        The slope of the cummulative source number counts.
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
        Profile model parameterization (letter case independent):

            * `nfw` (default);
            * `einasto` - valid in numcosmo only;
            * `hernquist` - valid in numcosmo only;

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * `mean` (default);
            * `critical` - not in cluster_toolkit;
            * `virial` - not in cluster_toolkit;

    z_src_model : str, optional
        Source redshift model, with the following supported options:
            `single_plane` (default) - all sources at one redshift (if
            `z_source` is a float) or known individual source galaxy redshifts
            (if `z_source` is an array and `r_proj` is a float);


    Returns
    -------
    magnification_bias : array_like
        magnification bias
    """
    if np.any(np.array(z_source) <= z_cluster):
        warnings.warn(
            'Some source redshifts are lower than the cluster redshift.'
            ' magnification = 1 for those galaxies.')
    if z_src_model == 'single_plane':

        gcm.validate_input = validate_input
        gcm.set_cosmo(cosmo)
        gcm.set_halo_density_profile(
            halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef)
        gcm.set_concentration(cdelta)
        gcm.set_mass(mdelta)

        magnification_bias = gcm.eval_magnification_bias(r_proj, z_cluster, z_source, alpha)

    else:
        raise ValueError("Unsupported z_src_model")


    gcm.validate_input = True
    return magnification_bias
