"""@file func_layer.py
Main functions to encapsule oo calls
"""
# pylint: disable=too-many-lines
# pylint: disable=invalid-name
# Thin functonal layer on top of the class implementation of CLMModeling .
# The functions expect a global instance of the actual CLMModeling named
# "_modeling_object".

import numpy as np

if "_modeling_object" not in globals():
    _modeling_object = None


def compute_3d_density(
    r3d,
    mdelta,
    cdelta,
    z_cl,
    cosmo,
    delta_mdef=200,
    halo_profile_model="nfw",
    massdef="mean",
    alpha_ein=None,
    verbose=False,
    validate_input=True,
):
    r"""Retrieve the 3d density :math:`\rho(r)`.

    Profiles implemented so far are:

        `nfw`: :math:`\rho(r) = \frac{\rho_0}{\frac{c}{(r/R_{vir})}
        \left(1+\frac{c}{(r/R_{vir})}\right)^2}` (Navarro et al. 1996)

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

            * 'nfw' (default)
            * 'einasto' - not in cluster_toolkit
            * 'hernquist' - not in cluster_toolkit

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * 'mean' (default)
            * 'critical'
            * 'virial'

    alpha_ein : float, None, optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        If None, use the default value of the backend. (0.25 for the NumCosmo backend and a
        cosmology-dependent value for the CCL backend.)
    verbose : boolean, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only available for the NC and CCL
        backends.
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.

    Returns
    -------
    rho : numpy.ndarray, float
        3-dimensional mass density in units of :math:`M_\odot\ Mpc^{-3}`

    Notes
    -----
    Need to refactor later so we only require arguments that are necessary for all profiles
    and use another structure to take the arguments necessary for specific models
    """
    _modeling_object.validate_input = validate_input
    _modeling_object.set_cosmo(cosmo)
    _modeling_object.set_halo_density_profile(
        halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef
    )
    _modeling_object.set_concentration(cdelta)
    _modeling_object.set_mass(mdelta)
    if halo_profile_model == "einasto" or alpha_ein is not None:
        _modeling_object.set_einasto_alpha(alpha_ein)

    rho = _modeling_object.eval_3d_density(r3d, z_cl, verbose=verbose)

    _modeling_object.validate_input = True
    return rho


def compute_surface_density(
    r_proj,
    mdelta,
    cdelta,
    z_cl,
    cosmo,
    delta_mdef=200,
    halo_profile_model="nfw",
    massdef="mean",
    alpha_ein=None,
    verbose=False,
    use_projected_quad=False,
    validate_input=True,
):
    r"""Computes the surface mass density

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

            * 'nfw' (default)
            * 'einasto' - not in cluster_toolkit
            * 'hernquist' - not in cluster_toolkit

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * 'mean' (default)
            * 'critical'
            * 'virial'

    alpha_ein : float, None, optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        If None, use the default value of the backend. (0.25 for the NumCosmo backend and a
        cosmology-dependent value for the CCL backend.)
    verbose : boolean, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only available for the NC and CCL
        backends.
    use_projected_quad : bool
        Only available for Einasto profile with CCL as the backend. If True, CCL will use
        quad_vec instead of default FFTLog to calculate the surface density profile.
        Default: False
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.

    Returns
    -------
    sigma : numpy.ndarray, float
        2D projected surface density in units of :math:`M_\odot\ Mpc^{-2}`

    Notes
    -----
    Need to refactory so we only require arguments that are necessary for all models and use
    another structure to take the arguments necessary for specific models.
    """
    _modeling_object.validate_input = validate_input
    _modeling_object.set_cosmo(cosmo)
    _modeling_object.set_halo_density_profile(
        halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef
    )
    _modeling_object.set_concentration(cdelta)
    _modeling_object.set_mass(mdelta)
    if halo_profile_model == "einasto" or alpha_ein is not None:
        _modeling_object.set_einasto_alpha(alpha_ein)
    if halo_profile_model == "einasto" and _modeling_object.backend=="ccl":
        _modeling_object.set_projected_quad(use_projected_quad)

    sigma = _modeling_object.eval_surface_density(r_proj, z_cl, verbose=verbose)

    _modeling_object.validate_input = True
    return sigma


def compute_mean_surface_density(
    r_proj,
    mdelta,
    cdelta,
    z_cl,
    cosmo,
    delta_mdef=200,
    halo_profile_model="nfw",
    massdef="mean",
    alpha_ein=None,
    verbose=False,
    validate_input=True,
):
    r"""Computes the mean value of surface density inside radius `r_proj`

    .. math::
        \bar{\Sigma}(<R) = \frac{2}{R^2} \int^R_0 dR' R' \Sigma(R'),

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

            * 'nfw' (default)
            * 'einasto' - not in cluster_toolkit
            * 'hernquist' - not in cluster_toolkit

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * 'mean' (default)
            * 'critical'
            * 'virial'

    alpha_ein : float, optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope. Option only
        available for the NumCosmo backend
    verbose : boolean, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only available for the NC and CCL
        backends.
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.


    Returns
    -------
    sigma : numpy.ndarray, float
        2D projected surface density in units of :math:`M_\odot\ Mpc^{-2}`

    Notes
    -----
    Need to refactory so we only require arguments that are necessary for all models and use
    another structure to take the arguments necessary for specific models.
    """
    _modeling_object.validate_input = validate_input
    _modeling_object.set_cosmo(cosmo)
    _modeling_object.set_halo_density_profile(
        halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef
    )
    _modeling_object.set_concentration(cdelta)
    _modeling_object.set_mass(mdelta)
    if alpha_ein is not None:
        _modeling_object.set_einasto_alpha(alpha_ein)

    sigma_bar = _modeling_object.eval_mean_surface_density(r_proj, z_cl, verbose=verbose)

    _modeling_object.validate_input = True
    return sigma_bar


def compute_excess_surface_density(
    r_proj,
    mdelta,
    cdelta,
    z_cl,
    cosmo,
    delta_mdef=200,
    halo_profile_model="nfw",
    massdef="mean",
    alpha_ein=None,
    verbose=False,
    validate_input=True,
):
    r"""Computes the excess surface density

    .. math::
        \Delta\Sigma(R) = \bar{\Sigma}(<R)-\Sigma(R),

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

            * 'nfw' (default)
            * 'einasto' - not in cluster_toolkit
            * 'hernquist' - not in cluster_toolkit

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * 'mean' (default)
            * 'critical'
            * 'virial'

    alpha_ein : float, None, optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        If None, use the default value of the backend. (0.25 for the NumCosmo backend and a
        cosmology-dependent value for the CCL backend.)
    verbose : boolean, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only available for the NC and CCL
        backends.
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.

    Returns
    -------
    deltasigma : numpy.ndarray, float
        Excess surface density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    _modeling_object.validate_input = validate_input
    _modeling_object.set_cosmo(cosmo)
    _modeling_object.set_halo_density_profile(
        halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef
    )
    _modeling_object.set_concentration(cdelta)
    _modeling_object.set_mass(mdelta)
    if halo_profile_model == "einasto" or alpha_ein is not None:
        _modeling_object.set_einasto_alpha(alpha_ein)

    deltasigma = _modeling_object.eval_excess_surface_density(r_proj, z_cl, verbose=verbose)

    _modeling_object.validate_input = True
    return deltasigma


def compute_excess_surface_density_2h(
    r_proj,
    z_cl,
    cosmo,
    halobias=1.0,
    logkbounds=(-5, 5),
    ksteps=1000,
    loglbounds=(0, 6),
    lsteps=500,
    validate_input=True,
):
    r"""Computes the 2-halo term excess surface density from eq.(13) of Oguri & Hamana (2011)

    .. math::
        \Delta\Sigma_{\text{2h}}(R) = \frac{\rho_m(z)b(M)}{(1 + z)^3D_A(z)^2}
        \int\frac{ldl}{(2\pi)} P_{\text{mm}}(k_l, z)J_2(l\theta)

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
    logkbounds : tuple(float,float), shape(2,), optional
        Log10 of the upper and lower bounds for the linear matter power spectrum
    ksteps : int, optional
        Number of steps in k-space
    loglbounds : tuple(float,float), shape(2,), optional
        Log10 of the upper and lower bounds for numerical integration
    lsteps : int, optional
        Steps for the numerical integration
    validate_input: bool
        Validade each input argument

    Returns
    -------
    deltasigma_2h : numpy.ndarray, float
        2-halo term excess surface density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    _modeling_object.validate_input = validate_input
    _modeling_object.set_cosmo(cosmo)

    deltasigma_2h = _modeling_object.eval_excess_surface_density_2h(
        r_proj,
        z_cl,
        halobias=halobias,
        logkbounds=logkbounds,
        ksteps=ksteps,
        loglbounds=loglbounds,
        lsteps=lsteps,
    )

    _modeling_object.validate_input = True
    return deltasigma_2h


def compute_surface_density_2h(
    r_proj,
    z_cl,
    cosmo,
    halobias=1,
    logkbounds=(-5, 5),
    ksteps=1000,
    loglbounds=(0, 6),
    lsteps=500,
    validate_input=True,
):
    r"""Computes the 2-halo term surface density from eq.(13) of Oguri & Hamana (2011)

    .. math::
        \Sigma_{\text{2h}}(R) = \frac{\rho_\text{m}(z)b(M)}{(1 + z)^3D_A(z)^2}
        \int\frac{ldl}{(2\pi)}P_{\text{mm}}(k_l, z)J_0(l\theta)

    where

    .. math::
        k_l = \frac{l}{D_A(z)(1 + z)}

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
    logkbounds : tuple(float,float), shape(2,), optional
        Log10 of the upper and lower bounds for the linear matter power spectrum
    ksteps : int, optional
        Number of steps in k-space
    loglbounds : tuple(float,float), shape(2,), optional
        Log10 of the upper and lower bounds for numerical integration
    lsteps : int, optional
        Steps for the numerical integration
    validate_input: bool
        Validade each input argument

    Returns
    -------
    sigma_2h : numpy.ndarray, float
        2-halo term surface density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    _modeling_object.validate_input = validate_input
    _modeling_object.set_cosmo(cosmo)

    sigma_2h = _modeling_object.eval_surface_density_2h(
        r_proj,
        z_cl,
        halobias=halobias,
        logkbounds=logkbounds,
        ksteps=ksteps,
        loglbounds=loglbounds,
        lsteps=lsteps,
    )

    _modeling_object.validate_input = True
    return sigma_2h


def compute_critical_surface_density_eff(cosmo, z_cluster, pzbins, pzpdf, validate_input=True):
    r"""Computes the 'effective critical surface density'

    .. math::
        \langle \Sigma_{\text{crit}}^{-1}\rangle^{-1} =
        \left(\int \frac{1}{\Sigma_{\text{crit}}(z)}p(z) \mathrm{d}z\right)^{-1}

    where :math:`p(z)` is the source photoz probability density function.
    This comes from the maximum likelihood estimator for evaluating a :math:`\Delta\Sigma`
    profile.

    For the standard :math:`\Sigma_{\text{crit}}(z)` definition, use the `eval_sigma_crit` method of
    the CLMM cosmology object.

    Parameters
    ----------
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    z_cluster : float
        Galaxy cluster redshift
    pzbins : array-like
        Bins where the source redshift pdf is defined
    pzpdf : array-like
        Values of the source redshift pdf
    validate_input: bool
        Validade each input argument


    Returns
    -------
    sigma_c : numpy.ndarray, float
        Cosmology-dependent (effective) critical surface density in units of
        :math:`M_\odot\ Mpc^{-2}`
    """

    _modeling_object.validate_input = validate_input
    _modeling_object.set_cosmo(cosmo)
    sigma_c = _modeling_object.eval_critical_surface_density_eff(z_cluster, pzbins, pzpdf)

    _modeling_object.validate_input = True
    return sigma_c


def compute_tangential_shear(
    r_proj,
    mdelta,
    cdelta,
    z_cluster,
    z_src,
    cosmo,
    delta_mdef=200,
    halo_profile_model="nfw",
    massdef="mean",
    alpha_ein=None,
    z_src_info="discrete",
    verbose=False,
    validate_input=True,
):
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
    z_src : array_like, float, function
        Information on the background source galaxy redshift(s). Value required depends on
        `z_src_info` (see below).
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization (letter case independent):

            * 'nfw' (default)
            * 'einasto' - not in cluster_toolkit
            * 'hernquist' - not in cluster_toolkit

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * 'mean' (default)
            * 'critical'
            * 'virial'

    alpha_ein : float, None, optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        If None, use the default value of the backend. (0.25 for the NumCosmo backend and a
        cosmology-dependent value for the CCL backend.)
    z_src_info : str, optional
        Type of redshift information provided by the `z_src` argument.
        The following supported options are:

            * 'discrete' (default) : The redshift of sources is provided by `z_src`.
              It can be individual redshifts for each source galaxy when `z_src` is an array
              or all sources are at the same redshift when `z_src` is a float.

            * 'distribution' : A redshift distribution function is provided by `z_src`.
              `z_src` must be a one dimensional function.

            * 'beta' : The averaged lensing efficiency is provided by `z_src`.
              `z_src` must be a tuple containing
              ( :math:`\langle \beta_s \rangle, \langle \beta_s^2 \rangle`),
              the lensing efficiency and square of the lensing efficiency averaged over
              the galaxy redshift distribution repectively.

                .. math::
                    \langle \beta_s \rangle = \left\langle \frac{D_{LS}}{D_S}\frac{D_\infty}
                    {D_{L,\infty}}\right\rangle

                .. math::
                    \langle \beta_s^2 \rangle = \left\langle \left(\frac{D_{LS}}
                    {D_S}\frac{D_\infty}{D_{L,\infty}}\right)^2 \right\rangle

    verbose : bool, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and CCL
        backends.
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.

    Returns
    -------
    gammat : numpy.ndarray, float
        Tangential shear
    """

    _modeling_object.validate_input = validate_input
    _modeling_object.set_cosmo(cosmo)
    _modeling_object.set_halo_density_profile(
        halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef
    )
    _modeling_object.set_concentration(cdelta)
    _modeling_object.set_mass(mdelta)
    if halo_profile_model == "einasto" or alpha_ein is not None:
        _modeling_object.set_einasto_alpha(alpha_ein)
    if np.min(r_proj) < 1.0e-11:
        raise ValueError(
            f"Rmin = {np.min(r_proj):.2e} Mpc/h! This value is too small "
            "and may cause computational issues."
        )

    tangential_shear = _modeling_object.eval_tangential_shear(
        r_proj,
        z_cluster,
        z_src,
        z_src_info=z_src_info,
        verbose=verbose,
    )

    _modeling_object.validate_input = True
    return tangential_shear


def compute_convergence(
    r_proj,
    mdelta,
    cdelta,
    z_cluster,
    z_src,
    cosmo,
    delta_mdef=200,
    halo_profile_model="nfw",
    massdef="mean",
    alpha_ein=None,
    z_src_info="discrete",
    verbose=False,
    validate_input=True,
):
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
    z_src : array_like, float, function
        Information on the background source galaxy redshift(s). Value required depends on
        `z_src_info` (see below).
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization (letter case independent):

            * 'nfw' (default)
            * 'einasto' - not in cluster_toolkit
            * 'hernquist' - not in cluster_toolkit

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * 'mean' (default)
            * 'critical'
            * 'virial'

    alpha_ein : float, None, optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        If None, use the default value of the backend. (0.25 for the NumCosmo backend and a
        cosmology-dependent value for the CCL backend.)
    z_src_info : str, optional
        Type of redshift information provided by the `z_src` argument.
        The following supported options are:

            * 'discrete' (default) : The redshift of sources is provided by `z_src`.
              It can be individual redshifts for each source galaxy when `z_src` is an array
              or all sources are at the same redshift when `z_src` is a float.

            * 'distribution' : A redshift distribution function is provided by `z_src`.
              `z_src` must be a one dimensional function.

            * 'beta' : The averaged lensing efficiency is provided by `z_src`.
              `z_src` must be a tuple containing
              ( :math:`\langle \beta_s \rangle, \langle \beta_s^2 \rangle`),
              the lensing efficiency and square of the lensing efficiency averaged over
              the galaxy redshift distribution repectively.

                .. math::
                    \langle \beta_s \rangle = \left\langle \frac{D_{LS}}{D_S}\frac{D_\infty}
                    {D_{L,\infty}}\right\rangle

                .. math::
                    \langle \beta_s^2 \rangle = \left\langle \left(\frac{D_{LS}}
                    {D_S}\frac{D_\infty}{D_{L,\infty}}\right)^2 \right\rangle

    verbose : bool, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and CCL
        backends.
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.

    Returns
    -------
    kappa : numpy.ndarray, float
        Mass convergence, kappa.

    """

    _modeling_object.validate_input = validate_input
    _modeling_object.set_cosmo(cosmo)
    _modeling_object.set_halo_density_profile(
        halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef
    )
    _modeling_object.set_concentration(cdelta)
    _modeling_object.set_mass(mdelta)
    if halo_profile_model == "einasto" or alpha_ein is not None:
        _modeling_object.set_einasto_alpha(alpha_ein)

    convergence = _modeling_object.eval_convergence(
        r_proj,
        z_cluster,
        z_src,
        z_src_info=z_src_info,
        verbose=verbose,
    )

    _modeling_object.validate_input = True
    return convergence


def compute_reduced_tangential_shear(
    r_proj,
    mdelta,
    cdelta,
    z_cluster,
    z_src,
    cosmo,
    delta_mdef=200,
    halo_profile_model="nfw",
    massdef="mean",
    z_src_info="discrete",
    approx=None,
    integ_kwargs=None,
    alpha_ein=None,
    validate_input=True,
    verbose=False,
):
    r"""Computes the reduced tangential shear

    .. math::
        g_t = \frac{\gamma_t}{1-\kappa}

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
    z_src : array_like, float, function
        Information on the background source galaxy redshift(s). Value required depends on
        `z_src_info` (see below).
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization (letter case independent):

            * 'nfw' (default)
            * 'einasto' - not in cluster_toolkit
            * 'hernquist' - not in cluster_toolkit

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * 'mean' (default);
            * 'critical' - not in cluster_toolkit;
            * 'virial' - not in cluster_toolkit;

    alpha_ein : float, None, optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        If None, use the default value of the backend. (0.25 for the NumCosmo backend and a
        cosmology-dependent value for the CCL backend.)
    z_src_info : str, optional
        Type of redshift information provided by the `z_src` argument.
        The following supported options are:

            * 'discrete' (default) : The redshift of sources is provided by `z_src`.
              It can be individual redshifts for each source galaxy when `z_src` is an array
              or all sources are at the same redshift when `z_src` is a float
              (Used for `approx=None`).

            * 'distribution' : A redshift distribution function is provided by `z_src`.
              `z_src` must be a one dimensional function (Used when `approx=None`).

            * 'beta' : The averaged lensing efficiency is provided by `z_src`.
              `z_src` must be a tuple containing
              ( :math:`\langle \beta_s \rangle, \langle \beta_s^2 \rangle`),
              the lensing efficiency and square of the lensing efficiency averaged over
              the galaxy redshift distribution repectively.

                .. math::
                    \langle \beta_s \rangle = \left\langle \frac{D_{LS}}{D_S}\frac{D_\infty}
                    {D_{L,\infty}}\right\rangle

                .. math::
                    \langle \beta_s^2 \rangle = \left\langle \left(\frac{D_{LS}}
                    {D_S}\frac{D_\infty}{D_{L,\infty}}\right)^2 \right\rangle

    approx : str, optional
        Type of computation to be made for reduced tangential shears, options are:

            * None (default): Requires `z_src_info` to be 'discrete' or 'distribution'.
              If `z_src_info='discrete'`, full computation is made for each
              `r_proj, z_src` pair individually. If `z_src_info='distribution'`, reduced
              tangential shear at each value of `r_proj` is calculated as

              .. math::
                  g_t
                  =\left<\frac{\beta_s\gamma_{\infty}}{1-\beta_s\kappa_{\infty}}\right>
                  =\frac{\int_{z_{min}}^{z_{max}}\frac{\beta_s(z)\gamma_{\infty}}
                  {1-\beta_s(z)\kappa_{\infty}}N(z)\text{d}z}
                  {\int_{z_{min}}^{z_{max}} N(z)\text{d}z}

            * 'order1' : Same approach as in Weighing the Giants - III (equation 6 in
              Applegate et al. 2014; https://arxiv.org/abs/1208.0605).
              `z_src_info` must be 'beta':

              .. math::
                  g_t\approx\frac{\left<\beta_s\right>\gamma_{\infty}}
                  {1-\left<\beta_s\right>\kappa_{\infty}}

            * 'order2' : Same approach as in Cluster Mass Calibration at High
              Redshift (equation 12 in Schrabback et al. 2017;
              https://arxiv.org/abs/1611.03866).
              `z_src_info` must be 'beta':

              .. math::
                  g_t\approx\frac{\left<\beta_s\right>\gamma_{\infty}}
                  {1-\left<\beta_s\right>\kappa_{\infty}}
                  \left(1+\left(\frac{\left<\beta_s^2\right>}
                  {\left<\beta_s\right>^2}-1\right)\left<\beta_s\right>\kappa_{\infty}\right)

    integ_kwargs: None, dict
        Extra arguments for the redshift integration (when
        `approx=None, z_src_info='distribution'`). Possible keys are:

            * 'zmin' (None, float) : Minimum redshift to be set as the source of the galaxy
              when performing the sum. (default=None)
            * 'zmax' (float) : Maximum redshift to be set as the source of the galaxy
              when performing the sum. (default=10.0)
            * 'delta_z_cut' (float) : Redshift cut so that `zmin` = `z_cl` + `delta_z_cut`.
              `delta_z_cut` is ignored if `z_min` is already provided. (default=0.1)

    alpha_ein : float, None, optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        If None, use the default value of the backend. (0.25 for the NumCosmo backend and a
        cosmology-dependent value for the CCL backend.)
    verbose : bool, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and CCL
        backends.
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.

    Returns
    -------
    gt : numpy.ndarray, float
        Reduced tangential shear

    """
    _modeling_object.validate_input = validate_input
    _modeling_object.set_cosmo(cosmo)
    _modeling_object.set_halo_density_profile(
        halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef
    )
    _modeling_object.set_concentration(cdelta)
    _modeling_object.set_mass(mdelta)
    if halo_profile_model == "einasto" or alpha_ein is not None:
        _modeling_object.set_einasto_alpha(alpha_ein)

    red_tangential_shear = _modeling_object.eval_reduced_tangential_shear(
        r_proj,
        z_cluster,
        z_src,
        z_src_info=z_src_info,
        approx=approx,
        integ_kwargs=integ_kwargs,
        verbose=verbose,
    )

    _modeling_object.validate_input = True
    return red_tangential_shear


def compute_magnification(
    r_proj,
    mdelta,
    cdelta,
    z_cluster,
    z_src,
    cosmo,
    delta_mdef=200,
    halo_profile_model="nfw",
    massdef="mean",
    alpha_ein=None,
    z_src_info="discrete",
    approx=None,
    integ_kwargs=None,
    verbose=False,
    validate_input=True,
):
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
    z_src : array_like, float, function
        Information on the background source galaxy redshift(s). Value required depends on
        `z_src_info` (see below).
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_model : str, optional
        Profile model parameterization (letter case independent):

            * 'nfw' (default)
            * 'einasto' - not in cluster_toolkit
            * 'hernquist' - not in cluster_toolkit

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * 'mean' (default)
            * 'critical'
            * 'virial'

    alpha_ein : float, None, optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        If None, use the default value of the backend. (0.25 for the NumCosmo backend and a
        cosmology-dependent value for the CCL backend.)
    z_src_info : str, optional
        Type of redshift information provided by the `z_src` argument.
        The following supported options are:

            * 'discrete' (default) : The redshift of sources is provided by `z_src`.
              It can be individual redshifts for each source galaxy when `z_src` is an array
              or all sources are at the same redshift when `z_src` is a float
              (Used for `approx=None`).

            * 'distribution' : A redshift distribution function is provided by `z_src`.
              `z_src` must be a one dimensional function (Used when `approx=None`).

            * 'beta' : The averaged lensing efficiency is provided by `z_src`.
              `z_src` must be a tuple containing
              ( :math:`\langle \beta_s \rangle, \langle \beta_s^2 \rangle`),
              the lensing efficiency and square of the lensing efficiency averaged over
              the galaxy redshift distribution repectively.

                .. math::
                    \langle \beta_s \rangle = \left\langle \frac{D_{LS}}{D_S}\frac{D_\infty}
                    {D_{L,\infty}}\right\rangle

                .. math::
                    \langle \beta_s^2 \rangle = \left\langle \left(\frac{D_{LS}}
                    {D_S}\frac{D_\infty}{D_{L,\infty}}\right)^2 \right\rangle

    approx : str, optional
        Type of computation to be made for magnifications, options are:

            * None (default): Requires `z_src_info` to be 'discrete' or 'distribution'.
              If `z_src_info='discrete'`, full computation is made for each
              `r_proj, z_src` pair individually. If `z_src_info='distribution'`, magnification
              at each value of `r_proj` is calculated as

              .. math::
                  \mu
                  =\left<\frac{1}{\left(1-\beta_s\kappa_{\infty}\right)^2
                  -\left(\beta_s\gamma_{\infty}\right)^2}\right>
                  =\frac{\int_{z_{min}}^{z_{max}}\frac{N(z)\text{d}z}
                  {\left(1-\beta_s(z)\kappa_{\infty}\right)^2
                  -\left(\beta_s(z)\gamma_{\infty}\right)^2}}
                  {\int_{z_{min}}^{z_{max}} N(z)\text{d}z}

            * 'order1' : Uses the weak lensing approximation of the magnification with up to
              first-order terms in :math:`\kappa_{\infty}` or :math:`\gamma_{\infty}`
              (`z_src_info` must be 'beta'):

              .. math::
                  \mu \approx 1 + 2 \left<\beta_s\right>\kappa_{\infty}

            * 'order2' : Uses the weak lensing approximation of the magnification with up to
              second-order terms in :math:`\kappa_{\infty}` or :math:`\gamma_{\infty}`
              (`z_src_info` must be 'beta'):

              .. math::
                  \mu \approx 1 + 2 \left<\beta_s\right>\kappa_{\infty}
                  + 3 \left<\beta_s^2\right>\kappa_{\infty}^2
                  + \left<\beta_s^2\right>\gamma_{\infty}^2

    integ_kwargs: None, dict
        Extra arguments for the redshift integration (when
        `approx=None, z_src_info='distribution'`). Possible keys are:

            * 'zmin' (None, float) : Minimum redshift to be set as the source of the galaxy
              when performing the sum. (default=None)
            * 'zmax' (float) : Maximum redshift to be set as the source of the galaxy
              when performing the sum. (default=10.0)
            * 'delta_z_cut' (float) : Redshift cut so that `zmin` = `z_cl` + `delta_z_cut`.
              `delta_z_cut` is ignored if `z_min` is already provided. (default=0.1)

    verbose : bool, optional
        If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and CCL
        backends.
    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.

    Returns
    -------
    magnification : numpy.ndarray, float
        Magnification :math:`\mu`.

    """

    _modeling_object.validate_input = validate_input
    _modeling_object.set_cosmo(cosmo)
    _modeling_object.set_halo_density_profile(
        halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef
    )
    _modeling_object.set_concentration(cdelta)
    _modeling_object.set_mass(mdelta)
    if halo_profile_model == "einasto" or alpha_ein is not None:
        _modeling_object.set_einasto_alpha(alpha_ein)

    magnification = _modeling_object.eval_magnification(
        r_proj,
        z_cluster,
        z_src,
        z_src_info=z_src_info,
        approx=approx,
        integ_kwargs=integ_kwargs,
        verbose=verbose,
    )

    _modeling_object.validate_input = True
    return magnification


def compute_magnification_bias(
    r_proj,
    alpha,
    mdelta,
    cdelta,
    z_cluster,
    z_src,
    cosmo,
    delta_mdef=200,
    halo_profile_model="nfw",
    massdef="mean",
    alpha_ein=None,
    z_src_info="discrete",
    approx=None,
    integ_kwargs=None,
    verbose=False,
    validate_input=True,
):
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
    z_src : array_like, float, function
        Information on the background source galaxy redshift(s). Value required depends on
        `z_src_info` (see below).
    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    delta_mdef : int, optional
        Mass overdensity definition.  Defaults to 200.
    alpha_ein : float, None, optional
        If `halo_profile_model=='einasto'`, set the value of the Einasto slope.
        Option only available for the NumCosmo and CCL backends.
        If None, use the default value of the backend. (0.25 for the NumCosmo backend and a
        cosmology-dependent value for the CCL backend.)
    halo_profile_model : str, optional
        Profile model parameterization (letter case independent):

            * 'nfw' (default)
            * 'einasto' - not in cluster_toolkit
            * 'hernquist' - not in cluster_toolkit

    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):

            * 'mean' (default)
            * 'critical'
            * 'virial'

    z_src_info : str, optional
        Type of redshift information provided by the `z_src` argument.
        The following supported options are:

            * 'discrete' (default) : The redshift of sources is provided by `z_src`.
              It can be individual redshifts for each source galaxy when `z_src` is an array
              or all sources are at the same redshift when `z_src` is a float
              (Used for `approx=None`).

            * 'distribution' : A redshift distribution function is provided by `z_src`.
              `z_src` must be a one dimensional function (Used when `approx=None`).

            * 'beta' : The averaged lensing efficiency is provided by `z_src`.
              `z_src` must be a tuple containing
              ( :math:`\langle \beta_s \rangle, \langle \beta_s^2 \rangle`),
              the lensing efficiency and square of the lensing efficiency averaged over
              the galaxy redshift distribution repectively.

                .. math::
                    \langle \beta_s \rangle = \left\langle \frac{D_{LS}}{D_S}\frac{D_\infty}
                    {D_{L,\infty}}\right\rangle

                .. math::
                    \langle \beta_s^2 \rangle = \left\langle \left(\frac{D_{LS}}
                    {D_S}\frac{D_\infty}{D_{L,\infty}}\right)^2 \right\rangle

    approx : str, optional
        Type of computation to be made for magnification biases, options are:

            * None (default): Requires `z_src_info` to be 'discrete' or 'distribution'.
              If `z_src_info='discrete'`, full computation is made for each
              `r_proj, z_src` pair individually. If `z_src_info='distribution'`, magnification
              bias at each value of `r_proj` is calculated as

              .. math::
                  \mu^{\alpha-1}
                  &=\left(\left<\frac{1}{\left(1-\beta_s\kappa_{\infty}\right)^2
                  -\left(\beta_s\gamma_{\infty}\right)^2}\right>\right)^{\alpha-1}
                  \\\\
                  &=\frac{\int_{z_{min}}^{z_{max}}\frac{N(z)\text{d}z}
                  {\left(\left(1-\beta_s(z)\kappa_{\infty}\right)^2
                  -\left(\beta_s(z)\gamma_{\infty}\right)^2\right)^{\alpha-1}}}
                  {\int_{z_{min}}^{z_{max}} N(z)\text{d}z}

            * 'order1' : Uses the weak lensing approximation of the magnification bias with up
              to first-order terms in :math:`\kappa_{\infty}` or :math:`\gamma_{\infty}`
              (`z_src_info` must be 'beta'):

              .. math::
                  \mu^{\alpha-1} \approx
                  1 + \left(\alpha-1\right)\left(2 \left<\beta_s\right>\kappa_{\infty}\right)

            * 'order2' : Uses the weak lensing approximation of the magnification bias with up
              to second-order terms in :math:`\kappa_{\infty}` or :math:`\gamma_{\infty}`
              `z_src_info` must be 'beta':

              .. math::
                  \mu^{\alpha-1} \approx
                  1 &+ \left(\alpha-1\right)\left(2 \left<\beta_s\right>\kappa_{\infty}\right)
                  \\\\
                  &+ \left(\alpha-1\right)\left(\left<\beta_s^2\right>\gamma_{\infty}^2\right)
                  \\\\
                  &+ \left(2\alpha-1\right)\left(\alpha-1\right)
                  \left(\left<\beta_s^2\right>\kappa_{\infty}^2\right)

    integ_kwargs: None, dict
        Extra arguments for the redshift integration (when
        `approx=None, z_src_info='distribution'`). Possible keys are:

            * 'zmin' (None, float) : Minimum redshift to be set as the source of the galaxy
              when performing the sum. (default=None)
            * 'zmax' (float) : Maximum redshift to be set as the source of the galaxy
              when performing the sum. (default=10.0)
            * 'delta_z_cut' (float) : Redshift cut so that `zmin` = `z_cl` + `delta_z_cut`.
              `delta_z_cut` is ignored if `z_min` is already provided. (default=0.1)

    validate_input : bool, optional
        If True (default), the types of the arguments are checked before proceeding.

    Returns
    -------
    magnification_bias : numpy.ndarray
        magnification bias
    """

    _modeling_object.validate_input = validate_input
    _modeling_object.set_cosmo(cosmo)
    _modeling_object.set_halo_density_profile(
        halo_profile_model=halo_profile_model, massdef=massdef, delta_mdef=delta_mdef
    )
    _modeling_object.set_concentration(cdelta)
    _modeling_object.set_mass(mdelta)
    if halo_profile_model == "einasto" or alpha_ein is not None:
        _modeling_object.set_einasto_alpha(alpha_ein)

    magnification_bias = _modeling_object.eval_magnification_bias(
        r_proj,
        z_cluster,
        z_src,
        alpha,
        z_src_info=z_src_info,
        approx=approx,
        integ_kwargs=integ_kwargs,
        verbose=verbose,
    )

    _modeling_object.validate_input = True
    return magnification_bias
