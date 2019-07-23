"""@file.py modeling.py
Functions for theoretical models.  Default is NFW.
"""

import astropy
from astropy import constants, cosmology, units
import cluster_toolkit as ct
import numpy as np

def _cclify_astropy_cosmo(apy_cosmo) :
    '''
    Generates a ccl-looking cosmology object (with all values needed for modeling) from an astropy cosmology object.

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
    cosmo_ccl = create_ccl_cosmo_object_from_astropy(astropy_cosmology_object)``
    '''
    if type(apy_cosmo) == astropy.cosmology.core.FlatLambdaCDM:
        ccl_cosmo = {'Omega_c': apy_cosmo.Odm0,
                 'Omega_b': apy_cosmo.Ob0,
                 'h': apy_cosmo.h,
                 'H0': apy_cosmo.H0.value}
    else:
        ccl_cosmo = apy_cosmo
    return ccl_cosmo

def _get_a_from_z(z):
    '''
    Convert redshift to scale factor

    Parameters
    ----------
    z : array-like, float
        redshift

    Returns
    -------
    a : array-like, float
        scale factor
    '''
    a = 1. / (1. + z)
    return a

def _get_z_from_a(a):
    '''
    Convert scale factor to redshift

    Parameters
    ----------
    a : array-like, float
        scale factor

    Returns
    -------
    z : array-like, float
        redshift
    '''
    z = 1. / a - 1.
    return z

def get_3d_density(r3d, mdelta, cdelta, cosmo, Delta=200, halo_profile_parameterization='nfw'):
    '''
    Retrieve the 3d density $\rho(r)$

    Parameters
    ----------
    r3d : array-like, float
        The radial positions in Mpc/h.
    mdelta : float
        Galaxy cluster mass in Msun/h.
    cdelta : float
        Galaxy cluster NFW concentration.
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    Delta : int, optional
        Mass overdensity definition; defaults to 200.
    halo_profile_parameterization : str, optional
        Profile model parameterization, with the following supported options:
        `nfw` (default) - $\rho(r)=\frac{\rho_0}{c/(r/R_{vir})(1+c/(r/R_{vir}))^2}$ [insert citation here]
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.

    Returns
    -------
    rho : array-like, float
        3-dimensional mass density

    Notes
    -----
    AIM: We should only require arguments that are necessary for all profiles and use another structure to take the arguments necessary for specific models
    '''
    cosmo = _cclify_astropy_cosmo(cosmo)
    Omega_m = cosmo['Omega_c'] + cosmo['Omega_b']

    if halo_profile_parameterization == 'nfw':
        rho = ct.density.rho_nfw_at_r(r3d, mdelta, cdelta, Omega_m, delta=Delta)
        return rho
    else:
        pass

def predict_surface_density(r_proj, mdelta, cdelta, cosmo, Delta=200, halo_profile_parameterization='nfw'):
    '''
    Computes the surface mass density $\Sigma(R) = \Omega_m \rho_{crit} \int^\inf_{-\inf} dz \Xi_{hm} (\sqrt{R^2+z^2})$, where $\Xi_{hm}$ is the halo mass function.

    Parameters
    ----------
    r_proj : array-like
        The projected radial positions in Mpc/h.
    mdelta : float
        Galaxy cluster mass in Msun/h.
    cdelta : float
        Galaxy cluster NFW concentration.
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    Delta : int, optional
        Mass overdensity definition; defaults to 200.
    halo_profile_parameterization : str, optional
        Profile model parameterization, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.

    Returns
    -------
    sigma : array-like, float
        Surface density, Sigma in units of [h M_\\odot/$pc^2$]

    Notes
    -----
    AIM: We should only require arguments that are necessary for all models and use another structure to take the arguments necessary for specific models.
    '''
    cosmo = _cclify_astropy_cosmo(cosmo)
    Omega_m = cosmo['Omega_c'] + cosmo['Omega_b']

    if halo_profile_parameterization == 'nfw':
        sigma = ct.deltasigma.Sigma_nfw_at_R(r_proj, mdelta, cdelta, Omega_m, delta=Delta)
        return sigma
    else:
        #return ct.Sigma_at_R(r_proj, mdelta, cdelta, cosmo.Omegam, delta=Delta)
        pass

def predict_excess_surface_density(r_proj, mdelta, cdelta, cosmo, Delta=200, halo_profile_parameterization='nfw'):
    '''
    Computes the excess surface density $\Delta\Sigma(R) = \bar{\Sigma}(<R)-\Sigma(R)$, where $\bar{\Sigma}(<R) = \frac{2}{R^2} \int^R_0 dR' R' \Sigma(R')$

    Parameters
    ----------
    r_proj : array-like
        The projected radial positions in Mpc/h.
    mdelta : float
        Galaxy cluster mass in Msun/h.
    cdelta : float
        Galaxy cluster NFW concentration.
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    Delta : int, optional
        Mass overdensity definition; defaults to 200.
    halo_profile_parameterization : str, optional
        Profile model parameterization, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.

    Returns
    -------
    deltasigma : array-like, float
        Excess surface density, DeltaSigma in units of [h M_\\odot/$pc^2$].
    '''
    cosmo = _cclify_astropy_cosmo(cosmo)
    Omega_m = cosmo['Omega_c'] + cosmo['Omega_b']

    if halo_profile_parameterization == 'nfw':
        Sigma_r_proj = np.logspace(-3, 4, 1000)
        Sigma = ct.deltasigma.Sigma_nfw_at_R(Sigma_r_proj, mdelta, cdelta, Omega_m, delta=Delta)
        # ^ Note: Let's not use this naming convention when transfering ct to ccl....
        deltasigma = ct.deltasigma.DeltaSigma_at_R(r_proj, Sigma_r_proj, Sigma, mdelta, cdelta, Omega_m, delta=Delta)
        return deltasigma
    else:
        pass

def _get_comoving_angular_distance_a(cosmo, aexp2, aexp1=1.):
    '''
    This is a function to calculate d_LS (angular distance between lens and source) because CCL cannot yet do it.  Temporarily using the astropy implementation.

    Parameters
    ----------
    cosmo : pyccl.core.Cosmology object
            CCL Cosmology object
    aexp2 : float
        smaller scale factor
    aexp1 : float, optional
        larger scale factor; defaults to 1.

    Returns
    -------
    da: float
        angular diameter distance

    Notes
    -----
    This is definitely broken if other cosmological parameter specifications differ, so we'll have to revise this later.
    # AIM: needs a docstring for args
    '''
    z1 = _get_z_from_a(aexp1)
    z2 = _get_z_from_a(aexp2)
    Omega_m = cosmo['Omega_b'] + cosmo['Omega_c']
    if type(cosmo) == astropy.cosmology.core.FlatLambdaCDM:
        ap_cosmo = cosmo
    else:
        ap_cosmo = astropy.cosmology.core.FlatLambdaCDM(H0=cosmo['H0'], Om0=Omega_m, Ob0=cosmo['Omega_b'])
    # astropy angular diameter distance in Mpc
    # need to return in pc/h
    da = ap_cosmo.angular_diameter_distance_z1z2(z1, z2).to_value(units.pc) * ap_cosmo.h
    return da

def get_critical_surface_density(cosmo, z_cluster, z_source):
    '''
    Computes the critical surface density $\Sigma_{crit} = \frac{c^2}{4\pi G} \frac{D_s}{D_LD_{LS}}$

    Parameters
    ----------
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    z_cluster : float
        Galaxy cluster redshift
    z_source : array-like, float
        Background source galaxy redshift(s)

    Returns
    -------
    sigmacrit : float
        Cosmology-dependent critical surface density

    Notes
    -----
    We will need gamma inf and kappa inf for alternative z_src_models using Beta_s
    '''
    # m_to_pc = 3.2408e-17
    # kg_to_msun = 5.0279e-31
    # if type(cosmo) == pyccl.core.Cosmology:
    #     c = ccl.physical_constants.CLIGHT * m_to_pc
    #     G = ccl.physical_constants.GNEWT * (m_to_pc)**3 / (kg_to_msun)
    # else:

    c = constants.c.to(units.pc/units.s).value
    G = constants.G.to(units.pc**3/units.M_sun/units.s**2).value
    
    aexp_cluster = _get_a_from_z(z_cluster)
    aexp_src = _get_a_from_z(z_source)

    d_l = _get_comoving_angular_distance_a(cosmo, aexp_cluster)
    d_s = _get_comoving_angular_distance_a(cosmo, aexp_src)
    d_ls = _get_comoving_angular_distance_a(cosmo, aexp_src, aexp_cluster)

    # will need to deal with units: distances in Mpc and some CCL constants in SI
    sigmacrit = d_s / (d_l * d_ls) * c * c / (4 * np.pi * G)
    return sigmacrit

def predict_tangential_shear(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta=200, halo_profile_parameterization='nfw', z_src_model='single_plane'):
    '''
    Computes the tangential shear $\gamma_t = \frac{\Delta\Sigma}{\Sigma_{crit}} = \frac{\bar{\Sigma}-\Sigma}{\Sigma_{crit}}}$, or $\gamma_t = \gamma_\inf \times \Beta_s$

    Parameters
    ----------
    r_proj : array-like
        The projected radial positions in Mpc/h.
    mdelta : float
        Galaxy cluster mass in Msun/h.
    cdelta : float
        Galaxy cluster NFW concentration.
    z_cluster : float
        Galaxy cluster redshift
    z_source : array-like, float
        Background source galaxy redshift(s)
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    Delta : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_parameterization : str, optional
        Profile model parameterization, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.

    Returns
    -------
    gammat : array-like, float
        tangential shear

    Notes
    -----
    We will need gamma inf and kappa inf for alternative z_src_models using Beta_s.
    AIM: Don't we want to raise exceptions rather than errors here?
    '''
    delta_sigma = predict_excess_surface_density(r_proj, mdelta, cdelta, cosmo, Delta=Delta,
                                                   halo_profile_parameterization=halo_profile_parameterization)

    if z_src_model == 'single_plane':
        sigma_c = get_critical_surface_density(cosmo, z_source, z_cluster)
        gammat = delta_sigma / sigma_c
        return gammat
    elif z_src_model == 'known_z_src': # Discrete case
        NotImplementedError('Need to implemnt Beta_s functionality, or average delta_sigma/sigma_c gamma_t = Beta_s*gamma_inf')
    elif z_src_model == 'z_src_distribution' : # Continuous ( from a distribution) case
        NotImplementedError('Need to implement Beta_s calculation from integrating distribution of redshifts in each radial bin')

def predict_convergence(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta=200, halo_profile_parameterization='nfw', z_src_model='single_plane'):
    '''
    Computes the mass convergence $\kappa = \frac{\Sigma}{\Sigma_{crit}}$ or $\kappa = \kappa_\inf \times \Beta_s$

    Parameters
    ----------
    r_proj : array-like
        The projected radial positions in Mpc/h.
    mdelta : float
        Galaxy cluster mass in Msun/h.
    cdelta : float
        Galaxy cluster NFW concentration.
    z_cluster : float
        Galaxy cluster redshift
    z_source : array-like, float
        Background source galaxy redshift(s)
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    Delta : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_parameterization : str, optional
        Profile model parameterization, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.

    Returns
    -------
    kappa : array-like, float
        Mass convergence, kappa.

    Notes
    -----
    AIM: Don't we want to raise exceptions rather than errors here?
    '''
    sigma = predict_surface_density(r_proj, mdelta, cdelta, cosmo, Delta=Delta, halo_profile_parameterization=halo_profile_parameterization)

    if z_src_model == 'single_plane':
        sigma_c = get_critical_surface_density(cosmo, z_cluster, z_source)
        kappa = sigma / sigma_c
        return kappa
    elif z_src_model == 'known_z_src': # Discrete case
        NotImplementedError('Need to implemnt Beta_s functionality, or average sigma/sigma_c kappa_t = Beta_s*kappa_inf')
    elif z_src_model == 'z_src_distribution': # Continuous ( from a distribution) case
        NotImplementedError('Need to implement Beta_s calculation from integrating distribution of redshifts in each radial bin')

def predict_reduced_tangential_shear(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta=200, halo_profile_parameterization='nfw', z_src_model='single_plane'):
    '''
    Computes the reduced tangential shear $g_t = \frac{\\gamma_t}{1-\\kappa}$.

    Parameters
    ----------
    r_proj : array-like
        The projected radial positions in Mpc/h.
    mdelta : float
        Galaxy cluster mass in Msun/h.
    cdelta : float
        Galaxy cluster NFW concentration.
    z_cluster : float
        Galaxy cluster redshift
    z_source : array-like, float
        Background source galaxy redshift(s)
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    Delta : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_parameterization : str, optional
        Profile model parameterization, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.

    Returns
    -------
    gt : array-like, float
        Reduced tangential shear

    Notes
    -----
    AIM: Don't we want to raise exceptions rather than errors here?
    '''
    if z_src_model == 'single_plane':
        kappa = predict_convergence(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta,
                                         halo_profile_parameterization,
                                        z_src_model)
        gamma_t = predict_tangential_shear(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta,
                                         halo_profile_parameterization,
                                        z_src_model)
        gt = gamma_t / (1 - kappa)
        return gt
    elif z_src_model == 'known_z_src': # Discrete case
        NotImplementedError('Need to implemnt Beta_s functionality, or average sigma/sigma_c kappa_t = Beta_s*kappa_inf')
    elif z_src_model == 'z_src_distribution': # Continuous ( from a distribution) case
        NotImplementedError('Need to implement Beta_s and Beta_s2 calculation from integrating distribution of redshifts in each radial bin')
