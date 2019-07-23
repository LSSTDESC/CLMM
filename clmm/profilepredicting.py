"""@file.py profilepredicting.py
Functions to compute profiles from theory.  Default is NFW.
"""
import numpy as np
import pyccl as ccl
import cluster_toolkit as ct

# AIM: standard nomenclature for verbs
# compute for heavy computation
# calculate for straightforward calculations
# get for lookup existing value

def set_omega_m(cosmo):
    '''
    Retrieves matter energy density from cosmology

    Parameters
    ----------
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object

    Returns
    -------
    cosmo : pyccl.core.Cosmology object
        modified CCL Cosmology object
    '''
    cosmo['Omega_m'] = cosmo['Omega_c'] + cosmo['Omega_b']
    return cosmo

def get_a_from_z(z):
    '''
    [write the docstring]
    '''
    a = 1. / (1. + z)
    return a

def get_z_from_a(a):
    '''
    [write the docstring]
    '''
    z = 1. / a - 1.
    return z

def get_3d_density_profile(r3d, mdelta, cdelta, cosmo, Delta=200, halo_profile_parameterization='nfw'):
    '''
    Computes the 3d density profile:
    $\rho(r)$

    e.g. For halo_profile_parameterization='nfw,

    $\rho(r)=\frac{\rho_0}{c/(r/R_{vir})(1+c/(r/R_{vir}))^2}$

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
        `nfw` (default) - [insert citation here]
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.

    Returns
    -------
    rho : array-like, float
        3-dimensional mass density profile

    Notes
    -----
    AIM: We should only require arguments that are necessary for all profiles and use another structure to take the arguments necessary for specific profiles.
    '''
    cosmo = set_omega_m(cosmo)

    if halo_profile_parameterization=='nfw':
        rho = ct.density.rho_nfw_at_r(r3d, mdelta, cdelta, cosmo['Omega_m'] delta=Delta)
        return rho
    else:
        pass

def calculate_surface_density(r_proj, mdelta, cdelta, cosmo, Delta=200, halo_profile_parameterization='nfw'):
    '''
    Computes the surface mass density profile:
    $\Sigma(R) = \Omega_m\rho_{crit}\int^\inf_{-\inf} dz \Xi_{hm}(\sqrt{R^2+z^2})$, where $\Xi_{hm}$ is the halo mass function.

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
    AIM: We should only require arguments that are necessary for all profiles and use another structure to take the arguments necessary for specific profiles.
    '''
    cosmo = set_omega_m(cosmo)

    if halo_profile_parameterization=='nfw':
        sigma = ct.deltasigma.Sigma_nfw_at_R(r_proj, mdelta, cdelta, cosmo['Omega_m'], delta=Delta)
        return sigma
    else:
        #return ct.Sigma_at_R(r_proj, mdelta, cdelta, cosmo.Omegam, delta=Delta)
        pass

def calculate_excess_surface_density(r_proj, mdelta, cdelta, cosmo, Delta=200, halo_profile_parameterization='nfw'):
    '''
    Computes the excess surface density profile:
    $\Delta\Sigma(R) = \bar{\Sigma}(<R)-\Sigma(R)$, where $\bar{\Sigma}(<R)=\frac{2}{R^2}\int^R_0 dR' R'\Sigma(R')$

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
    halo_profile_parameterization :obj:`str`, optional
        Profile model parameterization, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model : `str`, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.

    Returns
    -------
    deltsigma : array-like, float
        Excess surface density, DeltaSigma in units of [h M_\\odot/$pc^2$].
    '''
    cosmo = set_omega_m(cosmo)

    if halo_profile_parameterization == 'nfw':

        Sigma = ct.deltasigma.Sigma_nfw_at_R(r_proj, mdelta, cdelta, cosmo['Omega_m'], delta=Delta)
        # ^ Note: Let's not use this naming convention when transfering ct to ccl....
        deltasigma = ct.deltasigma.DeltaSigma_at_R(r_proj, r_proj, Sigma, mdelta, cdelta, cosmo['Omega_m'], delta=Delta)
        return deltasigma
    else:
        pass

def comoving_angular_distance_aexp1_aexp2(cosmo, aexp1, aexp2):
    '''
    This is a monkey-patched method to calculate d_LS (angular
    distance between lens and source) because CCL does not yet have
    this PR completed.  Temporarily using the astropy implementation.

    # AIM: needs a docstring for args
    '''
    z1 = get_z_from_a(aexp1)
    z2 = get_z_from_a(aexp2)
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    ap_cosmo = FlatLambdaCDM(H0=ccl_cosmo['H_0'], Om0=ccl_cosmo['Omega_m'])
    # astropy angular diameter distance in Mpc
    # need to return in pc/h
    da = ap_cosmo.angular_diameter_distance_z1z2(z1, z2).to_value(u.pc) * cosmo.h
    return da

def comoving_angular_distance(cosmo, aexp):
    '''
    Calculates comoving angular distance to an aexp given a cosmology
    
    Parameters
    ----------
    cosmo : ccl cosmology object
    
    Returns
    -------

    ''' 
    mpc_to_pc = 1e6
    
    try :
        import pyccl as ccl
        return ccl.comoving_angular_distance(cosmo, aexp) * aexp * cosmo['h'] * mpc_to_pc

    except ImportError :
        from astropy.cosmology import FlatLambdaCDM
        from astropy import units as u

        z = get_z_from_a(aexp)
        ap_cosmo = FlatLambdaCDM(H0=cosmo['H_0'], Om0=cosmo['Omega_m'])
        # astropy angular diameter distance in Mpc
        # need to return in pc/h
        da = ap_cosmo.angular_diameter_distance(z).to_value(u.pc) * cosmo.h
        return da

    

def get_critical_surface_density(cosmo, z_cluster, z_source):
    '''
    Computes the critical surface density:
    $\Sigma_{crit} = \frac{c^2}{4\pi G}\frac{D_s}{D_LD_{LS}}$

    Parameters
    ----------
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    z_cluster : float
        Galaxy cluster redshift
    z_source : float
        Background source galaxy redshift

    Returns
    -------
    sigmacrit : float
        Cosmology-dependent critical surface density

    Notes
    -----
    We will need gamma inf and kappa inf for alternative z_src_models using Beta_s
    '''        

    m_to_pc = 3.2408e-17
    kg_to_msun = 5.0279e-31

    try :
        import pyccl as ccl
        c = ccl.physical_constants.CLIGHT * m_to_pc
        G = ccl.physical_constants.GNEWT * (m_to_pc)**3 / (kg_to_msun)
    except ImportError :
        from astropy import constants, units
        c = constants.c.to_value(units.('pc/s'))
        G = constants.G.to_value(units.(....))

        
    aexp_cluster = get_a_from_z(z_cluster)
    aexp_src = get_a_from_z(z_source)
    
    d_l = comoving_angular_distance(cosmo, aexp_cluster)
    d_s = comoving_angular_distance(cosmo, aexp_src)
    d_ls = comoving_angular_distance_aexp1_aexp2(cosmo, aexp_cluster, aexp_src)

    # will need to deal with units: distances in Mpc and some CCL constants in SI
    sigmacrit = d_s / (d_l * d_ls) * c * c / (4 * np.pi * G)
    return sigmacrit

def compute_tangential_shear_profile(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta=200, halo_profile_parameterization='nfw', z_src_model='single_plane'):
    '''
    Computes the tangential shear profile:
    $\gamma_t = \frac{\Delta\Sigma}{\Sigma_{crit}} = \frac{\bar{\Sigma}-\Sigma}{\Sigma_{crit}}}$
    or
    $\gamma_t = \gamma_\inf \times \Beta_s$

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
    z_source : float
        Background source galaxy redshift
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
        tangential shear profile

    Notes
    -----
    We will need gamma inf and kappa inf for alternative z_src_models using Beta_s.
    AIM: Don't we want to raise exceptions rather than errors here?
    '''
    delta_sigma = calculate_excess_surface_density(r_proj, mdelta, cdelta, cosmo, Delta=Delta,
                                                   halo_profile_parameterization=halo_profile_parameterization)

    if z_src_model == 'single_plane':
        sigma_c = get_critical_surface_density(cosmo, z_cluster, z_source)
        gammat = delta_sigma / sigma_c
        return gammat
    elif z_src_model == 'known_z_src': # Discrete case
        NotImplementedError('Need to implemnt Beta_s functionality, or average delta_sigma/sigma_c gamma_t = Beta_s*gamma_inf')
    elif z_src_model == 'z_src_distribution' : # Continuous ( from a distribution) case
        NotImplementedError('Need to implement Beta_s calculation from integrating distribution of redshifts in each radial bin')

def compute_convergence_profile(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta=200, halo_profile_parameterization='nfw', z_src_model='single_plane'):
    '''
    Computes the mass convergence profile:
    $\kappa = \frac{\Sigma}{\Sigma_{crit}}$
    or
    $\kappa = \kappa_\inf \times \Beta_s$

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
    z_source : float
        Background source galaxy redshift
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    Delta : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_parameterization :obj:`str`, optional
        Profile model parameterization that we wish to use, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.

    Returns
    -------
    kappa : array-like, float
        Mass convergence profile, kappa.

    Notes
    -----
    AIM: Don't we want to raise exceptions rather than errors here?
    '''
    sigma = calculate_surface_density(r_proj, mdelta, cdelta, cosmo, Delta=Delta,
                                                      halo_profile_parameterization=halo_profile_parameterization)

    if z_src_model == 'single_plane':
        sigma_c = get_critical_surface_density(cosmo, z_cluster, z_source)
        kappa = sigma / sigma_c
        return kappa
    elif z_src_model == 'known_z_src': # Discrete case
        NotImplementedError('Need to implemnt Beta_s functionality, or average sigma/sigma_c kappa_t = Beta_s*kappa_inf')
    elif z_src_model == 'z_src_distribution' : # Continuous ( from a distribution) case
        NotImplementedError('Need to implement Beta_s calculation from integrating distribution of redshifts in each radial bin')

def compute_reduced_tangential_shear_profile(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta=200, halo_profile_parameterization='nfw', z_src_model='single_plane'):
    '''
    Computes the reduced tangential shear profile:
    $g_t = \frac{\\gamma_t}{1-\\kappa}$.

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
    z_source : float
        Background source galaxy redshift
    cosmo : pyccl.core.Cosmology object
        CCL Cosmology object
    Delta : int, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_parameterization :obj:`str`, optional
        Profile model parameterization that we wish to use, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model : str, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.

    Returns
    -------
    gt : array-like, float
        Reduced tangential shear.

    Notes
    -----
    AIM: Don't we want to raise exceptions rather than errors here?
    '''
    if z_src_model == 'single_plane':
        kappa = compute_convergence_profile(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta,
                                         halo_profile_parameterization,
                                        z_src_model)
        gamma_t = compute_tangential_shear_profile(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta,
                                         halo_profile_parameterization,
                                        z_src_model)
        gt = gamma_t / (1 - kappa)
        return gt
    elif z_src_model == 'known_z_src': # Discrete case
        NotImplementedError('Need to implemnt Beta_s functionality, or average sigma/sigma_c kappa_t = Beta_s*kappa_inf')
    elif z_src_model == 'z_src_distribution' : # Continuous ( from a distribution) case
        NotImplementedError('Need to implement Beta_s and Beta_s2 calculation from integrating distribution of redshifts in each radial bin')

def create_ccl_cosmo_object_from_astropy(astropy_cosmology_object) :
    ''' 
    Generates a ccl looking cosmology object (with all values needed for profilepredicting) 
    from an astropy cosmology object.  THIS IS A MONKEY PATCH NEED TO CHANGE LATER!!!
    
    Example:
    -------
    Need to replace: 
    cosmo_ccl = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)

    with 

    cosmo_ccl = create_ccl_cosmo_object_from_astropy()

    '''
    apy_cosmo = astropy_cosmology_object
    ccl_cosmo = { 'Omega_c':0.27,
                  'Omega_b':0.03,
                  'h':apy_cosmo.h
                  'H0':apy_cosmo.H0.value,
    }

    
    
    return ccl_cosmo

    
