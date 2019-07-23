"""@file.py profilepredicting.py
Functions to compute profiles from theory.  Default is NFW.
"""
import numpy as np
import pyccl as ccl
import cluster_toolkit as ct

# Define CCL cosmology object
cosmo_ccl = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)

# Select density profile and profile parametrization options 
density_profile_parametrization = 'nfw'
mass_Delta = 200
cluster_mass = 1.e15
cluster_concentration = 4

def get_3d_density_profile(r3d, mdelta, cdelta, cosmo, Delta=200, halo_profile_parameterization='nfw'):
    '''
    Computes the 3d density profile:
    $\rho(r)$

    e.g. For halo_profile_parameterization='nfw, 

    $\rho(r)=\frac{\rho_0}{c/(r/R_{vir})(1+c/(r/R_{vir}))^2}$
    
    Parameters
    ----------
    r3d : array-like
        The radial positions in Mpc/h.
    mdelta : float
        Galaxy cluster mass in Msun/h.
    cdelta : float
        Galaxy cluster NFW concentration.
    cosmo : object
        ccl cosmology object
    Delta : :obj:`int`, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_parameterization :obj:`str`, optional
        Profile model parameterization that we wish to use, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model :obj:`str`, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.


    Returns
    -------
    array-like
        3-dimensional mass density profile 

    '''

    Om_m = cosmo['Omega_c']+cosmo['Omega_b']

    if halo_profile_parameterization=='nfw':
        return ct.density.rho_nfw_at_r(r3d, mdelta, cdelta, Om_m, delta=Delta)
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
    cosmo : object
        ccl cosmology object
    Delta : :obj:`int`, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_parameterization :obj:`str`, optional
        Profile model parameterization that we wish to use, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model :obj:`str`, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.


    Returns
    -------
    array-like
        Excess surface density, DeltaSigma in units of [h M_\\odot/$pc^2$]


    '''

    Om_m = cosmo['Omega_c']+cosmo['Omega_b']
    
    if halo_profile_parameterization=='nfw':
        return ct.deltasigma.Sigma_nfw_at_R(r_proj, mdelta, cdelta, Om_m, delta=Delta)
    else:
        #return ct.Sigma_at_R(r_proj, mdelta, cdelta, cosmo.Omegam, delta=Delta)
        pass

def calculate_excess_surface_density(r_proj, mdelta, cdelta, cosmo, Delta=200, 
                                     halo_profile_parameterization='nfw'):
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
    cosmo : object
        ccl cosmology object
    Delta : :obj:`int`, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_parameterization :obj:`str`, optional
        Profile model parameterization that we wish to use, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model :obj:`str`, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.


    Returns
    -------
    array-like
        Excess surface density, DeltaSigma in units of [h M_\\odot/$pc^2$].


    '''

    Om_m = cosmo['Omega_c']+cosmo['Omega_b']
    
    if halo_profile_parameterization == 'nfw' :
        Sigma_r_proj = np.logspace(-3,4,1000)
        Sigma = ct.deltasigma.Sigma_nfw_at_R(Sigma_r_proj, mdelta, cdelta, Om_m, delta=Delta)
        # ^ Note: Let's not use this naming convention when transfering ct to ccl.... 
        return ct.deltasigma.DeltaSigma_at_R(r_proj, Sigma_r_proj, Sigma, mdelta, cdelta, Om_m, delta=Delta)
    else :
        pass


def _comoving_angular_distance_aexp1_aexp2(cosmo, aexp1, aexp2):
    '''

    This is a monkey-patched method to calculate d_LS (angular
    distance between lens and source) because CCL does not yet have
    this PR completed.  Temporarily using the astropy implementation.

    '''
    z1 = 1./aexp1 - 1.
    z2 = 1./aexp2 - 1. 
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    # astropy angular diameter distance in Mpc
    # need to return in pc/h
    return cosmo.angular_diameter_distance_z1z2(1, 2).to_value(u.pc) * cosmo.h
    
    

def get_critical_surface_density(cosmo, z_cluster, z_source):
    '''
    Note:  We will need gamma inf and kappa inf for alternative z_src_models using Beta_s

    Computes the critical surface density:
    $\Sigma_{crit} = \frac{c^2}{4\pi G}\frac{D_s}{D_LD_{LS}}$
    
    Parameters
    ----------
    cosmo : object
        ccl cosmology object
    z_cluster : float
        Galaxy cluster redshift
    z_source : float
        Background source galaxy redshift


    Returns
    -------
    float
        Cosmology dependent critical surface density

    '''

    m_to_pc = 3.2408e-17
    kg_to_msun = 5.0279e-31

    
    c = ccl.physical_constants.CLIGHT * m_to_pc
    G = ccl.physical_constants.GNEWT * (m_to_pc)**3/(kg_to_msun)

    h = cosmo['h']
    mpc2pc = 1e6
    
    aexp_cluster = 1./(1.+z_cluster)
    aexp_src = 1./(1.+z_source)
    d_l = ccl.comoving_angular_distance(cosmo, aexp_cluster) * aexp_cluster * h * mpc2pc
    d_s = ccl.comoving_angular_distance(cosmo, aexp_src) * aexp_src * h * mpc2pc
    d_ls = _comoving_angular_distance_aexp1_aexp2(cosmo, aexp_cluster, aexp_src)

    # will need to deal with units: distances in Mpc and some CCL constants in SI
    return c*c/(4*np.pi*G) * d_s/(d_l*d_ls) 

def compute_tangential_shear_profile(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta=200, 
                                     halo_profile_parameterization='nfw', 
                                     z_src_model='single_plane'):
    '''
    Note:  We will need gamma inf and kappa inf for alternative z_src_models using Beta_s

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
    cosmo : object
        ccl cosmology object
    Delta : :obj:`int`, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_parameterization :obj:`str`, optional
        Profile model parameterization that we wish to use, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model :obj:`str`, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.


    Returns
    -------
    array-like
        Mass convergence profile, kappa.

    '''
    delta_sigma = calculate_excess_surface_density(r_proj, mdelta, cdelta, cosmo, Delta=Delta, 
                                                   halo_profile_parameterization=halo_profile_parameterization)
    if z_src_model == 'single_plane':
        sigma_c = get_critical_surface_density(cosmo, z_cluster, z_source)
        return delta_sigma / sigma_c
    elif z_src_model == 'known_z_src': # Discrete case
        NotImplementedError('Need to implemnt Beta_s functionality, or average delta_sigma/sigma_c gamma_t = Beta_s*gamma_inf')
    elif z_src_model == 'z_src_distribution' : # Continuous ( from a distribution) case
        NotImplementedError('Need to implement Beta_s calculation from integrating distribution of redshifts in each radial bin')

def compute_convergence_profile(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta=200, 
                                     halo_profile_parameterization='nfw', 
                                    z_src_model='single_plane'):
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
    cosmo : object
        ccl cosmology object
    Delta : :obj:`int`, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_parameterization :obj:`str`, optional
        Profile model parameterization that we wish to use, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model :obj:`str`, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.


    Returns
    -------
    array-like
        Mass convergence profile, kappa.
        
        
    '''

    sigma = calculate_surface_density(r_proj, mdelta, cdelta, cosmo, Delta=Delta, 
                                                      halo_profile_parameterization=halo_profile_parameterization)
    
    if z_src_model == 'single_plane':
        sigma_c = get_critical_surface_density(cosmo, z_cluster, z_source)
        return sigma / sigma_c
    elif z_src_model == 'known_z_src': # Discrete case
        NotImplementedError('Need to implemnt Beta_s functionality, or average sigma/sigma_c kappa_t = Beta_s*kappa_inf')
    elif z_src_model == 'z_src_distribution' : # Continuous ( from a distribution) case
        NotImplementedError('Need to implement Beta_s calculation from integrating distribution of redshifts in each radial bin')

def compute_reduced_tangential_shear_profile(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta=200, 
                                             halo_profile_parameterization='nfw', 
                                                z_src_model='single_plane'):
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
    cosmo : object
        ccl cosmology object
    Delta : :obj:`int`, optional
        Mass overdensity definition.  Defaults to 200.
    halo_profile_parameterization :obj:`str`, optional
        Profile model parameterization that we wish to use, with the following supported options:
        `nfw` (default) - [insert citation here]
    z_src_model :obj:`str`, optional
        Source redshift model, with the following supported options:
        `single_plane` (default) - all sources at one redshift
        `known_z_src` - known individual source galaxy redshifts, e.g. discrete case
        `z_src_distribution` - known source redshift distribution, e.g. continuous case requiring integration.


    Returns
    -------
    array-like
        Reduced tangential shear.
        
        
    '''
    
    if z_src_model == 'single_plane':
        kappa = compute_convergence_profile(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta, 
                                         halo_profile_parameterization,
                                        z_src_model)
        gamma_t = compute_tangential_shear_profile(r_proj, mdelta, cdelta, z_cluster, z_source, cosmo, Delta, 
                                         halo_profile_parameterization,
                                        z_src_model)
        return gamma_t / (1 - kappa)
    elif z_src_model == 'known_z_src': # Discrete case
        NotImplementedError('Need to implemnt Beta_s functionality, or average sigma/sigma_c kappa_t = Beta_s*kappa_inf')
    elif z_src_model == 'z_src_distribution' : # Continuous ( from a distribution) case
        NotImplementedError('Need to implement Beta_s and Beta_s2 calculation from integrating distribution of redshifts in each radial bin')


# Quick test of functions

r3d = np.logspace(-2,2,100)

rho = get_3d_density_profile(r3d,mdelta=cluster_mass, cdelta=cluster_concentration, cosmo=cosmo_ccl)

Sigma = calculate_surface_density(r3d, cluster_mass, cluster_concentration, cosmo=cosmo_ccl, Delta=200, 
                                  halo_profile_parameterization='nfw')

DeltaSigma = calculate_excess_surface_density(r3d, cluster_mass, cluster_concentration, cosmo=cosmo_ccl, Delta=200, 
                                              halo_profile_parameterization='nfw')

Sigmac = get_critical_surface_density(cosmo_ccl, z_cluster=1.0, z_source=2.0)

gammat = compute_tangential_shear_profile(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, z_cluster=1.0, 
                                          z_source=2.0, cosmo=cosmo_ccl, Delta=200, 
                                          halo_profile_parameterization='nfw', z_src_model='single_plane')

kappa = compute_convergence_profile(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, 
                            z_cluster=1.0, z_source=2.0,
                                 cosmo=cosmo_ccl, Delta=200, 
                                     halo_profile_parameterization='nfw', 
                                    z_src_model='single_plane')

gt = compute_reduced_tangential_shear_profile(r3d, mdelta=cluster_mass, cdelta=cluster_concentration, 
                                         z_cluster=1.0, z_source=2.0, cosmo=cosmo_ccl, Delta=200, 
                                         halo_profile_parameterization='nfw', z_src_model='single_plane')


import matplotlib.pyplot as plt

def plot_profile(r, profile_vals, profile_label='rho'):
    plt.loglog(r, profile_vals)
    plt.xlabel('r [Mpc]', fontsize='xx-large')
    plt.ylabel(profile_label, fontsize='xx-large')



def check_import3():
    print("Imported profilepredicting.py")


