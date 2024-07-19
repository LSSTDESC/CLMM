import numpy as np
import clmm
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline



## CALCULATE MONOPOLE COMPONENT OF GAMMA_T
def gamma_tangential_monopole(r, mdelta, cdelta, z_cl, cosmo, hpmd='nfw'):
    ## Calculate the monopole component of gamma_T given by
    ## 2 * r^-2 * Integrate[ r'*Sigma(r', mdelta, cdelta, z_cl, **kwargs), 0, r] - Sigma(r, mdelta, cdelta, z_cl, **kwargs)

    kappa_0 = clmm.compute_surface_density(r, mdelta, cdelta, z_cl, cosmo, delta_mdef=200, 
                                     halo_profile_model=hpmd, massdef='mean', alpha_ein=None, 
                                     verbose=False, validate_input=True)
    
    f = lambda r_i: r_i*clmm.compute_surface_density(r_i, mdelta, cdelta, z_cl, cosmo, delta_mdef=200, 
                                         halo_profile_model=hpmd, massdef='mean', alpha_ein=None, 
                                          verbose=False, validate_input=True)
    integrate = np.vectorize(pyfunc = scipy.integrate.quad_vec)
    
    integral, err = integrate(f,0,r)
 
    
    return (2/r**2)*integral - kappa_0



## CALCULATE MONOPOLE COMPONENT OF GAMMA_T WITH EPSILON^2 CORRECTIONS
def gamma_tangential_monopole_e2corrected(r, ell, mdelta, cdelta, z_cl, cosmo, hpmd='nfw'):
    ## Calculate the monopole component of gamma_T given by
    ## 2 * r^-2 * Integrate[ r'*Sigma0(r', mdelta, cdelta, z_cl, **kwargs), 0, r] - Sigma0(r, mdelta, cdelta, z_cl, **kwargs)
    ## where Sigma0 = Sigma*(1 + e^2/8 * (eta_0 + eta_0^2/2 + r*(deta_0/dr)/2))

    kappa = clmm.compute_surface_density(r, mdelta, cdelta, z_cl, cosmo, delta_mdef=200,
                                     halo_profile_model=hpmd, massdef='mean', alpha_ein=None, 
                                     verbose=False, validate_input=True)
    
    eta_0  = r * np.gradient(np.log(kappa), r)
    Deta_0 = r * np.gradient(eta_0, r)
    
    kappa_0 = kappa * (1 + ell**2/8 * (eta_0 + eta_0**2/2 + Deta_0/2))
    
    f = lambda r_i, kapp_0_i: r_i * kappa_0_i

    integrate = np.vectorize(pyfunc = scipy.integrate.quad_vec)
    
    integral, err = integrate(f,0,r)
 
    
    return (2/r**2)*integral - kappa_0



def g_tangential_quadrupole(r, mdelta, cdelta, z_cl, cosmo, hpmd='nfw'):
    kappa_0 = clmm.compute_surface_density(r, mdelta, cdelta, z_cl, cosmo, delta_mdef=200, 
                                     halo_profile_model=hpmd, massdef='mean', alpha_ein=None, 
                                     verbose=False, validate_input=True)
    
    f1 = lambda r_i: (r_i**3)*clmm.compute_surface_density(r_i, mdelta, cdelta, z_cl, cosmo, delta_mdef=200, 
                                         halo_profile_model=hpmd, massdef='mean', alpha_ein=None, 
                                          verbose=False, validate_input=True)
    f2 = lambda r_i: (r_i**3)*clmm.compute_surface_density(r_i, mdelta, cdelta, z_cl, cosmo, delta_mdef=200, 
                                         halo_profile_model=hpmd, massdef='mean', alpha_ein=None, 
                                          verbose=False, validate_input=True)
    
    integrate = np.vectorize(pyfunc = scipy.integrate.quad_vec)
    
    integral1, err = integrate(f1,0,r)
    integral2, err = integrate(f2,r,np.inf)
 
    
    return (2/r**2)*integral - kappa_0




def _delta_sigma_4theta(ell_, r, mdelta, cdelta, z_cl, cosmo, hpmd='nfw', sample_N=10000, delta_mdef=200):

    ### DEFINING INTEGRALS:
    r_arr = np.linspace(0.01, np.max(r), sample_N)
    sigma_0_arr = clmm.compute_surface_density(r_arr, mdelta, cdelta, z_cl, cosmo, delta_mdef=delta_mdef, 
                                     halo_profile_model=hpmd, massdef='mean', alpha_ein=None, 
                                     verbose=False, validate_input=True)
    eta_0_arr = np.gradient(np.log(sigma_0_arr),r_arr)*r_arr
    f = InterpolatedUnivariateSpline(r_arr, (r_arr**3)*sigma_0_arr*eta_0_arr, k=3)  # k=3 order of spline
    integral_vec = np.vectorize(f.integral)
    ###
    
    ### ACTUAL COMPUTATION:
    I_1 = (3/(r**4)) * integral_vec(0, r)
    sigma_0 = clmm.compute_surface_density(r, mdelta, cdelta, z_cl, cosmo, delta_mdef=200, 
                                     halo_profile_model=hpmd, massdef='mean', alpha_ein=None, 
                                     verbose=False, validate_input=True)
    #eta_0 = np.gradient(np.log(sigma_0),r)
    eta_0_interpolation_func = InterpolatedUnivariateSpline(r_arr, eta_0_arr)
    eta_0 = eta_0_interpolation_func(r) 
    
    return (ell_/4.0)*(2*I_1 - sigma_0*eta_0), eta_0_interpolation_func



def _delta_sigma_const(ell_, r, mdelta, cdelta, z_cl, cosmo, hpmd='nfw', sample_N=10000 ,delta_mdef=200):

    ### DEFINING INTEGRALS:
    r_arr = np.linspace(0.01, np.max(r), sample_N)
    sigma_0_arr = clmm.compute_surface_density(r_arr, mdelta, cdelta, z_cl, cosmo, delta_mdef=delta_mdef, 
                                     halo_profile_model=hpmd, massdef='mean', alpha_ein=None, 
                                     verbose=False, validate_input=True)
    eta_0_arr = np.gradient(np.log(sigma_0_arr),r_arr)*r_arr
    f = InterpolatedUnivariateSpline(r_arr, sigma_0_arr*eta_0_arr/r_arr, k=3)  # k=3 order of spline
    integral_vec = np.vectorize(f.integral)
    ###
    
    ### ACTUAL COMPUTATION:
    I_2 = integral_vec(r, np.inf)
    sigma_0 = clmm.compute_surface_density(r, mdelta, cdelta, z_cl, cosmo, delta_mdef=delta_mdef, 
                                     halo_profile_model=hpmd, massdef='mean', alpha_ein=None, 
                                     verbose=False, validate_input=True)
    #eta_0 = np.gradient(np.log(sigma_0), r)*r
    eta_0_interpolation_func = InterpolatedUnivariateSpline(r_arr, eta_0_arr)
    eta_0 = eta_0_interpolation_func(r) 
    
    return (ell_/4.0)*(2*I_2 - sigma_0*eta_0), eta_0_interpolation_func


def _sigma(r, mdelta, cdelta, z_cl, cosmo, hpmd='nfw', sample_N=1000, delta_mdef=200):
    return clmm.compute_surface_density(r, mdelta, cdelta, z_cl, cosmo, delta_mdef=delta_mdef, 
                                     halo_profile_model=hpmd, massdef='mean', alpha_ein=None, 
                                     verbose=False, validate_input=True)

def _delta_sigma_excess(ell_, r, mdelta, cdelta, z_cl, cosmo, hpmd='nfw', sample_N=1000, delta_mdef=200):
    return clmm.compute_excess_surface_density(r, mdelta, cdelta, z_cl, cosmo, delta_mdef=delta_mdef, 
                                     halo_profile_model=hpmd, massdef='mean', alpha_ein=None, 
                                     verbose=False, validate_input=True)