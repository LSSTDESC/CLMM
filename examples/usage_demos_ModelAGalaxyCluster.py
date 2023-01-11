
##|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
##\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\| MODEL A GALAXY CLUSTER |/////////////////////////////////////////////
##A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAVAV/////////////////////////////////////////////////////A
##AA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAV/////////////////////////////////////////////////////AA
##AAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAV/////////////////////////////////////////////////////AAA
##AAAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V/////////////////////////////////////////////////////AAAA

## THIS WILL DEMONSTRATE THE STEPS TO GENERATE MODEL DATA FOR GALAXY CLUSTER WEAK LENSING OBSERVABLES.
## ROUGH OUTLINE:
##      @ DEFINE A GALAXY CLUSTER MODEL FOLLOWING AN NFW PROFILE
##      @ GENERATE VARIOUS PROFILES FOR THE MODEL (MASS, DENSITY, CONVERGENCE, SHEAR, ETC.)
## A FULL PIPELINE TO MEASURE A GALAXY CLUSTER WEAK LENSING MASS REQUIRES FITTING THE OBSERVED (OR MOCK) DATA.

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', message='.*(!).*')
import os
os.environ['CLMM_MODELING_BACKEND'] = 'ccl'
import clmm
import clmm.theory as m
from clmm import Cosmology


## DEFINE A COSMOLOGY WITH astropy
H0 = 70.
Omega_b0 = 0.045
Omega_dm0 = 0.27 - Omega_b0
Omega_k0 = 0.
cosmo = Cosmology(H0=H0, Omega_dm0=Omega_dm0, Omega_b0=Omega_b0, Omega_k0=Omega_k0)

## DEFINE A GALAXY CLUSTER MODEL.
density_profile_parametrization = 'nfw'
mass_Delta = 200
cl_mass = 1.e15
cl_conc = 4.
cl_z = 1.

## SOURCE PROPERTIES
z_src = 2.  ## ALL SOURCES ARE IN A PLANE (i.e. SAME REDSHIFT)
z_distrib_func = clmm.utils._chang_z_distrib    ## SOURCES' REDSHIFTS FOLLOW A GIVEN DISTRIBUTION
alpha = [2, -0.5]


## QUICK TEST OF ALL THEORY FUNCTIONALITY
r3d = np.logspace(-2, 2, 100)
rho = m.compute_3d_density(r3d, mdelta=cl_mass, cdelta=cl_conc, z_cl=cl_z, cosmo=cosmo)
Sigma = m.compute_surface_density(
        r3d, cl_mass, cl_conc, cl_z, cosmo=cosmo, delta_mdef=mass_Delta, halo_profile_model=density_profile_parametrization)
DeltaSigma = m.compute_excess_surface_density(
        r3d, cl_mass, cl_conc, cl_z, cosmo=cosmo, delta_mdef=mass_Delta, halo_profile_model=density_profile_parametrization)
Sigmac = m.compute_critical_surface_density(cosmo, z_cluster=cl_z, z_source=z_src)
gammat = m.compute_tangential_shear(
        r3d, mdelta=cl_mass, cdelta=cl_conc, z_cluster=cl_z, z_source=z_src, cosmo=cosmo, delta_mdef=mass_Delta,
        halo_profile_model=density_profile_parametrization, z_src_model='single_plane')
kappa = m.compute_convergence(
        r3d, mdelta=cl_mass, cdelta=cl_conc, z_cluster=cl_z, z_source=z_src, cosmo=cosmo, delta_mdef=mass_Delta,
        halo_profile_model=density_profile_parametrization, z_src_model='single_plane')
gt = m.compute_reduced_tangential_shear(
        r3d, mdelta=cl_mass, cdelta=cl_conc, z_cluster=cl_z, z_source=z_src, cosmo=cosmo, delta_mdef=mass_Delta,
        halo_profile_model=density_profile_parametrization, z_src_model='single_plane')
mu = m.compute_magnification(
        r3d, mdelta=cl_mass, cdelta=cl_conc, z_cluster=cl_z, z_source=z_src, cosmo=cosmo, delta_mdef=mass_Delta,
        halo_profile_model=density_profile_parametrization, z_src_model='single_plane')
mu_bias = m.compute_magnification_bias(
        r3d, alpha=alpha, mdelta=cl_mass, cdelta=cl_conc, z_cluster=cl_z, z_source=z_src, cosmo=cosmo,
        delta_mdef=mass_Delta, halo_profile_model=density_profile_parametrization, z_src_model='single_plane')

## LENSING QUNATITIES ASSUMING SOURCES FOLLOW A GIVEN REDSHIFT DISTRIBUTION:
gt_z = m.compute_reduced_tangential_shear(
        r3d, mdelta=cl_mass, cdelta=cl_conc, z_cluster=cl_z, z_source=z_src, cosmo=cosmo, delta_mdef=mass_Delta,
        halo_profile_model=density_profile_parametrization, z_src_model='schrabback18', z_distrib_func=z_distrib_func)


## PLOT THE PREDICTED PROFILES:
def plot_profile(r, profile_vals, profile_label='rho', label='', show_label=False, show=True) :
    plt.loglog(r, profile_vals, label=label)
    plt.xlabel('r [Mpc]', fontsize='x-large')
    plt.ylabel(profile_label, fontsize='x-large')
    if show_label :
        plt.legend()
    if show :
        plt.show()

plot_profile(r3d, rho, '$\\rho_{\\rm 3d}$')
plot_profile(r3d, Sigma, '$\\Sigma{\\rm 2d}$')
plot_profile(r3d, DeltaSigma, '$\\Delta\\Sigma_{\\rm 2d}$')
plot_profile(r3d, kappa, '$\\kappa$')
plot_profile(r3d, gammat, '$\\gamma_t$')

plot_profile(r3d, gt, '$g_t$', label='single plane', show=False)
plot_profile(r3d, gt_z, '$g_t$', label='redshift distribution', show_label=True, show=True)

plot_profile(r3d, mu, '$\mu$')

plot_profile(r3d, mu_bias[0]-1, profile_label='#\Delta_{\mu}$', label='$\\alpha$ = {}'.format(alpha[0]), show=False)
plot_profile(r3d, mu_bias[1]-1, '$\delta_{\mu}$', label='$\\alpha = {}$'.format(alpha[1]), show=False)

plt.legend(fontsize='x-large')
plt.yscale('linear')
plt.grid()
plt.ylim(-3,5)
plt.show()


## THE 2-HALO TERM EXCESS SURFACE DENSITY IS ONLY IMPLEMENTED FOR THE CCL AND NC BACKENDS.
DeltaSigma_2h = m.compute_excess_surface_density_2h(r3d, cl_z, cosmo=cosmo, halobias=0.3)
Sigma_2h = m.compute_surface_density_2h(r3d, cl_z, cosmo=cosmo, halobias=0.3)

plot_profile(r3d, DeltaSigma_2h, '$\\Delta\\Sigma_{\\rm 2h}$')
plot_profile(r3d, Sigma_2h, '$\\Sigma_{\\rm 2h}$')




