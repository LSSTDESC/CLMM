
##|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
##\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\| MEASURE A PROFILE |///////////////////////////////////////////////
##A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAVAV/////////////////////////////////////////////////////A
##AA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAV/////////////////////////////////////////////////////AA
##AAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAV/////////////////////////////////////////////////////AAA
##AAAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V/////////////////////////////////////////////////////AAAA

##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||||||||||||||| SETUP ||||||||||||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

import matplotlib.pyplot as plt
import numpy as np
import clmm
import clmm.dataops
from clmm.dataops import compute_tangential_and_cross_components, make_radial_profile, make_bins
from clmm.galaxycluster import GalaxyCluster
import clmm.utils as u
from clmm import Cosmology
from clmm.support import mock_data as mock

np.random.seed(42)

## DEFINE THE COSMOLOGY
H0 = 70.
Omega_b0 = 0.045
Omega_dm0 = 0.27 - Omega_b0
Omega_k0 = 0.
mock_cosmo = Cosmology(H0=H0, Omega_dm0=Omega_dm0, Omega_b0=Omega_b0, Omega_k0=Omega_k0)



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||| GENERATE CLUSTER OBJECT FROM MOCK DATA |||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## BELOW, THE MOCK DATA INCLUDES:
##      @ SHAPE NOISE
##      @ GALAXIES DRAWN FROM REDSHIFT DISTRIBUTION
##      @ PHOTOZ ERRORS

cosmo = mock_cosmo
cl_id = 'Awesome_cluster'
cl_mass = 1.e15
cl_z = 0.3
cl_conc = 4.
ngals = int(1e4)
cl_ra = 0.
cl_dec = 90.

zsrc_min = cl_z + 0.1

noisy_data_z = mock.generate_galaxy_catalog(
        cl_mass, 
        cl_z, 
        cl_conc, 
        cosmo, 
        'chang13', 
        zsrc_min=zsrc_min, 
        shapenoise=0.05, 
        photoz_sigma_unscaled=0.05, 
        ngals=ngals, 
        cluster_ra=cl_ra, 
        cluster_dec=cl_dec)

## LOAD THIS INTO A CLMM CLUSTER OBJECT CENTERED ON (O,O).
cl = GalaxyCluster(cl_id, cl_ra, cl_dec, cl_z, noisy_data_z)



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||||||| BASIC CHECKS AND PLOTS |||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

f, axs = plt.subplots(1,2, figsize=(12,4))

axs[0].scatter(cl.galcat['ra'], cl.galcat['dec'], color='b', s=1, alpha=0.3)
axs[0].plot(cl.ra, cl.dec, 'ro')
axs[0].set_ylabel('dec', fontsize='large')
axs[0].set_xlabel('ra', fontsize='large')

hist = axs[1]. hist(cl.galcat['z'], bins=40)[0]
axs[1].axvline(cl.z, c='r', ls='--')
axs[1].set_xlabel('$z_{source}$', fontsize='large')
xt = {t:f'{t}' for t in axs[1].get_xticks() if t!=0}
xt[cl.z] = '$z_{cl}$'
xto = sorted(list(xt.keys())+[cl.z])
axs[1].set_xticks(xto)
axs[1].set_xticklabels(xt[t] for t in xto)
axs[1].get_xticklabels()[xto.index(cl.z)].set_color('r')
plt.xlim(0, max(xto))
plt.tight_layout()

plt.show()


## CHECK ELLIPTICITIES

fig, ax1 = plt.subplots(1,1)
ax1.scatter(cl.galcat['e1'], cl.galcat['e2'], s=1, alpha=0.2)
ax1.set_xlabel('e1')
ax1.set_ylabel('e2')
ax1.set_aspect('equal', 'datalim')
ax1.axvline(0, linestyle='dotted', color='k')
ax1.axhline(0, linestyle='dotted', color='k')
plt.tight_layout()

plt.show()



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||||||| COMPUTE SHEAR PROFILES |||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## COMPUTE THE ANGULAR SEPARATION, CROSS AND TANGENTIAL SHEAR FOR EACH SOURCE GALAXY
theta, e_t, e_x = cl.compute_tangential_and_cross_components()

cl.galcat['et', 'ex'].pprint(max_width=-1)  ## PRINT OUT THE COMPONENTS

## PLOT TANGENTIAL AND CROSS ELLIPTICITY DISTRIBUTIONS FOR VERIFICATION
fig, axs = plt.subplots(1,2, figsize=(10,4))

axs[0].hist(cl.galcat['et'], bins='scott')
axs[0].set_xlabel('$\epsilon_t$', fontsize='xx-large')
#axs[0].set_yscale('log')

axs[1].hist(cl.galcat['ex'], bins='scott')
axs[1].set_xlabel('$\epsilon_x$', fontsize='xx-large')
#axs[1].set_yscale('log')

plt.tight_layout()
plt.show()



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||| COMPUTE SHEAR PROFILE IN RADIAL BINS ||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

cl.make_radial_profile('kpc', cosmo=cosmo)

print(cl.profile)

## USE A ClusterObject FUNCTION TO PLOT THE PROFILES

fig, axs = cl.plot_profiles(xscale='log')
\
new_profiles = cl.make_radial_profile('degrees', cosmo=cosmo)
fig, axs = cl.plot_profiles()

plt.show()



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||||||||| USER-DEFINED BINNING ||||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## EXAMPLE: GENERATE 20 BINS BETWEEN 1 AND 6 Mpc, LINEARLY SPACED
new_bins = make_bins(1, 6, nbins=20, method='evenwidth')
new_profiles = cl.make_radial_profile('Mpc', bins=new_bins, cosmo=cosmo)

fig, axs = cl.plot_profiles()
#plt.show()


## EXAMPLE: GENERATE 20 BINS BETWEEN 1 AND 6 Mpc, LOG SPACED
new_bins = make_bins(1, 6, nbins=20, method='evenlog10width')
new_profiles = cl.make_radial_profile('Mpc', bins=new_bins, cosmo=cosmo)

fig, axs = cl.plot_profiles()
axs.set_xscale('log')
#plt.show()


## EXAMPLE: GENERATE 20 BINS BETWEEN 1 AND 6 Mpc, EACH CONTAINING SAME NUMBER OF GALAXIES
##      @ FIRST CONVERT THE SOURCE SEPARATION TABLE TO Mpc
seps = u.convert_units(cl.galcat['theta'], 'radians', 'Mpc', redshift=cl.z, cosmo=cosmo)

new_bins = make_bins(1, 6, nbins=20, method='equaloccupation', source_seps=seps)
new_profiles = cl.make_radial_profile('Mpc', bins=new_bins, cosmo=cosmo)

print('number of galaxies in each bin: {}'.format(list(cl.profile['n_src'])))
fig, axs = cl.plot_profiles()
plt.show()



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||||||| OTHER PROFILE QUANTITIES ||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

plt.title('Average redshift in radial bins')
plt.errorbar(new_profiles['radius'], new_profiles['z'], new_profiles['z_err'], marker='o')
plt.axhline(cl.z, ls=':', c='r')
plt.text(1, cl.z*1.1, '$z_{cl}$', color='r')
plt.xlabel('Radius [Mpc]')
plt.ylabel('$\langle z \\rangle$')

plt.show()
