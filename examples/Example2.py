

##|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
##\\\\\\\\\\\\\\\\\\\\\\\| FIT HALO MASS TO SHEAR PROFILE: 2. REALISTIC DATA AND WRONG MODEL |///////////////////////
##A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAVAVAV////////////////////////////////////////////////////A
##AA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAVAV////////////////////////////////////////////////////AA
##AAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAV////////////////////////////////////////////////////AAA
##AAAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAV////////////////////////////////////////////////////AAAA
##AAAAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V////////////////////////////////////////////////////AAAAA


## THIS WILL DEMONSTRATE HOW TO USE clmm TO ESTIMATE A WL HALO MASS FROM OBSERVATIONS OF A GALAXY CLUSTER.
## IT WILL ALSO DEMONSTRATE THE BIAS INTRODUCED ON THE RECONSTRUCTED MASS BY A NAIVE FIT, WHEN THE REDSHIFT
## DISTRIBUTION OF THE BACKGROUND GALAXIES IS NOT PROPERLY ACCOUNTED FO IN THE MODEL.
## ORGANIZATION:
##      @ SETTING THINGS UP
##      @ GENERATE 3 DATASETS:
##          - IDEAL WITH A SINGLE SOURCE PLANE
##          - IDEAL WITH SOURCE GALAXIES FOLLOWING THE CHANGE ET AL. (2013) REDSHIFT DISTRIBUTION
##          - A NOISY DATASET WHERE PHOTOZ ERRORS AND SHAPE NOISE ARE ALSO INCLUDED.
##      @ COMPUTE THE BINNED REDUCED TANGENTIAL SHEAR PROFILE
##      @ SETTING UP THE 'SINGLE SOURCE PLANE' MODEL TO BE FITTED (WE EXPECT BIAS IN THE LATTER TWO DATASETS)
##      @ PERFORM SIMPLE FIT WITH scipy.optimize.curve_fit.




##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||||||||||||||| SETUP |||||||||||||||||||||||||||||||||||||||||||||||||||||||
#VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## BASIC IMPORTS
import clmm
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from clmm.support.sampler import fitters

## IMPORT clmm'S CORE MODULES
import clmm.dataops as da
import clmm.galaxycluster as gc
import clmm.theory as theory
from clmm import Cosmology

## TO GENERATE MOCK DATA
from clmm.support import mock_data as mock




##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||||||||| MAKING MOCK DATA |||||||||||||||||||||||||||||||||||||||||||||||||
#VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

np.random.seed(42)

H0 = 70.
Omega_b0  = 0.045
Omega_dm0 = 0.27 - Omega_b0
Omega_k0  = 0.
mock_cosmo = Cosmology(H0=H0, Omega_dm0=Omega_dm0, Omega_b0=Omega_b0, Omega_k0=Omega_k0)


## SET SOME PARAMETERS FOR A MOCK GALAXY CLUSTER
cosmo = mock_cosmo
cl_mass = 1.e15
cl_z    = 0.3
cl_conc = 4
ngals   = int(1e4)
Delta   = 200
cl_ra   = 20.
cl_dec  = 70.


## USE mock_data SUPORT TO GENERATE 3 GALAXY CATALOGS
ideal_data   = mock.generate_galaxy_catalog(cl_mass, cl_z, cl_conc, cosmo, 0.8, ngals=ngals, cluster_ra=cl_ra, cluster_dec=cl_dec)
ideal_data_z = mock.generate_galaxy_catalog(cl_mass, cl_z, cl_conc, cosmo, 'chang13', ngals=ngals, cluster_ra=cl_ra, cluster_dec=cl_dec)
noisy_data_z = mock.generate_galaxy_catalog(cl_mass, cl_z, cl_conc, cosmo, 'chang13', ngals=ngals, cluster_ra=cl_ra, cluster_dec=cl_dec, shapenoise=0.5, photoz_sigma_unscaled=0.05)


## THESE CATALOGS ARE THEN CONVERTED TO A clmm.GalaxyCluster OBJECT.
cl_id = 'CL_ideal'
gc_object = clmm.GalaxyCluster(cl_id, cl_ra, cl_dec, cl_z, ideal_data)
gc_object.save('ideal_GC.pkl')

cl_id = 'CL_ideal_z'
gc_object = clmm.GalaxyCluster(cl_id, cl_ra, cl_dec, cl_z, ideal_data_z)
gc_object.save('ideal_GC_z.pkl')

cl_id = 'CL_noisy_z'
gc_object = clmm.GalaxyCluster(cl_id, cl_ra, cl_dec, cl_z, noisy_data_z)
gc_object.save('noisy_GC_z.pkl')


## WE THEN (FOR DEMONSTRATION) READ IN THE clmm.GalaxyCluster OBJECTS FOR ANALYSIS.
cl1 = clmm.GalaxyCluster.load('ideal_GC.pkl')       ## ALL BACKGROUND GALAXIES ARE AT THE SAME REDSHIFT
cl2 = clmm.GalaxyCluster.load('ideal_GC_z.pkl')     ## BACKGROUND GALAXIES FOLLOW AND EMPIRICAL REDSHIFT DISTRIBUTION
cl3 = clmm.GalaxyCluster.load('noisy_GC_z.pkl')     ## SAME AS cl2 BUT WITH PHOTOZ ERROR AND SHAPE NOISE

h = plt.hist(cl2.galcat['z'], bins='scott')
plt.xlabel('Redshift')

#plt.show()




##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||||||||| COMPUTING SHEAR ||||||||||||||||||||||||||||||||||||||||||||||||||
#VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

theta1, gt1, gx1 = cl1.compute_tangential_and_cross_components()
theta2, gt2, gx2 = cl2.compute_tangential_and_cross_components()
theta3, gt3, gx3 = cl3.compute_tangential_and_cross_components()



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||||| RADIALLY BINNING THE DATA |||||||||||||||||||||||||||||||||||||||||||||
#VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

bin_edges = da.make_bins(0.7, 4, 15, method='evenlog10width')

## clmm.dataops.make_radial_profile EVALUATES THE AVERAGE SHEAR OF THE GALAXY CATALOG IN BINS OF RADIUS
profile1 = cl1.make_radial_profile('Mpc', bins=bin_edges, cosmo=cosmo)
profile2 = cl2.make_radial_profile('Mpc', bins=bin_edges, cosmo=cosmo)
profile3 = cl3.make_radial_profile('Mpc', bins=bin_edges, cosmo=cosmo)

## NOW THE clmm.GalaxyCluster OBJECTS HAVE ACQUIRED THE clmm.GalaxyCluster.profile ATTRIBUTE.
for n in cl1.profile.colnames: cl1.profile[n].format = '%6.3e'
cl1.profile.pprint(max_width=-1)


fig = plt.figure(figsize=(10,6))
fsize = 14

fig.gca().errorbar(profile1['radius'], profile1['gt'], yerr=profile1['gt_err'], marker='o', label='z_src = 0.8')
fig.gca().errorbar(profile2['radius'], profile2['gt'], yerr=profile2['gt_err'], marker='o', label='z_src = Chang et al. (2013)')
fig.gca().errorbar(profile3['radius'], profile3['gt'], yerr=profile3['gt_err'], marker='o', label='z_src = Chang et al. (2013) + photoz err + shape noise')

plt.gca().set_title(r'Binned shear of source galaxies', fontsize=fsize)
plt.gca().set_xlabel(r'$r$ [Mpc]', fontsize=fsize)
plt.gca().set_ylabel(r'$g_t$', fontsize=fsize)
plt.legend()

plt.show()




##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||||||| CREATE THE HALO MODEL |||||||||||||||||||||||||||||||||||||||||||||||
#VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## MODEL DEFINITION TO BE USED WITH scipy.optimize.curve_fit
def shear_profile_model(r, logm, z_src) :
    m = 10.**logm
    gt_model = clmm.compute_reduced_tangential_shear(r, m, cl_conc, cl_z, z_src, cosmo, delta_mdef=200, halo_profile_model='nfw')
    return gt_model




##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||||||| FITTING A HALO MASS ||||||||||||||||||||||||||||||||||||||||||||||||
##|||||||||||| HIGHLIGHTING BIAS WHEN NOT ACCOUNTING FOR THE SOURCE REDSHIFT DISTRIBUTION IN THE MODEL ||||||||||||||
#VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## TO BUILD THE MODEL WE MAKE THE WRONG ASSUMPTION THAT THE AVERAGE SHEAR IN BIN i EQUALS THE SHEAR AT THE AVERAGE
## REDSHIFT IN THE BIN; i.e. WE ASSUME THAT <g_t>_i = g_t(<z>_i).
## THIS SHOULD NOT IMPACT cl1 AS ALL SOURCES ARE LOCATED AT THE SAME REDSHIFT.
## HOWEVER, THIS YIELDS BIAS IN THE RECONSTRUCTED MASS OF cl2 AND cl3.

## FOR CLUSTER 1:
popt1, pcov1 = fitters['curve_fit'](lambda r, logm:shear_profile_model(r, logm, profile1['z']), profile1['radius'], profile1['gt'], profile1['gt_err'], bounds=[13., 17.])
m_est1 = 10.**popt1[0]
m_est_err1 = m_est1 * np.sqrt(pcov1[0][0]) * np.log(10) ## CONVERT THE ERROR ON logm TO ERROR ON m

## FOR CLUSTER 2:
popt2, pcov2 = fitters['curve_fit'](lambda r, logm:shear_profile_model(r, logm, profile2['z']), profile2['radius'], profile2['gt'], profile2['gt_err'], bounds=[13., 17.])
m_est2 = 10.**popt2[0]
m_est_err2 = m_est2 * np.sqrt(pcov2[0][0]) * np.log(10)

## FOR CLUSTER 3:
popt3, pcov3 = fitters['curve_fit'](lambda r, logm:shear_profile_model(r, logm, profile3['z']), profile3['radius'], profile3['gt'], profile3['gt_err'], bounds=[13., 17.])
m_est3 = 10.**popt3[0]
m_est_err3 = m_est3 * np.sqrt(pcov3[0][0]) * np.log(10)


print(f'Best fit mass for cluster 1 = {m_est1:.2e} +/- {m_est_err1:.2e} Msun')
print(f'Best fit mass for cluster 2 = {m_est2:.2e} +/- {m_est_err2:.2e} Msun')
print(f'Best fit mass for cluster 3 = {m_est3:.2e} +/- {m_est_err3:.2e} Msun')

## AS WE CAN SEE THE LATTER TWO MODELS ARE BIASED SINCE A SINGLE REDSHIFT WAS NOT ACCOUNTED FOR IN THE MODEL.






##|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
##\\\\\\\\\\\\\\\\\\\\\\\\\\| FIT HALO MASS TO SHEAR PROFILE: 2. OBJECT-ORIENTED MODELING |//////////////////////////
##A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAVAVAV////////////////////////////////////////////////////A
##AA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAVAV////////////////////////////////////////////////////AA
##AAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAV////////////////////////////////////////////////////AAA
##AAAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAV////////////////////////////////////////////////////AAAA
##AAAAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V////////////////////////////////////////////////////AAAAA


## WE WILL DO THE SAME THING AS ABOVE BUT NOW IN AN OBJECT-ORIENTED WAY.
## THAT'S IT.

## NOTHING CHANGES UNTIL WE GET DOWN TO THE SECTION: CREATE THE HALO MODEL

##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||||||| CREATE THE HALO MODEL |||||||||||||||||||||||||||||||||||||||||||||||
#VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## FIRST DEFINE A clmm THEORY OBJECT TO USE WITH THE NC OR CCL BACKEND
moo = clmm.Modeling(massdef='mean', delta_mdef=200, halo_profile_model='nfw')
moo.set_cosmo(mock_cosmo)
moo.set_concentration(cl_conc)

## MODEL DEFINITION TO BE USED WITH scipy.optimize.curve_fit
def shear_profile_model_(r, logm, z_src) :
    moo.set_mass(10.**logm)
    gt_model = moo.eval_reduced_tangential_shear(r, cl_z, z_src)
    return gt_model




##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||||||| FITTING A HALO MASS ||||||||||||||||||||||||||||||||||||||||||||||||
##|||||||||||| HIGHLIGHTING BIAS WHEN NOT ACCOUNTING FOR THE SOURCE REDSHIFT DISTRIBUTION IN THE MODEL ||||||||||||||
#VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

popt1, pcov1 = fitters['curve_fit'](lambda r, logm:shear_profile_model(r, logm, profile1['z']), profile1['radius'], profile1['gt'], profile1['gt_err'], bounds=[13., 17.])
m_est1 = 10.**popt1[0]
m_est_err1 = m_est1 * np.sqrt(pcov1[0][0]) * np.log(10)

popt2, pcov2 = fitters['curve_fit'](lambda r, logm:shear_profile_model(r, logm, profile2['z']), profile2['radius'], profile2['gt'], profile2['gt_err'], bounds=[13., 17.])
m_est2 = 10.**popt2[0]
m_est_err2 = m_est2 * np.sqrt(pcov2[0][0]) * np.log(10)

popt3, pcov3 = fitters['curve_fit'](lambda r, logm:shear_profile_model(r, logm, profile3['z']), profile3['radius'], profile3['gt'], profile3['gt_err'], bounds=[13., 17.])
m_est3 = 10.**popt3[0]
m_est_err3 = m_est3 * np.sqrt(pcov3[0][0]) * np.log(10)


print(f'Best fit mass for cluster 1 = {m_est1:.2e} +/- {m_est_err1:.2e} Msun')
print(f'Best fit mass for cluster 2 = {m_est2:.2e} +/- {m_est_err2:.2e} Msun')
print(f'Best fit mass for cluster 3 = {m_est3:.2e} +/- {m_est_err3:.2e} Msun')


## CALCULATE THE REDUCED TANGENTIAL SHEAR PREDICTED BY THE MODEL WHEN USING THE AVERAGE REDSHIFT OF THE CATALOG.
rr = np.logspace(-0.5, np.log10(5), 100)
moo.set_mass(m_est1)
gt_model1 = moo.eval_reduced_tangential_shear(rr, cl_z, np.mean(cl1.galcat['z']))

moo.set_mass(m_est2)
gt_model2 = moo.eval_reduced_tangential_shear(rr, cl_z, np.mean(cl2.galcat['z']))

moo.set_mass(m_est3)
gt_model3 = moo.eval_reduced_tangential_shear(rr, cl_z, np.mean(cl3.galcat['z']))




fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,6), sharey=True)

axs[0].errorbar(profile1['radius'], profile1['gt'], profile1['gt_err'], color='red', label='ideal_data, M_input = %.3e Msun' % cl_mass, fmt='.')
axs[0].plot(rr, gt_model1, color='red', label='best fit model 1, M_fit = %.2e +/- %.2e' % (m_est1, m_est_err1))

axs[0].errorbar(profile2['radius'], profile2['gt'], profile2['gt_err'], color='green', label='ideal_data_z, M_input = %.3e Msun' % cl_mass, fmt='.')
axs[0].plot(rr, gt_model2, color='green', label='best fit model 2, M_fit = %.2e +/- %.2e' % (m_est2, m_est_err2))

axs[0].set_title('Ideal data w/wo src redshift distribution', fontsize=fsize)
axs[0].semilogx()
axs[0].semilogy()
axs[0].legend(fontsize=fsize)
axs[0].set_xlabel('R [Mpc]', fontsize=fsize)
axs[0].set_ylabel('reduced tangential shear', fontsize=fsize)

axs[1].errorbar(profile3['radius'], profile3['gt'], profile3['gt_err'], color='red', label='noisy_data_z, M_input = %.3e Msun' % cl_mass, fmt='.')
axs[1].plot(rr, gt_model3, color='red', label='best fit model 3, M_fit = %.2e +/- %.2e' % (m_est3, m_est_err3))
axs[1].set_title('Noisy data with src redshift distribution', fontsize=fsize)
axs[1].semilogx()
axs[1].semilogy()
axs[1].legend(fontsize=fsize)
axs[1].set_xlabel('R [Mpc]', fontsize=fsize)
#axs[1].set_ylabel('reduced tangential shear', fontsize=fsize)

fig.tight_layout()
plt.show()
