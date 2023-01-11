
##|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
##\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\| FIT HALO MASS TO SHEAR PROFILE: 1. IDEAL DATA |/////////////////////////////////
##A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAVAVAV////////////////////////////////////////////////////A
##AA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAVAV////////////////////////////////////////////////////AA
##AAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAV////////////////////////////////////////////////////AAA
##AAAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAV////////////////////////////////////////////////////AAAA
##AAAAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V////////////////////////////////////////////////////AAAAA


## THIS WILL DEMONSTRATE HOW TO USE clmm TO ESTIMATE A WL HALO MASS OF A GALAXY CLUSTER IN THE IDEAL CASE:
##        i) ALL GALAXIES ON A SINGLE SOURCE PLANE
##       ii) NO REDSHIFT ERRORS
##      iii) NO SHAPE NOISE
## THE STEPS BELOW CORRESPOND TO:
##          @ SETTING THINGS UP, WITH THE PROPER IMPORTS,
##          @ GENERATING AN IDEAL MOCK DATASET,
##          @ COMPUTING THE BINNED REDUCED TANGENTIAL SHEAR PROFILE, FOR TWO DIFFERENT BINNING SCHEMES,
##          @ SETTING UP THE MODEL TO BE FITTED TO THE DATA,
##          @ PERFORM A SIMPLE FIT USING scipy.optimize.basinhopping AND VISUALIZE THE RESULTS.

## FIRST WE IMPORT SOME STANDARD PACKAGES.

import clmm
import numpy as np
import matplotlib.pyplot as plt
from clmm.dataops import compute_tangential_and_cross_components, make_radial_profile, make_bins

from numpy import random



## NEXT IMPORT clmm CORE MODEULES.

import clmm.dataops as da
import clmm.galaxycluster as gc
import clmm.theory as theory
from clmm import Cosmology
from clmm.utils import convert_units



## CHECK VERSION
print('clmm version: {}'.format(clmm.__version__))




##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||||||||| MAKING MOCK DATA ||||||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

from clmm.support import mock_data as mock

np.random.seed(42)

## FIRST DEFINE THE "TRUE" COSMOLOGY
H0 = 70.
Omega_b0 = 0.045
Omega_dm0 = 0.27 - Omega_b0
Omega_k0 = 0.
mock_cosmo = Cosmology(H0=H0, Omega_dm0=Omega_dm0, Omega_b0=Omega_b0, Omega_k0=Omega_k0)

## NOW SET THE PARAMETERS FOR MOCK GALAXY CLUSTER
cosmo   = mock_cosmo
cl_id   = 'Awesome_cluster'
cl_mass = 1.e15
cl_z    = 0.3
src_z   = 0.8
cl_conc = 4
ngals   = int(1e4)
cl_ra   = 20.
cl_dec  = 30.

## THEN USE mock_data SUPPORT MODULE TO GENERATE A NEW GALAXY CATALOG.
ideal_data = mock.generate_galaxy_catalog(cl_mass, cl_z, cl_conc, cosmo, src_z, ngals=ngals, cluster_ra=cl_ra, cluster_dec=cl_dec)

## CONVERT THE GALAXY CLUSTER TO A clmm.GalaxyCluster OBJECT.
gc_object = clmm.GalaxyCluster(cl_id, cl_ra, cl_dec, cl_z, ideal_data)

## A clmm.GalaxyCluster OBJECT CAN BE PICKLED AND SAVED FOR LATER USE.
gc_object.save('mock_GC.pkl')

## ANY SAVED clmm.GalaxyCluster OBJECT MAY BE READ IN FOR ANALYSIS.
cl = clmm.GalaxyCluster.load('mock_GC.pkl')
print('ID: {} \nRA: {} \nDEC: {} \nz_l: {}'.format(cl.unique_id, cl.ra, cl.dec, cl.z))
print('# of source galaxies: {}'.format(len(cl.galcat)))

ra_l = cl.ra
dec_l = cl.dec
z = cl.z
e1 = cl.galcat['e1']
e2 = cl.galcat['e2']
ra_s = cl.galcat['ra']
dec_s = cl.galcat['dec']
z_s = cl.galcat['z']

## WE CAN VISUALIZE THE DISTRIBUTION OF GALAXIES ON THE SKY.
fsize=15
fig = plt.figure(figsize=(10,6))
hb = fig.gca().hexbin(ra_s-ra_l, dec_s-dec_l, gridsize=50)

cb = fig.colorbar(hb)
cb.set_label('Number of sources in bin', fontsize=fsize)

plt.gca().set_xlabel(r'$\Delta$RA', fontsize=fsize)
plt.gca().set_ylabel(r'$\Delta$DEC', fontsize=fsize)
plt.gca().set_title('Source Galaxies', fontsize=fsize)

#plt.show()




##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||| DERIVING OBSERVABLES: SHEAR ||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## CALCULATE THE TANGENTIAL AND CROSS SHEARS FOR EACH SOURCE GALAXY IN THE CLUSTER.
theta, gt, gx = da.compute_tangential_and_cross_components(ra_l, dec_l, ra_s, dec_s, e1, e2)
r_s = convert_units(theta, 'radians', 'mpc', cl.z, cosmo)

## VISUALIZE THE SHEAR FIELD AT EACH GALAXY LOCATION.
fig = plt.figure(figsize=(10,6))
fig.gca().loglog(theta, gt, '.')
plt.ylabel('reduced shear', fontsize=fsize)
plt.xlabel('angular distance [rad]', fontsize=fsize)

#plt.show()


##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||| DERIVING OBSERVABLES: RADIALLY BINNING THE DATA ||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## HERE WE COMPARE THE RECONSTRUCTED MASS UNDER TWO DIFFERENT BIN DEFINITIONS.
## KEEP IN MIND THAT BINNING WILL CAUSE THE FITTED MASS TO BE SLIGHTLY LARGER THAN THE INPUT MASS.
## THE REASON IS THAT g(r), THE TANGENTIAL REDUCED SHEAR ALONG CLUSTER RADIUS, IS A CONVEX FUNCTION -- THE FUNCTION
## VALUE AFTER BINNING WOULD BE LARGER, BUT THE BIAS BECOMES SMALLER AS BIN NUMBER INCREASES.

bin_edges1 = da.make_bins(0.01, 3.7, 50)
bin_edges2 = da.make_bins(0.01, 3.7, 10)

## EVALUATE THE AVERAGE SHEAR OF THE GALAXY CATALOG IN BINS OF RADIUS.
res1 = da.make_radial_profile([gt,gx,z_s], theta, 'radians', 'Mpc', bins=bin_edges1, cosmo=cosmo, z_lens=z, include_empty_bins=False)
res2 = da.make_radial_profile([gt,gx,z_s], theta, 'radians', 'Mpc', bins=bin_edges2, cosmo=cosmo, z_lens=z, include_empty_bins=False)

## FOR LATER USE, DEFINE SOME VARIABLES FOR THE BINNED RADIUS AND TANGENTIAL SHEAR.
gt_profile1 = res1['p_0']
r1 = res1['radius']
z1 = res1['p_2']

gt_profile2 = res2['p_0']
r2 = res2['radius']
z2 = res2['p_2']

## TAKE A LOOK AT THE RADIALY BINNED SHEAR FOR THE MOCK GALAXIES.
fig = plt.figure(figsize=(10,6))
fig.gca().loglog(r_s, gt, '.', color='k', markersize=2, label='expected')
fig.gca().loglog(r1, gt_profile1, '.', label='50 bins')
fig.gca().loglog(r2, gt_profile2, '+', markersize=15, label='10 bins')
plt.legend(fontsize=fsize)

plt.gca().set_title(r'Binned shear of source galaxies', fontsize=fsize)
plt.gca().set_xlabel(r'$r$ [Mpc]', fontsize=fsize)
plt.gca().set_ylabel(r'$g_t$', fontsize=fsize)

#plt.show()


## WE CAN ALSO JUST RUN make_radial_profile DIRECT ON A clmm.GalaxyCluster OBJECT.
cl.compute_tangential_and_cross_components()
cl.make_radial_profile('Mpc', bins=1000, cosmo=cosmo, include_empty_bins=False)

## AFTER RUNNING clmm.GalaxyCluster.make_radial_profile THE OBJECT ACQUIRES THE clmm.GalaxyCluster.profile ATTRIBUTE.
for n in cl.profile.colnames: cl.profile[n].format = '%6.3e'
cl.profile.pprint(max_width=-1)




##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||||||||| MODELING THE DATA |||||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## THIS WILL DEMONSTRATE A FEW OF THE PROCEDURES ONE CAN PERFORM ONCE A COSMOLOGY HAS BEEN CHOSEN.

## CHOOSE A HALO MODEL:
##      @ NOTICE: SOME SAMPLERS MAY PASS ARRAYS OF SIZE 1 TO VARIABLES BEING MINIMIZED WHICH CAN CAUSE PROBLEMS
##        SINCE clmm FUNCTIONS TYPE CHECK THE INPUT.
##        THE FUNCTION BELOW CORRECTS FOR THIS BY CONVERTING THE MASS TO A FLOAT.

def nfw_to_shear_profile(logm, profile_info) :
    [r, gt_profile, z_src_rbin] = profile_info
    m = float(10.**logm)
    gt_model = clmm.compute_reduced_tangential_shear(r, m, cl_conc, cl_z, z_src_rbin, cosmo, delta_mdef=200, halo_profile_model='nfw')
    return np.sum((gt_model - gt_profile)**2)

## FITTING A HALO MASS:
## WE OPTIMIZE TO FINE THE FEST-FIT MASS FOR THE DATA UNDER THE TWO RADIAL BINNING SCHEMES.
## NOTE:
##      @ THE samplers['minimize'] IS A LOCAL OPTIMIZATION FUNCTION SO IT DOES NOT GUARANTEE CONSISTENT RESULTS FOR
##        ALL logm_0.
##      @ THE samplers['basinhopping'] IS A GLOBAL OPTIMIZATION FUNCTION GIVING MUCH MORE STABLE RESULTS.
from clmm.support.sampler import samplers

logm_0 = random.uniform(13., 17., 1)[0]
#logm_est1 = samplers['minimize'](nfw_to_shear_profile, logm_0, args=[r1, gt_profile1, z1])[0]
logm_est1 = samplers['basinhopping'](nfw_to_shear_profile, logm_0, minimizer_kwargs={'args': ([r1, gt_profile1, z1])})[0]

#logm_est2 = samplers['minimize'](nfw_to_shear_profile, logm_0, args=[r2, gt_profile2, z2])[0]
logm_est2 = samplers['basinhopping'](nfw_to_shear_profile, logm_0, minimizer_kwargs={'args': ([r2, gt_profile2, z2])})[0]

m_est1 = 10.**logm_est1
m_est2 = 10.**logm_est2

print((m_est1, m_est2))

## NEXT, CALCULATE THE REDUCED TANGENTIAL SHEAR PREDICTED BY THE TWO MODELS.
rr = np.logspace(-2, np.log10(5), 100)
gt_model1 = clmm.compute_reduced_tangential_shear(
        rr,
        m_est1,
        cl_conc,
        cl_z,
        src_z,
        cosmo,
        delta_mdef=200,
        halo_profile_model='nfw')

gt_model2 = clmm.compute_reduced_tangential_shear(
        rr,
        m_est2,
        cl_conc,
        cl_z,
        src_z,
        cosmo,
        delta_mdef=200,
        halo_profile_model='nfw')

## NOW VISUALIZE THE TWO PREDICTION OF REDUCED TANGENTIAL SHEAR.
fig = plt.figure(figsize=(10,6))
fig.gca().scatter(r1, gt_profile1, color='orange', label='binned mock data 1, M_input = %.3e Msun' % cl_mass)
fig.gca().plot(rr, gt_model1, color='orange', label='best fit model 1, M_fit = %.3e' % m_est1)
fig.gca().scatter(r2, gt_profile2, color='blue', alpha=0.5, label='binned mock data2, M_input = %.3e Msun' % cl_mass)
fig.gca().plot(rr, gt_model2, color='blue', linestyle='--', alpha=0.5, label='best fit model 2, M_fit = %3.e' % m_est2)

plt.semilogx()
plt.semilogy()

plt.legend()
plt.xlabel('R [Mpc]', fontsize=fsize)
plt.ylabel('reduced tangential shear', fontsize=fsize)

#plt.show()








##|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
##\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\| FIT HALO MASS TO SHEAR PROFILE: 1. MISCENTERED DATA |//////////////////////////////
##A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAVAVAV////////////////////////////////////////////////////A
##AA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAVAV////////////////////////////////////////////////////AA
##AAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAV////////////////////////////////////////////////////AAA
##AAAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAV////////////////////////////////////////////////////AAAA
##AAAAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V////////////////////////////////////////////////////AAAAA

## THIS WILL DEMONSTRATE THE IMPACT OF MISCENTERING THE CLUSTER ON THE MASS ESTIMATE.

from clmm.galaxycluster import GalaxyCluster
from clmm.support.sampler import fitters

from astropy.coordinates import SkyCoord
import astropy.units as u



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||| GENERATE CLUSTER OBJECTS FROM MOCK DATA ||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## DEFINE A NEW MOCK CLUSTER.
cosmo = mock_cosmo
cl_id = 'Awesome_cluster'
cl_mass = 1.e15
cl_z = 0.3
cl_conc = 4
ngal_density = 50   ## gal/arcmin2
cl_ra = 20.
cl_dec = 40.
zsrc_min = cl_z + 0.1   ## WE ONLY WANT TO DRAW BACKGROUND GALAXIES
field_size = 20     ## Mpc

ideal_data_z = mock.generate_galaxy_catalog(
        cl_mass, cl_z, cl_conc,     ## CLUSTER MASS, REDSHIFT, AND CONCENTRATION
        cosmo,                      ## CHOSEN COSMOLOGY
        zsrc='chang13',             ## SOURCE GALAXY REDSHIFT DISTRIBUTION; HERE TAKEN FROM CHANG et al. 2013
        delta_so=200,               ## OVERDENSITY-DENSTY CONTRAST (so = Spherical Overdensity)
        massdef='mean',             ## DEFINE THE MASS OVERDENSITY WRT THE 'mean' OR 'critical' DENSITY.
        zsrc_min=zsrc_min, ngal_density=ngal_density,
        cluster_ra=cl_ra, cluster_dec=cl_dec,   ## CLUSTER POSITION
        field_size=field_size)


## TO DEMONSTRATE THE AFFECT OF MISCENTERING WE WILL LOAD THE MOCK DATA INTO SEVERAL CLUSTER OBJECTS CENTERED IN A
## 0.4x0.4 DEG WINDOW AROUND THE TRUE CLUSTER POSITION.
## THE USER CAN CHANGE THE NUMBER OF CLUSTER CENTERS IF DESIRED.
## WE SET THE FIRST CENTER TO THE TRUE CENTER FOR COMPARISON PURPOSES (CORRESPONDING TO a == 0 BELOW).

center_number = 5
cl_list = []
coord = []

for a in range(0,center_number) :
    if a == 0 :
        cl_ra_new = cl_ra
        cl_dec_new = cl_dec
    else :
        cl_ra_new = random.uniform(cl_ra - 0.2, cl_ra + 0.2)
        cl_dec_new = random.uniform(cl_dec - 0.2, cl_dec + 0.2)

    cl = clmm.GalaxyCluster(cl_id, cl_ra_new, cl_dec_new, cl_z, ideal_data_z)
    
    print(f'ID: {cl.unique_id} \nRA: {cl.ra:.2f} \nDEC: {cl.dec:.2f} \nz_l: {cl.z}')
    print(f'The number of source galaxies is : {len(cl.galcat)}')

    cl_list.append(cl)
    coord.append(SkyCoord(cl.ra*u.deg, cl.dec*u.deg))

## OFFSET OF THE DIFFERENT CLUSTER CENTERS FROM THE POSITION 0,0 (IN DEGREE).
offset = [coord[0].separation(coord[i]).value for i in range(5)]



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||||||| BASIC CHECKS AND PLOTS ||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

##  @ GALAXY POSITIONS
##  @ REDSHIFT DISTRIBUTION

## FOR A BETTER VISUALIZTION, WE PLOT ALL THE DIFFERENT CLUSTER CENTERS, REPRESENTED BY THE RED DOTS.

f, ax = plt.subplots(1,2, figsize=(12,4))

for cl in cl_list :
    ax[0].scatter(cl.galcat['ra'], cl.galcat['dec'], color='blue', s=1, alpha=0.3)
    ax[0].plot(cl.ra, cl.dec, 'ro')
    ax[0].set_ylabel('dec', fontsize='large')
    ax[0].set_xlabel('ra', fontsize='large')

    hist = ax[1].hist(cl.galcat['z'], bins=20)[0]

    ax[1].axvline(cl.z, c='r', ls='--')
    ax[1].set_xlabel('$z_{source}$', fontsize='large')
    xt = {t:f'{t}' for t in ax[1].get_xticks() if t!=0}
    xt[cl.z] = '$z_{cl}$'
    xto = sorted(list(xt.keys())+[cl.z])
    ax[1].set_xticks(xto)
    ax[1].set_xticklabels(xt[t] for t in xto)
    ax[1].get_xticklabels()[xto.index(cl.z)].set_color('red')
    plt.xlim(0, max(xto))

#plt.show()




##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||| COMPUTE CENTER EFFECT ON THE SHEAR PROFILES |||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## NEXT, GENERATE THE PROFILES FOR ALL THE CLUSTER OBJECTS AND SAVE THE PROFILES INTO A LIST.
## ALSO SAVE THE gt, gx, AND radius COLUMNS OF EACH profile INTO LISTS TO MAKE A PLOT OF THESE COMPONENTS.

bin_edges = make_bins(0.3, 6, 10)       ## WE WILL USE THE SAME BINS FOR ALL THE CENTERS.

profile_list = []

for cl in cl_list :
    theta, e_t, e_x = compute_tangential_and_cross_components(
            ra_lens=cl.ra, dec_lens=cl.dec,
            ra_source=cl.galcat['ra'], dec_source=cl.galcat['dec'],
            shear1=cl.galcat['e1'], shear2=cl.galcat['e2'])
    cl.compute_tangential_and_cross_components(add=True)
    cl.make_radial_profile('Mpc', cosmo=cosmo, bins=bin_edges, include_empty_bins=False)
    profile_list.append(cl.profile)

fig = plt.figure(figsize=(10,6))

for a in range(0, len(profile_list)) :
    fig.gca().errorbar(profile_list[a]['radius'], profile_list[a]['gt'], profile_list[a]['gt_err'], linestyle='-', marker='o',
            label=f'offset = {"{:.2f}".format(offset[a])}$\deg$')

plt.xlabel('log(radius)', size=fsize)
plt.ylabel('gt', size=fsize)

plt.legend()

#plt.show()

fig2 = plt.figure(figsize=(10,6))

for a in range(0,len(profile_list)) :
    fig2.gca().errorbar(profile_list[a]['radius'], profile_list[a]['gx'], profile_list[a]['gx_err'], linestyle='-', marker='o',
            label=f'offset = {"{:.2f}".format(offset[a])}$\deg$')

plt.xlabel('log(radius)', size=fsize)
plt.ylabel('gx', size=fsize)
plt.legend(loc=4)

plt.show()




##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||| COMPUTE CENTER EFFECT BY FITTING HALO MASS ||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## IN THE STEP WE COMPUTE THE FITTING HALO MASS WITH THE nfw MODEL AND COMPARE THE IMPACT OF THE LCUSTER CENTRS ON
## THE WEAK LENSING MASS.

## THE FUNCTION BELOW DEFINES THE HALO MODEL.
logm_0 = random.uniform(13., 17., 1)[0]
def shear_profile_model(r, logm, z_src) :
    m = 10.**logm
    gt_model = clmm.compute_reduced_tangential_shear(r, m, cl_conc, cl_z, z_src, cosmo, delta_mdef=200, halo_profile_model='nfw')

    return gt_model

for a in range(0, len(cl_list)) :
    popt, pcov = fitters['curve_fit'](lambda r, logm:shear_profile_model(r, logm, profile_list[a]['z']),
            profile_list[a]['radius'],
            profile_list[a]['gt'],
            profile_list[a]['gt_err'],
            bounds=[13.,17.])

    m_est1 = 10.**popt[0]
    m_est_err1 = m_est1 * np.sqrt(pcov[0][0]) * np.log(10)      ## CONVER THE ERROR ON logm TO ERROR ON m.

    print(f'The fitted mass is : {m_est1:.2e}, for the offset distance: {offset[a]:.2f} deg')
    plt.errorbar(offset[a], 1-m_est1/cl_mass, yerr=m_est_err1/cl_mass, fmt='.', markersize=5, ls='none')

plt.xlabel('offset [deg]', size=fsize)
plt.ylabel('$\%\Delta$ fitted mass $M_{200,m}$ [M$_{\odot}$]', size=fsize)
plt.axhline(0)
plt.legend(loc='best')

plt.show()
