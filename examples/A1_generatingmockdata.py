
import numpy as np
from clmm import Cosmology
import clmm.support.mock_data as mock

import matplotlib.pyplot as plt




#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\| GENERATING MOCK DATA |////////////////////////////////
#A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V//////////////////////////////////////////A
#AA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V/////////////////////////////////////////AA


## FOR REPRODUCIBILITY
np.random.seed(14)

## SET COSMOLOGY OF MOCK DATA
H0        = 70.
Omega_b0  = 0.045
Omega_dm0 = 0.27 - Omega_b0
Omega_k0  = 0.
cosmo = Cosmology(H0=H0, Omega_dm0=Omega_dm0, Omega_b0=Omega_b0, Omega_k0=Omega_k0)


## CLUSTER INFO
cl_mass = 1.e15
cl_conc = 4.
cl_z    = 0.3
cl_ra   = 0.
cl_dec  = 0.


## CATALOG INFO
field_size = 10     ## i.e. 10X10 Mpc FIELD AT THE CLUSTER REDSHIFT (CENTERED ON CLUSTER)

## MAKE MOCK GALAXIES
mock_galaxies = mock.generate_galaxy_catalog(
        cluster_m=cl_mass, cluster_z=cl_z, cluster_c=cl_conc,   ## CLUSTER DATA
        cosmo=cosmo,                    ## COSMOLOGY OBJECT
        zsrc='desc_srd',                ## GALAXY REDSHIFT DISTRIBUTION
        zsrc_min=0.4,                   ## MINIMUM REDSHIFT OF THE GALAXIES
        shapenoise=0.05,                ## GAUSSIAN SHAPE NOISE TO THE GALAXY SHAPES
        photoz_sigma_unscaled=0.05,     ## PHOTO-z ERRORS TO SOURCE REDSHIFTS
        field_size=field_size,
        ngal_density=20                 ## NUMBER OF GAL/ARCMIN2 FOR z IN [0,INFTY]
        )['ra', 'dec', 'e1', 'e2', 'z', 'ztrue', 'pzbins', 'pzpdf', 'id']


ngals_init = len(mock_galaxies)     ## INITIAL NUMBER OF MOCK GALAXIES (ngal_density INTEGRATED OVER PATCH OF SKY)

## SOME GALAXIES HAVE REDSHIFT LOWER THAN THE CLUSTER.
## GET RID OF THESE SO WE CAN ONLY LOOK AT THOSE GALAXIES THAT 'COULD' GET LENSED.
mock_galaxies = mock_galaxies[(mock_galaxies['z']>cl_z)]
ngals_good = len(mock_galaxies)     ## NUMBER OF MOCK GALAXIES WITH REDSHIFT GREATER THAN LENS


if ngals_good < ngals_init :
    print(f'Number of excluded galaxies (with photoz < cluster_z): {ngals_init-ngals_good:,}')
    mock_galaxies['id'] = np.arange(ngals_good)     ## RESET IDs FOR LATER USE



## PUT GALAXY VALUES ON ARRAYS.
gal_ra     = mock_galaxies['ra']        ## GALAXIES' RA IN deg
gal_dec    = mock_galaxies['dec']       ## GALAXIES' DEC IN deg
gal_e1     = mock_galaxies['e1']        ## GALAXIES' ELLIPTICITY ALONG AXES 1
gal_e2     = mock_galaxies['e2']        ## GALAXIES' ELLIPTICITY ALONG AXES 2
gal_z      = mock_galaxies['z']         ## GALAXIES' OBSERVED REDSHIFTS
gal_ztrue  = mock_galaxies['ztrue']     ## GALAXIES' TRUE REDSHIFTS
gal_pzbins = mock_galaxies['pzbins']    ## GALAXIES' P(Z) BINS
gal_pzpdf  = mock_galaxies['pzpdf']     ## GALAXIES' P(Z)
gal_id     = mock_galaxies['id']        ## GALAXIES' ID

gal_ra[gal_ra>180] = gal_ra[gal_ra>180] - 360

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gal_ra, gal_z, gal_dec, s=2)
ax.set_xlabel('RA')
ax.set_zlabel('DEC')
ax.set_ylabel('redshift')

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.scatter(gal_ra, gal_dec, s=2)
ax.set_xlabel('RA')
ax.set_ylabel('DEC')

plt.show()






 
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\| MEASURING SHEAR PROFILES |//////////////////////////////
#A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V//////////////////////////////////////////A
#AA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V/////////////////////////////////////////AA


## FROM SOURCE GALAXY QUANTITIES, WE CAN COMPUTE THE ELLIPTICITIES AND CORRESPONDING RADIAL 
## PROFILE USING clmm.dataops FUNCTIONS:

import clmm.dataops as da
import clmm.cosmology.parent_class as pc

## CONVERT ELLIPTICITIES INTO SHEARS
gal_ang_dist, gal_gt, gal_gx = da.compute_tangential_and_cross_components(
        cl_ra, cl_dec,      ## CLUSTER POSITION
        gal_ra, gal_dec,    ## GALAXIES' POSITIONS
        gal_e1, gal_e2,     ## GALAXIES' ELLIPTICITIES
        geometry='flat')


## MEASURE PROFILE
bin_start = 0.01
bin_end = 3.7
Nbins = 50
profile = da.make_radial_profile(
        [gal_gt, gal_gx, gal_z],
        gal_ang_dist, 'radians', 'Mpc',
        bins=da.make_bins(bin_start, bin_end, Nbins),
        cosmo=cosmo,
        z_lens=cl_z,
        include_empty_bins=False)


## OVERLAY THE BINS ON THE SKY MAP
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
patches = []
DA = cosmo.eval_da_z1z2(0,cl_z) * np.pi/180     ## PREPARE ANGULAR DISTANCE TO CONVERT BIN EDGES (IN Mpc) TO AN ANGLE ON THE SKY IN RADIANS
patches += [Wedge((cl_ra, cl_dec), profile['radius_max'][i]/DA, 0, 360, width=0.99*(profile['radius_max'][i] - profile['radius_min'][i])/DA) for i in range(Nbins)]
p = PatchCollection(patches, alpha=0.5)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.scatter(gal_ra, gal_dec, s=2)
ax.add_collection(p)
ax.set_xlabel('RA')
ax.set_ylabel('DEC')

fig, axs = plt.subplots(1,1, figsize=(8,6))
axs.errorbar(profile['radius'], profile['p_0'], xerr=None, yerr=profile['p_0_err'], ls='none', marker='o', color='k', ms=1.5, label='$g_t$')
axs.errorbar(profile['radius'], profile['p_1'], xerr=None, yerr=profile['p_1_err'], ls='none', marker='o', color='r', ms=1.5, label='$g_X$')
plt.legend()



## THE ABOVE JUST PASS EXPLICIT ARGUMENTS TO  THE make_radial_profile FUNCTION WHICH RETURNS
## THE PROFILE. WE COULD ALSO CALL THE FUNCTION AS A METHOD OF A GalaxyCluster CLASS:

import clmm

## CREATE A GCData WITH THE GALAXIES
galaxies = clmm.GCData(
        [gal_ra, gal_dec, gal_e1, gal_e2, gal_z, gal_ztrue, gal_id],
        names=['ra', 'dec', 'e1', 'e2', 'z', 'ztrue', 'id'])

## CREATE A GalaxyCluster
cluster = clmm.GalaxyCluster('Name of the cluster', cl_ra, cl_dec, cl_z, galaxies)

## CONVERT ELIPTICITIES INTO SHEARS FOR THE MEMBERS
cluster.compute_tangential_and_cross_components(geometry='flat')

## MEASURE PROFILE AND ADD PROFILE TABLE TO THE CLUSTER
profile = cluster.make_radial_profile(
        bins=da.make_bins(0.1, field_size/2., 25, method='evenlog10width'),
        bin_units='Mpc',
        cosmo=cosmo,
        include_empty_bins=False,
        gal_ids_in_bins=True,)


fig, ax1 = cluster.plot_profiles()
plt.show()






#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\| THEORETICAL PREDICTIONS |//////////////////////////////
#A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V//////////////////////////////////////////A
#AA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V/////////////////////////////////////////AA


## HERE WE WILL CONSIDER THREE MODELS:
##      1) MODEL CONSIDERING ALL SOURCES LOCATED AT THE AVERAGE REDSHIFT
##                              g_t^avg = g(R_i, <z>)
##      2) MODEL RELYING ON THE OVERALL REDSHIFT DISTRIBUTION OF THE SOURCES, NOT USING INDIVIDUAL REDSHIFT INFORMATION
##                              g_t^N = ( <beta_s> gamma_t(R_i, z->infty) ) / ( 1 - <beta_s^2>/<beta_s> kappa(R_i, z->infty) )
##      3) MODEL USING INDIVIDUAL REDSHIFT AND RADIAL INFORMATION TO COMPUTE THE AVERAGED SHEAR OF THE GALAXIES IN EACH RADIAL BIN
##                              g_t^(z,R) = (1/N_i) SUM_(gal_j in bin_i) g_t(R_j, z_j)

## MODEL 1
def predict_reduced_tangential_shear_mean_z(profile, logm) :
    return clmm.compute_reduced_tangential_shear(
            r_proj=profile['radius'],       ## RADIAL COMPONENT OF THE PROFILE
            mdelta=10**logm,                ## MASS OF THE CLUSTER IN M_sun
            cdelta=4,                       ## CONCENTRAION OF THE CLUSTER
            z_cluster=cl_z,                 ## CLUSTER REDSHIFT
            z_source=np.mean(cluster.galcat['z']),  ## MEAN VALUE OF SOURCE GALAXIES' REDSHIFTS
            cosmo=cosmo,
            delta_mdef=200,                 ## MASS OVERDENSITY DEFINITION (DEFAULTS TO 200)
            halo_profile_model='nfw')       ## CHOOSE DM PROFILE: 'nfw' (DEFAULT), ['einasto','hernquist'] (ONLY IN numcosmo CURRENTLY)



## MODEL 2
z_inf = 1e3
dl_inf = cosmo.eval_da_z1z2(cl_z, z_inf)
d_inf = cosmo.eval_da(z_inf)

def betas(z) :
    dls = cosmo.eval_da_z1z2(cl_z, z)
    ds = cosmo.eval_da(z)
    return dls * d_inf / (ds * dl_inf)

def predict_reduced_tangential_shear_approx(profile, logm) :
    bs_mean = np.mean(betas(cluster.galcat['z']))
    bs2_mean = np.mean(betas(cluster.galcat['z'])**2)

    gamma_t_inf = clmm.compute_tangential_shear(
            r_proj=profile['radius'],       ## RADIAL COMPONENT OF THE PROFILE
            mdelta=10**logm,                ## MASS OF THE CLUSTER IN M_sun
            cdelta=4,                       ## CONCENTRATION OF THE CLUSTER
            z_cluster=cl_z,                 ## CLUSTER'S REDSHIFT
            z_source=z_inf,                 ## REDSHIFT VALUE AT INFINITY
            cosmo=cosmo,
            delta_mdef=200,                 ## MASS OVERDENSITY DEFINITION (DEFAULTS TO 200)
            halo_profile_model='nfw')       ## CHOOSE DM PROFILE: 'nfw' (DEFAULT), ['einasto','hernquist'] (ONLY IN numcosmo CURRENTLY)

    convergence_inf = clmm.compute_convergence(
            r_proj=profile['radius'],       ## RADIAL COMPONENT OF THE PROFILE
            mdelta=1**logm,                 ## MASS OF THE CLUSTER IN M_sun
            cdelta=4,                       ## CONCENTRAION OF THE CLUSTER
            z_cluster=cl_z,                 ## CLUSTER'S REDSHIFT
            z_source=z_inf,                 ## REDSHIFT VALUE AT INFINITY
            cosmo=cosmo,
            delta_mdef=200,                 ## MASS OVERDENSITY DEFINITION (DEFAULTS TO 200)
            halo_profile_model='nfw')       ## CHOOSE DM PROFILE: 'nfw' (DEFAULT), ['einasto', hernquist'] (ONLY IN numcosmo CURRENTLY)

    return bs_mean*gamma_t_inf / (1-(bs2_mean/bs_mean)*convergence_inf)


## MODEL 3
from clmm.utils import convert_units
cluster.galcat['theta_mpc'] = convert_units(cluster.galcat['theta'], 'radians', 'mpc', cluster.z, cosmo)

def predict_reduced_tangential_shear_exact(profile, logm) :
    return np.array([np.mean(
        clmm.compute_reduced_tangential_shear(
            r_proj=cluster.galcat[radial_bin['gal_id']]['theta_mpc'],   ## RADIAL COMPONENT OF THE PROFILE
            mdelta=10**logm,                ## MASS OF THE CLUSTER IN M_sun
            cdelta=4,                       ## CONCENTRAION OF THE CLUSTER
            z_cluster=cl_z,                 ## CLUSTER REDSHIFT
            z_source=cluster.galcat[radial_bin['gal_id']]['z'],     ## MEAN VALUE OF SOURCE GALAXIES' REDSHIFTS
            cosmo=cosmo,
            delta_mdef=200,                 ## MASS OVERDENSITY DEFINITION (DEFAULTS TO 200)
            halo_profile_model='nfw'))      ## CHOOSE DM PROFILE: 'nfw' (DEFAULT), ['einasto','hernquist'] (ONLY IN numcosmo CURRENTLY)
        for radial_bin in profile])






#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\| MASS FITTING |////////////////////////////////////
#A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V//////////////////////////////////////////A
#AA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V/////////////////////////////////////////AA

data_for_fit = cluster.profile[cluster.profile['n_src']>5]
remainder = cluster.profile[cluster.profile['n_src']<=5]

from clmm.support.sampler import fitters

## SINGLE z VALUE MODEL (MODEL 1)
popt, pcov = fitters['curve_fit'](      ## USING scipy.curve_fit
        predict_reduced_tangential_shear_mean_z,    ## MODEL FUNCTION
        data_for_fit,                               ## X-DATA
        data_for_fit['gt'],                         ## Y-DATA
        data_for_fit['gt_err'],                     ## Y-ERROR
        bounds=[10.,17.])

logm_1z, logm_1z_err = popt[0], np.sqrt(pcov[0][0])


## OVERALL REDSHIFT DISTRIBUTION (MODEL 2; FROM Applegate et al. 2014)
popt, pcov = fitters['curve_fit'](
        predict_reduced_tangential_shear_approx,
        data_for_fit,
        data_for_fit['gt'],
        data_for_fit['gt_err'],
        bounds=[10.,17.])

logm_ap, logm_ap_err = popt[0], np.sqrt(pcov[0][0])


## CONSIDERING ALL REDSHIFTS AND RADII (MODEL 3)

popt, pcov = fitters['curve_fit'](
        predict_reduced_tangential_shear_exact,
        data_for_fit,
        data_for_fit['gt'],
        data_for_fit['gt_err'],
        bounds=[10.,17.])

logm_exact, logm_exact_err = popt[0], np.sqrt(pcov[0][0])


## REPRODUCE FIG. 5 OF arxiv:2107.10857
model1 = predict_reduced_tangential_shear_mean_z(data_for_fit, logm_1z)
model1_h = predict_reduced_tangential_shear_mean_z(data_for_fit, logm_1z+logm_1z_err)
model1_l = predict_reduced_tangential_shear_mean_z(data_for_fit, logm_1z-logm_1z_err)
model2 = predict_reduced_tangential_shear_approx(data_for_fit, logm_ap)
model2_h = predict_reduced_tangential_shear_approx(data_for_fit, logm_ap+logm_ap_err)
model2_l = predict_reduced_tangential_shear_approx(data_for_fit, logm_ap-logm_ap_err)
model3 = predict_reduced_tangential_shear_exact(data_for_fit, logm_exact)
model3_h = predict_reduced_tangential_shear_exact(data_for_fit, logm_exact+logm_exact_err)
model3_l = predict_reduced_tangential_shear_exact(data_for_fit, logm_exact-logm_exact_err)


labels = [
        '$\log(M_{{input}}) = {:.3f} + \log(M_{{\odot}})$'.format(np.log10(cl_mass)),
        '$\log(M_{{fit}}^{{avg(z)}}) = ({:.3f}\pm{:.3f}) + \log(M_{{\odot}})$'.format(logm_1z, logm_1z_err),
        '$\log(M_{{fit}}^{{N(z)}}) = ({:.3f}\pm{:.3f}) + \log(M_{{\odot}})$'.format(logm_ap, logm_ap_err),
        '$\log(M_{{fit}}^{{z,R}}) = ({:.3f}\pm{:.3f}) + \log(M_{{\odot}})$'.format(logm_exact, logm_exact_err)]


fig, axs = plt.subplots(2,1, figsize=(6,6), sharex=True, gridspec_kw={'height_ratios': [3,1]})
plt.subplots_adjust(wspace=0.5, hspace=0.5)
axs[0].errorbar(data_for_fit['radius'], data_for_fit['gt'], xerr=None, yerr=data_for_fit['gt_err'], color='k', marker='o', ms=1, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=1, label=labels[0])
axs[0].errorbar(remainder['radius'], remainder['gt'], xerr=None, yerr=remainder['gt_err'], color='gray', marker='o', ms=1, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=1)
axs[0].plot(data_for_fit['radius'], model1, lw=1, label=labels[1], c='C0')
axs[0].fill_between(data_for_fit['radius'], model1_l, model1_h, color='C0', alpha=0.3)
axs[0].plot(data_for_fit['radius'], model2, lw=1, label=labels[2], c='C1')
axs[0].fill_between(data_for_fit['radius'], model2_l, model2_h, color='C1', alpha=0.3)
axs[0].plot(data_for_fit['radius'], model3, lw=1, label=labels[3], c='C2')
axs[0].fill_between(data_for_fit['radius'], model3_l, model3_h, color='C2', alpha=0.3)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
#axs[0].set_xlabel('$R$ [Mpc]')
axs[0].set_ylabel('$g_t$')
axs[0].legend()

axs[1].errorbar(data_for_fit['radius'], model1/data_for_fit['gt']-1, xerr=None, yerr=abs(model1*data_for_fit['gt_err']/data_for_fit['gt']**2), color='C0', marker='o', ms=1, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=1)
axs[1].errorbar((1+0.025)*data_for_fit['radius'], model2/data_for_fit['gt']-1, xerr=None, yerr=abs(model2*data_for_fit['gt_err']/data_for_fit['gt']**2), color='C1', marker='o', ms=1, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=1)
axs[1].errorbar((1+0.050)*data_for_fit['radius'], model3/data_for_fit['gt']-1, xerr=None, yerr=abs(model3*data_for_fit['gt_err']/data_for_fit['gt']**2), color='C2', marker='o', ms=1, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=1)
axs[1].set_xlabel('$R$ [Mpc]')
axs[1].set_ylabel('$g_t^{mod}/g_t^{data}-1$')
axs[1].set_ylim(-0.6,0.6)

plt.tight_layout(pad=0.1)
plt.show()
