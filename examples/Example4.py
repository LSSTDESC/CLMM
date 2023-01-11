

##|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
##\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\| USING CLMM ON REAL DATASETS |//////////////////////////////////////////
##A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAVAVAV////////////////////////////////////////////////////A
##AA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAVAV////////////////////////////////////////////////////AA
##AAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAV////////////////////////////////////////////////////AAA
##AAAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAV////////////////////////////////////////////////////AAAA
##AAAAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V////////////////////////////////////////////////////AAAAA

## DEMONSTRATE HOW TO RUN CLMM ON REAL OBSERVATIONAL DATASETS.
## AS AN EXAMPLE, USE THE DATA FROM THE Hyper Suprime-Cam Subaru Strategic Program (HSC SSP) PUBLIC RELEASES WHICH
## HAVE SIMILAR OBSERVATION CONDITIONS AND DATA FROMATS TO THE RUBIN LSST.

## OUTLINE:
##      @ SET THINGS UP
##      @ SELECT A CLUSTER
##      @ DOWNLOAD THE PUBLISHED CATALOG AT THE CLUSTER FIELD
##      @ LOAD THE CATALOG INTO CLMM
##      @ RUN CLMM ON THE DATASET

## LINKS:
##      @ DATA ACCESS OF THE HSC SSP PUBLIC DATA RELEASE: https://hsc-release.mtk.nao.ac.jp/doc/index.php/data-access__pdr3/
##      @ SHAPE CATALOG: https://hsc-release.mtk.nao.ac.jp/doc/index.php/s16a-shape-catalog-pdr2/
##      @ FAQ: https://hsc-release.mtk.nao.ac.jp/doc/index.php/faq__pdr3/
##      @ PHOTOMETRIC REDSHIFTS: https://hsc-release.mtk.nao.ac.jp/doc/index.php/photometric-redshifts/
##      @ CLUSTER CATALOG: https://hsc-release.mtk.nao.ac.jp/doc/index.php/camira_pdr2/



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||||||||||||||| SETUP ||||||||||||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import pickle as pkl
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', message='.*(!).*')


##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||||||||| SELECTING A CLUSTER |||||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## WE USE THE HSC SSP PUBLICATIONS (https://hsc.mtk.nao.ac.jp/ssp/publications/) TO SELECT A LIST OF REPORTED MASSIVE
## GALAXY CLUSTERS THAT HAVE BEEN MEASURED BY WEAK LENSING.
## IN THE TABLE BELOW, WE ASSUME h=0.7 AND THE COORDINATES ARE GIVEN FOR THE LENSING PEAKS UNLESS OTHERWISE STATED.
## WL COLUMN IS GIVEN IN UNITS [M_{200,500c}(1E14*M_sun)]

## NAME             z_cl        RA(deg)     DEC(deg)    WL          REF.                COMMENTS
## HWL16a-094       0.592       223.0801    0.1689      15.3, 7.8   HAMANA+2020         CAMIRA ID 1417; MIYAZAKI+2018 RANK 34
## HWL16a-026       0.424       130.5895    1.6473      8.7, 4.7    HAMANA+2020         --
## HWL16a-034       0.315       139.0387    -0.3966     8.1, 5.6    HAMANA+2020         ABELL 776; MACS J0916.1-0023; MIYAZAKI+2018 RANK 8; SEE ALSO MEDEZINSKI+2018
## Rank 9           0.312       37.3951     -3.6099     --, 5.9     MIYAZAKI+2018       --
## Rank 48          0.529       220.7900    1.0509      --, 10.4    MIYAZAKI+2018       --
## Rank 62          0.592       216.6510    0.7982      --, 10.2    MIYAZAKI+2018       --
## MaxBCG           0.2701      140.54565   3.77820     44.3, 25.1  MEDEZINSKI+2018     BCG CENTER (CLOSE TO THE X-RAY CENTER); PSZ2 G228.50+34.95; DOUBLE BCGs
##  J140.53188+03.76632
## XLSSC006         0.429       35.439      -3.772      9.6, 5.6    UMETSU+2020         X-RAY CENTER




##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||| DOWNLOADING THE CATALOG AT THE CLUSTER FIELD ||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## THE 3 MOST MASSIVE CLUSTER-CANDIDATES ARE MAXBCG J140.53188+03.76632, MIYAZAKI+2018 (M18 HEREAFTER) RANK 48 AND 62.
## WE USE object_id TO CROSS MATCH THE SHAPE CATALOG, PHOTO-Z CATALOG, AND PHOTOMETRY CATALOG.
## SINCE THE CLUSTERS ARE AT REDSHIFT OF ABOUT 0.4, A RADIUS OF 10 armin WOULD BE ABOUT 3Mpc.
## HOWEVER, WE MAKE A QUERY FOR THE WHOLE FIELD TO SAVE TIME.
## THE FINAL CATALOG INCLUDES SHAPE INFO, PHOTO-Z, AND PHOTOMETRY.
## BELOW IS AN EXAMPLE OF THE QUERY SQL COMMAND -- COULD TAKE UP TO AN HOUR AND PRODUCE A FILE ~400MB (.csv.gz).
## THE REQUEST IS MADE HERE: https://hsc-release.mtk.nao.ac.jp/datasearch/helps/sql_search

##  select
##   b.*, 
##   c.ira, c.idec, 
##   a.ishape_hsm_regauss_e1, a.ishape_hsm_regauss_e2, 
##   a.ishape_hsm_regauss_resolution, a.ishape_hsm_regauss_sigma, 
##   d1.photoz_best as ephor_ab_photoz_best, d1.photoz_risk_best as ephor_ab_photoz_risk_best, 
##   d2.photoz_best as frankenz_photoz_best, d2.photoz_risk_best as frankenz_photoz_risk_best, 
##   d3.photoz_best as nnpz_photoz_best, d3.photoz_risk_best as nnpz_photoz_risk_best, 
##   e.icmodel_mag, e.icmodel_mag_err, 
##   e.detect_is_primary, 
##   e.iclassification_extendedness, 
##   e.icmodel_flux_flags, 
##   e.icmodel_flux, e.icmodel_flux_err, 
##   c.iblendedness_abs_flux
##  from
##   s16a_wide.meas2 a
##   inner join s16a_wide.weaklensing_hsm_regauss b using (object_id)
##   inner join s16a_wide.meas c using (object_id)
##   -- inner join s16a_wide.photoz_demp d using (object_id)
##   -- inner join s16a_wide.photoz_ephor d using (object_id)
##    inner join s16a_wide.photoz_ephor_ab d1 using (object_id)
##    inner join s16a_wide.photoz_frankenz d2 using (object_id)
##   -- inner join s16a_wide.photoz_mizuki d using (object_id)
##   -- inner join s16a_wide.photoz_mlz d using (object_id)
##    inner join s16a_wide.photoz_nnpz d3 using (object_id)
##    inner join s16a_wide.forced e using (object_id)
##  -- Uncomment the specific lines depending upon the field to be used
##   -- where s16a_wide.search_xmm(c.skymap_id)
##   -- where s16a_wide.search_wide01h(c.skymap_id)
##   -- where s16a_wide.search_vvds(c.skymap_id)
##   -- where s16a_wide.search_hectomap(c.skymap_id)
##   -- where s16a_wide.search_gama15h(c.skymap_id)
##   where s16a_wide.search_gama09h(c.skymap_id)
##   --AND e.detect_is_primary
##   --AND conesearch(c.icoord, 140.54565, 3.77820, 600) 
##   --AND NOT e.icmodel_flux_flags
##   --AND e.iclassification_extendedness>0.5
##   --LIMIT 5

import os
cwd = os.getcwd()
filename = cwd + '/datasets/283416.csv.gz.0'
catalog = filename.replace('.csv', '.pkl')
if not Path(catalog).is_file() :
    data = Table.read(filename, format='ascii.csv')
    pkl.dump(data, open(catalog, 'wb'))
else :
    data = pkl.load(open(catalog, 'rb'))

print(data.colnames)


## WE SELECT THE 'frankenz' FOR THE TEST, BUT THERE ARE OTHER MOTHODS: ('nnpz', 'ephor_ab')
photoz_type = 'frankenz'

## CUTS
def make_cuts(catalog_in) :
    ## HERE WE CONSIDER THE CUTS MADE IN MANDELBAUM ET AL. 2018 (HSC SSP Y1 SHEAR CATALOG).
    select  = catalog_in['detect_is_primary'] == 'True'
    select &= catalog_in['icmodel_flux_flags'] == 'False'
    select &= catalog_in['iclassification_extendedness'] > 0.5
    select &= catalog_in['icmodel_mag_err']  <= 2.5/np.log(10.)/10.
    select &= catalog_in['ishape_hsm_regauss_e1']**2 + catalog_in['ishape_hsm_regauss_e2']**2 < 4.
    select &= catalog_in['icmodel_mag'] <= 24.5
    select &= catalog_in['iblendedness_abs_flux'] < (10**(-0.375))
    select &= catalog_in['ishape_hsm_regauss_resolution'] >= 0.3
    select &= catalog_in['ishape_hsm_regauss_sigma'] <= 0.4
    select &= catalog_in['%s_photoz_risk_best'%photoz_type] < 0.5

    catalog_out = catalog_in[select]
    return catalog_out

data = make_cuts(data)
print(len(data))




## REFERENCE: MANDELBAUM ET AL. 2018 'THE FIRST-YEAR SHEAR CATALOG OF THE SUBARU HYPER SUPRIME-CAM SUBARU STRATEGIC
## PROGRAM SURVEY'.
## SECTION A.3.2: 'PER-OBJECT GALAXY SHEAR ESTIMATE'
def apply_shear_calibration(catalog_in) :
    e1 = catalog_in['ishape_hsm_regauss_e1']
    e2 = catalog_in['ishape_hsm_regauss_e2']
    e_rms = catalog_in['ishape_hsm_regauss_derived_rms_e']
    m = catalog_in['ishape_hsm_regauss_derived_shear_bias_m']
    c1 = catalog_in['ishape_hsm_regauss_derived_shear_bias_c1']
    c2 = catalog_in['ishape_hsm_regauss_derived_shear_bias_c2']
    ## NOTE: IN THE MASS FIT WE HAVE YET TO IMPLEMENT THE WEIGHTS.
    weight = catalog_in['ishape_hsm_regauss_derived_shape_weight']

    R = 1. - np.sum(weight * e_rms**2) / np.sum(weight)
    m_mean = np.sum(weight * m) / np.sum(weight)
    c1_mean = np.sum(weight * c1) / np.sum(weight)
    c2_mean = np.sum(weight * c2) / np.sum(weight)
    print('R, m_mean, c1_mean, c2_mean: ', R, m_mean, c1_mean, c2_mean)

    g1 = (e1 / (2.*R) - c1) / (1. + m_mean)
    g2 = (e2 / (2.*R) - c2) / (1. + m_mean)

    return g1, g2

## ADJUST COLUMN NAMES.
def adjust_column_names(catalog_in) :
    column_name_map = {
            'ra': 'ira',
            'dec': 'idec',
            'z': '%s_photoz_best'%photoz_type,
            'id': '# object_id',}
    catalog_out = Table()
    for i in column_name_map :
        catalog_out[i] = catalog_in[column_name_map[i]]

    g1, g2 = apply_shear_calibration(catalog_in)
    catalog_out['e1'] = g1
    catalog_out['e2'] = g2

    return catalog_out

data = adjust_column_names(data)
print(data.colnames)


## MAKE SOME FIGURES.
def make_plots(catalog_in) :
    ## SCATTER
    plt.figure()
    plt.scatter(catalog_in['ra'], catalog_in['dec'], c=catalog_in['z'], s=1., alpha=0.2)
    plt.colorbar()
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.title('z')
    plt.tight_layout()

    ## HISTOGRAM
    plt.figure()
    plt.hist(catalog_in['z'], bins='scott')
    plt.xlabel('z')
    plt.ylabel('count')
    plt.tight_layout()

    ## RELATION
    plt.figure()
    plt.plot(catalog_in['e1'], catalog_in['e2'], ',')
    plt.xlabel('e1')
    plt.ylabel('e2')
    plt.tight_layout()

    fig, axs = plt.subplots(1,2, figsize=(8,5), sharey=True)
    axs[0].hist(catalog_in['e1'], bins='scott')
    axs[0].set_xlabel('e1')
    axs[1].hist(catalog_in['e2'], bins='scott')
    axs[1].set_xlabel('e2')
    plt.tight_layout()


    plt.show()

make_plots(data)




##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||||| RUNNING CLMM ON THE DATASET |||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

from clmm import Cosmology
H0 = 70.
Om_b = 0.045
Om_dm = 0.27 - Om_b
Om_k = 0.0
cosmo = Cosmology(H0=H0, Omega_dm0=Om_dm, Omega_b0=Om_b, Omega_k0=Om_k)

## CONSIDER THE CLUSTER: MaxBCG J140.53188+03.76632
cl_z   = 0.2701
cl_ra  = 140.54565
cl_dec = 3.77820

obs_galaxies = data
obs_galaxies = obs_galaxies[(obs_galaxies['z'] > (cl_z + 0.1))]

## AREA CUT
select = obs_galaxies['ra'] < cl_ra + 0.2/np.cos(cl_dec/180.*np.pi)
select = obs_galaxies['ra'] > cl_ra - 0.2/np.cos(cl_dec/180.*np.pi)
select = obs_galaxies['dec'] < cl_dec + 0.2
select = obs_galaxies['dec'] > cl_dec - 0.2
obs_galaxies = obs_galaxies[select]

obs_galaxies['id'] = np.arange(len(obs_galaxies))

## THROW THE GALAXY VALUES INTO ARRAYS.
gal_ra  = obs_galaxies['ra']
gal_dec = obs_galaxies['dec']
gal_e1  = obs_galaxies['e1']
gal_e2  = obs_galaxies['e2']
gal_z   = obs_galaxies['z']
gal_id  = obs_galaxies['id']



##wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
##|||||||||||||||||||||||||||||||||||||||||||||||||| SHEAR PROFILE ||||||||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

import clmm.dataops as da

## CONVERT ELIPTICITIES INTO SHEARS
gal_ang_dist, gal_gt, gal_gx = da.compute_tangential_and_cross_components(cl_ra, cl_dec, gal_ra, gal_dec, gal_e1, gal_e2)

## MEASURE PROFILE
field_size = 6.     ## IN Mpc
profile = da.make_radial_profile(
        [gal_gt, gal_gx, gal_z],
        gal_ang_dist,
        'radians',
        'Mpc',
        bins=da.make_bins(0.01, field_size/2., 50),
        cosmo=cosmo,
        z_lens=cl_z,
        include_empty_bins=False)

print(f'Profile table has columns: {", ".join(profile.colnames)},')
print('where p_(0, 1, 2) = (gt, gx, z)')


## USING THE GalaxyCluster OBJECT
import clmm
from clmm.utils import convert_units

galaxies = clmm.GCData([gal_ra, gal_dec, gal_e1, gal_e2, gal_z, gal_id], names=['ra', 'dec', 'e1', 'e2', 'z', 'id'])
cluster = clmm.GalaxyCluster('Name of Cluster', cl_ra, cl_dec, cl_z, galaxies)

cluster.compute_tangential_and_cross_components()   ## CONVERT ELIPTICITIES TO SHEARS
print(cluster.galcat.colnames)

seps = convert_units(cluster.galcat['theta'], 'radians', 'Mpc', cluster.z, cosmo)   ## MEASURE PROFILE

cluster.make_radial_profile(
        bins=da.make_bins(0.3, field_size/2., 15, method='evenlog10width'),
        bin_units='Mpc',
        cosmo=cosmo,
        include_empty_bins=False,
        gal_ids_in_bins=True)
print(cluster.profile.colnames)

import sys
sys.path.append('Paper_v1.0')
from paper_formating import prep_plot
fig = prep_plot(figsize=(9,9))
ax = fig.add_axes((0,0,1,1))
errorbar_kwargs = dict(
        linestyle='',
        marker='o',
        markersize=1,
        elinewidth=.5,
        capthick=.5)
ax.errorbar(cluster.profile['radius'], cluster.profile['gt'], yerr=cluster.profile['gt_err'], c='k', **errorbar_kwargs)
ax.set_xlabel('r [Mpc]', fontsize=10)
ax.set_ylabel(r'$g_t$', fontsize=10)
plt.show()



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||||||| THEORETICAL PREDICTIONS |||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## MODEL 1: 
## RELIES ON THE OVERALL REDSHIFT DISTRIBUTION OF THE SOURCES (WtG III APPLEGATE ET AL. 2014).
## NOTICE THAT THE CONCENTRATION OF MaxBCG J140.53188+03.76632 WAS NOT REPORTED.
## THE VALUE FROM THE STACKED SAMPLE IN THE PAPER IS ~7.
## FOR THE MASS SCALE, A TYPICAL c-M RELATION (e.g. CHILD ET AL. 2018) WOULD GIVE c~3 THOUGH.
## WE HAVE NOT CONSIDERED A c-M RELATION IN THE FITTING.
z_inf = 1.e3
dl_inf = cosmo.eval_da_z1z2(cl_z, z_inf)
d_inf = cosmo.eval_da(z_inf)
cl_conc = 4.

def betas(z) :
    dls = cosmo.eval_da_z1z2(cl_z, z)
    ds = cosmo.eval_da(z)
    return dls*d_inf / (ds*dl_inf)

def predict_reduced_tangential_shear_redshift_distribution(profile, logm) :
    bs_mean = np.mean(betas(cluster.galcat['z']))
    bs2_mean = np.mean(betas(cluster.galcat['z'])**2)

    gamma_t_inf = clmm.compute_tangential_shear(
            r_proj = profile['radius'],     ## RADIAL COMPONENT OF THE PROFILE
            mdelta = 10.**logm,             ## MASS OF THE CLUSTER [M_sun]
            cdelta = cl_conc,               ## CONCENTRATION OF THE CLUSTER
            z_cluster = cl_z,               ## REDSHIF OF THE CLUSTER
            z_source = z_inf,               ## RESHIFT VALUE AT INFINITY
            cosmo = cosmo,
            delta_mdef = 200,
            massdef = 'critical',           ## FOR M200c
            halo_profile_model = 'nfw')
    convergence_inf = clmm.compute_convergence(
            r_proj = profile['radius'],     ## RADIAL COMPONENT OF THE PROFILE
            mdelta = 10.**logm,             ## MASS OF THE CLUSTER [M_sun]
            cdelta = cl_conc,               ## CONCENTRATION OF THE CLUSTER
            z_cluster = cl_z,               ## REDSHIF OF THE CLUSTER
            z_source = z_inf,               ## RESHIFT VALUE AT INFINITY
            cosmo = cosmo,
            delta_mdef = 200,
            massdef = 'critical',           ## FOR M200c
            halo_profile_model = 'nfw')

    return bs_mean*gamma_t_inf / (1 - (bs2_mean/bs_mean) *convergence_inf)


## MODEL 2:
## USES INDIVIDUAL REDSHIFT AND RADIAL INFORMATION TO COMPUTE THE AVERAGE SHEAR IN EACH RADIAL BIN BASED ON THE
## GALAXIES ACTUALLY PRESENT IN THAT BIN.
cluster.galcat['theta_mpc'] = convert_units(cluster.galcat['theta'], 'radians', 'Mpc', cluster.z, cosmo)

def predict_reduced_tangential_shear_individual_redshift(profile, logm) :
    return np.array([np.mean(clmm.compute_reduced_tangential_shear(
        r_proj = cluster.galcat[radial_bin['gal_id']]['theta_mpc'],     ## RADIAL COMPONENT OF SOURCE GALAXIES IN BIN
        mdelta = 10.**logm,
        cdelta = cl_conc,
        z_cluster = cl_z,
        z_source = cluster.galcat[radial_bin['gal_id']]['z'],   ## REDSHIFT VALUE OF SOURCE GALAXIES IN BIN
        cosmo = cosmo,
        delta_mdef = 200,
        massdef = 'critical',
        halo_profile_model = 'nfw')) for radial_bin in profile])



## MASS FITTING

mask_for_fit = cluster.profile['n_src'] > 2
data_for_fit = cluster.profile[mask_for_fit]

from clmm.support.sampler import fitters
def fit_mass(predict_function) :
    popt, pcov = fitters['curve_fit'](predict_function, data_for_fit, data_for_fit['gt'], data_for_fit['gt_err'],
            bounds=[10., 17.])
    logm, logm_err = popt[0], np.sqrt(pcov[0][0])
    return {'logm':logm, 'logm_err':logm_err, 'm':10.**logm, 'm_err':(10.**logm)*logm_err*np.log(10.)}

## IN THE PAPER THE MEASURED MASS IS 44.3 {+30.3}{-19.9} * 10^14 M_sun (M200c, WL).
## FOR CONVENIENCE, WE CONSIDER A MEAN VALUE FOR THE ERRORBAR.
## WE BUILD A DICTIONARY BASED ON THAT RESULT.
m_paper = 44.4e14
m_err_paper = 25.1e14
logm_paper = np.log10(m_paper)
logm_err_paper = m_err_paper / (10.**logm_paper) / np.log(10.)
paper_value = {'logm':logm_paper, 'logm_err':logm_err_paper, 'm':10.**logm_paper,
        'm_err':(10.**logm_paper)*logm_err_paper*np.log(10.)}

fit_redshift_distribution = fit_mass(predict_reduced_tangential_shear_redshift_distribution)
fit_individual_redshift = fit_mass(predict_reduced_tangential_shear_individual_redshift)

print('Best fit mass for N(z) model = {:.3e} +- {:.3e}'.format(
    fit_redshift_distribution['m'], fit_redshift_distribution['m_err']))

print('Best fit mass for individual redshift and radius = {:.3e} +- {:.3e}'.format(
    fit_individual_redshift['m'], fit_individual_redshift['m_err']))


def get_predicted_shear(predict_function, fit_values) :
    gt_est = predict_function(data_for_fit, fit_values['logm'])
    gt_est_err = [predict_function(data_for_fit, fit_values['logm'] + i*fit_values['logm_err']) for i in (-3,3)]
    return gt_est, gt_est_err

gt_redshift_distribution, gt_err_redshift_distribution = get_predicted_shear(
        predict_reduced_tangential_shear_redshift_distribution, fit_redshift_distribution)
gt_individual_redshift, gt_err_individual_redshift = get_predicted_shear(
        predict_reduced_tangential_shear_individual_redshift, fit_individual_redshift)

gt_paper1, gt_err_paper1 = get_predicted_shear(predict_reduced_tangential_shear_redshift_distribution, paper_value)
gt_paper2, gt_err_paper2 = get_predicted_shear(predict_reduced_tangential_shear_individual_redshift, paper_value)

chi2_redshift_distribution_dof = np.sum((gt_redshift_distribution-data_for_fit['gt'])**2/(data_for_fit['gt_err'])**2)/(
        len(data_for_fit)-1)
chi2_individual_redshift_dof = np.sum((gt_individual_redshift-data_for_fit['gt'])**2/(data_for_fit['gt_err'])**2)/(
        len(data_for_fit)-1)

print('Reduced chi2 (N(z) model)             = {:.3e}'.format(chi2_redshift_distribution_dof))
print('Reduced chi2 (individual (R,z) model) = {:.3e}'.format(chi2_individual_redshift_dof))


from matplotlib.ticker import MultipleLocator
fig = prep_plot(figsize=(9,9))
gt_ax = fig.add_axes((.25, .42, .7, .55))
gt_ax.errorbar(data_for_fit['radius'], data_for_fit['gt'], data_for_fit['gt_err'], c='k', **errorbar_kwargs)
## POINTS IN GREY HAVE NOT BEEN USED FOR THE FIT
gt_ax.errorbar(cluster.profile['radius'][~mask_for_fit], cluster.profile['gt'][~mask_for_fit],
        cluster.profile['gt_err'][~mask_for_fit], c='grey', **errorbar_kwargs)

pow10 = 15.
mlabel = lambda name, fits: fr'$M_{{fit}}^{{{name}}} = {fits["m"]/10.**pow10:.3f}\pm{fits["m_err"]/10.**pow10:.3f}\times10^{{{pow10}}} M_\odot$'

gt_ax.loglog(data_for_fit['radius'], gt_redshift_distribution, '-C1', label=mlabel('N(z)', fit_redshift_distribution), lw=0.5)
gt_ax.fill_between(data_for_fit['radius'], *gt_err_redshift_distribution, lw=0, color='C1', alpha=0.2)

gt_ax.loglog(data_for_fit['radius'], gt_individual_redshift, '-C2', label=mlabel('z,R', fit_individual_redshift), lw=0.5)
gt_ax.fill_between(data_for_fit['radius'], *gt_err_individual_redshift, lw=0, color='C2', alpha=0.2)

gt_ax.loglog(data_for_fit['radius'], gt_paper1, '-C3', label=mlabel('paper; N(z)', paper_value), lw=0.5)
gt_ax.fill_between(data_for_fit['radius'], *gt_err_paper1, lw=0, color='C3', alpha=0.2)

gt_ax.loglog(data_for_fit['radius'], gt_paper2, '-C4', label=mlabel('paper; z,R', paper_value), lw=0.5)
gt_ax.fill_between(data_for_fit['radius'], *gt_err_paper2, lw=0, color='C4', alpha=0.2)

gt_ax.set_ylabel(r'$g_t$', fontsize=5)
gt_ax.legend(fontsize=4)
gt_ax.set_xticklabels([])
gt_ax.tick_params('x', labelsize=5)
gt_ax.tick_params('y', labelsize=5)

errorbar_kwargs2 = {k:v for k, v in errorbar_kwargs.items() if 'marker' not in k}
errorbar_kwargs2['markersize'] = 3
errorbar_kwargs2['markeredgewidth'] = 0.5
res_ax = fig.add_axes((0.25, 0.2, 0.7, 0.2))
delta = (cluster.profile['radius'][1]/cluster.profile['radius'][0])**0.25

res_ax.errorbar(data_for_fit['radius'], data_for_fit['gt']/gt_redshift_distribution-1,
        yerr=data_for_fit['gt_err']/gt_redshift_distribution, marker='s', c='C1', **errorbar_kwargs2)
errorbar_kwargs2['markersize'] = 3
errorbar_kwargs2['markeredgewidth'] = 0.5

res_ax.errorbar(data_for_fit['radius']*delta, data_for_fit['gt']/gt_individual_redshift-1,
        yerr=data_for_fit['gt_err']/gt_individual_redshift, marker='*', c='C2', **errorbar_kwargs2)

res_ax.set_xlabel(r'$R$ [Mpc]', fontsize=5)
res_ax.set_ylabel(r'$g_t^{data}/g_t^{model}-1$', fontsize=5)
res_ax.set_xscale('log')
res_ax.set_ylim(-5,5)
res_ax.yaxis.set_minor_locator(MultipleLocator(10))
res_ax.tick_params('x', labelsize=5)
res_ax.tick_params('y', labelsize=5)
plt.show()
