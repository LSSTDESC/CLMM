
##|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
##\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\| GENERATE MOCK DATA |///////////////////////////////////////////////
##A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAVAV/////////////////////////////////////////////////////A
##AA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAVAV/////////////////////////////////////////////////////AA
##AAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\VAV/////////////////////////////////////////////////////AAA
##AAAA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V/////////////////////////////////////////////////////AAAA

## GENERATE MOCK DATA WITH A VARIETY OF SYSTEMATIC EFFECTS INCLUDING:
##      @ PHOTOMETRIC REDSHIFTS
##      @ SOURCE GALAXY DISTRIBUTIONS
##      @ SHAPE NOISE
## OUTLINE:
##      @ IMPORTS & CONFIGURATION SETUP
##      @ GENERATE MOCK DATA WITH DIFFERENT SOURCE GALAXY OPTIONS
##      @ GENERATE MOCK DATA WITH DIFFERENT FIELD OF VIEW OPTIONS
##      @ GENERATE MOCK DATA WITH DIFFERENT GALAXY CLUSTER OPTIONS
## THE LAST STEP IS ONLY AVAILABLE USING THE Numcosmo OR CCL BACKENDS.
## USE THE os.environ['CLMM_MODELING_BACKEND'] LINE BELOW TO SELECT THE BACKEND.


##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||||||||||||||| SETUP ||||||||||||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

import warnings
warnings.filterwarnings('ignore', message='.*(!).*')
import os

## CHOOSE THE MODELING BACKEND: Numcosmo = 'nc'; CCL = 'ccl' (DEFAULT); cluster-toolkit = 'ct'
#os.environ['CLMM_MODELING_BACKEND'] = 'nc'


import clmm
import numpy as np
import matplotlib.pyplot as plt

from clmm.support import mock_data as mock
from clmm import Cosmology

np.random.seed(42)

##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||||||||| CLUSTER CONFIGURATION |||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

H0 = 70.
Omega_b0  = 0.045
Omega_dm0 = 0.27 - Omega_b0
Omega_k0  = 0.
mock_cosmo = Cosmology(H0=H0, Omega_dm0=Omega_dm0, Omega_b0=Omega_b0, Omega_k0=Omega_k0)

## NOW CHOOSE THE CLUSTER INFORMATION.
## DEFAULT IS TO WORK WITH AN NFW PROFILE USING THE '200,mean' MASS DEFINITION.
## THE Numcosmo AND CCL BACKENDS ALLOW FOR MORE FLEXIBILITY.

cosmo = mock_cosmo
cl_id = 'Awesome_cluster'
cl_mass = 1.e15     ## M200,m
cl_z = 0.3
src_z = 0.8
cl_conc = 4
ngals = int(1e3)
cl_ra = 50.
cl_dec = 87.



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||||| GENERATE MOCK DATA CATALOGS ||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV


zsrc_min = cl_z + 0.1   ## BACKGROUND GALAXIES MUST BE BEHIND CLUSTER

## CLEAN DATA: NO NOISE; ALL GALAXIES AT THE SAME REDSHIFT
ideal_data = mock.generate_galaxy_catalog(cl_mass, cl_z, cl_conc, cosmo, src_z, ngals=ngals, cluster_ra=cl_ra, cluster_dec=cl_dec)

## NOISY DATA: SHAPE NOISE; ALL GALAXIES AT THE SAME REDSHIFT
noisy_data_src_z = mock.generate_galaxy_catalog(cl_mass, cl_z, cl_conc, cosmo, src_z, ngals=ngals, cluster_ra=cl_ra, cluster_dec=cl_dec, shapenoise=0.05)

## NOISY DATA: SHAPE NOISE PLUS MEASUREMENT ERROR; ALL GALAXIES AT THE SAME REDSHIFT
noisy_data_src_z_e_err = mock.generate_galaxy_catalog(cl_mass, cl_z, cl_conc, cosmo, src_z, ngals=ngals, cluster_ra=cl_ra, cluster_dec=cl_dec, shapenoise=0.05, mean_e_err=0.05)

## NOTE: UNCERTAINTIES ARE CREATED BY SIMPOLY DRAWING RANDOM NUMBERS NEAR THE VALUE SPECIFIED BY mean_e_err.
##       USE AT YOUR OWN RISK.
##       THIS WILL BE IMPROVED IN FUTURE RELEASES.

## NOISY DATA: PHOTO-Z ERRORS (AND PDFS); ALL GALAXIES AT THE SAME REDSHIFT
noisy_data_photoz = mock.generate_galaxy_catalog(cl_mass, cl_z, cl_conc, cosmo, src_z, ngals=ngals, cluster_ra=cl_ra, cluster_dec=cl_dec, shapenoise=0.05, photoz_sigma_unscaled=0.05)

## CLEAN DATA: SOURCE GALAXY REDHSHIFTS DRAWN FROM A REDSHIFT DISTRIBUTION ('chang13' OR 'desc_srd') INSTEAD OF A
##             FIXED VALUE; NO SHAPE NOISE OR PHOTOZ ERRORS
ideal_with_src_dist = mock.generate_galaxy_catalog(cl_mass, cl_z, cl_conc, cosmo, 'chang13', zsrc_min=zsrc_min, zsrc_max=7., ngals=ngals, cluster_ra=cl_ra, cluster_dec=cl_dec)

## NOISY DATA: GALAXIES FOLLOWING REDSHIFT DISTRIBUTION; REDSHIFT ERROR; SHAPE NOISE
allsystematics1 = mock.generate_galaxy_catalog(
        cl_mass, cl_z, cl_conc, cosmo, 'chang13', zsrc_min=zsrc_min, ngals=ngals, cluster_ra=cl_ra, cluster_dec=cl_dec, photoz_sigma_unscaled=0.05)
allsystematics2 = mock.generate_galaxy_catalog(
        cl_mass, cl_z, cl_conc, cosmo, 'desc_srd', zsrc_min=zsrc_min, zsrc_max=7., ngals=ngals, cluster_ra=cl_ra, cluster_dec=cl_dec, shapenoise=0.05, photoz_sigma_unscaled=0.05)


## SANITY CHECK: CHECKING THAT NO GALAXIES WERE ORIGINALLY DRAWN BELOW zsrc_min, BEFORE PHOTOZ ERRORS ARE APPLIED (WHEN RELEVANT)
print('Number of galaxes below zsrc_min: ')
print('         ideal_data: {}'.format(np.sum(ideal_data['ztrue']<zsrc_min)))
print('   noisy_data_src_z: {}'.format(np.sum(noisy_data_src_z['ztrue']<zsrc_min)))
print('  noisy_data_photoz: {}'.format(np.sum(noisy_data_photoz['ztrue']<zsrc_min)))
print('ideal_with_src_dist: {}'.format(np.sum(ideal_with_src_dist['ztrue']<zsrc_min)))
print('    allsystematics1: {}'.format(np.sum(allsystematics1['ztrue']<zsrc_min)))



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||||||| INSPECT THE CATALOG DATA ||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## IDEAL CATALOG FIRST ENTRIES: NO NOISE ON THE SHAPE MEASUREMENT; ALL GALAXIES AT z=0.8; NO REDSHIFT ERRORS
for n in ideal_data.colnames :
    if n != 'id' :
        ideal_data[n].format = '%6.3e'

print(ideal_data[0:3])


## WITH PHOTO-Z ERRORS:
for n in noisy_data_photoz.colnames :
    if n != 'id' :
        noisy_data_photoz[n].format = '%6.3e'

print(noisy_data_photoz[0:3])


## HISTOGRAM OF THE RESHIFT DISTRIBUTION OF BACKGROUND GALAXIES.
## BY CONSTRUCTION, NO TRUE RESHIFTS OCCUR BELOW zsrc_min BUT SOME OBSERVED REDSHIFTS MIGHT BE.

fig, axs = plt.subplots(2,1, figsize=(5,8), sharex=True)
axs[0].hist(allsystematics1['z'], bins=50, alpha=0.3, density=True, label='measured z; chang13')
axs[0].hist(allsystematics1['ztrue'], bins=50, alpha=0.3, density=True, label='true z; chang13')
axs[0].axvline(zsrc_min, color='r', label='requested zmin')
axs[0].legend()

axs[1].hist(allsystematics2['z'], bins=50, alpha=0.3, density=True, label='measured z; desc_srd')
axs[1].hist(allsystematics2['ztrue'], bins=50, alpha=0.3, density=True, label='true z; desc_srd')
axs[1].axvline(zsrc_min, color='r', label='requested zmin')
axs[1].set_xlabel('Source Redshift')
axs[1].legend()

plt.tight_layout()

plt.show()



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||||||||| POPULATE A GALAXY CLUSTER OBJECT ||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

gc_object = clmm.GalaxyCluster(cl_id, cl_ra, cl_dec, cl_z, allsystematics1)

## FROM A GalaxyCluster OBJECT THAT HAS PHOTOZ INFORMATION, draw_gal_z_from_pdz ALLOWS TO GENERATE nobj RANDOM
## REDSHIFTS OF EACH GALAXY IN galacat, FROM ITS PHOTOZ PDF, AND STORE THE RESULT IN A NEW zcol_out COLUMN.
z_random = gc_object.draw_gal_z_from_pdz(zcol_out='z_random', overwrite=False, nobj=1)

## BELOW PLOT:
##      @ BLUE: OBSERVED PHOTOZ PDF
##      @ RED: OBSERVED Z
##      @ GREEN: TRUE REDSHIFT FROM WHICH THE SHEAR WERE COMPUTED
##      @ ORANGE: RANDOM REDSHIFT COMPUTED FROM THE PDF

galid = 0
plt.plot(gc_object.galcat['pzbins'][galid], gc_object.galcat['pzpdf'][galid], label='Photoz pdf')
plt.axvline(gc_object.galcat['z'][galid], label='Observed z', color='r')
plt.axvline(gc_object.galcat['ztrue'][galid], label='True z', color='g')
plt.axvline(gc_object.galcat['z_random'][galid], label='Random z from pdf', color='orange')
plt.xlabel('Redshft')
plt.ylabel('Photo-z Probability Distribution')
plt.legend(loc=1)

plt.tight_layout()

plt.show()


## PLOT SOURCE GALAXY ELLIPTICITIES
plt.scatter(gc_object.galcat['e1'], gc_object.galcat['e2'])
plt.xlim(-0.2, 0.2)
plt.ylim(-0.2, 0.2)
plt.xlabel('Ellipticity 1', fontsize='x-large')
plt.ylabel('Ellipticity 2', fontsize='x-large')

plt.tight_layout()

plt.show()



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||| GENERATE MOCK DATA CATALOGS WITH DIFFERENT FIELD-OF-VIEW OPTIONS ||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## ABOVE WE HAD SIMULATED ngals=1e3 GALAXIES IN A FIELD CORRESPONDING TO AN (8Mpc)X(8Mpc) PROPER DISTANCE SQUARE BOX
## AT THE CLUSTER REDSHIFT (DEFAULT).
## THE USER MAY INSTEAD VARY THE FIELD SIZE AND/OR PROVIDE A GALAXY DENSITY INSTEAD OF ngals.
## THIS IS WHAT WE DO BELOW.

## ngals=1e3 IN A (4Mpc)X(4Mpc) BOX
allsystematics2 = mock.generate_galaxy_catalog(
        cl_mass, cl_z, cl_conc, cosmo, 'chang13', zsrc_min=zsrc_min, zsrc_max=7., field_size=4., ngals=ngals, cluster_ra=cl_ra, cluster_dec=cl_dec, shapenoise=0.05, photoz_sigma_unscaled=0.05)

plt.scatter(allsystematics1['ra'], allsystematics1['dec'], marker='.', s=2, label='default 8x8 Mpc FoV')
plt.scatter(allsystematics2['ra'], allsystematics2['dec'], marker='.', s=2, label='user-defined FoV')
plt.legend()

plt.show()

## ALTERNATIVELY, THE USER MAY PROVIDE A GALAXY DENSITY.
allsystematics3 = mock.generate_galaxy_catalog(
        cl_mass, cl_z, cl_conc, cosmo, 'chang13', zsrc_min=zsrc_min, zsrc_max=7., ngal_density=1.3, cluster_ra=cl_ra, cluster_dec=cl_dec, shapenoise=0.05, photoz_sigma_unscaled=0.05)

allsystematics4 = mock.generate_galaxy_catalog(
        cl_mass, cl_z, cl_conc, cosmo, 'desc_srd', zsrc_min=zsrc_min, zsrc_max=7., ngal_density=1.3, cluster_ra=cl_ra, cluster_dec=cl_dec, shapenoise=0.05, photoz_sigma_unscaled=0.05)

plt.scatter(allsystematics1['ra'], allsystematics1['dec'], marker='.', s=2, label='ngals= 1000')
plt.scatter(allsystematics3['ra'], allsystematics3['dec'], marker='.', s=2, label='ngal_density = 1 gal/armin2')
plt.legend()

plt.show()



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||| GENERATE MOCK DATA WITH DIFFERENT GALAXY CLUSTER OPTIONS ||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## NOTE: AVAILABLE OPTIONS DEPEND ON THE MODELING BACKEND:
##      @ CLUSTER-TOOLKIT ALLOWS FOR OTHER VALUES OF THE OVERDENSITY PARAMETER, BUT IS RESTRICTED TO MEAN MASS DEF.
##      @ BOTH CCL AND NUMCOSMO ALLOW FOR OTHER VALUES OF THE OVERDENSITY PARAMTER, BUT WORK WITH BOTH THE MEAN AND
##        CRITICAL MASS DEFINITION
##      @ NUMCOSMO FURTHER ALLOWS FOR THE EINASTO OR BURKERT DENSITY PROFILES IN ADDITION TO NFW

## CHANGING THE OVERDENSITY PARAMETER (ALL BACKENDS) - delta_so (DEFAULT=200)
allsystematics_500mean = mock.generate_galaxy_catalog(
        cl_mass,
        cl_z,
        cl_conc,
        cosmo,
        'chang13',
        delta_so=500,
        zsrc_min=zsrc_min,
        zsrc_max=7.,
        ngals=ngals,
        cluster_ra=cl_ra,
        cluster_dec=cl_dec,
        shapenoise=0.05,
        photoz_sigma_unscaled=0.05)

## USING THE CRITCAL MASS DEFINITION (NUMBOSMO AND CCL ONLY) - massdef (DEFAULT='mean')
allsystematics_200critical = mock.generate_galaxy_catalog(
        cl_mass,
        cl_z,
        cl_conc,
        cosmo,
        'chang13',
        massdef='critical',
        zsrc_min=zsrc_min,
        zsrc_max=7.,
        ngals=ngals,
        cluster_ra=cl_ra,
        cluster_dec=cl_dec,
        shapenoise=0.05,
        photoz_sigma_unscaled=0.05)

## CHANGING THE HALO DENSITY PROFILE (NUMCOSMO AND CCL ONLY) - halo_profile_model (DEFAULT='nfw')
allsystematics_200m_einasto = mock.generate_galaxy_catalog(
        cl_mass,
        cl_z,
        cl_conc,
        cosmo,
        'chang13',
        halo_profile_model='einasto',
        delta_so=500,
        zsrc_min=zsrc_min,
        zsrc_max=7.,
        ngals=ngals,
        cluster_ra=cl_ra,
        cluster_dec=cl_dec,
        shapenoise=0.05,
        photoz_sigma_unscaled=0.05)

