
## USED TO MAKE FIG. 4 OF THE CLMM v1.0 PAPER.


##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||||||||||||||| SETUP ||||||||||||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

import os
import sys
import gi

from gi.repository import GObject
from gi.repository import NumCosmo as NC
from gi.repository import NumCosmoMath as NCM

from scipy.stats import chi2

import numpy as np
import corner

import clmm
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from clmm.support.sampler import fitters

clmm.__version__

import clmm.dataops as da
import clmm.galaxycluster as gc
import clmm.theory as theory
from clmm import Cosmology

from clmm.support import mock_data as mock


##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||||||||| MAKING MOCK DATA |||||||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

np.random.seed(42)

## SET THE COSMOLOGY
H0 = 70.
Omega_b0 = 0.045
Omega_dm0 = 0.27 - Omega_b0
Omega_k0 = 0.
mock_cosmo = Cosmology(H0=H0, Omega_dm0=Omega_dm0, Omega_b0=Omega_b0, Omega_k0=Omega_k0)


## SET THE MOCK CLUSTER
cosmo = mock_cosmo
cl_mass = 1.e15
cl_z = 0.3
cl_conc = 4.
ngals = int(1e4)
Delta = 200
cl_ra = 0.
cl_dec = 0.

## SET GALAXY CATALOG
noisy_data_z = mock.generate_galaxy_catalog(
        cl_mass, cl_z, cl_conc, cosmo, 'chang13', shapenoise=0.05, photoz_sigma_unscaled=0.05, ngals=ngals)

## DEFINE CLUSTER OBJECT WITH THE GALAXY CATALOG
cl_id = 'CL'
gc_object = clmm.GalaxyCluster(cl_id, cl_ra, cl_dec, cl_z, noisy_data_z)
gc_object.save('noisy_GC_z.pkl')

cl = clmm.GalaxyCluster.load('noisy_GC_z.pkl')

print('ID: {} \nRA: {} \nDEC: {} \nz: {}'.format(cl.unique_id, cl.ra, cl.dec, cl.z))
print('The number of source galaxies is: {}'.format(len(cl.galcat)))



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||||||| DERIVING OBSERVABLES |||||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## COMPUTING SHEAR:
cl.compute_tangential_and_cross_components(geometry='flat')

## RADIALLY BINNING THE DATA
bin_edges = da.make_bins(0.7, 4, 15, method='evenlog10width')
cl.make_radial_profile('Mpc', bins=bin_edges, cosmo=cosmo)

for n in cl.profile.colnames : cl.profile[n].format = '%6.3e'
cl.profile.pprint(max_width=-1)

fsize = 14
plt.errorbar(cl.profile['radius'], cl.profile['gt'], yerr=cl.profile['gt_err'])
plt.title(r'Binned reduced tangential shear profile', fontsize=fsize)
plt.xlabel(r'$r$ [Mpc]', fontsize=fsize)
plt.ylabel(r'$g_t$', fontsize=fsize)
plt.tight_layout()

plt.show()



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||| BIAS WHEN NOT ACCOUNTING FOR REDSHIFT DISTRIBUTION ||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## NOW WE ESTIMATE THE BEST-FIT MASS USING A SIMPLE IMPLEMENTATION OF THE LIKELIHOOD USING NcmDataGaussDiag OBJECT.
## TO BUILD THE MODEL, WE PURPOSELY MAKE THE WRONG ASSUMPTION THAT THE AVERAGE SHEAR IN BIN-i EQUALS THE SHEAR AT THE
## AVERAGE REDSHIFT IN THE BIN: <g_t>_i = g_t(<z>_i).

#class GaussGammaTErr(Ncm.DataGaussDiag) :
