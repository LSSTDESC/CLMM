

import sys
import os
import numpy as np
from astropy.table import Table
from numpy import random
import scipy
import matplotlib.pyplot as plt
import clmm
from clmm import GalaxyCluster, ClusterEnsemble, GCData
from clmm import Cosmology
from clmm.support import mock_data as mock

np.random.seed(241093)

H0 = 71.
Om_b0 = 0.0448
Om_dm0 = 0.265 - Om_b0
Om_k0 = 0.
cosmo = Cosmology(H0=H0, Omega_dm0=Om_dm0, Omega_b0=Om_b0, Omega_k0=Om_k0)


##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||| GENERATE CLUSTER CATALOG AND ASSOCIATED SOURCE CATALOGS |||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## RANDOMLY GENERATE THE MASSES, REDSHIFTS, CONCENTRATIONS, AND COORDINATES OF AN ENSEMBLE OF n_clusters CLUSTERS.
## FOR SIMPLICITY, THE DRAWING IS UNIFORM IN logm AND REDSHIFT SPACE INSTEAD OF FOLLOWING A HALO MASS FUNCTION.

## REDSHIFT AND MASS (IN M_sun) RANGE FOR GALAXY CLUSTERS
z_bin = [0.2, 0.25]
logm_bin = np.array([14., 14.1])

## NUMBER OF CLUSTERS IN THE ENSEMBLE
n_clusters = 30

## POPULATE THE ENSEMBLE REDSHIFTS, MASSES, CONCENTRATIONS, AND POSITIONS
cl_mass = 10.**((logm_bin[1] - logm_bin[0])*np.random(n_clusters) + logm_bin[0])
cl_z = (z_bin[1] - z_bin[0])*np.random.random(n_clusters) + z_bin[0]

c_mean = 4.
lnc = abs(np.log(c_mean) + 0.01*np.random.randn(n_clusters))
cl_conc = np.exp(lnc)

ra = np.random.random(n_clusters) * 360
sindec = 2*np.random.random(n_clusters) - 1
dec = np.arcsin(sindec)*180/np.pi
