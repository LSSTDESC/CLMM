import sys, os
import numpy as np
from astropy.table import QTable, Table, vstack, join
import pickle 
import pandas as pd
import clmm
import cmath
import GCRCatalogs
r"""
extract background galaxy catalog with qserv for:
cosmodc2:
- true shapes
- true redshift
and GCRCatalogs:
- photoz addons
"""
def _fix_axis_ratio(q_bad):
    # back out incorrect computation of q using Johnsonb function
    e_jb = np.sqrt((1 - q_bad**2)/(1 + q_bad**2))
    q_new = np.sqrt((1 - e_jb)/(1 + e_jb)) # use correct relationship to compute q from e_jb 
    return q_new

def _fix_ellipticity_disk_or_bulge(ellipticity):
    # back out incorrect computation of q using Johnsonb function 
    q_bad = (1-ellipticity)/(1+ellipticity) #use default e definition to calculate q
    # q_bad incorrectly computed from e_jb using q_bad = sqrt((1 - e_jb^2)/(1 + e_jb^2))
    q_new = _fix_axis_ratio(q_bad)
    e_new = (1 - q_new)/(1 + q_new)  # recompute e using default (1-q)/(1+q) definition
    return e_new

def correct_shear_ellipticity(ellipticity_uncorr_e1, ellipticity_uncorr_e2):
    ellipticity_uncorr_norm = (ellipticity_uncorr_e1**2+ellipticity_uncorr_e2**2)**.5
    complex_ellipticity_uncorr = ellipticity_uncorr_e1 + 1j*ellipticity_uncorr_e2
    phi = np.array([cmath.phase(c) for c in complex_ellipticity_uncorr])
    ellipticity_corr_norm = _fix_ellipticity_disk_or_bulge(ellipticity_uncorr_norm)
    ellipticity_corr = ellipticity_corr_norm*np.exp(1j*phi)
    ellipticity_corr_e1, ellipticity_corr_e2 = ellipticity_corr.real, ellipticity_corr.imag
    return ellipticity_corr_e1, ellipticity_corr_e2