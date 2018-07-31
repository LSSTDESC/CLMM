#!/usr/bin/env python
import numpy as np
from astropy.table import Table
# model from Dallas group
import colossus.cosmology.cosmology as Cosmology # used for distances

from clmm import models, summarizer
import clmm.models.CLMM_densityModels_beforeConvertFromPerH as dm
# the code we want to test:
from clmm.summarizer.shear_azimuthal_averager import ShearAzimuthalAverager

def generate_perfect_data(ngals, src_redshift, cluster_mass, cluster_redshift, concentration, chooseCosmology):
    '''
    This generates a fake dataset of background galaxies using the Dallas group software. 
    Data are perfect, i.e. no shape noise, all galaxies at the same redshift.

    Parameters
    ----------
        ngals: int
            Number of galaxies in the fake catalog
        cluster_mass: double
            Mass of the cluster
        cluster_redshift: double
            Redshift of vluster
        concentration: double
            Concentration of the cluster
        chooseCosmology: Cosmology object
            String defining the cosmological parameter set in colossus, e.g. WMAP7-ML
    Returns
    -------
        astropy table 
            Containing the galaxy catalog [id, ra, dec, gamma1, gamma2, z]
    '''

    ngals = 10000
    seqnr = np.arange(ngals)
    zL = cluster_redshift # cluster redshift

    mdef = '200c'
    cosmo = Cosmology.setCosmology(chooseCosmology)
    M = cluster_mass*cosmo.h
    c = concentration  
    r = np.linspace(0.25, 10., 1000) #Mpc
    r = r*cosmo.h #Mpc/h

    testProf= dm.nfwProfile(M = M, c = c, zL = zL, mdef = mdef, \
                            chooseCosmology = chooseCosmology, esp = None)

    x_mpc = np.random.uniform(-4, 4, size=ngals)
    y_mpc = np.random.uniform(-4, 4, size=ngals)
    r_mpc = np.sqrt(x_mpc**2 + y_mpc**2)

    Dl = cosmo.angularDiameterDistance(zL)

    x_deg = (x_mpc/Dl)*(180./np.pi)
    y_deg = (y_mpc/Dl)*(180./np.pi)

    gamt= testProf.deltaSigma(r_mpc)/testProf.Sc(src_redshift)

    posangle = np.arctan2(y_mpc, x_mpc)
    cos2phi = np.cos(2*posangle)
    sin2phi = np.sin(2*posangle)

    e1 = -gamt*cos2phi
    e2 = -gamt*sin2phi

    return Table([seqnr, -x_deg, y_deg, e1, e2, np.zeros(ngals)+src_redshift], \
                names=('id', 'ra','dec','gamma1','gamma2', 'z'))

def test_shear_azimuthal_averager():
    '''
    This tests the shear azimuthal averager by comparing the fake data g_t 
    profile to the theoretical binned profile. 

    This is done by checking whether the residuals in each bin are below 
    a given tolerance
    '''

    # define the set up for fake data
    ngals = 1e4
    cluster_redshift = 0.3
    src_redshift = 2 * cluster_redshift
    cluster_mass = 1.e15
    concentration = 4
    chooseCosmology = 'WMAP7-ML' #Choose cosmology used
    cosmo = Cosmology.setCosmology(chooseCosmology)

    # make the fake data catalog
    t = generate_perfect_data(ngals, src_redshift, cluster_mass, cluster_redshift, concentration, chooseCosmology)
    cl_dict={'ra':0.0, 'dec':0.0, 'z':cluster_redshift}
  
    # create an object, given cluster dictionary and galaxy astropy table
    saa = ShearAzimuthalAverager(cl_dict,t)

    # compute tangential and cross shear for each galaxy
    saa.compute_shear()

    # make the binned profile
    binned_profile = saa.make_shear_profile()


    # compute the theoretical tangential shear at the bin locations
    mdef = '200c'
    M = cluster_mass*cosmo.h # NOTE! model code operate with h
    c = concentration # NOTE! n=not sure
#    r = binned_profile['radius'] #Mpc
    theta = binned_profile['ang_separation'] #Mpc
#    r = r*cosmo.h #Mpc/h

    testProf = dm.nfwProfile(M = M, c = c, zL = cluster_redshift, mdef = mdef, \
                            chooseCosmology = chooseCosmology, esp = None)

    gt_mod= testProf.deltaSigma(theta*cosmo.angularDiameterDistance(cluster_redshift))/testProf.Sc(src_redshift)

   
    nr = len(binned_profile['g_t']) * 1.
    # equivalent to chi_sqr/nbins
    g_t_residual = sum(abs(binned_profile['g_t'] - gt_mod)/binned_profile['g_t_err'])/nr
    g_x_residual = sum(abs(binned_profile['g_x'])/binned_profile['g_x_err'])/nr

    print(g_t_residual, g_x_residual)

    tolerance = 1. # 
    assert g_t_residual < tolerance
    assert g_x_residual < tolerance
    

