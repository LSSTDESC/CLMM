#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
#import FoFCatalogMatching
#import GCRCatalogs


# see shear_azimuthal_averager_demo.py for demo

class ShearAzimuthalAverager(object):
    def __init__(self, ra_src, dec_src, z_src, ra_cl, dec_cl, z_cl, gamma_1, gamma_2):
        # TODO: pass catalog table directly!
        # TODO: check whether property exists in the catalog table
        self.ra_src = ra_src
        self.dec_src = dec_src
        self.z_src = z_src
        self.ra_cl = ra_cl
        self.dec_cl = dec_cl
        self.z_cl = z_cl
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


    def read_catalog(self):
        pass# done outside!
        """ 
        This reads in the galaxy catalog, which can either:
        - DC2 extragalactic catalog (shear1, shear2)
        - DC2 coadd catalog (shapeHSM_e1,shapeHSM_e2)
        - DM stack catalog (shapeHSM_e1,shapeHSM_e2)
        - ...
        
        Returns:
        - Homogenised astropy table containing at least: ra[deg], dec[deg], e1, e2, (z or p(z)), relevant flags...
     
        """
        
    def compute_shear(self):
        """
        Input: galaxy catalog astropy table
        Output: new columns on the astropy table containing 
            - tangential shear 
            - cross shear
            - physical projected distance to cluster center
            
        phi = numpy.arctan2(dec, ra)
        gamt = - (e1 * numpy.cos(2.0 * phi) + e2 * numpy.sin(2.0 * phi))
        gamc =  e1 * numpy.sin(2.0 * phi) - e2 * numpy.cos(2.0 * phi)
        dist = fcn(cosmology, phi, z)
        """
        shear1 = self.gamma_1
        shear2 = self.gamma_2
        phi = np.arctan2(self.dec_src-self.dec_cl, self.ra_src-self.ra_cl)
        self.gamt = - (shear1 * np.cos(2.0 * phi) - shear2 * np.sin(2.0 * phi))
        self.gamc = - shear1 * np.sin(2.0 * phi) + shear2 * np.cos(2.0 * phi)
        #plt.hist(self.gamt)


    def make_bins(self, user_defined=False):
        """
        Bin definition. Various options:
        - User-defined number of galaxies in each bin
        - regular radial binning
        - log binning
        - anything else?
        
        Returns array of bin edges (physical units), to be used in make_shear_profile 
        """
        if user_defined==False:
            self.r_arr = np.linspace(0.75,3,10)#np.exp(lnr_arr)
        else:
            print('not implemented yet')

        
    def make_shear_profile(self):
        """
        Input: 
        - galaxy catalog astropy table, including results from compute_shear()
        - is_binned flag
        
        Options:
        - bin edge array
        
        If is_binned == False: simply returns dist and gamt from astropy table
        If is_binned == True: average gamt in each bin defined by bin edge array
        """
        self.make_bins()
        theta = np.sqrt((self.ra_src-self.ra_cl)**2+(self.dec_src-self.dec_cl)**2)*np.pi/180.
        
        phys_dist = theta * self.cosmo.angular_diameter_distance(self.z_cl).value

        r_arr = self.r_arr
        nr = len(r_arr)-1
        rmean_arr = np.zeros(nr)
        gamt_arr = np.zeros(nr)
        gamt_err_arr = np.zeros(nr)
        gamc_arr = np.zeros(nr)
        gamc_err_arr = np.zeros(nr)
        for ir in range(nr):
            r_min = r_arr[ir]
            r_max = r_arr[ir+1]
            select = (phys_dist >= r_min) & (phys_dist < r_max)
            rmean_arr[ir] = np.mean(phys_dist[select])
            print(np.mean(self.gamt[select]), len(self.gamt[select]))
            gamt_arr[ir] = np.mean(self.gamt[select])
            gamt_err_arr[ir] = np.std(self.gamt[select])
            gamc_arr[ir] = np.mean(self.gamc[select])
            gamc_err_arr[ir] = np.std(self.gamc[select])

        self.rmean_arr = rmean_arr
        self.gamt_arr = gamt_arr
        self.gamc_arr = gamc_arr
        self.gamt_err_arr = gamt_err_arr
        self.gamc_err_arr = gamc_err_arr

    def plot_profile(self):
        plt.errorbar(self.rmean_arr, self.gamt_arr, yerr=self.gamt_err_arr, label=r'$\gamma_t$')
        plt.errorbar(self.rmean_arr, self.gamc_arr, yerr=self.gamc_err_arr, label=r'$\gamma_\times$')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('physical Mpc')
        plt.legend()
        plt.savefig('plots/profile_test.pdf')


   
