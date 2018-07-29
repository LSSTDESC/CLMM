#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

class ShearAzimuthalAverager(object):
    """
    a class that calculates the average azimuthal profile for shear
    see shear_azimuthal_averager_demo.ipynb for demo

    Attributes
    ----------
    cl_dict : dictionary
        containing the information of a cluster

    src_table: astropy table
        containing the information of sources

    Methods
    ----------
    compute_shear :
        computes the shear for all galaxies in the catalog

    make_shear_profile : 
        returns an astropy table containing the radial profile
        
    plot_profile : 
        plots the profile

    """
    def __init__(self, cl_dict, src_table):
        """
        Parameters
        ----------

        """
        # TODO: check whether property exists in the catalog table
        

        self.cl_dict = cl_dict
        self.src_table = src_table

        
    def compute_shear(self):
        """
        Computes the tangential and cross shear for each source in the catalog
        """

      # cluster information
        ra_cl = self.cl_dict['ra']
        dec_cl = self.cl_dict['dec']
        z_cl = self.cl_dict['z']

        
       # source infomration
        # first, check whether it has convergence
        ra_src = self.src_table['ra']
        dec_src = self.src_table['dec']
        z_src = self.src_table['z']

        if 'kappa' in self.src_table.keys():
            gamma_1 = self.src_table['gamma1']
            gamma_2 = self.src_table['gamma2']
            kappa = self.src_table['kappa']
            g1 = gamma_1/(1-kappa)
            g2 = gamma_2/(1-kappa)

        else:
            print('read g1, g2 directly')
            g1 = self.src_table['gamma1']
            g2 = self.src_table['gamma2']

        self.theta = np.sqrt((ra_src-ra_cl)**2+(dec_src-dec_cl)**2)*np.pi/180.
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.phys_dist = self.theta * cosmo.angular_diameter_distance(z_cl).value

        phi = np.arctan2(dec_src-dec_cl, ra_src-ra_cl)
        self.shear_t = - (g1 * np.cos(2.0 * phi) - g2 * np.sin(2.0 * phi))
        self.shear_c = - g1 * np.sin(2.0 * phi) + g2 * np.cos(2.0 * phi)
        #plt.hist(self.shear_t)

        
    def make_shear_profile(self, is_binned=True, bins=None):
        """
        Returns: 
             astropy table containing the shear profile
 
        Args:
            is_binned: boolean
            bins: user-defined array of bins
        """
        
        if bins is not None:
            self.r_arr = bins
            is_binned = True
        
        if bins is None and is_binned:
            self.r_arr = np.linspace(0.75,3,10)

        if is_binned == True:
            nr = len(self.r_arr)-1
            r_mean_arr = np.zeros(nr)
            angsep_mean_arr = np.zeros(nr)
            gamt_arr = np.zeros(nr)
            gamt_err_arr = np.zeros(nr)
            gamc_arr = np.zeros(nr)
            gamc_err_arr = np.zeros(nr)
            for ir in range(nr):
                r_min = self.r_arr[ir]
                r_max = self.r_arr[ir+1]
                select = (self.phys_dist >= r_min) & (self.phys_dist < r_max)
                r_mean_arr[ir] = np.mean(self.phys_dist[select])
                angsep_mean_arr[ir] = np.mean(self.theta[select])
                gamt_arr[ir] = np.mean(self.shear_t[select])
                gamt_err_arr[ir] = np.std(self.shear_t[select])
                gamc_arr[ir] = np.mean(self.shear_c[select])
                gamc_err_arr[ir] = np.std(self.shear_c[select])
        else:
            r_mean_arr = self.phys_dist
            angsep_mean_arr = self.theta 
            gamt_arr = self.shear_t
            gamt_err_arr = 0.*self.shear_t
            gamc_arr = self.shear_c
            gamc_err_arr = 0.*self.shear_c

        order = np.argsort(r_mean_arr)
        self.r_mean_arr = r_mean_arr[order]
        self.angsep_mean_arr = angsep_mean_arr[order]
        self.shear_t_arr = gamt_arr[order]
        self.shear_c_arr = gamc_arr[order]
        self.shear_t_err_arr = gamt_err_arr[order]
        self.shear_c_err_arr = gamc_err_arr[order]

        shear_profile = Table([self.r_mean_arr, self.angsep_mean_arr, self.shear_t_arr, self.shear_c_arr, self.shear_t_err_arr, self.shear_c_err_arr],\
                                  names=('radius','ang_seperation','g_t','g_x', 'g_t_err','g_x_err'))

        shear_profile['radius'].unit='Mpc'
        shear_profile['ang_seperation'].unit='rad'
        return shear_profile
        
    def plot_profile(self):
        plt.errorbar(self.r_mean_arr, self.shear_t_arr, yerr=self.shear_t_err_arr, label=r'$g_t$')
        plt.errorbar(self.r_mean_arr, self.shear_c_arr, yerr=self.shear_c_err_arr, label=r'$g_\times$')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('physical Mpc')
        plt.legend()

