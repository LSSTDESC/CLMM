#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

class ShearAzimuthalAverager(object):
    """
    a class that calculate the average azimuthal profile for shear
    see shear_azimuthal_averager_demo.py for demo

    Attributes
    ----------

    """
    def __init__(self, cl_dict, src_table):
        """
        Parameters
        ----------
        cl_dict : dictionary
            containing the infomration of cluster

        st_table: astropy table
            containing the infomration of sources

        """
        # TODO: check whether property exists in the catalog table
        

        # cluster information
        self.ra_cl = cl_dict['ra']
        self.dec_cl = cl_dict['dec']
        self.z_cl = cl_dict['z']

        # source infomration
        # first, check whether it has convergence
        self.id_src = src_table['id']
        self.ra_src = src_table['ra']
        self.dec_src = src_table['dec']
        self.z_src = src_table['z']

        if 'kappa' in src_table.keys():
            print('this is sim!')
            self.gamma_1 = src_table['gamma1']
            self.gamma_2 = src_table['gamma2']
            self.kappa = src_table['kappa']
            self.g1 = self.gamma_1/(1-self.kappa)
            self.g2 = self.gamma_2/(1-self.kappa)

        else:
            print('read g1, g2 directly')


        # source information
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


    def read_catalog(self):
        """ 
        This reads in the galaxy catalog, which can either:
        - DC2 extragalactic catalog (shear1, shear2)
        - DC2 coadd catalog (shapeHSM_e1,shapeHSM_e2)
        - DM stack catalog (shapeHSM_e1,shapeHSM_e2)
        - ...
        
        Returns:
        - Homogenised astropy table containing at least: ra[deg], dec[deg], e1, e2, (z or p(z)), relevant flags...
     
        """
        pass# done outside!

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
        shear1 = self.g1
        shear2 = self.g2
        phi = np.arctan2(self.dec_src-self.dec_cl, self.ra_src-self.ra_cl)
        self.shear_t = - (shear1 * np.cos(2.0 * phi) - shear2 * np.sin(2.0 * phi))
        self.shear_c = - shear1 * np.sin(2.0 * phi) + shear2 * np.cos(2.0 * phi)
        #plt.hist(self.shear_t)


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
            print(np.mean(self.shear_t[select]), len(self.shear_t[select]))
            gamt_arr[ir] = np.mean(self.shear_t[select])
            gamt_err_arr[ir] = np.std(self.shear_t[select])
            gamc_arr[ir] = np.mean(self.shear_c[select])
            gamc_err_arr[ir] = np.std(self.shear_c[select])

        self.rmean_arr = rmean_arr
        self.shear_t_arr = gamt_arr
        self.shear_c_arr = gamc_arr
        self.shear_t_err_arr = gamt_err_arr
        self.shear_c_err_arr = gamc_err_arr

    def plot_profile(self):
        plt.errorbar(self.rmean_arr, self.shear_t_arr, yerr=self.shear_t_err_arr, label=r'$g_t$')
        plt.errorbar(self.rmean_arr, self.shear_c_arr, yerr=self.shear_c_err_arr, label=r'$g_\times$')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('physical Mpc')
        plt.legend()
        plt.savefig('plots/profile_test.pdf')


if __name__ == "__main__":
    import GCRCatalogs
    from astropy.table import Table

    extragalactic_cat = GCRCatalogs.load_catalog('proto-dc2_v2.1.2_test')
    # get a massive halo!
    massive_halos = extragalactic_cat.get_quantities(['halo_mass', 'redshift','ra', 'dec'], filters=['halo_mass > 1e14','is_central==True'])

    m = massive_halos['halo_mass']
    select = (m == np.max(m))
    ra_cl = massive_halos['ra'][select][0]
    dec_cl = massive_halos['dec'][select][0]
    z_cl = massive_halos['redshift'][select][0]

    # make a dictionary for clusterr
    cl_dict = {'z':z_cl, 'ra':ra_cl, 'dec': dec_cl}

    # make a astropy table for source

    # get galaxies around it
    ra_min, ra_max = ra_cl-0.3, ra_cl+0.3
    dec_min, dec_max = dec_cl-0.3, dec_cl+0.3
    z_min = z_cl + 0.1
    z_max = 1.5

    coord_filters = [
        'ra >= {}'.format(ra_min),
        'ra < {}'.format(ra_max),
        'dec >= {}'.format(dec_min),
        'dec < {}'.format(dec_max),
    ]
    z_filters = ['redshift >= {}'.format(z_min),'redshift < {}'.format(z_max)]
    gal_cat = extragalactic_cat.get_quantities(['ra', 'dec', 'shear_1', 'shear_2', 'shear_2_phosim', 'shear_2_treecorr','redshift','galaxy_id','convergence'], filters=(coord_filters + z_filters))

    t = Table([gal_cat['galaxy_id'],gal_cat['ra'],gal_cat['dec'],gal_cat['shear_1'],\
          gal_cat['shear_2'],gal_cat['redshift'],gal_cat['convergence']], \
          names=('id','ra','dec', 'gamma1', 'gamma2', 'z', 'kappa'))

    # crerate an object
    saa = ShearAzimuthalAverager(cl_dict=cl_dict, src_table=t)
    saa.compute_shear()
    saa.make_shear_profile()
    saa.plot_profile()

    plt.show()
   
