"""@file clusterensemble.py
The Cluster Ensemble class
"""
import numpy as np
from scipy.stats import binned_statistic
from astropy.table import Table
from .plotting import mpl
import healpy

from .gcdata import GCData
from .galaxycluster import GalaxyCluster
from .utils import compute_radial_averages
from .dataops import make_radial_profile, make_stacked_radial_profile

class ClusterEnsemble():
    """Object that contains a list of GalaxyCluster objects

    Attributes
    ----------
    unique_id : int or string
        Unique identifier of the galaxy cluster ensemble
    data : GCData
        Table with galaxy clusters data (i. e. ids, profiles, redshifts).
    id_dict: dict
        Dictionary of indicies given the cluster id
    """
    def __init__(self, unique_id, colnames, gclist, *args, **kwargs):
        """Initializes a ClusterEnsemble object
        Parameters
        ----------
        unique_id : int or string
            Unique identifier of the galaxy cluster ensemble
        """
        if isinstance(unique_id, (int, str)):
            unique_id = str(unique_id)
        else:
            raise TypeError('unique_id incorrect type: %s'%type(unique_id))
        self.unique_id = unique_id
        self.data = GCData(names=colnames, meta={'bin_units': None},
                           dtype=[np.object for c in colnames])
        if len(args)>0 or len(kwargs)>0:
            self._add_values(gclist, **kwargs)

    def _add_values(self, gc_list, **kwargs):
        """Add values for all attributes
        Parameters
        ----------
        gc_list : list, tuple
            List of GalaxyCluster objects.
        gc_cols : list, tuple
            List of GalaxyCluster objects.
        """
        for gc in gc_list:
            self.make_individual_radial_profile(gc, **kwargs)

    def __getitem__(self, item):
        """Returns self.data[item]"""
        return self.data[item]

    def __len__(self):
        """Returns length of ClusterEnsemble"""
        return len(self.data)

    def stack(self):
        """Produces a GalaxyCluster object by stacking elements of gclist

        Parameters
        ---------

        Returns
        ---------
        gc_stack : GalaxyCluster
            Stacked galaxy cluster generated from elements of self.gclist
        """
        return

    def make_individual_radial_profile(self, galaxycluster, cosmo = None, tan_component='et',
            cross_component='ex', sep='theta', weights = 'w_ls', bins = None):
        """Compute the individual shear profile from a single GalaxyCluster object 
        and adds the averaged data in the data attribute.
        Parameters:
        ----------
        galaxycluster : GalaxyCluster
            GalaxyCluster object with cluster metadata and background galaxy data
        cosmo : Comsology
            cosmology object
        tan_component : str
            componenent to be used to compute average tangential shear profile
        cross_component : str
            componenent to be used to compute average cross shear profile
        sep : str
            distance to the cluster center to be used in radial binning
        weights : str
            weights to be used in the weighted average tangential and cross profiles
        bins : array
            bin edges for the radial binning
        Returns:
        -------
        mean_r : array
            mean radial position (arithmetic mean of galaxy positions in the bin
        mean_gt : array
            mean tangential profile
        mean_gx :
            mean cross profile
        """
        weights = galaxycluster.galcat[weights]
        separation = galaxycluster.galcat[sep]
        gt = galaxycluster.galcat[tan_component]
        gx = galaxycluster.galcat[cross_component]
        mean_sep, mean_gt, err_gt, num_objects, binnumber = compute_radial_averages(separation,
                                                                                     gt, bins, 
                                                                                     yerr=None, 
                                                                                     error_model='ste',
                                                                                     weights=weights)
        mean_sep, mean_gx, err_gx, num_objects, binnumber = compute_radial_averages(separation, 
                                                                                     gx, bins, 
                                                                                     yerr=None, 
                                                                                     error_model='ste', 
                                                                                     weights=weights)
        Wl, bin_edges, binnumber = binned_statistic(separation, weights, 
                                                    statistic = 'sum', 
                                                    bins=bins, range=None)
        data_to_save = [galaxycluster.id, galaxycluster.ra, galaxycluster.dec, galaxycluster.z, 
                        mean_sep, mean_gt, mean_gx, Wl]
        self.data.add_row(data_to_save)
            
    def make_stacked_radial_profile(self, stacked_data):
        """Compute stacked profile, and mean separation distances.
        Parameters:
        ----------
        stacked_data : dict
            data with individual cross and tangential profile for each clusters in 
            the ensemble
        Returns:
        -------
        r : array
            mean radial distance in each radial bins
        gt : array, array
            stacked tangential profile
        gx : array, array
            stacked cross profile
        """
        r = np.average(stacked_data['r'], axis = 0, weights = None)
        gt = np.average(stacked_data['gt'], axis = 0, weights = stacked_data['W_l'])
        gx = np.average(stacked_data['gx'], axis = 0, weights = stacked_data['W_l'])
        return r, gt, gx
        
    def compute_sample_covariance(self):
        """Compute Sample covariance matrix for cross and tangential and cross 
        stacked profiles adds as attributes.
        Returns:
        -------
        sample_tangential_covariance_matrix : ndarray
            The sample covariance matrix for the stacked tangential profile
        sample_cross_covariance_matrix : ndarray
            The sample covariance matrix for the stacked cross profile
        """
        stacked_data = self.data
        n_catalogs = len(stacked_data['id'])
        self.sample_tangential_covariance_matrix = np.cov(np.array(stacked_data['gt']).T, 
                                                          bias = False)/n_catalogs
        self.sample_cross_covariance_matrix = np.cov(np.array(stacked_data['gx']).T, 
                                                          bias = False)/n_catalogs
    
    def compute_bootstrap_covariance(self, n_bootstrap=10):
        """Compute the bootstrap covariance matrix, add boostrap covariance matrix for 
        tangential and cross profiles as attributes.
        Parameters:
        ----------
        n_bootstrap : int
            number of bootstrap resamplings
        """
        stacked_data = self.data
        stacked_data_table = Table(stacked_data)
        cluster_index = np.arange(len(stacked_data_table['id']))
        gt_boot, gx_boot = [], []
        for n_boot in range(n_bootstrap):
            cluster_index_bootstrap = np.random.choice(cluster_index, len(cluster_index))
            stacked_data_table_bootstrap = stacked_data_table[cluster_index_bootstrap]
            r, gt, gx = self.make_stacked_radial_profile(stacked_data_table_bootstrap)
            gt_boot.append(gt), gx_boot.append(gx)
        self.bootstrap_tangential_covariance_matrix = np.cov(np.array(gt_boot).T, bias = False,ddof=0)
        self.bootstrap_cross_covariance_matrix = np.cov(np.array(gx_boot).T, bias = False)
    
    def compute_jackknife_covariance(self, n_side=2):
        """Compute the jackknife covariance matrix, add boostrap covariance matrix for 
        tangential and cross profiles as attributes.
        Uses healpix sky area sub-division : https://healpix.sourceforge.io
        Parameters:
        ----------
        n_side : int
            healpix sky area division parameter (number of sky area : 12*n_side^2)
        """
        #may induce artificial noise if there are some healpix pixels 
        #not covering entirely the 2D map of clusters
        stacked_data = self.data
        ra, dec =  stacked_data['ra'], stacked_data['dec']
        index = np.arange(len(self.data['id']))
        healpix = healpy.ang2pix(2**n_side, ra, dec, nest=True, lonlat=True)
        healpix_list_unique = np.unique(healpix)
        n_jack = len(healpix_list_unique)
        gt_jack, gx_jack = [], []
        for i, hp_list_delete in enumerate(healpix_list_unique):
                mask_in_area = np.isin(healpix, hp_list_delete)
                mask_out_area = np.invert(mask_in_area)
                stacked_data_table_jackknife = Table(stacked_data)[mask_out_area]
                r, gt, gx = self.make_stacked_radial_profile(stacked_data_table_jackknife)
                gt_jack.append(gt), gx_jack.append(gx)
        coeff = (n_jack - 1)**2/(n_jack)
        self.jackknife_tangential_covariance_matrix = coeff * np.cov(np.array(gt_jack).T, 
                                                                    bias = False, ddof=0)
        self.jackknife_cross_covariance_matrix = coeff * np.cov(np.array(gx_jack).T, 
                                                               bias = False, ddof=0)
