"""@file clusterensemble.py
The Cluster Ensemble class
"""

from .galaxycluster import GalaxyCluster
import numpy as np
from astropy.table import Table
from collections import Sequence
from scipy.stats import binned_statistic
from .utils import compute_radial_averages

class ClusterEnsemble():
    """Object that contains a list of GalaxyCluster objects

    Attributes
    ----------
    unique_id : int or string
        Unique identifier of the galaxy cluster ensemble
    gclist : list
        List of galaxy cluster objects
    """
    def __init__(self, unique_id, gclist):
        """Initializes a ClusterEnsemble object

        Parameters
        ----------
        unique_id : int or string
            Unique identifier of the galaxy cluster ensemble
        gclist : collections.Sequence
            Array-like Sequence of galaxy cluster objects

        Returns
        ---------

        """
        if isinstance(unique_id, (int, str)):
            unique_id = str(unique_id)
        else:
            raise TypeError('unique_id incorrect type: %s'%type(unique_id))
        if isinstance(gclist, Sequence):
            gclist = list(gclist)
        else:
            raise TypeError('gclist incorrect type: %s'%type(gclist))
        for gc in gclist:
            if ~isinstance(gc, GalaxyCluster):
                raise TypeError('gclist entry incorrect type: %s'%type(gc))

        self.unique_id = unique_id
        self.gclist = gclist

    def __getitem__(self, key):
        """Returns GalaxyCluster object at key in gclist"""
        if ~isinstance(key, int):
            raise TypeError('key incorrect type: %s'%type(key))
        
        return gclist[key]

    def __len__(self):
        """Returns length of ClusterEnsemble"""
        return len(gclist)

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

    def make_individual_radial_profile(self, galaxycluster, cosmo, tan_component='et',
            cross_component='ex', sep='theta', weights = 'w_ls', bins = None):
        """Compute the individual shear profile from a single GalaxyCluster object 
        and adds the averaged data in the stack file
        
        Attributes:
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
        for i, key in enumerate(self.data.keys()):
            self.data[key].append(data_to_save[i])
        return mean_sep, mean_gt, mean_gx
    
    def make_stacked_radial_profile(self, stacked_data):
        """Compute stacked profile, add mean separation distances,
        stacked cross/tangential profiles and associated errors
        --------
        r : array
            mean radial distance in each radial bins
        gt, gt_err : array, array
            stacked tangential profile, standard deviation of the stacked tangential profile
        gx, gx_err : array, array
            stacked cross profile, standard deviation of the stacked cross profile
        """
        n_catalogs = len(stacked_data['id'])
        r = np.average(stacked_data['r'], axis = 0, weights = None)
        gt = np.average(stacked_data['gt'], axis = 0, weights = stacked_data['W_l'])
        gt_err = np.std(stacked_data['gt'], axis = 0)/np.sqrt(n_catalogs)
        gx = np.average(stacked_data['gx'], axis = 0, weights = stacked_data['W_l'])
        gx_err = np.std(stacked_data['gx'], axis = 0)/np.sqrt(n_catalogs)
        
        return r, gt, gt_err, gx, gx_err
        
    def compute_sample_covariance(self, stacked_data):
        """Compute Sample covariance matrix for cross and tangential stacked profile 
        add sample covariance matrix for tangential and cross profiles as attributes
        Returns:
        -------
        sample_tangential_covariance_matrix : ndarray
            The sample covariance matrix for the stacked tangential profile
        sample_cross_covariance_matrix : ndarray
            The sample covariance matrix for the stacked cross profile
        """
        n_catalogs = len(stacked_data['id'])
        self.sample_tangential_covariance_matrix = np.cov(np.array(stacked_data['gt']).T, 
                                                          bias = False)/n_catalogs
        self.sample_cross_covariance_matrix      = np.cov(np.array(stacked_data['gx']).T, 
                                                          bias = False)/n_catalogs
    
    def compute_bootstrap_covariance(self, stacked_data, n_bootstrap = 10):
        """Compute the bootstrap covariance matrix, add boostrap covariance matrix for 
        tangential and cross profiles as attributes
        Attributes:
        -----------
        n_bootstrap : int
            number of bootstrap resampling
        """
        stacked_data_table = Table(stacked_data)
        cluster_index = stacked_data_table['id']
        gt_boot, gx_boot = [], []
        for n_boot in range(n_bootstrap):
            cluster_index_bootstrap = np.random.choice(cluster_index, len(cluster_index))
            mask_cluster_index_bootstrap = np.isin(cluster_index, cluster_index_bootstrap)
            index_bootstrap = np.arange(
            stacked_data_table_bootstrap = stacked_data_table[mask_cluster_index_bootstrap]
            print(len(stacked_data_table_bootstrap))
            r, gt, gt_err, gx, gx_err = self.make_stacked_radial_profile(stacked_data_table_bootstrap)
            gt_boot.append(gt), gx_boot.append(gx)
        gt_boot_ = np.stack((np.array(gt_boot).astype(float)), axis = 1)
        self.bootstrap_tangential_covariance_matrix = np.cov(gt_boot_, bias = False,ddof=0)
        self.bootstrap_cross_covariance_matrix = np.cov(np.array(gx_boot).T, bias = False)
