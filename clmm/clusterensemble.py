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
    def __init__(self, unique_id, gclist, *args, **kwargs):
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
        self.data = GCData(meta={'bin_units': None})
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

        Returns
        ---------
        gc_stack : GalaxyCluster
            Stacked galaxy cluster generated from elements of self.gclist
        """
        return

    def make_individual_radial_profile(self, galaxycluster, bin_units, bins=10, error_model='ste',
                                       cosmo=None, tan_component_in='et', cross_component_in='ex',
                                       tan_component_out='gt', cross_component_out='gx',
                                       tan_component_in_err=None, cross_component_in_err=None,
                                       weights_in='w_ls', weights_out='W_l'):
        """Compute the individual shear profile from a single GalaxyCluster object
        and adds the averaged data in the data attribute.

        Parameters
        ----------
        galaxycluster : GalaxyCluster
            GalaxyCluster object with cluster metadata and background galaxy data
        bin_units : str
            Units to use for the radial bins of the shear profile
            Allowed Options = ["radians", "deg", "arcmin", "arcsec", "kpc", "Mpc"]
            (letter case independent)
        bins : array_like, optional
            User defined bins to use for the shear profile. If a list is provided, use that as
            the bin edges. If a scalar is provided, create that many equally spaced bins between
            the minimum and maximum angular separations in bin_units. If nothing is provided,
            default to 10 equally spaced bins.
        error_model : str, optional
            Statistical error model to use for y uncertainties. (letter case independent)

                * `ste` - Standard error [=std/sqrt(n) in unweighted computation] (Default).
                * `std` - Standard deviation.

        cosmo: dict, optional
            Cosmology parameters to convert angular separations to physical distances
        tan_component_in: string, optional
            Name of the tangential component column in `galcat` to be binned.
            Default: 'et'
        cross_component_in: string, optional
            Name of the cross component column in `galcat` to be binned.
            Default: 'ex'
        tan_component_out: string, optional
            Name of the tangetial component binned column to be added in profile table.
            Default: 'gt'
        cross_component_out: string, optional
            Name of the cross component binned profile column to be added in profile table.
            Default: 'gx'
        tan_component_in_err: string, None, optional
            Name of the tangential component error column in `galcat` to be binned.
            Default: None
        cross_component_in_err: string, None, optional
            Name of the cross component error column in `galcat` to be binned.
            Default: None
        weights_in : str, None
            Name of the weight column in `galcat` to be considered in binning.
        weights_out : str, None
            Name of the weight column to be used in the added to the profile table.
        """
        if self.data.meta['bin_units'] is None:
            self.data.meta['bin_units'] = bin_units
        elif self.data.meta['bin_units'] != bin_units:
            raise ValueError('inconsistent units')
        # This will be replaced when gc has weights on make_radial_profile
        profile_table = make_radial_profile(
            [galaxycluster.galcat[n].data for n in (tan_component_in, cross_component_in, 'z')],
            angsep=galaxycluster.galcat['theta'], angsep_units='radians',
            bin_units=bin_units, bins=bins, error_model=error_model,
            include_empty_bins=True, return_binnumber=False,
            cosmo=cosmo, z_lens=galaxycluster.z, #weights=galaxycluster.galcat[weights_in],
            components_error=[None if n is None else galaxycluster.galcat[n].data
                              for n in (tan_component_in_err, cross_component_in_err, None)],
            )
        profile_table[weights_out] = 1 # rm this line
        data_to_save = [galaxycluster.unique_id, galaxycluster.ra, galaxycluster.dec, galaxycluster.z,
                        *[np.array(profile_table[col]) for col in
                            ('radius', 'p_0', 'p_1', weights_out)]]
        # to be fixed down to here after issue 443 is merged
        if len(self.data)==0:
            for col, data in zip(['cluster_id', 'ra', 'dec', 'z', 'radius', tan_component_out,
                                  cross_component_out, weights_out], data_to_save):
                self.data[col] = [data]
        else:
            self.data.add_row(data_to_save)

    def make_stacked_radial_profile(self, tan_component='gt', cross_component='gx',
                                    weights='W_l'):
        """Computes stacked profile and mean separation distances and add it internally.

        Parameters
        ----------
        tan_component : string, optional
            Name of the tangential component column in `data`.
            Default: 'gt'
        cross_component : string, optional
            Name of the cross component column in `data`.
            Default: 'gx'
        weights : str
            Name of the weights column in `data`.
        """
        radius, components = make_stacked_radial_profile(
            self.data['radius'], self.data[weights],
            [self.data[tan_component], self.data[cross_component]])
        self.stacked_data = GCData([radius, *components], meta=self.data.meta,
                                    names=('radius', tan_component, cross_component))

    def compute_sample_covariance(self):
        """Compute Sample covariance matrix for cross and tangential and cross
        stacked profiles adds as attributes.

        Returns
        -------
        sample_tangential_covariance : ndarray
            The sample covariance matrix for the stacked tangential profile
        sample_cross_covariance : ndarray
            The sample covariance matrix for the stacked cross profile
        """
        n_catalogs = len(self.data)
        self.sample_tangential_covariance = np.cov(self.data['gt'].T, bias = False)/n_catalogs
        self.sample_cross_covariance = np.cov(self.data['gt'].T, bias = False)/n_catalogs

    def compute_bootstrap_covariance(self, n_bootstrap=10):
        """Compute the bootstrap covariance matrix, add boostrap covariance matrix for
        tangential and cross profiles as attributes.

        Parameters
        ----------
        n_bootstrap : int
            number of bootstrap resamplings
        """
        cluster_index = np.arange(len(self.data))
        gt_boot, gx_boot = [], []
        for n_boot in range(n_bootstrap):
            cluster_index_bootstrap = np.random.choice(cluster_index, len(cluster_index))
            data_bootstrap = self.data[cluster_index_bootstrap]
            r, (gt, gx) = make_stacked_radial_profile(
                            data_bootstrap['radius'], data_bootstrap['W_l'],
                            [data_bootstrap['gt'], data_bootstrap['gx']])
            gt_boot.append(gt), gx_boot.append(gx)
        self.bootstrap_tangential_covariance = np.cov(np.array(gt_boot).T, bias = False,ddof=0)
        self.bootstrap_cross_covariance = np.cov(np.array(gx_boot).T, bias = False)

    def compute_jackknife_covariance(self, n_side=2):
        """Compute the jackknife covariance matrix, add boostrap covariance matrix for
        tangential and cross profiles as attributes.
        Uses healpix sky area sub-division : https://healpix.sourceforge.io

        Parameters
        ----------
        n_side : int
            healpix sky area division parameter (number of sky area : 12*n_side^2)
        """
        #may induce artificial noise if there are some healpix pixels
        #not covering entirely the 2D map of clusters
        index = np.arange(len(self.data))
        pixels = healpy.ang2pix(2**n_side, self.data['ra'], self.data['dec'],
                                nest=True, lonlat=True)
        pixels_list_unique = np.unique(pixels)
        n_jack = len(pixels_list_unique)
        gt_jack, gx_jack = [], []
        for i, hp_list_delete in enumerate(pixels_list_unique):
            mask_in_area = np.isin(pixels, hp_list_delete)
            data_jk = self.data[~mask_in_area]
            r, (gt, gx) = make_stacked_radial_profile(data_jk['radius'], data_jk['W_l'],
                                                      [data_jk['gt'], data_jk['gx']])
            gt_jack.append(gt), gx_jack.append(gx)
        coeff = (n_jack - 1)**2/(n_jack)
        self.jackknife_tangential_covariance = coeff*np.cov(np.array(gt_jack).T,
                                                            bias=False, ddof=0)
        self.jackknife_cross_covariance = coeff*np.cov(np.array(gx_jack).T,
                                                       bias=False, ddof=0)
