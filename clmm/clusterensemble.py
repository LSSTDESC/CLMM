"""@file clusterensemble.py
The Cluster Ensemble class
"""
import pickle
import numpy as np
import healpy

from .gcdata import GCData
from .dataops import make_stacked_radial_profile
from .utils import DiffArray


class ClusterEnsemble:
    """Object that contains a list of GalaxyCluster objects

    Attributes
    ----------
    unique_id : int or string
        Unique identifier of the galaxy cluster ensemble
    data : GCData
        Table with galaxy clusters data (i. e. ids, profiles, redshifts).
    id_dict: dict
        Dictionary of indices given the cluster id
    stacked_data : GCData, None
        Stacked cluster profiles
    cov : dict
        Dictionary with the covariances:

        * "tan_sc" : tangential component computed with sample covariance
        * "cross_sc" : cross component computed with sample covariance
        * "tan_bs" : tangential component computed with bootstrap
        * "cross_bs" : cross component computed with bootstrap
        * "tan_jk" : tangential component computed with jackknife
        * "cross_jk" : cross component computed with jackknife

    """

    def __init__(self, unique_id, gc_list=None, **kwargs):
        """Initializes a ClusterEnsemble object
        Parameters
        ----------
        unique_id : int or string
            Unique identifier of the galaxy cluster ensemble
        """
        if isinstance(unique_id, (int, str)):
            unique_id = str(unique_id)
        else:
            raise TypeError(f"unique_id incorrect type: {type(unique_id)}")
        self.unique_id = unique_id
        self.data = GCData(meta={"bin_units": None, "radius_min": None, "radius_max": None})
        if gc_list is not None:
            self._add_values(gc_list, **kwargs)
        self.stacked_data = None
        self.cov = {
            "tan_sc": None,
            "cross_sc": None,
            "tan_bs": None,
            "cross_bs": None,
            "tan_jk": None,
            "cross_jk": None,
        }

    def _add_values(self, gc_list, **kwargs):
        """Add values for all attributes

        Parameters
        ----------
        gc_list : list, tuple
            List of GalaxyCluster objects.
        gc_cols : list, tuple
            List of GalaxyCluster objects.
        """
        for cluster in gc_list:
            self.make_individual_radial_profile(cluster, **kwargs)

    def __getitem__(self, item):
        """Returns self.data[item]"""
        return self.data[item]

    def __len__(self):
        """Returns length of ClusterEnsemble"""
        return len(self.data)

    def make_individual_radial_profile(
        self,
        galaxycluster,
        bin_units,
        bins=10,
        error_model="ste",
        cosmo=None,
        tan_component_in="et",
        cross_component_in="ex",
        tan_component_out="gt",
        cross_component_out="gx",
        tan_component_in_err=None,
        cross_component_in_err=None,
        use_weights=True,
        weights_in="w_ls",
        weights_out="W_l",
    ):
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
        bins : array_like or int, optional
            User defined bins to use for the shear profile. If a list is provided, use that as
            the bin edges. If a integer is provided, create that many equally spaced bins between
            the minimum and maximum angular separations in bin_units. If nothing is provided,
            defaults to 10 equally spaced bins.
        error_model : str, optional
            Statistical error model to use for y uncertainties. (letter case independent)

                * `ste` - Standard error [=std/sqrt(n) in unweighted computation] (Default).
                * `std` - Standard deviation.

        cosmo: clmm.Comology, optional
            CLMM Cosmology object
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
        # pylint: disable=unused-argument
        tb_kwargs = {}
        tb_kwargs.update(locals())
        tb_kwargs.pop("self")
        tb_kwargs.pop("tb_kwargs")
        tb_kwargs.pop("galaxycluster")

        cl_bin_units = galaxycluster.galcat.meta.get("bin_units", None)
        self.data.update_info_ext_valid("bin_units", self.data, cl_bin_units, overwrite=False)

        cl_cosmo = galaxycluster.galcat.meta.get("cosmo", None)
        self.data.update_info_ext_valid("cosmo", self.data, cl_cosmo, overwrite=False)

        profile_table = galaxycluster.make_radial_profile(
            include_empty_bins=True, gal_ids_in_bins=False, add=False, **tb_kwargs
        )

        self.add_individual_radial_profile(
            galaxycluster,
            profile_table,
            tan_component_out,
            cross_component_out,
            weights_out,
        )

    def add_individual_radial_profile(
        self,
        galaxycluster,
        profile_table,
        tan_component="gt",
        cross_component="gx",
        weights="W_l",
    ):
        """Compute the individual shear profile from a single GalaxyCluster object
        and adds the averaged data in the data attribute.

        Parameters
        ----------
        galaxycluster : GalaxyCluster
            GalaxyCluster object with cluster metadata and background galaxy data
        profile_table : GCData
            Table containing the radius grid points, the tangential, cross shear and weights
            profiles on that grid.
        tan_component: string, optional
            Name of the tangetial component binned column in the profile table.
            Default: 'gt'
        cross_component: string, optional
            Name of the cross component binned profile column in the profile table.
            Default: 'gx'
        weights : str, None
            Name of the weight binned column in the profile table.
        """
        cl_bin_units = profile_table.meta.get("bin_units", None)
        self.data.update_info_ext_valid("bin_units", self.data, cl_bin_units, overwrite=False)
        for col in ("radius_min", "radius_max"):
            value = DiffArray(profile_table[col])
            self.data.update_info_ext_valid(col, self.data, value, overwrite=False)

        cl_cosmo = profile_table.meta.get("cosmo", None)
        self.data.update_info_ext_valid("cosmo", self.data, cl_cosmo, overwrite=False)

        tbcols = ("radius", tan_component, cross_component, weights)
        data_to_save = [
            galaxycluster.unique_id,
            galaxycluster.ra,
            galaxycluster.dec,
            galaxycluster.z,
            *(np.array(profile_table[col]) for col in tbcols),
        ]
        if len(self.data) == 0:
            for col, data in zip(["cluster_id", "ra", "dec", "z", *tbcols], data_to_save):
                self.data[col] = [data]
        else:
            self.data.add_row(data_to_save)

    def _check_empty_data(self):
        if len(self.data) == 0:
            raise ValueError(
                "There is no single cluster profile data. Please run"
                + "'make_individual_radial_profile' or "
                "'add_individual_radial_profile' "
                "for each cluster in your catalog"
            )

    def make_stacked_radial_profile(self, tan_component="gt", cross_component="gx", weights="W_l"):
        """Computes stacked profile and mean separation distances and add it internally
        to `stacked_data`.

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
        self._check_empty_data()

        radius, components = make_stacked_radial_profile(
            self.data["radius"],
            self.data[weights],
            [self.data[tan_component], self.data[cross_component]],
        )
        self.stacked_data = GCData(
            [
                self.data.meta["radius_min"].value,
                self.data.meta["radius_max"].value,
                radius,
                *components,
            ],
            meta={k: v for k, v in self.data.meta.items() if k not in ("radius_min", "radius_max")},
            names=("radius_min", "radius_max", "radius", tan_component, cross_component),
        )

    def compute_sample_covariance(self, tan_component="gt", cross_component="gx"):
        """Compute Sample covariance matrix for cross and tangential and cross
        stacked profiles and updates .cov dict (`tan_sc`, `cross_sc`).

        Parameters
        ----------
        tan_component : string, optional
            Name of the tangential component column in `data`.
            Default: 'gt'
        cross_component : string, optional
            Name of the cross component column in `data`.
            Default: 'gx'
        """
        self._check_empty_data()

        n_catalogs = len(self.data)
        self.cov["tan_sc"] = np.cov(self.data[tan_component].T, bias=False) / n_catalogs
        self.cov["cross_sc"] = np.cov(self.data[cross_component].T, bias=False) / n_catalogs

    def compute_bootstrap_covariance(
        self, tan_component="gt", cross_component="gx", n_bootstrap=10
    ):
        """Compute the bootstrap covariance matrix, add boostrap covariance matrix for
        tangential and cross stacked profiles and updates .cov dict (`tan_jk`, `cross_bs`).

        Parameters
        ----------
        tan_component : string, optional
            Name of the tangential component column in `data`.
            Default: 'gt'
        cross_component : string, optional
            Name of the cross component column in `data`.
            Default: 'gx'
        n_bootstrap : int
            number of bootstrap resamplings
        """
        self._check_empty_data()
        n_catalogs = len(self)

        cluster_index = np.arange(n_catalogs)
        cluster_index_bootstrap = [
            np.random.choice(cluster_index, n_catalogs) for n_boot in range(n_bootstrap)
        ]

        gt_boot, gx_boot = make_stacked_radial_profile(
            self["radius"][None, cluster_index_bootstrap][0].transpose(1, 2, 0),
            self["W_l"][None, cluster_index_bootstrap][0].transpose(1, 2, 0),
            [
                self[tan_component][None, cluster_index_bootstrap][0].transpose(1, 2, 0),
                self[cross_component][None, cluster_index_bootstrap][0].transpose(1, 2, 0),
            ],
        )[1]

        coeff = (n_catalogs / (n_catalogs - 1)) ** 2
        self.cov["tan_bs"] = coeff * np.cov(np.array(gt_boot), bias=False, ddof=0)
        self.cov["cross_bs"] = coeff * np.cov(np.array(gx_boot), bias=False)

    def compute_jackknife_covariance(self, tan_component="gt", cross_component="gx", n_side=16):
        """Compute the jackknife covariance matrix, add boostrap covariance matrix for
        tangential and cross stacked profiles and updates .cov dict (`tan_jk`, `cross_jk`).

        Uses healpix sky area sub-division : https://healpix.sourceforge.io

        Parameters
        ----------
        tan_component : string, optional
            Name of the tangential component column in `data`.
            Default: 'gt'
        cross_component : string, optional
            Name of the cross component column in `data`.
            Default: 'gx'
        n_side : int
            healpix sky area division parameter (number of sky area : 12*n_side^2)
        """
        # may induce artificial noise if there are some healpix pixels
        # not covering entirely the 2D map of clusters
        self._check_empty_data()

        pixels = healpy.ang2pix(n_side, self.data["ra"], self.data["dec"], nest=True, lonlat=True)
        pixels_list_unique = np.unique(pixels)
        gt_jack, gx_jack = [], []
        for hp_list_delete in pixels_list_unique:
            mask = ~np.isin(pixels, hp_list_delete)
            gt, gx = make_stacked_radial_profile(
                self["radius"][mask],
                self["W_l"][mask],
                [self[tan_component][mask], self[cross_component][mask]],
            )[1]
            gt_jack.append(gt)
            gx_jack.append(gx)
        n_jack = pixels_list_unique.size
        coeff = (n_jack - 1) ** 2 / (n_jack)
        self.cov["tan_jk"] = coeff * np.cov(np.transpose(gt_jack), bias=False, ddof=0)
        self.cov["cross_jk"] = coeff * np.cov(np.transpose(gx_jack), bias=False, ddof=0)

    def save(self, filename, **kwargs):
        """Saves GalaxyCluster object to filename using Pickle"""
        with open(filename, "wb") as fin:
            pickle.dump(self, fin, **kwargs)

    @classmethod
    def load(cls, filename, **kwargs):
        """Loads GalaxyCluster object to filename using Pickle"""
        with open(filename, "rb") as fin:
            self = pickle.load(fin, **kwargs)
        return self
