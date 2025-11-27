"""@file clusterensemble.py
The Cluster Ensemble class
"""

import pickle
import warnings

import healpy
import numpy as np

from .dataops import make_stacked_radial_profile
from .gcdata import GCData
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
        * "quad_4theta_sc" : quadrupole 4theta component computed with sample covariance
        * "quad_const_sc" : quadrupole constant component computed with sample covariance
        * "tan_bs" : tangential component computed with bootstrap
        * "cross_bs" : cross component computed with bootstrap
        * "quad_4theta_bs" : quadrupole 4theta component computed with bootstrap
        * "quad_const_bs" : quadrupole constant component computed with bootstrap
        * "tan_jk" : tangential component computed with jackknife
        * "cross_jk" : cross component computed with jackknife
        * "quad_4theta_jk" : quadrupole 4theta component computed with jackknife
        * "quad_const_jk" : quadrupole constant component computed with jackknife

    """

    def __init__(self, unique_id, gc_list=None, include_quadrupole=False, **kwargs):
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
        self.include_quadrupole = include_quadrupole
        if gc_list is not None:
            self._add_values(gc_list, **kwargs)
            weights_out = kwargs.get("weights_out", "W_l")
            if (self.data[weights_out] == 0).any():
                warnings.warn(
                    "There are empty bins in some of the profile tables,"
                    f" filter them with {weights_out}>0 for visualization."
                )
        self.stacked_data = None
        self.cov = {
            "tan_sc": None,
            "cross_sc": None,
            "quad_4theta_sc": None,
            "quad_const_sc": None,
            "tan_bs": None,
            "cross_bs": None,
            "quad_4theta_bs": None,
            "quad_const_bs": None,
            "tan_jk": None,
            "cross_jk": None,
            "quad_4theta_jk": None,
            "quad_const_jk": None,
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

    # pylint: disable=R0914
    def make_individual_radial_profile(
        self,
        galaxycluster,
        bin_units,
        bins=10,
        error_model="ste",
        cosmo=None,
        tan_component_in="et",
        cross_component_in="ex",
        quad_4theta_component_in="e_quad_4theta",
        quad_const_component_in="e_quad_const",
        tan_component_out="gt",
        cross_component_out="gx",
        quad_4theta_component_out="g_quad_4theta",
        quad_const_component_out="g_quad_const",
        tan_component_in_err=None,
        cross_component_in_err=None,
        quad_4theta_component_in_err=None,
        quad_const_component_in_err=None,
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
        quad_4theta_component_in: string, optional
            Name of the quadrupole 4theta component column in `galcat` to be binned.
            Default: 'e_quad_4theta'
        quad_const_component_in: string, optional
            Name of the quadrupole constant component column in `galcat` to be binned.
            Default: 'e_quad_const'
        tan_component_out: string, optional
            Name of the tangetial component binned column to be added in profile table.
            Default: 'gt'
        cross_component_out: string, optional
            Name of the cross component binned profile column to be added in profile table.
            Default: 'gx'
        quad_4theta_component_out: string, optional
            Name of the quadrupole 4theta component binned profile column to be added
            in profile table.
            Default: 'g_quad_4theta'
        quad_const_component_out: string, optional
            Name of the quadrupole constant component binned profile column to be added
            in profile table.
            Default: 'g_quad_const'
        tan_component_in_err: string, None, optional
            Name of the tangential component error column in `galcat` to be binned.
            Default: None
        cross_component_in_err: string, None, optional
            Name of the cross component error column in `galcat` to be binned.
            Default: None
        quad_4theta_component_in_err: string, None, optional
            Name of the quadrupole 4theta component error column in `galcat` to be binned.
            Default: None
        quad_const_component_in_err: string, None, optional
            Name of the quadrupole constant component error column in `galcat` to be binned.
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
            tan_component=tan_component_out,
            cross_component=cross_component_out,
            quad_4theta_component=quad_4theta_component_out,
            quad_const_component=quad_const_component_out,
            weights=weights_out,
        )

    def add_individual_radial_profile(
        self,
        galaxycluster,
        profile_table,
        tan_component="gt",
        cross_component="gx",
        quad_4theta_component="g_quad_4theta",
        quad_const_component="g_quad_const",
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
        quad_4theta_component: string, optional
            Name of the quadrupole 4theta component binned profile column in the profile table.
            Default: 'g_quad_4theta'
        quad_const_component: string, optional
            Name of the quadrupole constant component binned profile column in the profile table.
            Default: 'g_quad_const'
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

        _quad_cols = ()
        if self.include_quadrupole:
            _quad_cols = (quad_4theta_component, quad_const_component)
        tbcols = ("radius", tan_component, cross_component, *_quad_cols, weights)

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

    def make_stacked_radial_profile(
        self,
        tan_component="gt",
        cross_component="gx",
        quad_4theta_component="g_quad_4theta",
        quad_const_component="g_quad_const",
        weights="W_l",
    ):
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
        quad_4theta_component : string, optional
            Name of the quadrupole 4theta component column in `data`.
            Default: 'g_quad_4theta'
        quad_const_component : string, optional
            Name of the quadrupole constant component column in `data`.
            Default: 'g_quad_const'
        weights : str
            Name of the weights column in `data`.
        """
        self._check_empty_data()

        _col_names = [tan_component, cross_component]
        if self.include_quadrupole:
            _col_names += [quad_4theta_component, quad_const_component]

        radius, components = make_stacked_radial_profile(
            self.data["radius"],
            self.data[weights],
            [self.data[_col] for _col in _col_names],
        )
        self.stacked_data = GCData(
            [
                self.data.meta["radius_min"].value,
                self.data.meta["radius_max"].value,
                radius,
                *components,
            ],
            meta={k: v for k, v in self.data.meta.items() if k not in ("radius_min", "radius_max")},
            names=("radius_min", "radius_max", "radius", *_col_names),
        )

    def compute_sample_covariance(
        self,
        tan_component="gt",
        cross_component="gx",
        quad_4theta_component="g_quad_4theta",
        quad_const_component="g_quad_const",
    ):
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
        quad_4theta_component : string, optional
            Name of the quadrupole 4theta component column in `data`.
            Default: 'g_quad_4theta'
        quad_const_component : string, optional
            Name of the quadrupole constant component column in `data`.
            Default: 'g_quad_const'
        """
        self._check_empty_data()

        _components = [("tan_sc", tan_component), ("cross_sc", cross_component)]
        if self.include_quadrupole:
            _components += [
                ("quad_4theta_sc", quad_4theta_component),
                ("quad_const_sc", quad_const_component),
            ]

        n_catalogs = len(self.data)
        for _cov_name_out, _col_name_in in _components:
            self.cov[_cov_name_out] = np.cov(self.data[_col_name_in].T, bias=False) / n_catalogs

    def compute_bootstrap_covariance(
        self,
        tan_component="gt",
        cross_component="gx",
        quad_4theta_component="g_quad_4theta",
        quad_const_component="g_quad_const",
        n_bootstrap=10,
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
        quad_4theta_component : string, optional
            Name of the quadrupole 4theta component column in `data`.
            Default: 'g_quad_4theta'
        quad_const_component : string, optional
            Name of the quadrupole constant component column in `data`.
            Default: 'g_quad_const'
        n_bootstrap : int
            number of bootstrap resamplings
        """
        self._check_empty_data()
        n_catalogs = len(self)

        cluster_index = np.arange(n_catalogs)
        cluster_index_bootstrap = [
            np.random.choice(cluster_index, n_catalogs) for n_boot in range(n_bootstrap)
        ]
        _shear_components = [tan_component, cross_component]
        if self.include_quadrupole:
            _shear_components += [quad_4theta_component, quad_const_component]

        g_boot_components = make_stacked_radial_profile(
            self["radius"][None, cluster_index_bootstrap][0].transpose(1, 2, 0),
            self["W_l"][None, cluster_index_bootstrap][0].transpose(1, 2, 0),
            [
                self[_component][None, cluster_index_bootstrap][0].transpose(1, 2, 0)
                for _component in _shear_components
            ],
        )[1]

        coeff = (n_catalogs / (n_catalogs - 1)) ** 2
        self.cov["tan_bs"] = coeff * np.cov(np.array(g_boot_components[0]), bias=False, ddof=0)
        self.cov["cross_bs"] = coeff * np.cov(np.array(g_boot_components[1]), bias=False)
        if self.include_quadrupole:
            gt_boot, gx_boot, g4theta_boot, gconst_boot = g_boot_components
            self.cov["quad_4theta_bs"] = coeff * np.cov(
                np.array(g_boot_components[2]), bias=False, ddof=0
            )
            self.cov["quad_const_bs"] = coeff * np.cov(
                np.array(g_boot_components[3]), bias=False, ddof=0
            )

    def compute_jackknife_covariance(
        self,
        tan_component="gt",
        cross_component="gx",
        quad_4theta_component="g_quad_4theta",
        quad_const_component="g_quad_const",
        n_side=16,
    ):
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
        quad_4theta_component : string, optional
            Name of the quadrupole 4theta component column in `data`.
            Default: 'g_quad_4theta'
        quad_const_component : string, optional
            Name of the quadrupole constant component column in `data`.
            Default: 'g_quad_const'
        n_side : int
            healpix sky area division parameter (number of sky area : 12*n_side^2)
        """
        # may induce artificial noise if there are some healpix pixels
        # not covering entirely the 2D map of clusters
        self._check_empty_data()

        pixels = healpy.ang2pix(n_side, self.data["ra"], self.data["dec"], nest=True, lonlat=True)
        pixels_list_unique = np.unique(pixels)
        gt_jack, gx_jack = [], []
        if self.include_quadrupole:
            g4theta_jack, gconst_jack = [], []
        for hp_list_delete in pixels_list_unique:
            mask = ~np.isin(pixels, hp_list_delete)
            if self.include_quadrupole:
                g_components = make_stacked_radial_profile(
                    self["radius"][mask],
                    self["W_l"][mask],
                    [
                        self[tan_component][mask],
                        self[cross_component][mask],
                        self[quad_4theta_component][mask],
                        self[quad_const_component][mask],
                    ],
                )[1]
                gt_jack.append(g_components[0])
                gx_jack.append(g_components[1])
                g4theta_jack.append(g_components[2])
                gconst_jack.append(g_components[3])
            else:
                g_components = make_stacked_radial_profile(
                    self["radius"][mask],
                    self["W_l"][mask],
                    [self[tan_component][mask], self[cross_component][mask]],
                )[1]
                gt_jack.append(g_components[0])
                gx_jack.append(g_components[1])
        n_jack = pixels_list_unique.size
        coeff = (n_jack - 1) ** 2 / (n_jack)
        self.cov["tan_jk"] = coeff * np.cov(np.transpose(gt_jack), bias=False, ddof=0)
        self.cov["cross_jk"] = coeff * np.cov(np.transpose(gx_jack), bias=False, ddof=0)
        if self.include_quadrupole:
            self.cov["quad_4theta_jk"] = coeff * np.cov(
                np.transpose(g4theta_jack), bias=False, ddof=0
            )
            self.cov["quad_const_jk"] = coeff * np.cov(
                np.transpose(gconst_jack), bias=False, ddof=0
            )

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
