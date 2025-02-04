"""@file galaxycluster.py
The GalaxyCluster class
"""

import pickle
import warnings
from .gcdata import GCData
from .dataops import (
    compute_tangential_and_cross_components,
    make_radial_profile,
    compute_galaxy_weights,
    compute_background_probability,
)
from .theory import compute_critical_surface_density_eff
from .plotting import plot_profiles
from .utils import (
    validate_argument,
    _validate_ra,
    _validate_dec,
    _draw_random_points_from_tab_distribution,
    _validate_coordinate_system,
)


class GalaxyCluster:
    """Object that contains the galaxy cluster metadata and background galaxy data

    Attributes
    ----------
    unique_id : int or string
        Unique identifier of the galaxy cluster
    ra : float
        Right ascension of galaxy cluster center (in degrees)
    dec : float
        Declination of galaxy cluster center (in degrees)
    z : float
        Redshift of galaxy cluster center
    galcat : GCData
        Table of background galaxy data containing at least galaxy_id, ra, dec, e1, e2, z
    coordinate_system : str, optional
        Coordinate system of the ellipticity components. Must be either 'celestial' or
        euclidean'. See https://doi.org/10.48550/arXiv.1407.7676 section 5.1 for more details.
        Default is 'euclidean'.
    validate_input: bool
        Validade each input argument
    """

    def __init__(self, *args, validate_input=True, **kwargs):
        self.unique_id = None
        self.ra = None
        self.dec = None
        self.z = None
        self.galcat = None
        self.validate_input = validate_input
        if len(args) > 0 or len(kwargs) > 0:
            self._add_values(*args, **kwargs)
            self._check_types()
            self.set_ra_lower(ra_low=0)

    def _add_values(
        self,
        unique_id: str,
        ra: float,
        dec: float,
        z: float,
        galcat: GCData,
        coordinate_system: str = "euclidean",
    ):
        """Add values for all attributes"""
        self.unique_id = unique_id
        self.ra = ra
        self.dec = dec
        self.z = z
        self.galcat = galcat
        self.coordinate_system = coordinate_system

    def _check_types(self):
        """Check types of all attributes"""
        validate_argument(vars(self), "unique_id", (int, str))
        _validate_ra(vars(self), "ra", False)
        _validate_dec(vars(self), "dec", False)
        validate_argument(vars(self), "z", (float, str), argmin=0, eqmin=True)
        validate_argument(vars(self), "galcat", GCData)
        _validate_coordinate_system(vars(self), "coordinate_system", str)
        self.unique_id = str(self.unique_id)
        self.ra = float(self.ra)
        self.dec = float(self.dec)
        self.z = float(self.z)

    def save(self, filename, **kwargs):
        """Saves GalaxyCluster object to filename using Pickle"""
        with open(filename, "wb") as fin:
            pickle.dump(self, fin, **kwargs)

    @classmethod
    def load(cls, filename, **kwargs):
        """Loads GalaxyCluster object to filename using Pickle"""
        with open(filename, "rb") as fin:
            self = pickle.load(fin, **kwargs)
        # pylint: disable=protected-access
        self._check_types()
        return self

    def _str_colnames(self):
        """Colnames in comma separated str"""
        return ", ".join(self.galcat.colnames)

    def __repr__(self):
        """Generates basic description of GalaxyCluster"""
        return (
            f"GalaxyCluster {self.unique_id}: "
            f"(ra={self.ra}, dec={self.dec}) at z={self.z}"
            f"\n> with columns: {self._str_colnames()}"
            f"\n> {len(self.galcat)} source galaxies"
        )

    def __str__(self):
        """Generates string for print(GalaxyCluster)"""
        table = "objects".join(self.galcat.__str__().split("objects")[1:])
        return self.__repr__() + "\n" + table

    def _repr_html_(self):
        """Generates string for display(GalaxyCluster)"""
        # pylint: disable=protected-access
        return (
            f"<b>GalaxyCluster:</b> {self.unique_id} "
            f"(ra={self.ra}, dec={self.dec}) at z={self.z}"
            f"<br>> <b>with columns:</b> {self._str_colnames()}"
            f"<br>> {len(self.galcat)} source galaxies"
            f"<br>{self.galcat._html_table()}"
        )

    def add_critical_surface_density(self, cosmo, use_pdz=False, force=False):
        r"""Computes the critical surface density for each galaxy in `galcat`.
        It only runs if input cosmo != galcat cosmo or if `sigma_c` not in `galcat`.

        Parameters
        ----------
        cosmo : clmm.Cosmology object
            CLMM Cosmology object
        use_pdz : bool
            Flag to specify the use of the photoz pdf. If `False` (default), `sigma_c` is computed
            using the redshift point estimates of the `z` column of the `galcat` table. If `True`,
            `sigma_c` is computed as 1/<1/Sigma_crit>, where the average is performed using
            the individual galaxy redshift pdf. In that case, the `galcat` table should have
            pzbins` and `pzpdf` columns.
        force : bool
            Force recomputation of sigma_c.

        Returns
        -------
        None
        """
        if cosmo is None:
            raise TypeError("To compute Sigma_crit, please provide a cosmology")
        sigmac_colname = "sigma_c_eff" if use_pdz else "sigma_c"
        if (
            cosmo.get_desc() != self.galcat.meta["cosmo"]
            or sigmac_colname not in self.galcat.columns
            or force
        ):
            if self.z is None:
                raise TypeError("Cluster's redshift is None. Cannot compute Sigma_crit")
            if not use_pdz and "z" not in self.galcat.columns:
                raise TypeError(
                    "Galaxy catalog missing the redshift column (which should be"
                    "called 'z'). Cannot compute Sigma_crit."
                )
            if use_pdz and not self.galcat.has_pzpdfs():
                raise TypeError(
                    "Galaxy catalog missing the pzpdfs. Cannot compute 1/<1/Sigma_crit>."
                )

            self.galcat.update_cosmo(cosmo, overwrite=True)
            if not use_pdz:
                self.galcat[sigmac_colname] = cosmo.eval_sigma_crit(self.z, self.galcat["z"])
            else:
                zdata = self._get_input_galdata({"pzpdf": "pzpdf", "pzbins": "pzbins"})
                self.galcat[sigmac_colname] = compute_critical_surface_density_eff(
                    cosmo=cosmo,
                    z_cluster=self.z,
                    pzbins=zdata["pzbins"],
                    pzpdf=zdata["pzpdf"],
                    validate_input=self.validate_input,
                )
        return sigmac_colname

    def _get_input_galdata(self, col_dict):
        """
        Checks required columns exist in galcat and returns kwargs dictionary
        to be passed to dataops functions.

        Parametters
        -----------
        col_dict : dict
            Dictionary with the names of the dataops arguments as keys and galcat columns
            as values, made to usually pass locals() here.

        Returns
        -------
        kwarg_data : dict
            Dictionary with the data to be passed to functions by **kwargs method.
        """
        use_cols = {**col_dict}
        kwarg_data = {}
        if "pzbins" in col_dict:
            if not self.galcat.has_pzpdfs():
                raise TypeError("Missing galaxy photoz distributions")
            use_cols.pop("pzbins")
            use_cols.pop("pzpdf")
            pzbins, pzpdfs = self.galcat.get_pzpdfs()
            kwarg_data.update({"pzbins": pzbins, "pzpdf": pzpdfs})
        missing_cols = ", ".join(
            [f"'{t_}'" for t_ in use_cols.values() if t_ not in self.galcat.columns]
        )
        if len(missing_cols) > 0:
            raise TypeError(f"Galaxy catalog missing required columns: {missing_cols}")
        kwarg_data.update({key: self.galcat[colname] for key, colname in use_cols.items()})
        return kwarg_data

    def compute_tangential_and_cross_components(
        self,
        shape_component1="e1",
        shape_component2="e2",
        tan_component="et",
        cross_component="ex",
        geometry="curve",
        is_deltasigma=False,
        use_pdz=False,
        cosmo=None,
        add=True,
    ):
        r"""Adds a tangential- and cross- components for shear or ellipticity to self

        Calls `clmm.dataops.compute_tangential_and_cross_components` with the following arguments:
        ra_lens: `cluster` Ra
        dec_lens: `cluster` Dec
        ra_source: `galcat` Ra
        dec_source: `galcat` Dec
        shear1: `galcat` shape_component1
        shear2: `galcat` shape_component2
        geometry: `input` geometry
        is_deltasigma: `input` is_deltasigma

        Parameters
        ----------
        shape_component1: string, optional
            Name of the column in the `galcat` astropy table of the cluster object that contains
            the shape or shear measurement along the first axis. Default: `e1`
        shape_component1: string, optional
            Name of the column in the `galcat` astropy table of the cluster object that contains
            the shape or shear measurement along the second axis. Default: `e2`
        tan_component: string, optional
            Name of the column to be added to the `galcat` astropy table that will contain the
            tangential component computed from columns `shape_component1` and `shape_component2`.
            Default: `et`
        cross_component: string, optional
            Name of the column to be added to the `galcat` astropy table that will contain the
            cross component computed from columns `shape_component1` and `shape_component2`.
            Default: `ex`
        geometry: str, optional
            Sky geometry to compute angular separation.
            Options are curve (uses astropy) or flat.
        is_deltasigma: bool
            If `True`, the tangential and cross components returned are multiplied by Sigma_crit.
            Results in units of :math:`M_\odot\ Mpc^{-2}`
        cosmo: astropy cosmology object
            Specifying a cosmology is required if `is_deltasigma` is True
        add: bool
            If `True`, adds the computed shears to the `galcat`

        Returns
        -------
        angsep: array_like
            Angular separation between lens and each source galaxy in radians
        tangential_component: array_like
            Tangential shear (or assimilated quantity) for each source galaxy
        cross_component: array_like
            Cross shear (or assimilated quantity) for each source galaxy
        """
        # Check is all the required data is available
        col_dict = {
            "ra_source": "ra",
            "dec_source": "dec",
            "shear1": shape_component1,
            "shear2": shape_component2,
        }
        if is_deltasigma:
            sigmac_colname = self.add_critical_surface_density(cosmo, use_pdz=use_pdz)
            col_dict.update({"sigma_c": sigmac_colname})
        cols = self._get_input_galdata(col_dict)

        # compute shears
        angsep, tangential_comp, cross_comp = compute_tangential_and_cross_components(
            is_deltasigma=is_deltasigma,
            ra_lens=self.ra,
            dec_lens=self.dec,
            geometry=geometry,
            validate_input=self.validate_input,
            coordinate_system=self.coordinate_system,
            **cols,
        )
        if add:
            self.galcat["theta"] = angsep
            self.galcat[tan_component] = tangential_comp
            self.galcat[cross_component] = cross_comp
            if is_deltasigma:
                sigmac_type = "effective" if use_pdz else "standard"
                self.galcat.meta[f"{tan_component}_sigmac_type"] = sigmac_type
                self.galcat.meta[f"{cross_component}_sigmac_type"] = sigmac_type
        return angsep, tangential_comp, cross_comp

    def compute_background_probability(
        self, use_pdz=False, add=True, p_background_name="p_background"
    ):
        r"""Probability for being a background galaxy

        Parameters
        ----------
        use_pdz : bool
            If True, computes the probability using the photoz pdf
        add : bool
            If True, add background probability columns to the galcat table
        p_background_name : str, optional
            User-defined name for the background probability column to be stored
            in the galcat table (i.e., if add=True)

        Returns
        -------
        p_background : array
            Probability for being a background galaxy
        """
        cols = self._get_input_galdata(
            {"pzpdf": "pzpdf", "pzbins": "pzbins"} if use_pdz else {"z_src": "z"}
        )
        p_background = compute_background_probability(
            self.z, use_pdz=use_pdz, validate_input=self.validate_input, **cols
        )
        if add:
            self.galcat[p_background_name] = p_background
        return p_background

    def compute_galaxy_weights(
        self,
        use_pdz=False,
        use_shape_noise=False,
        shape_component1="e1",
        shape_component2="e2",
        use_shape_error=False,
        shape_component1_err="e1_err",
        shape_component2_err="e2_err",
        weight_name="w_ls",
        cosmo=None,
        is_deltasigma=False,
        add=True,
    ):
        r"""Computes the individual lens-source pair weights

        Parameters
        ----------
        use_pdz : bool
            True for computing photometric weights
        use_shape_noise : bool
            True for considering shapenoise in the weight computation
        shape_component1: string
            column name : The measured shear (or reduced shear or ellipticity)
            of the source galaxies
        shape_component2: array
            column name : The measured shear (or reduced shear or ellipticity)
            of the source galaxies
        use_shape_error : bool
            True for considering measured shape error in the weight computation
        shape_component1_err: array
            column name : The measurement error on the 1st-component of ellipticity
            of the source galaxies
        shape_component2_err: array
            column name : The measurement error on the 2nd-component of ellipticity
            of the source galaxies
        weight_name : string
            Name of the new column for the weak lensing weights in the galcat table
        cosmo: clmm.Comology object, None
            CLMM Cosmology object.
        is_deltasigma: bool
            Indicates whether it is the excess surface density or the tangential shear
        add : bool
            If True, add weight column to the galcat table

        Returns
        -------
        w_ls: array
            the individual lens source pair weights
        """
        # input cols
        col_dict = {}
        if is_deltasigma:
            sigmac_colname = self.add_critical_surface_density(cosmo, use_pdz=use_pdz)
            col_dict.update({"sigma_c": sigmac_colname})
        if use_shape_noise:
            col_dict.update(
                {
                    "shape_component1": shape_component1,
                    "shape_component2": shape_component2,
                }
            )
        if use_shape_error:
            col_dict.update(
                {
                    "shape_component1_err": shape_component1_err,
                    "shape_component2_err": shape_component2_err,
                }
            )
        cols = self._get_input_galdata(col_dict)

        # computes weights
        w_ls = compute_galaxy_weights(
            is_deltasigma=is_deltasigma,
            use_shape_noise=use_shape_noise,
            use_shape_error=use_shape_error,
            validate_input=self.validate_input,
            **cols,
        )
        if add:
            self.galcat[weight_name] = w_ls
            if is_deltasigma:
                self.galcat.meta[f"{weight_name}_sigmac_type"] = (
                    "effective" if use_pdz else "standard"
                )
        return w_ls

    def draw_gal_z_from_pdz(self, zcol_out="z", overwrite=False, nobj=1, xmin=None, xmax=None):
        """Draw random redshifts from the photoz pdf for each galaxy
        of the galcat table.

        Parameters
        ----------
        zcol_out : string
            Name of the column of the galcat table where the random
            redshifts are to be stored. Default='z'
        overwrite : bool
            If True and if zcol_out already exists in the table, the column
            will be overwritten by the new random values
        nobj : int, optional
            Number of random samples to generate. Default is 1.
        xmin : float
            Lower bound to draw redshift. Default is the min(x_tab)
        xmax : float
            Upper bound to draw redshift. Default is the max(x_tab)

        Returns
        -------
        samples : ndarray
            Random points following the pdf_tab distribution
        """

        if zcol_out in self.galcat.columns and overwrite is False:
            raise TypeError(
                f"Column {zcol_out} already exists in galcat. \
                            Set overwrite=True to overwrite or use other column name"
            )

        zdata = self._get_input_galdata({"pzpdf": "pzpdf", "pzbins": "pzbins"})
        if self.galcat.pzpdf_info["type"] == "shared_bins":
            res = [
                _draw_random_points_from_tab_distribution(
                    zdata["pzbins"], pzpdf, nobj=nobj, xmin=xmin, xmax=xmax
                )
                for pzpdf in zdata["pzpdf"]
            ]
        else:
            res = [
                _draw_random_points_from_tab_distribution(
                    pzbin, pzpdf, nobj=nobj, xmin=xmin, xmax=xmax
                )
                for pzbin, pzpdf in zip(zdata["pzbins"], zdata["pzpdf"])
            ]

        self.galcat[zcol_out] = res
        return res

    def make_radial_profile(
        self,
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
        include_empty_bins=False,
        gal_ids_in_bins=False,
        add=True,
        table_name="profile",
        overwrite=True,
        use_weights=False,
        weights_in="w_ls",
        weights_out="W_l",
    ):
        r"""Compute the shear or ellipticity profile of the cluster

        We assume that the cluster object contains information on the cross and
        tangential shears or ellipticities and angular separation of the source galaxies

        Calls `clmm.dataops.make_radial_profile` with the following arguments:
        components: `galcat` components (tan_component_in, cross_component_in, z)
        angsep: `galcat` theta
        angsep_units: 'radians'
        bin_units: `input` bin_units
        bins: `input` bins
        include_empty_bins: `input` include_empty_bins
        cosmo: `input` cosmo
        z_lens: cluster z

        Parameters
        ----------
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

        cosmo: clmm.Comology object, None
            CLMM Cosmology object, used to convert angular separations to physical distances
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
        include_empty_bins: bool, optional
            Also include empty bins in the returned table
        gal_ids_in_bins: bool, optional
            Also include the list of galaxies ID belonging to each bin in the returned table
        add: bool, optional
            Attach the profile to the cluster object
        table_name: str, optional
            Name of the profile table to be add as `cluster.table_name`.
            Default 'profile'
        overwrite: bool, optional
            Overwrite profile table.
            Default True
        use_weights: bool, optional
            If True, use the column `weights_in` in `galcat` as the weights
            Default: False
        weights_in: str, optional
            Name of the weights column in `galcat`
            Default: 'w_ls'

        Returns
        -------
        profile : GCData
            Output table containing the radius grid points, the tangential and cross shear
            profiles on that grid, and the errors in the two shear profiles. The errors are
            defined as the standard errors in each bin.
        """
        # Too many local variables (19/15)
        # pylint: disable=R0914

        if not all(
            t_ in self.galcat.columns for t_ in (tan_component_in, cross_component_in, "theta")
        ):
            raise TypeError(
                "Shear or ellipticity information is missing. Galaxy catalog must have tangential"
                "and cross shears (gt, gx) or ellipticities (et, ex). "
                "Run compute_tangential_and_cross_components first."
            )
        if "z" not in self.galcat.columns:
            raise TypeError("Missing galaxy redshifts!")
        # Compute the binned averages and associated errors
        profile_table, binnumber = make_radial_profile(
            [self.galcat[n].data for n in (tan_component_in, cross_component_in, "z")],
            angsep=self.galcat["theta"],
            angsep_units="radians",
            bin_units=bin_units,
            bins=bins,
            error_model=error_model,
            include_empty_bins=include_empty_bins,
            return_binnumber=True,
            cosmo=cosmo,
            z_lens=self.z,
            validate_input=self.validate_input,
            components_error=[
                None if n is None else self.galcat[n].data
                for n in (tan_component_in_err, cross_component_in_err, None)
            ],
            weights=self.galcat[weights_in].data if use_weights else None,
        )
        # Reaname table columns
        for i, name in enumerate([tan_component_out, cross_component_out, "z"]):
            profile_table.rename_column(f"p_{i}", name)
            profile_table.rename_column(f"p_{i}_err", f"{name}_err")
        # Reaname weights columns
        profile_table.rename_column("weights_sum", weights_out)
        # add galaxy IDs
        if gal_ids_in_bins:
            if "id" not in self.galcat.columns:
                raise TypeError("Missing galaxy IDs!")
            nbins = len(bins) - 1 if hasattr(bins, "__len__") else bins
            gal_ids = [list(self.galcat["id"][binnumber == i + 1]) for i in range(nbins)]
            if not include_empty_bins:
                gal_ids = [g_id for g_id in gal_ids if len(g_id) > 1]
            profile_table["gal_id"] = gal_ids
        if add:
            profile_table.update_cosmo_ext_valid(self.galcat, cosmo, overwrite=False)
            if hasattr(self, table_name):
                if overwrite:
                    warnings.warn(f"overwriting {table_name} table.")
                    delattr(self, table_name)
                else:
                    raise AttributeError(
                        f"table {table_name} already exists, "
                        "set overwrite=True or use another name."
                    )
            setattr(self, table_name, profile_table)
        return profile_table

    def plot_profiles(
        self,
        tangential_component="gt",
        tangential_component_error="gt_err",
        cross_component="gx",
        cross_component_error="gx_err",
        table_name="profile",
        xscale="linear",
        yscale="linear",
    ):
        """Plot shear profiles using `plotting.plot_profiles` function

        Parameters
        ----------
        tangential_component: str, optional
            Name of the column in the galcat Table corresponding to the tangential component of
            the shear or reduced shear (Delta Sigma not yet implemented). Default: 'gt'
        tangential_component_error: str, optional
            Name of the column in the galcat Table corresponding to the uncertainty in tangential
            component of the shear or reduced shear. Default: 'gt_err'
        cross_component: str, optional
            Name of the column in the galcat Table corresponding to the cross component of the
            shear or reduced shear. Default: 'gx'
        cross_component_error: str, optional
            Name of the column in the galcat Table corresponding to the uncertainty in the cross
            component of the shear or reduced shear. Default: 'gx_err'
        table_name: str, optional
            Name of the GalaxyCluster() `.profile` attribute. Default: 'profile'
        xscale:
            matplotlib.pyplot.xscale parameter to set x-axis scale (e.g. to logarithmic axis)
        yscale:
            matplotlib.pyplot.yscale parameter to set y-axis scale (e.g. to logarithmic axis)

        Returns
        -------
        fig:
            The matplotlib figure object that has been plotted to.
        axes:
            The matplotlib axes object that has been plotted to.
        """
        if not hasattr(self, table_name):
            raise ValueError(f"GalaxyClusters does not have a '{table_name}' table.")
        profile = getattr(self, table_name)
        for col in (tangential_component, cross_component):
            if col not in profile.columns:
                raise ValueError(f"Column for plotting '{col}' does not exist.")
        for col in (tangential_component_error, cross_component_error):
            if col not in profile.columns:
                warnings.warn(f"Column for plotting '{col}' does not exist.")
        return plot_profiles(
            rbins=profile["radius"],
            r_units=profile.meta["bin_units"],
            tangential_component=profile[tangential_component],
            tangential_component_error=(
                profile[tangential_component_error]
                if tangential_component_error in profile.columns
                else None
            ),
            cross_component=profile[cross_component],
            cross_component_error=(
                profile[cross_component_error] if cross_component_error in profile.columns else None
            ),
            xscale=xscale,
            yscale=yscale,
            tangential_component_label=tangential_component,
            cross_component_label=cross_component,
        )

    def set_ra_lower(self, ra_low=0):
        """
        Set window of values for cluster and galcat RA to [ra_low, ra_low+360[

        Parameters
        ----------
        ra_low: float
            Lower value for RA range, can be -180 or 0

        """
        if ra_low not in (-180.0, 0.0):
            raise ValueError("ra_low must be -180 or 0")
        self.ra += 360.0 if self.ra < ra_low else 0
        self.ra -= 360.0 if self.ra >= ra_low + 360.0 else 0
        if "ra" in self.galcat.columns:
            self.galcat["ra"][self.galcat["ra"] < ra_low] += 360.0
            self.galcat["ra"][self.galcat["ra"] >= ra_low + 360.0] -= 360.0
