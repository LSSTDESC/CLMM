"""@file galaxycluster.py
The GalaxyCluster class
"""
import pickle
import warnings
from .gcdata import GCData
from .dataops import compute_tangential_and_cross_components, make_radial_profile,_compute_galaxy_weights
from .theory import compute_critical_surface_density
from .plotting import plot_profiles


class GalaxyCluster():
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
    """

    def __init__(self, *args, **kwargs):
        self.unique_id = None
        self.ra = None
        self.dec = None
        self.z = None
        self.galcat = None
        if len(args)>0 or len(kwargs)>0:
            self._add_values(*args, **kwargs)
            self._check_types()

    def _add_values(self, unique_id: str, ra: float, dec: float, z: float,
                 galcat: GCData):
        """Add values for all attributes"""
        self.unique_id = unique_id
        self.ra = ra
        self.dec = dec
        self.z = z
        self.galcat = galcat

    def _check_types(self):
        """Check types of all attributes"""
        if isinstance(self.unique_id, (int, str)): # should unique_id be a float?
            self.unique_id = str(self.unique_id)
        else:
            raise TypeError(f'unique_id incorrect type: {type(self.unique_id)}')
        try:
            self.ra = float(self.ra)
        except TypeError:
            print(f'ra incorrect type: {type(self.ra)}')
        try:
            self.dec = float(self.dec)
        except TypeError:
            print(f'dec incorrect type: {type(self.dec)}')
        try:
            self.z = float(self.z)
        except TypeError:
            print(f'z incorrect type: {type(self.z)}')
        if not isinstance(self.galcat, GCData):
            raise TypeError(f'galcat incorrect type: {type(self.galcat)}')
        if not -360. <= self.ra <= 360.:
            raise ValueError(f'ra={self.ra} not in valid bounds: [-360, 360]')
        if not -90. <= self.dec <= 90.:
            raise ValueError(f'dec={self.dec} not in valid bounds: [-90, 90]')
        if self.z < 0.:
            raise ValueError(f'z={self.z} must be greater than 0')

    def save(self, filename, **kwargs):
        """Saves GalaxyCluster object to filename using Pickle"""
        with open(filename, 'wb') as fin:
            pickle.dump(self, fin, **kwargs)

    @classmethod
    def load(cls, filename, **kwargs):
        """Loads GalaxyCluster object to filename using Pickle"""
        with open(filename, 'rb') as fin:
            self = pickle.load(fin, **kwargs)
        self._check_types()
        return self

    def _str_colnames(self):
        """Colnames in comma separated str"""
        return ', '.join(self.galcat.colnames)

    def __repr__(self):
        """Generates basic description of GalaxyCluster"""
        return (
            f'GalaxyCluster {self.unique_id}: '
            f'(ra={self.ra}, dec={self.dec}) at z={self.z}'
            f'\n> with columns: {self._str_colnames()}'
            f'\n> {len(self.galcat)} source galaxies'
            )

    def __str__(self):
        """Generates string for print(GalaxyCluster)"""
        table = 'objects'.join(self.galcat.__str__().split('objects')[1:])
        return self.__repr__()+'\n'+table

    def _repr_html_(self):
        """Generates string for display(GalaxyCluster)"""
        return (
            f'<b>GalaxyCluster:</b> {self.unique_id} '
            f'(ra={self.ra}, dec={self.dec}) at z={self.z}'
            f'<br>> <b>with columns:</b> {self._str_colnames()}'
            f'<br>> {len(self.galcat)} source galaxies'
            f'<br>{self.galcat._html_table()}'
            )

    def add_critical_surface_density(self, cosmo):
        r"""Computes the critical surface density for each galaxy in `galcat`.
        It only runs if input cosmo != galcat cosmo or if `sigma_c` not in `galcat`.

        Parameters
        ----------
        cosmo : clmm.Cosmology object
            CLMM Cosmology object

        Returns
        -------
        None
        """
        if cosmo is None:
            raise TypeError('To compute Sigma_crit, please provide a cosmology')
        if cosmo.get_desc() != self.galcat.meta['cosmo'] or 'sigma_c' not in self.galcat:
            if self.z is None:
                raise TypeError('Cluster\'s redshift is None. Cannot compute Sigma_crit')
            if 'z' not in self.galcat.columns:
                raise TypeError('Galaxy catalog missing the redshift column. '
                                'Cannot compute Sigma_crit')
            self.galcat.update_cosmo(cosmo, overwrite=True)
            self.galcat['sigma_c'] = compute_critical_surface_density(cosmo=cosmo, z_cluster=self.z,
                                                                  z_source=self.galcat['z'])

    def compute_tangential_and_cross_components(
            self, shape_component1='e1', shape_component2='e2', tan_component='et',
            cross_component='ex', geometry='curve', is_deltasigma=False, cosmo=None, add=True):
        r"""Adds a tangential- and cross- components for shear or ellipticity to self

        Calls `clmm.dataops.compute_tangential_and_cross_components` with the following arguments:
        ra_lens: cluster Ra
        dec_lens: cluster Dec
        ra_source: `galcat` Ra
        dec_source: `galcat` Dec
        shear1: `galcat` shape_component1
        shear2: `galcat` shape_component2
        geometry: `input` geometry
        is_deltasigma: `input` is_deltasigma
        sigma_c: `galcat` sigma_c | None

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
        missing_cols = ', '.join(
            [f"'{t_}'" for t_ in ('ra', 'dec', shape_component1, shape_component2)
                if t_ not in self.galcat.columns])
        if len(missing_cols)>0:
            raise TypeError('Galaxy catalog missing required columns: '+missing_cols+\
                            '. Do you mean to first convert column names?')
        if is_deltasigma:
            self.add_critical_surface_density(cosmo)
        # compute shears
        angsep, tangential_comp, cross_comp = compute_tangential_and_cross_components(
                ra_lens=self.ra, dec_lens=self.dec,
                ra_source=self.galcat['ra'], dec_source=self.galcat['dec'],
                shear1=self.galcat[shape_component1], shear2=self.galcat[shape_component2],
                geometry=geometry, is_deltasigma=is_deltasigma,
                sigma_c=self.galcat['sigma_c'] if 'sigma_c' in self.galcat.columns else None)
        if add:
            self.galcat['theta'] = angsep
            self.galcat[tan_component] = tangential_comp
            self.galcat[cross_component] = cross_comp
        return angsep, tangential_comp, cross_comp
    
    def compute_galaxy_weights(self, z_source='z', photoz_pdf='pzpdf', z_axis_photoz='pzbins', 
                           shape_component1='e1', shape_component2='e2', 
                           shape_component1_err='e1_err', shape_component2_err='e2_err', 
                           add_photoz=False, add_shapenoise = False, add_shape_error=False, 
                           weight_name='w_ls', p_background_name='p_background',
                           cosmo=None,
                           is_deltasigma=False, add=True):
        r"""
        Parameters:
        -----------
        z_source: string
            column name : source redshifts
        cosmo: clmm.Cosmology object
        photoz_pdf : string
            column name : photometric probablility density function of the source galaxies
        z_axis_photoz : string
            column name : redshift axis on which the individual photoz pdf is tabulated
        shape_component1: string
            column name : The measured shear (or reduced shear or ellipticity) 
            of the source galaxies
        shape_component2: array
            column name : The measured shear (or reduced shear or ellipticity) 
            of the source galaxies
        shape_component1_err: array
            column name : The measurement error on the 1st-component of ellipticity 
            of the source galaxies
        shape_component2_err: array
            column name : The measurement error on the 2nd-component of ellipticity 
            of the source galaxies
        add_photoz : boolean
            True for computing photometric weights
        add_shapenoise : boolean
            True for considering shapenoise in the weight computation
        add_shape_error : boolean
            True for considering measured shape error in the weight computation
        weight_name : string
            Name of the new column for the weak lensing weights in the galcat table
        p_background : string
            Name of the new column for the background probability in the galcat table
        is_deltasigma: boolean
            Indicates whether it is the excess surface density or the tangential shear
        add : boolean
            If True, add weight and background probability columns to the galcat table

        Returns:
        --------
        w_ls: array
            the individual lens source pair weights
        p_background : array
            the probability for being a background galaxy
    """
        
        w_ls, p_background = _compute_galaxy_weights(self.z, cosmo, 
                           z_source=self.galcat[z_source], pzpdf=self.galcat[photoz_pdf], 
                           pzbins=self.galcat[z_axis_photoz], 
                           shape_component1=self.galcat[shape_component1], 
                           shape_component2=self.galcat[shape_component2], 
                           shape_component1_err=self.galcat[shape_component1_err], 
                           shape_component2_err=self.galcat[shape_component2_err], 
                           add_photoz=add_photoz, add_shapenoise=add_shapenoise, 
                           add_shape_error=add_shape_error, 
                           is_deltasigma=is_deltasigma)
        
        if add == True:
            
            self.galcat[weight_name], self.galcat[p_background_name] = w_ls, p_background
            
        return w_ls, p_background

    def make_radial_profile(
            self, bin_units, bins=10, cosmo=None, tan_component_in='et', cross_component_in='ex',
            tan_component_out='gt', cross_component_out='gx', include_empty_bins=False,
            gal_ids_in_bins=False, add=True, table_name='profile', overwrite=True):
        r"""Compute the shear or ellipticity profile of the cluster

        We assume that the cluster object contains information on the cross and
        tangential shears or ellipticities and angular separation of the source galaxies

        Calls `clmm.dataops.make_radial_profile` with the following arguments:
        components: `galcat` components (tan_component_in, cross_component_in, z)
        angsep: `galcat` theta
        angsep_units: radians
        bin_units: `input` bin_units
        bins: `input` bins
        include_empty_bins: `input` include_empty_bins
        cosmo: `input` cosmo
        z_lens: cluster z

        Parameters
        ----------
        angsep_units : str
            Units of the calculated separation of the source galaxies
            Allowed Options = ["radians"]
        bin_units : str
            Units to use for the radial bins of the shear profile
            Allowed Options = ["radians", "deg", "arcmin", "arcsec", "kpc", "Mpc"]
            (letter case independent)
        bins : array_like, optional
            User defined bins to use for the shear profile. If a list is provided, use that as the
            bin edges. If a scalar is provided, create that many equally spaced bins between the
            minimum and maximum angular separations in bin_units. If nothing is provided, default to
            10 equally spaced bins.
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

        Returns
        -------
        profile : GCData
            Output table containing the radius grid points, the tangential and cross shear profiles
            on that grid, and the errors in the two shear profiles. The errors are defined as the
            standard errors in each bin.
        """
        #Too many local variables (19/15)
        #pylint: disable=R0914

        if not all([t_ in self.galcat.columns for t_ in
                (tan_component_in, cross_component_in, 'theta')]):
            raise TypeError(
                'Shear or ellipticity information is missing!  Galaxy catalog must have tangential '
                'and cross shears (gt, gx) or ellipticities (et, ex). '
                'Run compute_tangential_and_cross_components first.')
        if 'z' not in self.galcat.columns:
            raise TypeError('Missing galaxy redshifts!')
        # Compute the binned averages and associated errors
        profile_table, binnumber = make_radial_profile(
            [self.galcat[n].data for n in (tan_component_in, cross_component_in, 'z')],
            angsep=self.galcat['theta'], angsep_units='radians',
            bin_units=bin_units, bins=bins, include_empty_bins=include_empty_bins,
            return_binnumber=True,
            cosmo=cosmo, z_lens=self.z)
        # Reaname table columns
        for i, name in enumerate([tan_component_out, cross_component_out, 'z']):
            profile_table.rename_column(f'p_{i}', name)
            profile_table.rename_column(f'p_{i}_err', f'{name}_err')
        # add galaxy IDs
        if gal_ids_in_bins:
            if 'id' not in self.galcat.columns:
                raise TypeError('Missing galaxy IDs!')
            nbins = len(bins)-1 if hasattr(bins, '__len__') else bins
            gal_ids = [list(self.galcat['id'][binnumber==i+1])
                        for i in range(nbins)]
            if not include_empty_bins:
                gal_ids = [g_id for g_id in gal_ids if len(g_id)>1]
            profile_table['gal_id'] = gal_ids
        if add:
            profile_table.update_cosmo_ext_valid(self.galcat, cosmo, overwrite=False)
            if hasattr(self, table_name):
                if overwrite:
                    warnings.warn(f'overwriting {table_name} table.')
                    delattr(self, table_name)
                else:
                    raise AttributeError(
                        f'table {table_name} already exists, '
                        'set overwrite=True or use another name.')
            setattr(self, table_name, profile_table)
        return profile_table

    def plot_profiles(self, tangential_component='gt', tangential_component_error='gt_err',
                      cross_component='gx', cross_component_error='gx_err', table_name='profile',
                      xscale='linear', yscale='linear'):
        """Plot shear profiles using `plotting.plot_profiles` function

        Parameters
        ----------
        tangential_component: str, optional
            Name of the column in the galcat Table corresponding to the tangential component of the
            shear or reduced shear (Delta Sigma not yet implemented). Default: 'gt'
        tangential_component_error: str, optional
            Name of the column in the galcat Table corresponding to the uncertainty in tangential
            component of the shear or reduced shear. Default: 'gt_err'
        cross_component: str, optional
            Name of the column in the galcat Table corresponding to the cross component of the shear
            or reduced shear. Default: 'gx'
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
            if col not in profile.colnames:
                raise ValueError(f"Column for plotting '{col}' does not exist.")
        for col in (tangential_component_error, cross_component_error):
            if col not in profile.colnames:
                warnings.warn(f"Column for plotting '{col}' does not exist.")
        return plot_profiles(
            rbins=profile['radius'],
            r_units=profile.meta['bin_units'],
            tangential_component=profile[tangential_component],
            tangential_component_error=(profile[tangential_component_error] if
                tangential_component_error in profile.colnames else None),
            cross_component=profile[cross_component],
            cross_component_error=(profile[cross_component_error] if
                cross_component_error in profile.colnames else None),
            xscale=xscale, yscale=yscale,
            tangential_component_label=tangential_component,
            cross_component_label=cross_component)
