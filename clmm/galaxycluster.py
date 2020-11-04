"""@file galaxycluster.py
The GalaxyCluster class
"""
import pickle
import warnings
from .gcdata import GCData
from .dataops import compute_tangential_and_cross_components, make_binned_profile
from .modeling import get_critical_surface_density

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
        return
    def _check_types(self):
        """Check types of all attributes"""
        if isinstance(self.unique_id, (int, str)): # should unique_id be a float?
            self.unique_id = str(self.unique_id)
        else:
            raise TypeError(f'unique_id incorrect type: {type(unique_id)}')
        try:
            self.ra = float(self.ra)
        except ValueError:
            raise TypeError(f'ra incorrect type: {type(self.ra)}')
        try:
            self.dec = float(self.dec)
        except ValueError:
            raise TypeError(f'dec incorrect type: {type(self.dec)}')
        try:
            self.z = float(self.z)
        except ValueError:
            raise TypeError(f'z incorrect type: {type(self.z)}')
        if not isinstance(self.galcat, GCData):
            raise TypeError(f'galcat incorrect type: {type(self.galcat)}')
        if not -360. <= self.ra <= 360.:
            raise ValueError(f'ra={self.ra} not in valid bounds: [-360, 360]')
        if not -90. <= self.dec <= 90.:
            raise ValueError(f'dec={self.dec} not in valid bounds: [-90, 90]')
        if self.z < 0.:
            raise ValueError(f'z={self.z} must be greater than 0')
        return
    def save(self, filename, **kwargs):
        """Saves GalaxyCluster object to filename using Pickle"""
        with open(filename, 'wb') as fin:
            pickle.dump(self, fin, **kwargs)
        return
    def load(filename, **kwargs):
        """Loads GalaxyCluster object to filename using Pickle"""
        with open(filename, 'rb') as fin:
            self = pickle.load(fin, **kwargs)
        self._check_types()
        return self
    def __repr__(self):
        """Generates string for print(GalaxyCluster)"""
        output = f'GalaxyCluster {self.unique_id}: '+\
                 f'(ra={self.ra}, dec={self.dec}) at z={self.z}\n'+\
                 f'> {len(self.galcat)} source galaxies\n> With columns:'
        for colname in self.galcat.colnames:
            output+= f' {colname}'
        return output
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
            self.galcat['sigma_c'] = get_critical_surface_density(cosmo=cosmo, z_cluster=self.z,
                                                                  z_source=self.galcat['z'])
        return
    def compute_tangential_and_cross_components(self,
                      shape_component1='e1', shape_component2='e2',
                      tan_component='et', cross_component='ex',
                      geometry='flat', is_deltasigma=False, cosmo=None,
                      add=True):
        r"""Adds a tangential- and cross- components for shear or ellipticity to self

        Calls `clmm.dataops.compute_tangential_and_cross_components` with the following arguments::

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
            Flat is currently the only supported option.
        is_deltasigma: bool
            If `True`, the tangential and cross components returned are multiplied by Sigma_crit. Results in units of :math:`M_\odot\ Mpc^{-2}`
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
        if not all([t_ in self.galcat.columns for t_ in ('ra', 'dec', shape_component1, shape_component2)]):
            raise TypeError('Galaxy catalog missing required columns.'
                            'Do you mean to first convert column names?')
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
    def make_binned_profile(self,
                            bin_units, bins=10, cosmo=None,
                            tan_component_in='et', cross_component_in='ex',
                            tan_component_out='gt', cross_component_out='gx',
                            include_empty_bins=False, gal_ids_in_bins=False,
                            add=True, table_name='profile', overwrite=True):
        r"""Compute the shear or ellipticity profile of the cluster

        We assume that the cluster object contains information on the cross and
        tangential shears or ellipticities and angular separation of the source galaxies

        Calls `clmm.dataops.make_binned_profile` with the following arguments::

            components: `galcat` components (tan_component_in, cross_component_in, z)
            angsep: `galcat` theta
            angsep_units: radians
            bin_units: `input` bin_units
            bins: `input` bins
            include_empty_bins: `input` include_empty_bins
            cosmo: `input` cosmo
            z_source: `galcat` z

        Parameters
        ----------
        angsep_units : str
            Units of the calculated separation of the source galaxies
            Allowed Options = ["radians"]
        bin_units : str
            Units to use for the radial bins of the shear profile
            Allowed Options = ["radians", deg", "arcmin", "arcsec", kpc", "Mpc"]
        bins : array_like, optional
            User defined bins to use for the shear profile. If a list is provided, use that as
            the bin edges. If a scalar is provided, create that many equally spaced bins between
            the minimum and maximum angular separations in bin_units. If nothing is provided,
            default to 10 equally spaced bins.
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
        if not all([t_ in self.galcat.columns for t_ in (tan_component_in, cross_component_in, 'theta')]):
            raise TypeError('Shear or ellipticity information is missing!  Galaxy catalog must have tangential '
                            'and cross shears (gt, gx) or ellipticities (et, ex). Run compute_tangential_and_cross_components first.')
        if 'z' not in self.galcat.columns:
            raise TypeError('Missing galaxy redshifts!')
        # Compute the binned averages and associated errors
        profile_table, binnumber = make_binned_profile(
            [self.galcat[n].data for n in (tan_component_in, cross_component_in, 'z')],
            angsep=self.galcat['theta'], angsep_units='radians',
            bin_units=bin_units, bins=bins, include_empty_bins=include_empty_bins,
            cosmo=cosmo, z_source=self.galcat['z'])
        # Reaname table columns
        for i, n in enumerate([tan_component_out, cross_component_out, 'z']):
            profile_table.rename_column(f'p_{i}', n)
            profile_table.rename_column(f'p_{i}_err', f'{n}_err')
        # add galaxy IDs
        if gal_ids_in_bins:
            if 'id' not in self.galcat.columns:
                raise TypeError('Missing galaxy IDs!')
            nbins = len(bins) if hasattr(bins, '__len__') else bins
            gal_ids = [list(self.galcat['id'][binnumber==i+1])
                        for i in range(nbins-1)]
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
                    raise AttributeError(f'table {table_name} already exists, set overwrite=True or use another name.')
            setattr(self, table_name, profile_table)
        return profile_table
