"""@file galaxycluster.py
The GalaxyCluster class
"""
import pickle
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
    def get_critical_surface_density(self, cosmo):
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

        Calls `clmm.dataops.compute_tangential_and_cross_components with the following arguments::


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
            self.get_critical_surface_density(cosmo)
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
# Monkey patch functions onto Galaxy Cluster object
GalaxyCluster.make_binned_profile = make_binned_profile
