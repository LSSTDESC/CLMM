"""@file galaxycluster.py
The GalaxyCluster class
"""
import pickle
from astropy.table import Table


def load_cluster(filename, **kwargs):
    """Loads GalaxyCluster object from filename using Pickle"""
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)


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
    galcat : astropy Table
        Table of background galaxy data containing at least galaxy_id, ra, dec, e1, e2, z
    """
    def __init__(self, *args, **kargs):
        self.unique_id = None
        self.ra = None
        self.dec = None
        self.z = None
        self.galcat = None
        if len(args)>0 or len(kargs)>0:
            self._add_values(*args, **kargs)
            self._check_types()
    def _add_values(self, unique_id: str, ra: float, dec: float, z: float,
                 galcat: Table):
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
        if not isinstance(self.galcat, Table):
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
    def load(self, filename, **kwargs):
        """Loads GalaxyCluster object to filename using Pickle"""
        self = load_cluster(filename, **kwargs)
        self._check_types()
        return
    def __repr__(self):
        """Generates string for print(GalaxyCluster)"""
        output = f'GalaxyCluster {self.unique_id}: ' +\
                 f'(ra={self.ra}, dec={self.dec}) at z={self.z}\n' +\
                 f'> {len(self.galcat)} source galaxies\n> With columns:'
        for colname in self.galcat.colnames:
            output += f' {colname}'
        return output
