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
        Table of background galaxy data including galaxy_id, ra, dec, e1, e2, z, kappa
    """
    def __init__(self, unique_id: str, ra: float, dec: float, z: float,
                 galcat: Table):
        if isinstance(unique_id, (int, str)): # should unique_id be a float?
            unique_id = str(unique_id)
        else:
            raise TypeError(f'unique_id incorrect type: {type(unique_id)}')
        try:
            ra = float(ra)
        except ValueError:
            raise TypeError(f'ra incorrect type: {type(ra)}')
        try:
            dec = float(dec)
        except ValueError:
            raise TypeError(f'dec incorrect type: {type(dec)}')
        try:
            z = float(z)
        except ValueError:
            raise TypeError(f'z incorrect type: {type(z)}')
        if not isinstance(galcat, Table):
            raise TypeError(f'galcat incorrect type: {type(galcat)}')

        if not -360. <= ra <= 360.:
            raise ValueError(f'ra={ra} not in valid bounds: [-360, 360]')
        if not -90. <= dec <= 90.:
            raise ValueError(f'dec={dec} not in valid bounds: [-90, 90]')
        if z < 0.:
            raise ValueError(f'z={z} must be greater than 0')

        self.unique_id = unique_id
        self.ra = ra
        self.dec = dec
        self.z = z
        self.galcat = galcat

    def save(self, filename, **kwargs):
        """Saves GalaxyCluster object to filename using Pickle"""
        with open(filename, 'wb') as fin:
            pickle.dump(self, fin, **kwargs)

    def __repr__(self):
        """Generates string for print(GalaxyCluster)"""
        output = f'GalaxyCluster {self.unique_id}: ' +\
                 f'(ra={self.ra}, dec={self.dec}) at z={self.z}\n' +\
                 f'> {len(self.galcat)} source galaxies\n> With columns:'
        for colname in self.galcat.colnames:
            output += f' {colname}'
        return output
