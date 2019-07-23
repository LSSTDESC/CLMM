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
            raise TypeError('unique_id incorrect type: %s'%type(unique_id))
        if not isinstance(ra, float):
            raise TypeError('ra incorrect type: %s'%type(ra))
        if not isinstance(dec, float):
            raise TypeError('dec incorrect type: %s'%type(dec))
        if not isinstance(galcat, Table):
            raise TypeError('galcat incorrect type: %s'%type(galcat))

        if not -360. <= ra <= 360.:
            raise ValueError(r'ra %s not in valid bounds: [-360, 360]'%ra)
        if not -90. <= dec <= 90.:
            raise ValueError(r'dec %s not in valid bounds: [-90, 90]'%dec)
        if z < 0.:
            raise ValueError(r'z %s must be greater than 0'%z)

        self.unique_id = unique_id
        self.ra = ra
        self.dec = dec
        self.z = z
        self.galcat = galcat

    def save(self, filename, **kwargs):
        """Saves GalaxyCluster object to filename using Pickle"""
        with open(filename, 'wb') as fin:
            pickle.dump(self, fin, **kwargs)
