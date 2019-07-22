"""@file galaxycluster.py
The GalaxyCluster class
"""
import pickle
import astropy.table
from astropy.table import Table


class GalaxyCluster():
    '''
    Object that contains the galaxy cluster metadata and background galaxy data
    Attributes
    ----------
    id (int): Unique identifier of the galaxy cluster
    ra (float): Right ascension of galaxy cluster center (in degrees)
    dec (float): Declination of galaxy cluster center (in degrees)
    z (float): Redshift of galaxy cluster center (in degrees)
    richness (int): Number of member galaxies above prescribes luminosity cut
    '''
    def __init__(self, id: int=None,
                       ra: float=None, dec: float=None,
                       z: float=None, richness: int=None,
                       gals: astropy.table.table.Table=Table()
                ):
        self.id = id
        self.ra = ra
        self.dec = dec
        self.z = z
        self.richness = richness
        self.gals = gals
        #raise AttributeError('meh')

    def save(self, filename, **kwargs):
        """Saves GalaxyCluster object to filename using Pickle"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f, **kwargs)

    def load(self, filename, **kwargs):
        """Loads GalaxyCluster object from filename using Pickle"""
        with open(filename, 'rb') as f:
            new_cl = pickle.load(f, **kwargs)
        self.__init__(**(new_cl.__dict__))
