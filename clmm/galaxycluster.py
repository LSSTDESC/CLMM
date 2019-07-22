"""@file galaxycluster.py
The GalaxyCluster class
"""

import pickle
import astropy.table
from astropy.table import Table
from typing import Union

class GalaxyCluster():
    '''
    Object that contains the galaxy cluster metadata and background galaxy data
    Attributes
    ----------
    unique_id (int or string): Unique identifier of the galaxy cluster
    ra (float): Right ascension of galaxy cluster center (in degrees)
    dec (float): Declination of galaxy cluster center (in degrees)
    z (float): Redshift of galaxy cluster center
    richness (int): Number of member galaxies above prescribes luminosity cut
    galcat (astropy Table): table of background galaxy data including galaxy_id, ra, dec, e1, e2, z, kappa
    '''
    def __init__(self, unique_id: Union[int,str]=None,
                       ra: float=None, 
                       dec: float=None,
                       z: float=None, 
                       richness: int=None,
                       galcat: astropy.table.table.Table=Table()
                ):
        self.unique_id = unique_id
        self.ra = ra
        self.dec = dec
        self.z = z
        self.richness = richness
        self.galcat = galcat

    def save(self, filename, **kwargs):
        """Saves GalaxyCluster object to filename using Pickle"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f, **kwargs)

    def load(self, filename, **kwargs):
        """Loads GalaxyCluster object from filename using Pickle"""
        with open(filename, 'rb') as f:
            new_cl = pickle.load(f, **kwargs)
        self.__init__(**(new_cl.__dict__))
	

