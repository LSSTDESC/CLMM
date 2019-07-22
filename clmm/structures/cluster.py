import pickle
import astropy.table
from astropy.table import Table

class GalaxyCluster():
    '''
    Object that contains the galaxy cluster metadata and background galaxy data
    Attributes
    ----------
    cl_id (int): Unique identifier of the galaxy cluster
    cl_name (string): Name of galaxy cluster
    cl_ra (float): Right ascension of galaxy cluster center (in degrees)
    cl_dec (float): Declination of galaxy cluster center (in degrees)
    cl_z (float): Redshift of galaxy cluster center
    cl_richness (int): Number of member galaxies above prescribes luminosity cut
    '''
    def __init__(self, cl_id: int=None,
                       cl_name: str=None,
                       cl_ra: float=None, 
                       cl_dec: float=None,
                       cl_z: float=None, 
                       cl_richness: int=None,
                       gal_cat: astropy.table.table.Table=Table()
                ):
        self.cl_id = cl_id
        self.cl_name = cl_name
        self.cl_ra = cl_ra
        self.cl_dec = cl_dec
        self.cl_z = cl_z
        self.cl_richness = cl_richness
        self.gal_cat = gal_cat

    def save(self, filename, **kwargs):
        """Saves GalaxyCluster object to filename using Pickle"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f, **kwargs)

    def load(self, filename, **kwargs):
        """Loads GalaxyCluster object from filename using Pickle"""
        with open(filename, 'rb') as f:
            new_cl = pickle.load(f, **kwargs)
        self.__init__(**(new_cl.__dict__))
	

