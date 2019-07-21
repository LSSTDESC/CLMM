import pickle
from astropy.table import Table

class Cluster():
    def __init__(self, id=None, 
                       ra=None, dec=None,
                       z=None, richness=None,
                       gals=Table()
                ):
        self.id = id
        self.ra = ra
        self.dec = dec
        self.z = z
        self.richness = richness
        self.gals = gals

    def save(self, filename, **kwargs):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=3, **kwargs)

    def load(self, filename, **kwargs):
        with open(filename, 'rb') as f:
            new_cl = pickle.load(f, **kwargs)
        self.__dict__.update(new_cl.__dict__)
	

