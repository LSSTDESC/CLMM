"""@file lsst.io.py
Contains functions to interact with the General Catalog Reader
"""
from collections.abc import Sequence
import os
import numpy as np
from astropy.table import Table
from ..galaxycluster import GalaxyCluster
from ..modeling import get_reduced_shear_from_convergence

def load_GCR_catalog(catalog_name):
    """Loads a catalog from GCRCatalogs"""
    import GCRCatalogs
    return GCRCatalogs.load_catalog(catalog_name)

class _test_catalog():
    """Blank test catalog designed to mimic a GCRCatalogs catalog. For testing only."""
    def __init__(self, n=10):
        self.n = n
    def __len__(self):
        return self.n
    def get_quantities(self, quantities, *args, **kwargs):
        out_dict = {}
        for q in quantities:
            if q == 'galaxy_id':
                out_dict[q] = np.arange(self.n)
            else:
                out_dict[q] = np.arange(0, 1, 1./self.n)
        return out_dict

def _make_GCR_filter(filter_name, low_bound, high_bound):
    """Create a filter for a specified range of a certain quantity in the GCRCatalogs format.
    
    Parameters
    ----------
    filter_name: str
        Quantity to filter
    low_bound, high_bound: float
        Range of values that the quantity is restricted to
    """
     # checking filter_name type
    if not isinstance(filter_name, str):
        raise TypeError('filter_name not string.')
    # checking for valid filter
    if low_bound >= high_bound:
        raise ValueError('Invalid range: [%d, %d]'%(low_bound, high_bound))

    return ['%s >= %d'%(filter_name, low_bound),
            '%s < %d'%(filter_name, high_bound)]

def load_from_dc2(nclusters, catalog_name, save_dir, ra_range=(-0.3, 0.3), dec_range=(-0.3, 0.3),
                  z_range=(0.1, 1.5), verbose=False, _reader='GCR'):
    """Load random clusters from DC2 using GCRCatalogs

    This function saves a set of random clusters loaded from a DC2 catalog. The galaxies
    selected for each cluster are cut within a rectangular projected region around the true
    cluster center. The parameters for this cut are specified by the ra_range, dec_range,
    and z_range arguments.

    Parameters
    ----------
    nclusters: int
        Number of clusters to load and save.
    catalog_name: str
        Name of catalog (without '.yaml')
    save_dir: str
        Path to directory in which to save cluster objects
    ra_range: length-2 array-like of floats, optional
        Range of right ascension values (in degrees) to cut galaxies relative to the cluster center
    dec_range: length-2 array-like of floats, optional
        Range of declination values (in degrees)to cut galaxies relative to the cluster center
    z_range: length-2 array-like of floats, optional
        Range of redshift values to cut galaxies relative to the cluster center
    verbose: bool
        Sets the function to print the id of each cluster while loading
    _reader: str
        Reader argument used for testing. In practice, should be default to 'GCR'.
    """
    # check that ranges are length-2
    if len(ra_range)!=2:
        raise ValueError('ra_range incorrect length: %i'%len(ra_range))
    if len(dec_range)!=2:
        raise ValueError('dec_range incorrect length: %i'%len(dec_range))
    if len(z_range)!=2:
        raise ValueError('z_range incorrect length: %i'%len(z_range))

    # load catalog
    if _reader=='GCR':
        catalog = load_GCR_catalog(catalog_name)
    elif _reader=='test':
        catalog = _test_catalog(10)
    else:
        raise ValueError('Invalid reader name: %s'%_reader)
        
    # check range of nclusters
    if (nclusters<=0) | (nclusters>len(catalog)):
        raise ValueError('nclusters value out of range: %i'%nclusters)

    halos = catalog.get_quantities(['galaxy_id', 'halo_mass', 'redshift', 'ra', 'dec'],
                                   filters=['halo_mass > 1e14', 'is_central==True'])

    # generate GalaxyCluster objects
    for i in np.random.choice(range(len(halos['galaxy_id'])), nclusters, replace=False):
        # specify cluster information
        cl_id = halos['galaxy_id'][i]
        cl_ra = halos['ra'][i]
        cl_dec = halos['dec'][i]
        cl_z = halos['redshift'][i]

        if verbose:
            print(f'Loading cluster {cl_id}')

        # get galaxies around cluster
        
        filters = (_make_GCR_filter('ra', cl_ra + ra_range[0], cl_ra + ra_range[1]) + 
                   _make_GCR_filter('dec', cl_dec + dec_range[0], cl_dec + dec_range[1]) + 
                   _make_GCR_filter('redshift', cl_z + z_range[0], cl_z + z_range[1]))

        gals = catalog.get_quantities(['galaxy_id', 'ra', 'dec',
                                       'shear_1', 'shear_2',
                                       'redshift', 'convergence'],
                                      filters=filters)

        # calculate reduced shear
        g1 = get_reduced_shear_from_convergence(gals['shear_1'], gals['convergence'])
        g2 = get_reduced_shear_from_convergence(gals['shear_2'], gals['convergence'])

        # store the results into an astropy table
        t = Table([gals['galaxy_id'], gals['ra'], gals['dec'],
                   2*g1, 2*g2,
                   gals['redshift'], gals['convergence']],
                  names=('galaxy_id', 'ra', 'dec', 'e1', 'e2', 'z', 'kappa'))

        # save as GalaxyCluster object
        c = GalaxyCluster(unique_id=int(cl_id), ra=cl_ra, dec=cl_dec, z=cl_z,
                          galcat=t)

        c.save(os.path.join(save_dir, '%i.p'%cl_id))
