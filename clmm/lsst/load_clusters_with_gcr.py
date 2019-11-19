"""@file lsst.io.py
Contains functions to interact with the General Catalog Reader
"""
from collections.abc import Sequence
import os
import numpy as np
from astropy.table import Table
from clmm import GalaxyCluster

def load_catalog(catalog_name):
    """Loads a catalog from GCRCatalogs"""
    import GCRCatalogs
    return GCRCatalogs.load_catalog(catalog_name)

def _get_filters_from_range(filter_name, center, filter_range):
    """Creates filters for a certain quantity in the GCRCatalogs format. Creates a filter
    around a center value.
    
    Parameters
    ----------
    filter_name: str
        Quantity with which to filter
    center: float
        Value about which to construct the filter
    filter_range: length-2 array-like of floats
        Range around the center value about which to construct the filter
    """
     # checking filter_name type
    if not isinstance(filter_name, str):
        raise TypeError('filter_name not string.')
    # checking for valid filter
    if len(filter_range) != 2:
        raise TypeError('%s incorrect length: %s'%(key, len(ranges[key])))
    if filter_range[0] >= filter_range[1]:
        raise ValueError('%s invalid range'%key)

    return ['%s >= %d'%(filter_name, center+filter_range[0]),
            '%s < %d'%(filter_name, center+filter_range[1])]


def _calc_reduced_shear(shear, convergence):
    """Calculates reduced shear from shear and convergence"""
    return shear/(1-convergence)

def load_from_dc2_rect(nclusters, catalog_name, save_dir, ra_range=(-0.3, 0.3), dec_range=(-0.3, 0.3),
                       z_range=(0.1, 1.5), verbose=False):
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
        Range of right ascension values (in degrees) to cut galaxies around the cluster center
    dec_range: length-2 array-like of floats, optional
        Range of declination values (in degrees)to cut galaxies around the cluster center
    z_range: length-2 array-like of floats, optional
        Range of redshift values to cut galaxies around the cluster center
    verbose: bool
        Sets the function to print the id of each cluster while loading
    """

    # check that ra, dec are within bounds
    for i in ra_range:
        if not -360. <= i <= 360.:
            raise ValueError(r'ra %s not in valid bounds: [-360, 360]'%i)
    for i in dec_range:
        if not -90. <= i <= 90.:
            raise ValueError(r'dec %s not in valid bounds: [-90, 90]'%i)


    # load GCR catalog
    catalog = load_catalog(catalog_name)

    halos = catalog.get_quantities(['galaxy_id', 'halo_mass', 'redshift', 'ra', 'dec'],
                                   filters=['halo_mass > 1e14', 'is_central==True'])

    # generate GalaxyCluster objects
    for i in np.random.choice(range(len(halos)), nclusters, replace=False):
        # specify cluster information
        cl_id = halos['galaxy_id'][i]
        cl_ra = halos['ra'][i]
        cl_dec = halos['dec'][i]
        cl_z = halos['redshift'][i]

        if verbose:
            print('Loading cluster %s'%cl_id)

        # get galaxies around cluster
        
        filters = (_get_filters_from_range('ra', cl_ra, ra_range) + 
                   _get_filters_from_range('dec', cl_dec, dec_range) + 
                   _get_filters_from_range('redshift', cl_z, z_range))

        gals = catalog.get_quantities(['galaxy_id', 'ra', 'dec',
                                       'shear_1', 'shear_2',
                                       'redshift', 'convergence'],
                                      filters=filters)

        # calculate reduced shear
        g1 = _calc_reduced_shear(gals['shear_1'], gals['convergence'])
        g2 = _calc_reduced_shear(gals['shear_2'], gals['convergence'])

        # store the results into an astropy table
        t = Table([gals['galaxy_id'], gals['ra'], gals['dec'],
                   2*g1, 2*g2,
                   gals['redshift'], gals['convergence']],
                  names=('galaxy_id', 'ra', 'dec', 'e1', 'e2', 'z', 'kappa'))

        # save as GalaxyCluster object
        c = GalaxyCluster(unique_id=int(cl_id), ra=cl_ra, dec=cl_dec, z=cl_z,
                          galcat=t)

        c.save(os.path.join(save_dir, '%i.p'%cl_id))
