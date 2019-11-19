"""@file lsst.io.py
Contains functions to interact with the General Catalog Reader
"""

import os
import numpy as np
from astropy.table import Table
from clmm import GalaxyCluster

def load_from_dc2_rect(N, catalog, save_dir, ra_range=(-0.3, 0.3), dec_range=(-0.3, 0.3),
                       z_range=(0.1, 1.5), verbose=False):
    """Load random clusters from DC2 using GCRCatalogs
    
    This function saves a set of random clusters loaded from a DC2 catalog. The galaxies
    selected for each cluster are cut within a rectangular projected region around the true 
    cluster center. The parameters for this cut are specified by the ra_range, dec_range,
    and z_range arguments.
    
    Parameters
    ----------
    N: int
        Number of clusters to load and save. 
    catalog: str
        Name of catalog (without '.yaml')
    save_dir: str
        Path to directory in which to save cluster objects
    ra_range: length-2 tuple of floats, optional
        Range of right ascension values (in degrees) to cut galaxies around the cluster center
    dec_range: length-2 tuple of floats, optional
        Range of declination values (in degrees)to cut galaxies around the cluster center
    z_range: length-2 tuple of floats, optional
        Range of redshift values (in degrees)to cut galaxies around the cluster center
    verbose: bool
        Sets the function to print the id of each cluster while loading
    """
    import GCRCatalogs

    catalog = GCRCatalogs.load_catalog(catalog)
    
    halos = catalog.get_quantities(['galaxy_id', 'halo_mass', 'redshift','ra', 'dec'],
                                   filters=['halo_mass > 1e14','is_central==True'])

    for i in np.random.choice(range(len(halos)), N, replace=False):
        # specify cluster information
        cl_id = halos['galaxy_id'][i]
        cl_ra = halos['ra'][i]
        cl_dec = halos['dec'][i]
        cl_z = halos['redshift'][i]
        
        if verbose:
            print('Loading cluster %s'%cl_id)

        # get galaxies around cluster
        coord_filters = [
            'ra >= %s'%(cl_ra+ra_range[0]),
            'ra < %s'%(cl_ra+ra_range[1]),
            'dec >= %s'%(cl_dec+dec_range[0]),
            'dec < %s'%(cl_dec+dec_range[1]),
        ]
        z_filters = ['redshift >= %s'%(cl_z+z_range[0]),
                     'redshift < %s'%(cl_z+z_range[1])]

        gals = catalog.get_quantities(['galaxy_id', 'ra', 'dec', 
                                       'shear_1', 'shear_2',
                                       'redshift', 'convergence'], 
                                      filters=(coord_filters + z_filters))

        # calculate reduced shear
        g1 = gals['shear_1']/(1.-gals['convergence'])
        g2 = gals['shear_2']/(1.-gals['convergence'])

        # store the results into an astropy table
        t = Table([gals['galaxy_id'], gals['ra'], gals['dec'],
                   2*g1, 2*g2,
                   gals['redshift'], gals['convergence']],
                  names=('galaxy_id', 'ra','dec', 'e1', 'e2', 'z', 'kappa'))

        # save as GalaxyCluster object
        c = GalaxyCluster(unique_id=int(cl_id), ra=cl_ra, dec=cl_dec, z=cl_z,
                          galcat=t)

        c.save(os.path.join(save_dir, '%i.p'%cl_id))

