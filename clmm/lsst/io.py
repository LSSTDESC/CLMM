"""@file lsst.io.py
Contains functions to interact with the General Catalog Reader
"""

import os
import numpy as np
from astropy.table import Table
from clmm import GalaxyCluster

def load_from_dc2(N, catalog, save_dir, verbose=False):
    import GCRCatalogs

    catalog = GCRCatalogs.load_catalog(catalog)
    
    halos = catalog.get_quantities(['galaxy_id', 'halo_mass', 'redshift','ra', 'dec'],
                                   filters=['halo_mass > 1e14','is_central==True'])

    for i in np.random.choice(range(len(halos)), N, replace=False):

        cl_id = halos['galaxy_id'][i]
        cl_ra = halos['ra'][i]
        cl_dec = halos['dec'][i]
        cl_z = halos['redshift'][i]
        cl_m = halos['halo_mass'][i]
        
        if verbose:
            print('Loading cluster %s'%cl_id)

        # get galaxies around cluster
        ra_min, ra_max = cl_ra-0.3, cl_ra+0.3
        dec_min, dec_max = cl_dec-0.3, cl_dec+0.3
        z_min = cl_z + 0.1
        z_max = 1.5

        coord_filters = [
            'ra >= %s'%ra_min,
            'ra < %s'%ra_max,
            'dec >= %s'%dec_min,
            'dec < %s'%dec_max,
        ]
        z_filters = ['redshift >= %s'%z_min,
                     'redshift < %s'%z_max]

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


        c = GalaxyCluster(unique_id=cl_id, ra=cl_ra, dec=cl_dec, z=cl_z,
                          galcat=t)

        c.save(os.path.join(save_dir, '%s.p'%cl_id))

