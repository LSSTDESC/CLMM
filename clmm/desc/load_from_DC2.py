import os
import numpy as np
import GCRCatalogs
from astropy.table import Table
from clmm.structures import GalaxyCluster

def load_from_dc2(N, catalog, save_dir, verbose=False):
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

        gals = catalog.get_quantities(['galaxy_id', 'ra', 'dec', 'shear_1', 'shear_2',
                                       'redshift', 'convergence'], filters=(coord_filters + z_filters))

        # calculate reduced shear
        g1 = gals['shear_1']/(1.-gals['convergence'])
        g2 = gals['shear_2']/(1.-gals['convergence'])

        # store the results into an astropy table
        t = Table([gals['galaxy_id'],gals['ra'],gals['dec'],
                   2*g1, 2*g2,
                   gals['redshift'],gals['convergence']],
                  names=('gal_id','gal_ra','gal_dec', 'gal_e1', 'gal_e2', 'gal_z', 'gal_kappa'))


        c = GalaxyCluster(cl_id=cl_id, cl_ra=cl_ra, cl_dec=cl_dec,
                          cl_z=cl_z, cl_richness=None,
                          gal_cat=t)

        c.save(os.path.join(save_dir, '%s.p'%cl_id))

