
import os
import numpy as np
import GCRCatalogs
from astropy.table import Table

from clmm.structures import GalaxyCluster

def load_from_dc2(N, catalog, save_dir):
    catalog = GCRCatalogs.load_catalog(catalog)
    halos = catalog.get_quantities(['galaxy_id', 'halo_mass', 'redshift','ra', 'dec'],
                                   filters=['halo_mass > 1e14','is_central==True'])

    for _ in range(N):
        i = np.random.randint(len(halos))

        id_cl = halos['galaxy_id'][i]
        ra_cl = halos['ra'][i]
        dec_cl = halos['dec'][i]
        z_cl = halos['redshift'][i]
        m_cl = halos['halo_mass'][i]

        # get galaxies around cluster
        ra_min, ra_max = ra_cl-0.3, ra_cl+0.3
        dec_min, dec_max = dec_cl-0.3, dec_cl+0.3
        z_min = z_cl + 0.1
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
                  names=('id','ra','dec', 'e1', 'e2', 'z', 'kappa'))


        c = GalaxyCluster(id=id_cl, ra=ra_cl, dec=dec_cl,
                          z=z_cl, richness=None,
                          gals=t)

        c.save(os.path.join(save_dir, '%s.p'%id_cl))

