import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy import integrate
from scipy.interpolate import interp1d
import clmm


def compute_photoz_pdfs(galaxy_catalog, photoz_ref, ngals):
    """Add photo-z errors and compute photo-z pdfs for each source galaxy

    Parameters
    ----------
    galaxy_catalog : astropy.table.Table
        Source galaxy catalog
    photoz_ref : float
        Amount of scatter in the photo-zs
    ngals : float
        The number of source galaxies to draw

    Returns
    -------
    galaxy_catalog: astropy.table.Table
        Source galaxy catalog now with photoz errors and pdfs
    """
    galaxy_catalog['pzsigma'] = photoz_ref*(1.+galaxy_catalog['ztrue'])
    galaxy_catalog['z'] = galaxy_catalog['ztrue'] + galaxy_catalog['pzsigma']*np.random.standard_normal(ngals)

    pzbins_grid, pzpdf_grid = [], []
    for row in galaxy_catalog:
        zmin, zmax = row['ztrue'] - 0.5, row['ztrue'] + 0.5
        zbins = np.arange(zmin, zmax, 0.03)
        pzbins_grid.append(zbins)
        pzpdf_grid.append(np.exp(-0.5*((zbins - row['ztrue'])/row['pzsigma'])**2)/np.sqrt(2*np.pi*row['pzsigma']**2))
    galaxy_catalog['pzbins'] = pzbins_grid
    galaxy_catalog['pzpdf'] = pzpdf_grid

    return galaxy_catalog


def draw_sources_redshifts(zsrc, ngals, cluster_z, zsrc_max):
    """Set source galaxy redshifts either set to a fixed value or draw from Chang et al. 2013.

    Uses a sampling technique found in Numerical Recipes in C, Chap 7.2: Transformation Method.
    Pulling out random values from a given probability distribution.

    Parameters
    ----------
    zsrc : float or str
        Choose the source galaxy distribution to be fixed or according to a predefined
        distribution.
        float : All sources galaxies at this fixed redshift
        str : Draws individual source gal redshifts from predefined distribution. Options
              are: chang13
    ngals : float
        The number of source galaxies to draw
    cluster_z : float
        The cluster redshift
    zsrc_max : float
        The max redshift to draw sources at

    Returns
    -------
    galaxy_catalog : astropy.table.Table
        The source galaxy catalog with redshifts
    """
    # Set zsrc to constant value
    if isinstance(zsrc, float):
        zsrc_list = np.ones(ngals)*zsrc

    # Draw zsrc from Chang et al. 2013
    elif zsrc == 'chang13':
        def pzfxn(z):
            """Redshift distribution function"""
            alpha, beta, z0 = 1.24, 1.01, 0.51
            return (z**alpha)*np.exp(-(z/z0)**beta)

        def integrated_pzfxn(zmax, func):
            """Integrated redshift distribution function for transformation method"""
            val, err = integrate.quad(func, cluster_z+0.1, zmax)
            return val
        vectorization_integrated_pzfxn = np.vectorize(integrated_pzfxn)

        zsrc_min = cluster_z + 0.1
        zsrc_domain = np.arange(zsrc_min, zsrc_max, 0.001)
        probdist = vectorization_integrated_pzfxn(zsrc_domain, pzfxn)

        uniform_deviate = np.random.uniform(probdist.min(), probdist.max(), ngals)
        zsrc_list = interp1d(probdist, zsrc_domain, kind='linear')(uniform_deviate)           

    # Invalid entry
    else:
        raise ValueError("zsrc must be a float or chang13. You set: {}".format(zsrc))

    return Table([zsrc_list, zsrc_list], names=('ztrue', 'z'))


def draw_galaxy_positions(galaxy_catalog, ngals, cluster_z, cosmo):
    """Draw positions of source galaxies around lens

    Parameters
    ----------
    galaxy_catalog : astropy.table.Table
        Source galaxy catalog
    ngals : float
        The number of source galaxies to draw
    cluster_z : float
        The cluster redshift
    cosmo : dict
        Dictionary of cosmological parameters. Must contain at least, Omega_c, Omega_b,
        h, and H0

    Returns
    -------
    galaxy_catalog : astropy.table.Table
        Source galaxy catalog with positions added
    """
    Dl = clmm.get_angular_diameter_distance_a(cosmo, 1./(1.+cluster_z))
    galaxy_catalog['x_mpc'] = np.random.uniform(-4., 4., size=ngals)
    galaxy_catalog['y_mpc'] = np.random.uniform(-4., 4., size=ngals)
    galaxy_catalog['r_mpc'] = np.sqrt(galaxy_catalog['x_mpc']**2 + galaxy_catalog['y_mpc']**2)
    galaxy_catalog['ra'] = -(galaxy_catalog['x_mpc']/Dl)*(180./np.pi)
    galaxy_catalog['dec'] = (galaxy_catalog['y_mpc']/Dl)*(180./np.pi)

    return galaxy_catalog

def generate_galaxy_catalog(cluster_m, cluster_z, cluster_c, cosmo, ngals, mdef, zsrc,
                            zsrc_max=7., shapenoise=None, photoz_ref=None):
    """Generates a mock dataset of sheared background galaxies.

    Parameters
    ----------
    cluster_m : float
        Cluster mass
    cluster_z : float
        Cluster redshift
    cluster_c : float
        Cluster concentration
    cosmo : dict
        Dictionary of cosmological parameters. Must contain at least, Omega_c, Omega_b,
        h, and H0
    ngals : float
        Number of galaxies to generate
    mdef : float
        Mass definition in terms of rho_XXX???
    zsrc : float or str
        Choose the source galaxy distribution to be fixed or according to a predefined
        distribution.
        float : All sources galaxies at this fixed redshift
        str : Draws individual source gal redshifts from predefined distribution. Options
              are: chang13
    zsrc_max : float, optional
        If source redshifts are drawn, the maximum source redshift
    shapenoise : float, optional
        If set, applies Gaussian shape noise to the galaxy shapes
    photoz_ref : float, optional
        If set, applies photo-z errors to source redshifts

    Returns
    -------
    galaxy_catalog : astropy.table.Table
        Table of source galaxies with drawn and derived properties required for lensing studies

    Notes
    -----
    Much of this code in this function was adapted from the Dallas group
    """
    # Set the source galaxy redshifts
    galaxy_catalog = draw_sources_redshifts(zsrc, ngals, cluster_z, zsrc_max)

    # Add photo-z errors and pdfs to source galaxy redshifts
    if photoz_ref is not None:
        galaxy_catalog = compute_photoz_pdfs(galaxy_catalog, photoz_ref, ngals)


    # Draw galaxy positions
    galaxy_catalog = draw_galaxy_positions(galaxy_catalog, ngals, cluster_z, cosmo)

    # Compute the shear on each source galaxy
    gamt = clmm.predict_reduced_tangential_shear(galaxy_catalog['r_mpc'], mdelta=cluster_m,
                                                 cdelta=cluster_c, z_cluster=cluster_z,
                                                 z_source=galaxy_catalog['z'], cosmo=cosmo,
                                                 Delta=mdef, halo_profile_parameterization='nfw',
                                                 z_src_model='single_plane')
    galaxy_catalog['gammat'] = gamt

    # Add shape noise to source galaxy shears
    if shapenoise is not None:
        galaxy_catalog['gammat'] += shapenoise*np.random.standard_normal(ngals)

    # Compute ellipticities
    galaxy_catalog['posangle'] = np.arctan2(galaxy_catalog['y_mpc'], galaxy_catalog['x_mpc'])
    galaxy_catalog['e1'] = -galaxy_catalog['gammat']*np.cos(2*galaxy_catalog['posangle'])
    galaxy_catalog['e2'] = -galaxy_catalog['gammat']*np.sin(2*galaxy_catalog['posangle'])

    # galaxy_catalog['id'] = np.arange(ngals)

    if photoz_ref is not None:
        return galaxy_catalog['ra', 'dec', 'e1', 'e2', 'z', 'pzbins', 'pzpdf'] 
    else:
        return galaxy_catalog['ra', 'dec', 'e1', 'e2', 'z']

