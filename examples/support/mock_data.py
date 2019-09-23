"""Functions to generate mock source galaxy distributions to demo lensing code"""
import numpy as np
from astropy.table import Table
from scipy import integrate
from scipy.interpolate import interp1d
import clmm


def compute_photoz_pdfs(galaxy_catalog, photoz_sigma_unscaled, ngals):
    r"""Add photo-z errors and compute photo-z pdfs for each source galaxy

    We compute the photo-z error at a given redshift using

    ..math::
        \sigma_{\rm pz} = \sigma_{\rm pz, unscaled}(1+z)

    We then define the photo-z PDF as a Gaussian, centered on the true redshift
    of the cluster with a width given by :math:`\sigma_{\rm pz}` defined over
    the redshift range of :math:`z_{\rm true} \pm 0.5`.

    Parameters
    ----------
    galaxy_catalog : astropy.table.Table
        Source galaxy catalog
    photoz_sigma_unscaled : float
        Amount of scatter in the photo-zs
    ngals : float
        The number of source galaxies to draw

    Returns
    -------
    galaxy_catalog: astropy.table.Table
        Source galaxy catalog now with photoz errors and pdfs
    """
    galaxy_catalog['pzsigma'] = photoz_sigma_unscaled*(1.+galaxy_catalog['ztrue'])
    galaxy_catalog['z'] = galaxy_catalog['ztrue'] + \
                          galaxy_catalog['pzsigma']*np.random.standard_normal(ngals)

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
            val, _ = integrate.quad(func, cluster_z+0.1, zmax)
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
        and H0

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


def _find_aphysical_galaxies(galaxy_catalog):
    r"""Finds the galaxies that have aphysical derived values due to large systematic choices.

    Currently checks the following conditions
    e1 \in [-1, 1]
    e2 \in [-1, 1]
    This was converted to a seperate function to allow for ease of extension without needing
    to change the same code in multiple locations.

    Parameters
    ----------
    galaxy_catalog : astropy.table.Table
        Galaxy source catalog

    Returns
    -------
    nbad : int
        The number of aphysical galaxies in galaxy_catalog
    badgals : array_like
        A list of the indicies in galaxy_catalog that need to be redrawn
    """
    badgals = np.where((np.abs(galaxy_catalog['e1']) > 1.0) |
                       (np.abs(galaxy_catalog['e2']) > 1.0)
                      )[0]
    nbad = len(badgals)
    return nbad, badgals


def generate_galaxy_catalog(cluster_m, cluster_z, cluster_c, cosmo, ngals, mdef, zsrc,
                            zsrc_max=7., shapenoise=None, photoz_sigma_unscaled=None, nretry=5):
    """Generates a mock dataset of sheared background galaxies.

    This function also ensure that it does not return any nonsensical values for derived
    properties. We re-draw all galaxies with e1 or e2 outside the bounds of [-1, 1].
    After 5 (default) attempts to re-draw these properties, we return the catalog
    and throw a warning.

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
        and H0
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
    photoz_sigma_unscaled : float, optional
        If set, applies photo-z errors to source redshifts
    nretry : int, optional
        The number of times that we re-draw each galaxy with non-sensical derived properties

    Returns
    -------
    galaxy_catalog : astropy.table.Table
        Table of source galaxies with drawn and derived properties required for lensing studies

    Notes
    -----
    Much of this code in this function was adapted from the Dallas group
    """
    params = {'cluster_m' : cluster_m, 'cluster_z' : cluster_z, 'cluster_c' : cluster_c,
              'cosmo' : cosmo, 'mdef' : mdef, 'zsrc' : zsrc, 'zsrc_max' : zsrc_max,
              'shapenoise' : shapenoise, 'photoz_sigma_unscaled' : photoz_sigma_unscaled}
    galaxy_catalog = _generate_galaxy_catalog(ngals=ngals, **params)

    # Check for bad galaxies and replace them
    for i in range(nretry):
        nbad, badids = _find_aphysical_galaxies(galaxy_catalog)
        if nbad < 1:
            break
        replacements = _generate_galaxy_catalog(ngals=nbad, **params)
        galaxy_catalog[badids] = replacements

    # Final check to see if there are bad galaxies left
    nbad, _ = _find_aphysical_galaxies(galaxy_catalog)
    if nbad > 1:
        print("Not able to remove {} aphysical objects after {} iterations".format(nbad, nretry))

    # Now that the catalog is final, add an id column
    galaxy_catalog['id'] = np.arange(ngals)
    return galaxy_catalog


def _generate_galaxy_catalog(cluster_m, cluster_z, cluster_c, cosmo, ngals, mdef, zsrc,
                             zsrc_max=7., shapenoise=None, photoz_sigma_unscaled=None):
    """A private function that skips the sanity checks on derived properties. This
    function should only be used when called directly from `generate_galaxy_catalog`.
    Takes the same parameters and returns the same things as the before mentioned function.
    """
    # Set the source galaxy redshifts
    galaxy_catalog = draw_sources_redshifts(zsrc, ngals, cluster_z, zsrc_max)

    # Add photo-z errors and pdfs to source galaxy redshifts
    if photoz_sigma_unscaled is not None:
        galaxy_catalog = compute_photoz_pdfs(galaxy_catalog, photoz_sigma_unscaled, ngals)


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

    if photoz_sigma_unscaled is not None:
        return galaxy_catalog['ra', 'dec', 'e1', 'e2', 'z', 'pzbins', 'pzpdf']
    return galaxy_catalog['ra', 'dec', 'e1', 'e2', 'z']
