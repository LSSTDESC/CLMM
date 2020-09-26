"""Functions to generate mock source galaxy distributions to demo lensing code"""
import numpy as np
from clmm import GCData
from scipy import integrate
from scipy.interpolate import interp1d
from astropy import units
from clmm.modeling import predict_reduced_tangential_shear, angular_diameter_dist_a1a2

def generate_galaxy_catalog(cluster_m, cluster_z, cluster_c, cosmo, ngals, Delta_SO, zsrc, zsrc_min=0.4,
                            zsrc_max=7., shapenoise=None, photoz_sigma_unscaled=None, nretry=5):
    """Generates a mock dataset of sheared background galaxies.

    We build galaxy catalogs following a series of steps.

    1. Draw true redshifts of the source galaxy population. This step is described by the
    parameters `zsrc` and `zsrc_max`. `zsrc` can be a `float` in which case every source is
    at the given redshift or a `str` describing a specific model to use for the source
    distribution. Currently, the only supported model for source galaxy distribution is that
    of Chang et al. 2013 arXiv:1305.0793. When a model is used to describe the distribution,
    `zsrc_max` is the maximum allowed redshift of a source galaxy.

    2. Apply photometric redshift errors to the source galaxy population. This step is
    described by the parameter `photoz_sigma_unscaled`. If this parameter is set to a float,
    we add Gaussian uncertainty to the source redshift

    ..math::
        z \sim \mathcal{N}\left(z^{\rm true},
        \sigma_{\rm photo-z}^{\rm unscaled}(1+z^{\rm true}) \right)

    We additionally include two columns in the output catalog, `pzbins` and `pzpdf` which
    desribe the photo-z distribution as a Gaussian centered at :math:`z^{\rm true} with a
    width :math:`\sigma_{\rm photo-z} = \sigma_{\rm photo-z}^{\rm unscaled}(1+z^{\rm true})`

    If `photoz_sigma_unscaled` is `None`, the `z` column in the output catalog is the true
    redshift.

    3. Draw galaxy positions. Positions are drawn in a square box around the lens position with
    a side length of 4 Mpc. We then convert to right ascension and declination using the
    cosmology defined in `cosmo`.

    4. We predict the reduced tangential shear of each using the radial distances of each source
    from the lens, the source redshifts, and the lens mass, concentration, and redshift. In the
    given cosmology for an NFW halo.

    5. We apply shape noise to the tangential shears. This is described by the parameter
    `shapenoise`. If this is set to a float, we apply a Gaussian perturbation to the
    tangential shear with a width of `shapenoise`.

    6. Finally, we compute the two components of the ellipticity, e1 and e2.

    If the shape noise parameter is high, we may draw nonsensical values for ellipticities. We 
    ensure that we does not return any nonsensical values for derived properties. We re-draw
    all galaxies with e1 or e2 outside the bounds of [-1, 1]. After 5 (default) attempts to
    re-draw these properties, we return the catalog as is and throw a warning.

    Parameters
    ----------
    cluster_m : float
        Cluster mass
    cluster_z : float
        Cluster redshift
    cluster_c : float
        Cluster concentration in the same mass definition as Delta_SO
    cosmo : dict
        Dictionary of cosmological parameters. Must contain at least, Omega_c, Omega_b,
        and H0
    ngals : float
        Number of galaxies to generate
    Delta_SO : float
        Overdensity density contrast used to compute the cluster mass and concentration. The
        spherical overdensity mass is computed as the mass enclosed within the radius
        :math:`R_{\Delta{\rm SO}}` where the mean density is :math:`\Delta_{\rm SO}` times
        the mean density of the Universe at the cluster redshift
        :math:`M_{\Delta{\rm SO}}=4/3\pi\Delta_{\rm SO}\rho_{m}(z_{\rm lens})R_{\Delta{\rm SO}}^3`
    zsrc : float or str
        Choose the source galaxy distribution to be fixed or drawn from a predefined distribution.
        float : All sources galaxies at this fixed redshift
        str : Draws individual source gal redshifts from predefined distribution. Options
              are: chang13
    zsrc_min : float, optional
        The minimum source redshift allowed.
    zsrc_max : float, optional
        If source redshifts are drawn, the maximum source redshift
    shapenoise : float, optional
        If set, applies Gaussian shape noise to the galaxy shapes with a width set by `shapenoise`
    photoz_sigma_unscaled : float, optional
        If set, applies photo-z errors to source redshifts
    nretry : int, optional
        The number of times that we re-draw each galaxy with non-sensical derived properties

    Returns
    -------
    galaxy_catalog : clmm.GCData
        Table of source galaxies with drawn and derived properties required for lensing studies

    Notes
    -----
    Much of this code in this function was adapted from the Dallas group
    """
    params = {'cluster_m' : cluster_m, 'cluster_z' : cluster_z, 'cluster_c' : cluster_c,
              'cosmo' : cosmo, 'Delta_SO' : Delta_SO, 'zsrc' : zsrc, 'zsrc_min' : zsrc_min,
              'zsrc_max' : zsrc_max,'shapenoise' : shapenoise, 'photoz_sigma_unscaled' : photoz_sigma_unscaled}
    galaxy_catalog = _generate_galaxy_catalog(ngals=ngals, **params)
    # Check for bad galaxies and replace them
    for i in range(nretry):
        nbad, badids = _find_aphysical_galaxies(galaxy_catalog, zsrc_min)
        if nbad < 1:
            break
        replacements = _generate_galaxy_catalog(ngals=nbad, **params)
        galaxy_catalog[badids] = replacements

    # Final check to see if there are bad galaxies left
    nbad, _ = _find_aphysical_galaxies(galaxy_catalog, zsrc_min)
    if nbad > 1:
        print("Not able to remove {} aphysical objects after {} iterations".format(nbad, nretry))

    # Now that the catalog is final, add an id column
    galaxy_catalog['id'] = np.arange(ngals)
    return galaxy_catalog


def _generate_galaxy_catalog(cluster_m, cluster_z, cluster_c, cosmo, ngals, Delta_SO, zsrc,
                             zsrc_min=0.4, zsrc_max=7., shapenoise=None, photoz_sigma_unscaled=None):
    """A private function that skips the sanity checks on derived properties. This
    function should only be used when called directly from `generate_galaxy_catalog`.
    Takes the same parameters and returns the same things as the before mentioned function.

    For a more detailed description of each of the parameters, see the documentation of
    `generate_galaxy_catalog`.
    """
    # Set the source galaxy redshifts
    galaxy_catalog = _draw_source_redshifts(zsrc, cluster_z, zsrc_min, zsrc_max, ngals)

    # Add photo-z errors and pdfs to source galaxy redshifts
    if photoz_sigma_unscaled is not None:
        galaxy_catalog = _compute_photoz_pdfs(galaxy_catalog, photoz_sigma_unscaled, ngals)
    # Draw galaxy positions
    galaxy_catalog = _draw_galaxy_positions(galaxy_catalog, ngals, cluster_z, cosmo)
    # Compute the shear on each source galaxy
    gamt = predict_reduced_tangential_shear(galaxy_catalog['r_mpc'], mdelta=cluster_m,
                                            cdelta=cluster_c, z_cluster=cluster_z,
                                            z_source=galaxy_catalog['ztrue'], cosmo=cosmo,
                                            delta_mdef=Delta_SO, halo_profile_model='nfw',
                                            z_src_model='single_plane')
    galaxy_catalog['gammat'] = gamt
    galaxy_catalog['gammax'] = np.zeros(ngals)

    # Add shape noise to source galaxy shears
    if shapenoise is not None:
        galaxy_catalog['gammat'] += shapenoise*np.random.standard_normal(ngals)
        galaxy_catalog['gammax'] += shapenoise*np.random.standard_normal(ngals)

    # Compute ellipticities
    galaxy_catalog['posangle'] = np.arctan2(galaxy_catalog['y_mpc'], galaxy_catalog['x_mpc'])

    galaxy_catalog['e1'] = -galaxy_catalog['gammat']*np.cos(2*galaxy_catalog['posangle']) \
                           + galaxy_catalog['gammax']*np.sin(2*galaxy_catalog['posangle'])
    galaxy_catalog['e2'] = -galaxy_catalog['gammat']*np.sin(2*galaxy_catalog['posangle']) \
                           - galaxy_catalog['gammax']*np.cos(2*galaxy_catalog['posangle'])

    if photoz_sigma_unscaled is not None:
        return galaxy_catalog['ra', 'dec', 'e1', 'e2', 'z', 'ztrue', 'pzbins', 'pzpdf']
    return galaxy_catalog['ra', 'dec', 'e1', 'e2', 'z', 'ztrue']


def _draw_source_redshifts(zsrc, cluster_z, zsrc_min, zsrc_max, ngals):
    """Set source galaxy redshifts either set to a fixed value or draw from a predefined
    distribution. Return a table (GCData) of the source galaxies

    Uses a sampling technique found in Numerical Recipes in C, Chap 7.2: Transformation Method.
    Pulling out random values from a given probability distribution.
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
            val, _ = integrate.quad(func, zsrc_min, zmax)
            return val
        vectorization_integrated_pzfxn = np.vectorize(integrated_pzfxn)

        zsrc_domain = np.arange(zsrc_min, zsrc_max, 0.001)
        probdist = vectorization_integrated_pzfxn(zsrc_domain, pzfxn)

        uniform_deviate = np.random.uniform(probdist.min(), probdist.max(), ngals)
        zsrc_list = interp1d(probdist, zsrc_domain, kind='linear')(uniform_deviate)
    
    # Draw zsrc from a uniform distribution between zmin and zmax
    elif zsrc == 'uniform':
        zsrc_list = np.random.uniform(cluster_z + 0.1, zsrc_max, ngals)

    # Invalid entry
    else:
        raise ValueError("zsrc must be a float or chang13. You set: {}".format(zsrc))

    return GCData([zsrc_list, zsrc_list], names=('ztrue', 'z'))


def _compute_photoz_pdfs(galaxy_catalog, photoz_sigma_unscaled, ngals):
    r"""Add photo-z errors and PDFs to the mock catalog."""
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


def _draw_galaxy_positions(galaxy_catalog, ngals, cluster_z, cosmo):
    """Draw positions of source galaxies around lens

    We draw physical x and y positions from uniform distribution with -4 and 4 Mpc of the
    lensing cluster center. We then convert these to RA and DEC using the supplied cosmology

    Parameters
    ----------
    galaxy_catalog : clmm.GCData
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
    galaxy_catalog : clmm.GCData
        Source galaxy catalog with positions added
    """
    Dl = angular_diameter_dist_a1a2(cosmo, 1./(1.+cluster_z))*units.pc.to(units.Mpc)
    galaxy_catalog['x_mpc'] = np.random.uniform(-4., 4., size=ngals)
    galaxy_catalog['y_mpc'] = np.random.uniform(-4., 4., size=ngals)
    galaxy_catalog['r_mpc'] = np.sqrt(galaxy_catalog['x_mpc']**2 + galaxy_catalog['y_mpc']**2)
    galaxy_catalog['ra'] = -(galaxy_catalog['x_mpc']/Dl)*(180./np.pi)
    galaxy_catalog['dec'] = (galaxy_catalog['y_mpc']/Dl)*(180./np.pi)

    return galaxy_catalog


def _find_aphysical_galaxies(galaxy_catalog, zsrc_min):
    r"""Finds the galaxies that have aphysical derived values due to large systematic choices.

    Currently checks the following conditions
    e1 \in [-1, 1]
    e2 \in [-1, 1]
    z  < zsrc_min
    This was converted to a seperate function to allow for ease of extension without needing
    to change the same code in multiple locations.

    Parameters
    ----------
    galaxy_catalog : clmm.GCData
        Galaxy source catalog
    zsrc_min : float
        Minimum galaxy redshift allowed 

    Returns
    -------
    nbad : int
        The number of aphysical galaxies in galaxy_catalog
    badgals : array_like
        A list of the indicies in galaxy_catalog that need to be redrawn
    """
    badgals = np.where((np.abs(galaxy_catalog['e1']) > 1.0) |
                       (np.abs(galaxy_catalog['e2']) > 1.0) |
                       (galaxy_catalog['ztrue'] < zsrc_min)
                      )[0]
    nbad = len(badgals)
    return nbad, badgals





