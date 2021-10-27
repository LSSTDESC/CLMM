"""Functions to generate mock source galaxy distributions to demo lensing code"""
import warnings
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

from ..gcdata import GCData
from ..theory import compute_tangential_shear, compute_convergence
from ..utils import convert_units, compute_lensed_ellipticity


def generate_galaxy_catalog(
        cluster_m, cluster_z, cluster_c, cosmo, zsrc, delta_so=200, massdef='mean',
        halo_profile_model='nfw', zsrc_min=None, zsrc_max=7., field_size=8., shapenoise=None,
        photoz_sigma_unscaled=None, nretry=5, ngals=None, ngal_density=None):
    r"""Generates a mock dataset of sheared background galaxies.

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

    .. math::
        z \sim \mathcal{N}\left(z^{\rm true},
        \sigma_{\rm photo-z}^{\rm unscaled}(1+z^{\rm true}) \right)

    We additionally include two columns in the output catalog, `pzbins` and `pzpdf` which
    desribe the photo-z distribution as a Gaussian centered at :math:`z^{\rm true}` with a
    width :math:`\sigma_{\rm photo-z} = \sigma_{\rm photo-z}^{\rm unscaled}(1+z^{\rm true})`

    If `photoz_sigma_unscaled` is `None`, the `z` column in the output catalog is the true
    redshift.

    3. Draw galaxy positions. Positions are drawn in a square box around the lens position (with a
    default side length of 8 Mpc) at the lens redhsift. We then convert to right ascension and
    declination using the cosmology defined in `cosmo`.

    4. We predict the reduced tangential shear of each using the radial distances of each source
    from the lens, the source redshifts, and the lens mass, concentration, and redshift. In the
    given cosmology for an NFW halo. The reduced tangential shear is then transformed into `g1` and
    `g2`` components.

    5. If the `shapenoise=True`, intrinsic ellipticities (1,2) components are drawn from a Gaussian
    distribution of width of `shapenoise`.  These ellipticities components are then combined with
    `g1` and `g2` to provide lensed ellipticies `e1` and `e2`. If `shapenoise=False`, `g1` and `g2`
    are directly used as ellipticity components.

    If the shape noise parameter is high, we may draw nonsensical values for ellipticities. We
    ensure that we does not return any nonsensical values for derived properties. We re-draw
    all galaxies with e1 or e2 outside the bounds of [-1, 1]. After 5 (default) attempts to
    re-draw these properties, we return the catalog as is and throw a warning.

    Parameters
    ----------
    cluster_m : float
        Cluster mass in Msun
    cluster_z : float
        Cluster redshift
    cluster_c : float
        Cluster concentration in the same mass definition as delta_so
    cosmo : dict
        Dictionary of cosmological parameters. Must contain at least, Omega_c, Omega_b,
        and H0
    zsrc : float or str
        Choose the source galaxy distribution to be fixed or drawn from a predefined distribution.

        * `float` : All sources galaxies at this fixed redshift;
        * `str` : Draws individual source gal redshifts from predefined distribution. Options are:

            * `chang13` - Chang et al. 2013 (arXiv:1305.0793);
            * `desc_srd` - LSST/DESC Science Requirement Document (arxiv:1809.01669);

    delta_so : float, optional
        Overdensity density contrast used to compute the cluster mass and concentration. The
        spherical overdensity mass is computed as the mass enclosed within the radius
        :math:`R_{\Delta{\rm SO}}` where the mean matter density is :math:`\Delta_{\rm SO}` times
        the mean (or critical, depending on the massdef keyword) density of the Universe at the
        cluster redshift
        :math:`M_{\Delta{\rm SO}}=4/3\pi\Delta_{\rm SO}\rho_{m}(z_{\rm lens})R_{\Delta{\rm SO}}^3`
    massdef : string, optional
        Definition the mass overdensity with respect to the 'mean' or 'critical' density of the
        universe. Default is 'mean' as it works for all theory backends. The NumCosmo and CCL
        backends also allow the use of 'critical'.  (letter case independent)
    halo_profile_model : string, optional
        Halo density profile. Default is 'nfw', which works for all theory backends. The NumCosmo
        backend allow for more options, e.g. 'einasto' or 'burkert' profiles (letter case
        independent).
    zsrc_min : float, optional
        The minimum true redshift of the sources. If photoz errors are included, the observed
        redshift may be smaller than zsrc_min.
    zsrc_max : float, optional
        The maximum true redshift of the sources, apllied when galaxy redshifts are drawn from a
        redshift distribution. If photoz errors are included, the observed redshift may be larger
        than zsrc_max.
    field_size : float, optional
        The size of the field (field_size x field_size) to be simulated.
        Proper distance in Mpc  at the cluster redshift.
    shapenoise : float, optional
        If set, applies Gaussian shape noise to the galaxy shapes with a width set by `shapenoise`
    photoz_sigma_unscaled : float, optional
        If set, applies photo-z errors to source redshifts
    nretry : int, optional
        The number of times that we re-draw each galaxy with non-sensical derived properties
    ngals : float, optional
        Number of galaxies to generate
    ngal_density : float, optional
        The number density of galaxies (in galaxies per square arcminute, from z=0 to z=infty).
        The number of galaxies to be drawn will then depend on the redshift distribution and
        user-defined redshift range.  If specified, the ngals argument will be ignored.

    Returns
    -------
    galaxy_catalog : clmm.GCData
        Table of source galaxies with drawn and derived properties required for lensing studies

    Notes
    -----
    Much of this code in this function was adapted from the Dallas group
    """
    #pylint: disable=too-many-arguments
    #Too many local variables (25/15)
    #pylint: disable=R0914

    if zsrc_min is None:
        zsrc_min = cluster_z+0.1

    params = {'cluster_m': cluster_m, 'cluster_z': cluster_z, 'cluster_c': cluster_c,
              'cosmo': cosmo, 'delta_so': delta_so, 'zsrc': zsrc, 'massdef': massdef,
              'halo_profile_model': halo_profile_model,
              'zsrc_min': zsrc_min, 'zsrc_max': zsrc_max,
              'shapenoise': shapenoise, 'photoz_sigma_unscaled': photoz_sigma_unscaled,
              'field_size': field_size}

    if ngals is None and ngal_density is None:
        raise ValueError(
            'Either the number of galaxies "ngals" or the galaxy density "ngal_density"'
            ' keyword must be specified')

    if ngals is not None and ngal_density is not None:
        raise ValueError(
            'The "ngals" and "ngal_density" keywords cannot both be set. Please use one only')

    if ngal_density is not None:
        # Compute the number of galaxies to be drawn
        ngals = _compute_ngals(ngal_density, field_size, cosmo,
                               cluster_z, zsrc, zsrc_min=zsrc_min, zsrc_max=zsrc_max)

    galaxy_catalog = _generate_galaxy_catalog(ngals=ngals, **params)
    # Check for bad galaxies and replace them
    nbad, badids = _find_aphysical_galaxies(galaxy_catalog, zsrc_min)
    ntry = 0
    while (nbad > 0) and (ntry < nretry):
        replacements = _generate_galaxy_catalog(ngals=nbad, **params)
        #galaxy_catalog[badids] = replacements
        for badid, replacement in zip(badids, replacements):
            for col in galaxy_catalog.colnames:
                galaxy_catalog[col][badid] = replacement[col]
        nbad, badids = _find_aphysical_galaxies(galaxy_catalog, zsrc_min)
        ntry += 1

    # Final check to see if there are bad galaxies left
    if nbad > 1:
        warnings.warn(
            "Not able to remove {} aphysical objects after {} iterations".format(nbad, nretry))

    # Now that the catalog is final, add an id column
    galaxy_catalog['id'] = np.arange(ngals)
    return galaxy_catalog


def _chang_z_distrib(redshift, is_cdf=False):
    """
    A private function that returns the Chang et al (2013) unnormalized galaxy redshift distribution
    function, with the fiducial set of parameters.

    Parameters
    ----------
    redshift : float
        Galaxy redshift

    Returns
    -------
    The value of the distribution at z
    """
    alpha, beta, redshift0 = 1.24, 1.01, 0.51
    if is_cdf:
        return redshift0**(alpha+1)*gammainc((alpha+1)/beta, (redshift/redshift0)**beta)/beta*gamma((alpha+1)/beta)
    else:
        return (redshift**alpha)*np.exp(-(redshift/redshift0)**beta)


def _srd_z_distrib(redshift, is_cdf=False):
    """
    A private function that returns the unnormalized galaxy redshift distribution function used in
    the LSST/DESC Science Requirement Document (arxiv:1809.01669).

    Parameters
    ----------
    redshift : float
        Galaxy redshift

    Returns
    -------
    The value of the distribution at z
    """
    alpha, beta, redshift0 = 2., 0.9, 0.28
    if is_cdf:
        return redshift0**(alpha+1)*gammainc((alpha+1)/beta, (redshift/redshift0)**beta)/beta*gamma((alpha+1)/beta)
    else:
        return (redshift**alpha)*np.exp(-(redshift/redshift0)**beta)


def _compute_ngals(ngal_density, field_size, cosmo, cluster_z, zsrc, zsrc_min=None, zsrc_max=None):
    """
    A private function that computes the number of galaxies to draw given the user-defined
    field size, galaxy density, cosmology, cluster redshift, galaxy redshift distribution
    and requested redshift range.
    For a more detailed description of each of the parameters, see the documentation of
    `generate_galaxy_catalog`.
    """
    field_size_arcmin = convert_units(
        field_size, 'Mpc', 'arcmin', redshift=cluster_z, cosmo=cosmo)
    ngals = ngal_density*field_size_arcmin*field_size_arcmin

    if isinstance(zsrc, float):
        ngals = int(ngals)
    elif zsrc in ('chang13', 'desc_srd'):
        z_distrib_func = _chang_z_distrib if zsrc == 'chang13' else _srd_z_distrib
        # Compute the normalisation for the redshift distribution function (z=[0,\infty])
        norm, _ = integrate.quad(z_distrib_func, 0., 100)
        # Probability to find the galaxy in the requested redshift range
        prob = integrate.quad(z_distrib_func, zsrc_min, zsrc_max)[0]/norm
        ngals = int(ngals*prob)
    else:
        raise ValueError(f"zsrc (={zsrc}) must be float, 'chang13' or 'desc_src'")
    return ngals


def _generate_galaxy_catalog(
        cluster_m, cluster_z, cluster_c, cosmo, ngals, zsrc, delta_so=None, massdef=None,
        halo_profile_model=None, zsrc_min=None, zsrc_max=None, shapenoise=None,
        photoz_sigma_unscaled=None, field_size=None):
    """A private function that skips the sanity checks on derived properties. This
    function should only be used when called directly from `generate_galaxy_catalog`.
    For a detailed description of each of the parameters, see the documentation of
    `generate_galaxy_catalog`.
    """
    #Too many local variables (22/15)
    #pylint: disable=R0914

    # Set the source galaxy redshifts
    galaxy_catalog = _draw_source_redshifts(zsrc, zsrc_min, zsrc_max, ngals)

    # Add photo-z errors and pdfs to source galaxy redshifts
    if photoz_sigma_unscaled is not None:
        galaxy_catalog = _compute_photoz_pdfs(
            galaxy_catalog, photoz_sigma_unscaled)
    # Draw galaxy positions
    galaxy_catalog = _draw_galaxy_positions(
        galaxy_catalog, ngals, cluster_z, cosmo, field_size)
    # Compute the shear on each source galaxy
    gamt = compute_tangential_shear(galaxy_catalog['r_mpc'], mdelta=cluster_m,
                                    cdelta=cluster_c, z_cluster=cluster_z,
                                    z_source=galaxy_catalog['ztrue'], cosmo=cosmo,
                                    delta_mdef=delta_so, halo_profile_model=halo_profile_model,
                                    massdef=massdef,
                                    z_src_model='single_plane')

    gamx = np.zeros(ngals)
    kappa = compute_convergence(galaxy_catalog['r_mpc'], mdelta=cluster_m,
                                cdelta=cluster_c, z_cluster=cluster_z,
                                z_source=galaxy_catalog['ztrue'], cosmo=cosmo,
                                delta_mdef=delta_so, halo_profile_model=halo_profile_model,
                                massdef=massdef,
                                z_src_model='single_plane')

    galaxy_catalog['gammat'] = gamt
    galaxy_catalog['gammax'] = np.zeros(ngals)

    galaxy_catalog['posangle'] = np.arctan2(galaxy_catalog['y_mpc'],
                                            galaxy_catalog['x_mpc'])

    # corresponding shear1,2 components
    gam1 = -gamt*np.cos(2*galaxy_catalog['posangle']) + \
        gamx*np.sin(2*galaxy_catalog['posangle'])
    gam2 = -gamt*np.sin(2*galaxy_catalog['posangle']) - \
        gamx*np.cos(2*galaxy_catalog['posangle'])

    # instrinsic ellipticities
    e1_intrinsic = 0
    e2_intrinsic = 0

    # Add shape noise to source galaxy shears
    if shapenoise is not None:
        e1_intrinsic = shapenoise*np.random.standard_normal(ngals)
        e2_intrinsic = shapenoise*np.random.standard_normal(ngals)

    # Compute ellipticities
    galaxy_catalog['e1'], galaxy_catalog['e2'] = compute_lensed_ellipticity(
        e1_intrinsic, e2_intrinsic, gam1, gam2, kappa)

    if photoz_sigma_unscaled is not None:
        return galaxy_catalog['ra', 'dec', 'e1', 'e2', 'z', 'ztrue', 'pzbins', 'pzpdf']
    return galaxy_catalog['ra', 'dec', 'e1', 'e2', 'z', 'ztrue']


def _draw_random_points_from_distribution(xmin, xmax, nobj, dist_func, xstep=0.001):
    """Draw random points with a given distribution.

    Uses a sampling technique found in Numerical Recipes in C, Chap 7.2: Transformation Method.

    Parameters
    ----------
    xmin : float
        The minimum source redshift allowed.
    xmax : float, optional
        If source redshifts are drawn, the maximum source redshift
    nobj : float
        Number of galaxies to generate
    dist_func : function
        Function of the required distribution
    xstep : float
        Size of the step to interpolate the culmulative distribution.

    Returns
    -------
    ndarray
        Random points with dist_func distribution
    """
    xdomain = np.arange(xmin, xmax, xstep)
    # Cumulative probability function of the redshift distribution
    probdist = np.vectorize(lambda zmax: integrate.quad(
        dist_func, xmin, zmax)[0])(xdomain)
    # Get random values for probdist
    uniform_deviate = np.random.uniform(probdist.min(), probdist.max(), nobj)
    return interp1d(probdist, xdomain, kind='linear')(uniform_deviate)


def _draw_source_redshifts(zsrc, zsrc_min, zsrc_max, ngals):
    """Set source galaxy redshifts either set to a fixed value or draw from a predefined
    distribution. Return a table (GCData) of the source galaxies

    Uses a sampling technique found in Numerical Recipes in C, Chap 7.2: Transformation Method.
    Pulling out random values from a given probability distribution.

    Parameters
    ----------
    ngals : float
        Number of galaxies to generate
    zsrc : float or str
        Choose the source galaxy distribution to be fixed or drawn from a predefined distribution.
        float : All sources galaxies at this fixed redshift
        str : Draws individual source gal redshifts from predefined distribution. Options are:

            * `chang13` - Chang et al. 2013 (arXiv:1305.0793);
            * `desc_srd` - LSST/DESC Science Requirement Document (arxiv:1809.01669);

    zsrc_min : float
        The minimum source redshift allowed.
    zsrc_max : float, optional
        If source redshifts are drawn, the maximum source redshift

    Returns
    -------
    galaxy_catalog : clmm.GCData
        Table of true and 'measured' photometric redshifts, which here the same. Redshift
        photometric errors are then added using _compute_photoz_pdfs.

    Notes
    -----
    Much of this code in this function was adapted from the Dallas group
    """
    # Set zsrc to constant value
    if isinstance(zsrc, float):
        zsrc_list = np.ones(ngals)*zsrc

    # Draw zsrc from Chang et al. 2013
    elif zsrc == 'chang13':
        zsrc_list = _draw_random_points_from_distribution(
            zsrc_min, zsrc_max, ngals, _chang_z_distrib)

    # Draw zsrc from the distribution used in the DESC SRD (arxiv:1809.01669)
    elif zsrc == 'desc_srd':
        zsrc_list = _draw_random_points_from_distribution(
            zsrc_min, zsrc_max, ngals, _srd_z_distrib)

    # Draw zsrc from a uniform distribution between zmin and zmax
    elif zsrc == 'uniform':
        zsrc_list = np.random.uniform(zsrc_min, zsrc_max, ngals)

    # Invalid entry
    else:
        raise ValueError(
            "zsrc must be a float, chang13 or desc_srd. You set: {}".format(zsrc))

    return GCData([zsrc_list, zsrc_list], names=('ztrue', 'z'))


def _compute_photoz_pdfs(galaxy_catalog, photoz_sigma_unscaled):
    """Private function to add photo-z errors and PDFs to the mock catalog.

    Parameters
    ----------
    galaxy_catalog : clmm.GCData
        Input galaxy catalog to which photoz PDF will be added
    photoz_sigma_unscaled : float
        Width of the Gaussian PDF, without the (1+z) factor

    Returns
    -------
    galaxy_catalog : clmm.GCData
        Output galaxy catalog with columns corresponding to the bins
        and values of the redshift PDF for each galaxy.
    """
    galaxy_catalog['pzsigma'] = photoz_sigma_unscaled * \
        (1.+galaxy_catalog['ztrue'])
    galaxy_catalog['z'] = galaxy_catalog['ztrue'] +\
        galaxy_catalog['pzsigma'] * \
        np.random.standard_normal(len(galaxy_catalog))

    pzbins_grid, pzpdf_grid = [], []
    for row in galaxy_catalog:
        pdf_range = row['pzsigma']*10.
        zmin, zmax = row['z']-pdf_range, row['z']+pdf_range
        zbins = np.arange(zmin, zmax, 0.03)
        pzbins_grid.append(zbins)
        pzpdf_grid.append(
            np.exp(-0.5*((zbins-row['z'])/row['pzsigma'])**2)/np.sqrt(2*np.pi*row['pzsigma']**2))
    galaxy_catalog['pzbins'] = pzbins_grid
    galaxy_catalog['pzpdf'] = pzpdf_grid

    return galaxy_catalog


def _draw_galaxy_positions(galaxy_catalog, ngals, cluster_z, cosmo, field_size):
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
    field_size : float
        The size of the field (field_size x field_size) to be simulated around the cluster center.
        Proper distance in Mpc at the cluster redshift.

    Returns
    -------
    galaxy_catalog : clmm.GCData
        Source galaxy catalog with positions added
    """
    lens_distance = cosmo.eval_da(cluster_z)  # Mpc

    galaxy_catalog['x_mpc'] = np.random.uniform(
        -(field_size/2.), field_size/2., size=ngals)
    galaxy_catalog['y_mpc'] = np.random.uniform(
        -(field_size/2.), field_size/2., size=ngals)
    galaxy_catalog['r_mpc'] = np.sqrt(
        galaxy_catalog['x_mpc']**2+galaxy_catalog['y_mpc']**2)
    galaxy_catalog['ra'] = -np.rad2deg(galaxy_catalog['x_mpc']/lens_distance)
    galaxy_catalog['dec'] = np.rad2deg(galaxy_catalog['y_mpc']/lens_distance)

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
    etot = np.sqrt(galaxy_catalog['e1'] * galaxy_catalog['e1']
                   + galaxy_catalog['e2'] * galaxy_catalog['e2'])

#     badgals = np.where((np.abs(galaxy_catalog['e1']) > 1.0) |
#                        (np.abs(galaxy_catalog['e2']) > 1.0) |
#                        (galaxy_catalog['ztrue'] < zsrc_min)
#                       )[0]

    badgals = np.where((etot > 1.0) | (galaxy_catalog['ztrue'] < zsrc_min))[0]
    nbad = len(badgals)
    return nbad, badgals
