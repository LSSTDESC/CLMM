"""Functions to generate mock source galaxy distributions to demo lensing code"""
import numpy as np
import warnings
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.stats import chi2
from astropy import units

from clmm import GCData
from clmm.theory import compute_tangential_shear, compute_convergence
from clmm.utils import convert_units, compute_lensed_ellipticity


def generate_galaxy_catalog(cluster_m, cluster_z, cluster_c, cosmo, zsrc,
                            Delta_SO=200, massdef='mean',
                            halo_profile_model='nfw', zsrc_min=None,
                            zsrc_max=7., field_size=8., shapenoise=None,
                            mean_e_err=None, photoz_sigma_unscaled=None,
                            pz_bin_width=0.02, nretry=5, ngals=None,
                            ngal_density=None):
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

    3. Draw galaxy positions. Positions are drawn in a square box around the lens position (with
    a default side length of 8 Mpc) at the lens redhsift. We then convert to right ascension and declination using the
    cosmology defined in `cosmo`.

    4. We predict the reduced tangential shear of each using the radial distances of each source
    from the lens, the source redshifts, and the lens mass, concentration, and redshift. In the
    given cosmology for an NFW halo. The reduced tangential shear is then transformed into `g1` and `g2``
    components.

    5. If the `shapenoise=True`, intrinsic ellipticities (1,2) components are drawn from a Gaussian distribution of width of `shapenoise`.
    These ellipticities components are then combined with `g1` and `g2` to provide lensed ellipticies `e1` and `e2`. If `shapenoise=False`,
    `g1` and `g2` are directly used as ellipticity components.

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
        Cluster concentration in the same mass definition as Delta_SO
    cosmo : dict
        Dictionary of cosmological parameters. Must contain at least, Omega_c, Omega_b,
        and H0
    zsrc : float or str
        Choose the source galaxy distribution to be fixed or drawn from a predefined distribution.
        float : All sources galaxies at this fixed redshift
        str : Draws individual source gal redshifts from predefined distribution. Options are: chang13, desc_srd
    Delta_SO : float, optional
        Overdensity density contrast used to compute the cluster mass and concentration. The
        spherical overdensity mass is computed as the mass enclosed within the radius
        :math:`R_{\Delta{\rm SO}}` where the mean matter density is :math:`\Delta_{\rm SO}` times
        the mean (or critical, depending on the massdef keyword) density of the Universe at the cluster redshift
        :math:`M_{\Delta{\rm SO}}=4/3\pi\Delta_{\rm SO}\rho_{m}(z_{\rm lens})R_{\Delta{\rm SO}}^3`
    massdef : string, optional
        Definition the mass overdensity with respect to the 'mean' or 'critical' density of the universe. Default is 'mean' as it works
        for all theory backends. The NumCosmo and CCL backends also allow the use of 'critical'.
        (letter case independent)
    halo_profile_model : string, optional
        Halo density profile. Default is 'nfw', which works for all theory backends. The NumCosmo backend allow for more
        options, e.g. 'einasto' or 'burkert' profiles (letter case independent).
    zsrc_min : float, optional
        The minimum true redshift of the sources. If photoz errors are included, the observed redshift
        may be smaller than zsrc_min.
    zsrc_max : float, optional
        The maximum true redshift of the sources, apllied when galaxy redshifts are drawn from
        a redshift distribution. If photoz errors are included, the observed redshift may be larger than
        zsrc_max.
    field_size : float, optional
        The size of the field (field_size x field_size) to be simulated.
        Proper distance in Mpc  at the cluster redshift.
    shapenoise : float, optional
        If set, applies Gaussian shape noise to the galaxy shapes with a width set by `shapenoise`
    mean_e_err : float, optional
        Mean per-component ellipticity uncertainty. Currently,
        individual uncertainties are drawn from a uniform distribution
        in the range [0.9,1.1]*mean_e_err. If not provided, the output
        table will not include this column.
    photoz_sigma_unscaled : float, optional
        If set, applies photo-z errors to source redshifts
    nretry : int, optional
        The number of times that we re-draw each galaxy with non-sensical derived properties
    ngals : float, optional
        Number of galaxies to generate
    ngal_density : float, optional
        The number density of galaxies (in galaxies per square arcminute, from z=0 to z=infty).
        The number of galaxies to be drawn will then depend on the redshift distribution and user-defined redshift range.
        If specified, the ngals argument will be ignored.

    Returns
    -------
    galaxy_catalog : clmm.GCData
        Table of source galaxies with drawn and derived properties required for lensing studies

    Notes
    -----
    Much of this code in this function was adapted from the Dallas group
    """

    _test_input_params(cluster_m, cluster_z, cluster_c, cosmo, zsrc, Delta_SO,
                       massdef, halo_profile_model, zsrc_min, zsrc_max,
                       field_size, shapenoise, mean_e_err,
                       photoz_sigma_unscaled, pz_bin_width,
                       nretry, ngals, ngal_density)

    if zsrc_min is None: zsrc_min = cluster_z+0.1

    params = {'cluster_m' : cluster_m, 'cluster_z' : cluster_z, 'cluster_c' : cluster_c,
              'cosmo' : cosmo, 'Delta_SO' : Delta_SO, 'zsrc' : zsrc, 'massdef' : massdef,
              'halo_profile_model' : halo_profile_model,
              'zsrc_min' : zsrc_min, 'zsrc_max' : zsrc_max,
              'shapenoise' : shapenoise, 'mean_e_err': mean_e_err,
              'photoz_sigma_unscaled' : photoz_sigma_unscaled,
              'field_size' : field_size}

    if ngals is None and ngal_density is None:
        err = 'Either the number of galaxies "ngals" or the galaxy density' \
              ' "ngal_density" keyword must be specified'
        raise ValueError(err)

    if ngals is not None and ngal_density is not None:
        err = 'The "ngals" and "ngal_density" keywords cannot both be set.' \
              ' Please use one only'
        raise ValueError(err)

    if ngal_density is not None:
        # Compute the number of galaxies to be drawn
        ngals = _compute_ngals(ngal_density, field_size, cosmo, cluster_z, zsrc, zsrc_min=zsrc_min, zsrc_max=zsrc_max)

    galaxy_catalog = _generate_galaxy_catalog(ngals=ngals, **params)
    # Check for bad galaxies and replace them
    nbad, badids = _find_aphysical_galaxies(galaxy_catalog, zsrc_min)
    for i in range(nretry):
        if nbad < 1:
            break
        replacements = _generate_galaxy_catalog(ngals=nbad, **params)
        #galaxy_catalog[badids] = replacements
        for badid, replacement in zip(badids, replacements):
            for col in galaxy_catalog.colnames:
                galaxy_catalog[col][badid] = replacement[col]
        nbad, badids = _find_aphysical_galaxies(galaxy_catalog, zsrc_min)

    # Final check to see if there are bad galaxies left
    if nbad > 1:
        warnings.warn("Not able to remove {} aphysical objects after {} iterations".format(nbad, nretry))

    # Now that the catalog is final, add an id column
    galaxy_catalog['id'] = np.arange(ngals)
    return galaxy_catalog


def _chang_z_distrib(z):
    """
    A private function that returns the Chang et al (2013) unnormalized galaxy redshift distribution function,
    with the fiducial set of parameters.

    Parameters
    ----------
    z : float
        Galaxy redshift

    Returns
    -------
    The value of the distribution at z
    """
    alpha, beta, z0 = 1.24, 1.01, 0.51
    return (z**alpha)*np.exp(-(z/z0)**beta)

def _srd_z_distrib(z):
    """
    A private function that returns the unnormalized galaxy redshift distribution function used in
    the LSST/DESC Science Requirement Document (arxiv:1809.01669).

    Parameters
    ----------
    z : float
        Galaxy redshift

    Returns
    -------
    The value of the distribution at z
    """
    alpha, beta, z0 = 2., 0.9, 0.28
    return (z**alpha)*np.exp(-(z/z0)**beta)


def _compute_ngals(ngal_density, field_size, cosmo, cluster_z, zsrc, zsrc_min=None, zsrc_max=None):
    """
    A private function that computes the number of galaxies to draw given the user-defined
    field size, galaxy density, cosmology, cluster redshift, galaxy redshift distribution
    and requested redshift range.
    For a more detailed description of each of the parameters, see the documentation of
    `generate_galaxy_catalog`.
    """
    field_size_arcmin = convert_units(field_size, 'Mpc', 'arcmin', redshift=cluster_z, cosmo=cosmo)
    ngals = int(ngal_density*field_size_arcmin*field_size_arcmin)

    if isinstance(zsrc, float):
        return int(ngals)
    elif zsrc in ('chang13', 'desc_srd'):
        z_distrib_func = _chang_z_distrib if zsrc=='chang13' else _srd_z_distrib
        # Compute the normalisation for the redshift distribution function (z=[0,\infty])
        norm, _ = integrate.quad(z_distrib_func, 0., 100)
        # Probability to find the galaxy in the requested redshift range
        prob = integrate.quad(_chang_z_distrib, zsrc_min, zsrc_max)[0]/norm
        return int(ngals*prob)


def _generate_galaxy_catalog(cluster_m, cluster_z, cluster_c, cosmo, ngals,
                             zsrc, Delta_SO=None, massdef=None,
                             halo_profile_model=None, zsrc_min=None,
                             zsrc_max=None, shapenoise=None,
                             mean_e_err=None, photoz_sigma_unscaled=None,
                             pz_bin_width=0.02, field_size=None):
    """A private function that skips the sanity checks on derived properties. This
    function should only be used when called directly from `generate_galaxy_catalog`.
    For a detailed description of each of the parameters, see the documentation of
    `generate_galaxy_catalog`.
    """
    # Set the source galaxy redshifts
    galaxy_catalog = _draw_source_redshifts(zsrc, zsrc_min, zsrc_max, ngals)

    # Add photo-z errors and pdfs to source galaxy redshifts
    if photoz_sigma_unscaled is not None:
        galaxy_catalog = _compute_photoz_pdfs(
            galaxy_catalog, photoz_sigma_unscaled, zsrc_max,
            pz_bin_width=pz_bin_width)
    # Draw galaxy positions
    galaxy_catalog = _draw_galaxy_positions(galaxy_catalog, ngals, cluster_z, cosmo, field_size)
    # Compute the shear on each source galaxy
    gamt = compute_tangential_shear(galaxy_catalog['r_mpc'], mdelta=cluster_m,
                                    cdelta=cluster_c, z_cluster=cluster_z,
                                    z_source=galaxy_catalog['ztrue'],
                                    cosmo=cosmo, delta_mdef=Delta_SO,
                                    halo_profile_model=halo_profile_model,
                                    massdef=massdef,
                                    z_src_model='single_plane')

    gamx = np.zeros(ngals)

    kappa = compute_convergence(galaxy_catalog['r_mpc'], mdelta=cluster_m,
                                            cdelta=cluster_c, z_cluster=cluster_z,
                                            z_source=galaxy_catalog['z'], cosmo=cosmo,
                                            delta_mdef=Delta_SO, halo_profile_model=halo_profile_model,
                                            massdef=massdef,
                                            z_src_model='single_plane')

    posangle = np.arctan2(galaxy_catalog['y_mpc'], galaxy_catalog['x_mpc'])
    #corresponding shear1,2 components
    gam1 = -gamt*np.cos(2*posangle) + gamx*np.sin(2*posangle)
    gam2 = -gamt*np.sin(2*posangle) - gamx*np.cos(2*posangle)

    #instrinsic ellipticities
    e1_intrinsic = 0
    e2_intrinsic = 0

    # Add shape noise to source galaxy shears
    if shapenoise is not None:
        e1_intrinsic = shapenoise*np.random.standard_normal(ngals)
        e2_intrinsic = shapenoise*np.random.standard_normal(ngals)

    # Compute ellipticities
    galaxy_catalog['e1'], galaxy_catalog['e2'] = compute_lensed_ellipticity(
        e1_intrinsic, e2_intrinsic, gam1, gam2, kappa)

    cols = ['ra', 'dec', 'e1', 'e2']
    # if adding uncertainties
    if mean_e_err is not None:
        galaxy_catalog['e_err'] \
            = mean_e_err * np.random.uniform(0.9, 1.1, ngals)
        galaxy_catalog['e1'] = np.random.normal(
            galaxy_catalog['e1'], galaxy_catalog['e_err'])
        galaxy_catalog['e2'] = np.random.normal(
            galaxy_catalog['e2'], galaxy_catalog['e_err'])
        cols = cols + ['e_err']
    cols = cols + ['z', 'ztrue']
    if photoz_sigma_unscaled is not None:
        cols = cols + ['pzpdf']
    # so that we preserve special attributes
    galaxy_catalog.remove_columns(
        [col for col in galaxy_catalog.colnames if col not in cols])

    return galaxy_catalog


def _draw_random_points_from_distribution(xmin, xmax, nobj, dist_func, dx=0.001):
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
    dx : float
        Size of the step to interpolate the culmulative distribution.

    Returns
    -------
    ndarray
        Random points with dist_func distribution
    """
    xdomain = np.arange(xmin, xmax, dx)
    # Cumulative probability function of the redshift distribution
    probdist = np.vectorize(lambda zmax: integrate.quad(dist_func, xmin, zmax)[0])(xdomain)
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
        str : Draws individual source gal redshifts from predefined distribution. Options
              are: chang13 or desc_srd
    zsrc_min : float
        The minimum source redshift allowed.
    zsrc_max : float, optional
        If source redshifts are drawn, the maximum source redshift

    Returns
    -------
    galaxy_catalog : clmm.GCData
        Table of true and 'measured' photometric redshifts, which here the same. Redshift photometric errors
        are then added using _compute_photoz_pdfs.

    Notes
    -----
    Much of this code in this function was adapted from the Dallas group
    """
    # Set zsrc to constant value
    if isinstance(zsrc, float):
        zsrc_list = np.ones(ngals)*zsrc

    # Draw zsrc from Chang et al. 2013
    elif zsrc == 'chang13':
        zsrc_list = _draw_random_points_from_distribution(zsrc_min, zsrc_max, ngals, _chang_z_distrib)

    # Draw zsrc from the distribution used in the DESC SRD (arxiv:1809.01669)
    elif zsrc == 'desc_srd':
        zsrc_list = _draw_random_points_from_distribution(zsrc_min, zsrc_max, ngals, _srd_z_distrib)

    # Draw zsrc from a uniform distribution between zmin and zmax
    elif zsrc == 'uniform':
        zsrc_list = np.random.uniform(zsrc_min, zsrc_max, ngals)

    # Invalid entry
    else:
        raise ValueError("zsrc must be a float, chang13 or desc_srd. You set: {}".format(zsrc))

    return GCData([zsrc_list, zsrc_list], names=('ztrue', 'z'))


def _compute_photoz_pdfs(galaxy_catalog, photoz_sigma_unscaled,
                         zsrc_max, pz_bin_width=0.02):
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
    galaxy_catalog['pzsigma'] = photoz_sigma_unscaled*(1.+galaxy_catalog['ztrue'])
    galaxy_catalog['z'] = galaxy_catalog['ztrue']+\
                          galaxy_catalog['pzsigma']*np.random.standard_normal(len(galaxy_catalog))

    # start at zero to incorporate photoz errors
    pzbins = np.arange(0, zsrc_max+pz_bin_width, pz_bin_width)
    pzsigma = galaxy_catalog['pzsigma'][:,None]
    pzpdf_grid = np.exp(
            -(pzbins-galaxy_catalog['z'][:,None])**2/(2*pzsigma**2)) \
        / ((2*np.pi)**0.5*pzsigma)
    galaxy_catalog['pzpdf'] = pzpdf_grid
    galaxy_catalog.pzbins = pzbins
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
    Dl = cosmo.eval_da(cluster_z) # Mpc

    galaxy_catalog['x_mpc'] = np.random.uniform(-(field_size/2.), field_size/2., size=ngals)
    galaxy_catalog['y_mpc'] = np.random.uniform(-(field_size/2.), field_size/2., size=ngals)
    galaxy_catalog['r_mpc'] = np.sqrt(galaxy_catalog['x_mpc']**2+galaxy_catalog['y_mpc']**2)
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
    etot = np.sqrt(galaxy_catalog['e1'] * galaxy_catalog['e1']
                   + galaxy_catalog['e2'] * galaxy_catalog['e2'])

#     badgals = np.where((np.abs(galaxy_catalog['e1']) > 1.0) |
#                        (np.abs(galaxy_catalog['e2']) > 1.0) |
#                        (galaxy_catalog['ztrue'] < zsrc_min)
#                       )[0]

    badgals = np.where((etot > 1.0) | (galaxy_catalog['ztrue'] < zsrc_min))[0]
    nbad = len(badgals)
    return nbad, badgals


def _test_input_params(cluster_m, cluster_z, cluster_c, cosmo, zsrc, Delta_SO,
                       massdef, halo_profile_model, zsrc_min, zsrc_max,
                       field_size, shapenoise, mean_e_err,
                       photoz_sigma_unscaled, pz_bin_width, nretry, ngals,
                       ngal_density):
    # all positive ints or floats
    float_vars = (cluster_m, cluster_z, cluster_c, Delta_SO, zsrc_max,
                  pz_bin_width, field_size)
    float_names = ('cluster_m', 'cluster_z', 'cluster_c', 'Delta_S0',
                   'zsrc_max', 'pz_bin_width', 'field_size')
    int_vars = []
    int_names = []
    # will iterate over floats and then ints
    var_types = ((float,int,np.floating,np.integer), (int,np.integer))
    var_type_names = ('float', 'int')
    for vars, names, types, typename in zip((float_vars,int_vars),
                                            (float_names,int_names),
                                            var_types, var_type_names):
        for var, name in zip(vars, names):
            if not isinstance(var, types):
                err = f'{name} must be {typename},' \
                      f' received {type(var).__name__}'
                raise TypeError(err)
            if np.iterable(var):
                # in case it's a list or tuple
                var = np.array(var)
                if var.size > 1:
                    err = f'{name} must be {typename}, received {type(var)}'
                    raise TypeError(err)
            if var <= 0:
                err = f'{name} must be greater than zero, received {var}'
                raise ValueError(err)
    # all variables either None or semi-positive float
    float_vars = (zsrc_min, shapenoise, mean_e_err, photoz_sigma_unscaled)
    float_names = ('zsrc_min', 'shapenoise', 'mean_e_err',
                   'photoz_sigma_unscaled')
    int_vars = (nretry,)
    int_names = ('nretry',)
    for vars, names, types, typename in zip((float_vars,int_vars),
                                            (float_names,int_names),
                                            var_types, var_type_names):
        for var, name,  in zip(vars, names):
            if var is None:
                continue
            try:
                var / 1
            except TypeError:
                err = f'if specified, a{name} must be float,' \
                      f' received {type(var).__name__}'
                raise TypeError(err)
            if np.iterable(var):
                # in case it's a list or tuple
                var = np.array(var)
                if var.size > 1:
                    err = f'{name} must be None or float, received {type(var)}'
                    raise TypeError(err)
            if var < 0:
                err = f'{name} must be greater than or equal to zero,' \
                      f' received {var}'
                raise ValueError(err)
    return

