"""Functions to compute polar/azimuthal averages in radial bins"""
import math
import warnings
import numpy as np
import scipy
from astropy.coordinates import SkyCoord
from astropy import units as u
from .. gcdata import GCData
from . .utils import compute_radial_averages, make_bins, convert_units, arguments_consistency, validate_argument
from .. theory import compute_critical_surface_density


def compute_tangential_and_cross_components(
        ra_lens, dec_lens, ra_source, dec_source,
        shear1, shear2, geometry='curve',
        is_deltasigma=False, cosmo=None,
        z_lens=None, z_source=None, sigma_c=None, validate_input=True):
    r"""Computes tangential- and cross- components for shear or ellipticity

    To do so, we need the right ascension and declination of the lens and of all of the sources. We
    also need the two shape components of all of the sources.

    These quantities can be handed to `tangential_and_cross_components` in two ways

    1. Pass in each as parameters::

        compute_tangential_and_cross_components(ra_lens, dec_lens, ra_source, dec_source,
        shape_component1, shape_component2)

    2. As a method of `GalaxyCluster`::

        cluster.tangential_and_cross_components()

    The angular separation between the source and the lens, :math:`\theta`, and the azimuthal
    position of the source relative to the lens, :math:`\phi`, are computed within the function and
    the angular separation is returned.

    In the flat sky approximation, these angles are calculated using (_lens: lens, _source: source,
    RA is from right to left)

    .. math::

        \theta^2 = & \left(\delta_s-\delta_l\right)^2+
        \left(\alpha_l-\alpha_s\right)^2\cos^2(\delta_l)\\
        \tan\phi = & \frac{\delta_s-\delta_l}{\left(\alpha_l-\alpha_s\right)\cos(\delta_l)}

    The tangential, :math:`g_t`, and cross, :math:`g_x`, ellipticity/shear components are calculated
    using the two ellipticity/shear components :math:`g_1` and :math:`g_2` of the source galaxies,
    following Eq.7 and Eq.8 in Schrabback et al. (2018), arXiv:1611:03866 which is consistent with
    arXiv:0509252

    .. math::

        g_t =& -\left( g_1\cos\left(2\phi\right)+g_2\sin\left(2\phi\right)\right)\\
        g_x =& g_1 \sin\left(2\phi\right)-g_2\cos\left(2\phi\right)

    Finally, and if requested by the user throught the `is_deltasigma` flag, an estimate of the
    excess surface density :math:`\widehat{\Delta\Sigma}` is obtained from

    .. math::

        \widehat{\Delta\Sigma_{t,x}} = g_{t,x} \times \Sigma_c(cosmo, z_L, z_{\rm src})

    where :math:`\Sigma_c` is the critical surface density that depends on the cosmology and on the
    lens and source redshifts. If :math:`g_{t,x}` correspond to the shear, the above expression is
    an accurate. However, if :math:`g_{t,x}` correspond to ellipticities or reduced shear, this
    expression only gives an estimate :math:`\widehat{\Delta\Sigma_{t,x}}`, valid only in the weak
    lensing regime.

    Parameters
    ----------
    ra_lens: float
        Right ascension of the lensing cluster in degrees
    dec_lens: float
        Declination of the lensing cluster in degrees
    ra_source: array
        Right ascensions of each source galaxy in degrees
    dec_source: array
        Declinations of each source galaxy in degrees
    shear1: array
        The measured shear (or reduced shear or ellipticity) of the source galaxies
    shear2: array
        The measured shear (or reduced shear or ellipticity) of the source galaxies
    geometry: str, optional
        Sky geometry to compute angular separation.
        Options are curve (uses astropy) or flat.
    is_deltasigma: bool
        If `True`, the tangential and cross components returned are multiplied by Sigma_crit.
        Results in units of :math:`M_\odot\ Mpc^{-2}`
    cosmo: clmm.Cosmology, optional
        Required if `is_deltasigma` is True and `sigma_c` not provided.
        Not used if `sigma_c` is provided.
    z_lens: float, optional
        Redshift of the lens, required if `is_deltasigma` is True and `sigma_c` not provided.
        Not used if `sigma_c` is provided.
    z_source: array, optional
        Redshift of the source, required if `is_deltasigma` is True and `sigma_c` not provided.
        Not used if `sigma_c` is provided.
    sigma_c : float, optional
        Critical surface density in units of :math:`M_\odot\ Mpc^{-2}`,
        if provided, `cosmo`, `z_lens` and `z_source` are not used.
    validate_input: bool
        Validade each input argument

    Returns
    -------
    angsep: array_like
        Angular separation between lens and each source galaxy in radians
    tangential_component: array_like
        Tangential shear (or assimilated quantity) for each source galaxy
    cross_component: array_like
        Cross shear (or assimilated quantity) for each source galaxy
    """
    # pylint: disable-msg=too-many-locals
    # Note: we make these quantities to be np.array so that a name is not passed from astropy
    # columns
    if validate_input:
        validate_argument(locals(), 'ra_source', 'float_array',
                          argmin=-360, eqmin=True, argmax=360, eqmax=True)
        validate_argument(locals(), 'dec_source', 'float_array',
                          argmin=-90, eqmin=True, argmax=90, eqmax=True)
        validate_argument(locals(), 'ra_lens', 'float_array',
                          argmin=-360, eqmin=True, argmax=360, eqmax=True)
        validate_argument(locals(), 'dec_lens', 'float_array',
                          argmin=-90, eqmin=True, argmax=90, eqmax=True)
        validate_argument(locals(), 'shear1', 'float_array')
        validate_argument(locals(), 'shear2', 'float_array')
        validate_argument(locals(), 'geometry', str)
        validate_argument(locals(), 'is_deltasigma', bool)
        validate_argument(locals(), 'z_lens', float, argmin=0, eqmin=True, none_ok=True)
        validate_argument(locals(), 'z_source', 'float_array', argmin=0, eqmin=True, none_ok=True)
        validate_argument(locals(), 'sigma_c', 'float_array', none_ok=True)
        ra_source_, dec_source_, shear1_, shear2_ = arguments_consistency(
            [ra_source, dec_source, shear1, shear2],
            names=('Ra', 'Dec', 'Shear1', 'Shear2'),
            prefix='Tangential- and Cross- shape components sources')
    elif np.iterable(ra_source):
        ra_source_, dec_source_, shear1_, shear2_ = (
            np.array(col) for col in [ra_source, dec_source, shear1, shear2])
    else:
        ra_source_, dec_source_, shear1_, shear2_ = ra_source, dec_source, shear1, shear2
    # Compute the lensing angles
    if geometry == 'flat':
        angsep, phi = _compute_lensing_angles_flatsky(
            ra_lens, dec_lens, ra_source_, dec_source_)
    elif geometry == 'curve':
        angsep, phi = _compute_lensing_angles_astropy(
            ra_lens, dec_lens, ra_source_, dec_source_)
    else:
        raise NotImplementedError(
            f"Sky geometry {geometry} is not currently supported")
    # Compute the tangential and cross shears
    tangential_comp = _compute_tangential_shear(shear1_, shear2_, phi)
    cross_comp = _compute_cross_shear(shear1_, shear2_, phi)
    # If the is_deltasigma flag is True, multiply the results by Sigma_crit.
    if is_deltasigma:
        if sigma_c is None:
            # Need to verify that cosmology and redshifts are provided
            if any(t_ is None for t_ in (z_lens, z_source, cosmo)):
                raise TypeError(
                    'To compute DeltaSigma, please provide a '
                    'i) cosmology, ii) redshift of lens and sources')
            sigma_c = compute_critical_surface_density(cosmo, z_lens, z_source)
        tangential_comp *= sigma_c
        cross_comp *= sigma_c
    return angsep, tangential_comp, cross_comp

def compute_galaxy_weights(z_lens, cosmo, z_source=None, pzpdf=None, pzbins=None,
                           shape_component1=None, shape_component2=None,
                           shape_component1_err=None, shape_component2_err=None,
                           add_photoz=False, add_shapenoise=False, add_shape_error=False,
                           is_deltasigma=False, validate_input=True):

    r"""Compute the individual lens-source pair weights $w_{ls}$.

    The weights $w_{ls}$ expresses as : $w_{ls} = w_ls_geo * w_ls_shape$, following E. S. Sheldon et al.
    (2003), arXiv:astro-ph/0312036:

    1. the geometrical weight `w_ls_geo` depends on the lens and source redshift informations.
    This function allows the user to compute `w_ls_geo` using true (a.) or photometric (b.) redshifts of source galaxies.
        a. true background galaxy redshifts, considering the excess surface density:

        .. math::

        w_{ls, geo} = 1. / \Sigma_c(cosmo, z_L, z_{\rm src})^2

        .. math::

        b. photometric background galaxy redshifts:

        .. math::

        w_{ls, geo} = [\int_{\delta + z_L} dz_s p_{\rm photoz}(z_s) \Sigma_c(cosmo, z_L, z_s)^{-1}] ^2

        .. math::

        for the tangential shear, the weights 'w_ls_geo` are 1

    2. The shape weight `w_ls_shape` depends on shapenoise and/or shape measurement errors

        .. math::

        w_{ls, shape} = 1/(\sigma_{\rm shapenoise}^2 + \sigma_{\rm measurement}^2)

        .. math::

    The total weight `w_ls` is the product of the geometrical weight and the shape weight.

    The probability for a galaxy to be in the background of the cluster is defined by:

    .. math::

        P(z_s > z_l) = [\int_{z_L}^{+\infty} dz_s p_{\rm photoz}(z_s)

    .. math::

    The function return the probability for a galaxy to be in the background of the cluster;
    if photometric probability density functions are provoded, the function computes the above
    integral. In the case of true redshifts, it returns 1 if `z_s > z_l` else returns 0.

    Parameters:
    -----------
    z_lens: float
        Redshift of the lens.
    z_source: array, optional
        Redshift of the source. Used only with add_photoz=True.
    cosmo: clmm.Comology object, None
        CLMM Cosmology object.
    pzpdf : array, optional
        Photometric probablility density functions of the source galaxies
    pzbins : array, optional
        Redshift axis on which the individual photoz pdf is tabulated
    shape_component1: array
        The measured shear (or reduced shear or ellipticity) of the source galaxies
    shape_component2: array
        The measured shear (or reduced shear or ellipticity) of the source galaxies
    shape_component1_err: array
        The measurement error on the 1st-component of ellipticity of the source galaxies
    shape_component2_err: array
        The measurement error on the 2nd-component of ellipticity of the source galaxies
    add_photoz : boolean
        True if considering photometric redshifts
    add_shapenoise : boolean
        True for considering shapenoise in the weight computation
    add_shape_error : boolean
        True for considering measured shape error in the weight computation
    is_deltasigma: boolean
        Indicates whether it is the excess surface density or the tangential shear
    validate_input: bool
        Validade each input argument

    Returns:
    --------
    w_ls: array
        Individual lens source pair weights
    p_background : array
        Probability for being a background galaxy
    """
    if validate_input:
        validate_argument(locals(), 'z_lens', float, argmin=0, eqmin=True)
        validate_argument(locals(), 'z_source', 'float_array', argmin=0, eqmin=True, none_ok=True)
        #validate_argument(locals(), 'pzpdf', 'float_array')
        #validate_argument(locals(), 'pzbins', 'float_array')
        validate_argument(locals(), 'shape_component1', 'float_array')
        validate_argument(locals(), 'shape_component2', 'float_array')
        validate_argument(locals(), 'shape_component1_err', 'float_array', none_ok=True)
        validate_argument(locals(), 'shape_component2_err', 'float_array', none_ok=True)
        validate_argument(locals(), 'add_photoz', bool)
        validate_argument(locals(), 'add_shape_error', bool)
        validate_argument(locals(), 'add_shape_error', bool)
        validate_argument(locals(), 'is_deltasigma', bool)
        arguments_consistency(
            [shape_component1, shape_component2],
            names=('shape_component1', 'shape_component2'),
            prefix='Shape components sources')

    #computing w_ls_geo
    if add_photoz:
        # adding these lines to interpolate CLMM redshift grid for each galaxies
        # to a constant redshift grid for all galaxies. If there is a constant grid for all galaxies
        # these lines are not necessary and photoz_z_axis_grid_cut, photoz_matrix = pzbins, pzpdf
        photoz_z_axis_grid = np.linspace(0, 5, 100)
        photoz_z_axis_grid_cut = photoz_z_axis_grid[photoz_z_axis_grid > z_lens]
        photoz_matrix = np.array([np.interp(photoz_z_axis_grid_cut, pzbin, pdf)
                                    for pzbin, pdf in zip(pzbins, pzpdf)])
        ###
        p_background = scipy.integrate.simps(photoz_matrix, x=photoz_z_axis_grid_cut, axis = 1)
        w_ls_geo = 1
        if is_deltasigma == True:
            sigma_crit_grid = cosmo.eval_sigma_crit(z_lens, photoz_z_axis_grid_cut)
            # w_ls_geo = 1/(int sigma_c^-1)^2
            w_ls_geo = (scipy.integrate.simps(photoz_matrix * (1./sigma_crit_grid),
                                              x=photoz_z_axis_grid_cut, axis = 1))**-2
    else:
        p_background = np.zeros(len(z_source))
        p_background[z_source > z_lens] = 1
        norm = 1
        if is_deltasigma:
            norm = cosmo.eval_sigma_crit(z_lens, z_source)**2
        w_ls_geo = p_background/norm

    #computing w_ls_shape
    err_e_shapenoise = np.zeros(len(shape_component1))
    err_e_measurement = np.zeros(len(shape_component1))
    if add_shapenoise:
        err_e_shapenoise = np.sqrt(np.std(shape_component1)**2 + np.std(shape_component2)**2)
    if add_shape_error == True :
        err_e_measurement = np.sqrt(shape_component1_err**2 + shape_component2_err**2)
    w_ls_shape = 1./(err_e_measurement**2 + err_e_shapenoise**2)

    w_ls = w_ls_shape * w_ls_geo
    return w_ls, p_background

def _compute_lensing_angles_flatsky(ra_lens, dec_lens, ra_source_list, dec_source_list):
    r"""Compute the angular separation between the lens and the source and the azimuthal
    angle from the lens to the source in radians.

    In the flat sky approximation, these angles are calculated using
    .. math::
        \theta = \sqrt{\left(\delta_s-\delta_l\right)^2+
        \left(\alpha_l-\alpha_s\right)^2\cos^2(\delta_l)}

        \tan\phi = \frac{\delta_s-\delta_l}{\left(\alpha_l-\alpha_s\right)\cos(\delta_l)}

    For extended descriptions of parameters, see `compute_shear()` documentation.

    Parameters
    ----------
    ra_lens: float
        Right ascension of the lensing cluster in degrees
    dec_lens: float
        Declination of the lensing cluster in degrees
    ra_source_list: array
        Right ascensions of each source galaxy in degrees
    dec_source_list: array
        Declinations of each source galaxy in degrees

    Returns
    -------
    angsep: array
        Angular separation between the lens and the source in radians
    phi: array
        Azimuthal angle from the lens to the source in radians
    """
    # Put angles between -pi and pi
    r2pi = lambda x: x-np.round(x/(2.0*math.pi))*2.0*math.pi
    deltax = r2pi(np.radians(ra_source_list-ra_lens)) * \
        math.cos(math.radians(dec_lens))
    deltay = np.radians(dec_source_list-dec_lens)
    # Ensure that abs(delta ra) < pi
    #deltax[deltax >= np.pi] = deltax[deltax >= np.pi]-2.*np.pi
    #deltax[deltax < -np.pi] = deltax[deltax < -np.pi]+2.*np.pi
    angsep = np.sqrt(deltax**2+deltay**2)
    phi = np.arctan2(deltay, -deltax)
    # Forcing phi to be zero everytime angsep is zero. This is necessary due to arctan2 features.
    if np.iterable(phi):
        phi[angsep == 0.0] = 0.0
    else:
        phi = 0.0 if angsep == 0.0 else phi
    if np.any(angsep > np.pi/180.):
        warnings.warn(
            "Using the flat-sky approximation with separations >1 deg may be inaccurate")
    return angsep, phi


def _compute_lensing_angles_astropy(ra_lens, dec_lens, ra_source_list, dec_source_list):
    r"""Compute the angular separation between the lens and the source and the azimuthal
    angle from the lens to the source in radians.

    Parameters
    ----------
    ra_lens: float
        Right ascension of the lensing cluster in degrees
    dec_lens: float
        Declination of the lensing cluster in degrees
    ra_source_list: array
        Right ascensions of each source galaxy in degrees
    dec_source_list: array
        Declinations of each source galaxy in degrees

    Returns
    -------
    angsep: array
        Angular separation between the lens and the source in radians
    phi: array
        Azimuthal angle from the lens to the source in radians
    """
    sk_lens = SkyCoord(ra_lens*u.deg, dec_lens*u.deg, frame='icrs')
    sk_src = SkyCoord(ra_source_list*u.deg,
                      dec_source_list*u.deg, frame='icrs')
    angsep, phi = sk_lens.separation(
        sk_src).rad, sk_lens.position_angle(sk_src).rad
    # Transformations for phi to have same orientation as _compute_lensing_angles_flatsky
    phi += 0.5*np.pi
    if np.iterable(phi):
        phi[phi > np.pi] -= 2*np.pi
        phi[angsep == 0] = 0
    else:
        phi -= 2*np.pi if phi > np.pi else 0
        phi = 0 if angsep == 0 else phi
    return angsep, phi


def _compute_tangential_shear(shear1, shear2, phi):
    r"""Compute the tangential shear given the two shears and azimuthal positions for
    a single source or list of sources.

    We compute the tangential shear following Eq. 7 of Schrabback et al. 2018, arXiv:1611:03866
    .. math::
        g_t = -\left( g_1\cos\left(2\phi\right)+g_2\sin\left(2\phi\right)\right)

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    return -(shear1*np.cos(2.*phi)+shear2*np.sin(2.*phi))


def _compute_cross_shear(shear1, shear2, phi):
    r"""Compute the cross shear given the two shears and azimuthal position for a single
    source of list of sources.

    We compute the cross shear following Eq. 8 of Schrabback et al. 2018, arXiv:1611:03866
    also checked arxiv 0509252
    .. math::
        g_x = g_1 \sin\left(2\phi\right)-g_2\cos\left(2\phi\right)

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    return shear1*np.sin(2.*phi)-shear2*np.cos(2.*phi)


def make_radial_profile(components, angsep, angsep_units, bin_units,
                        bins=10, components_error=None, error_model='ste',
                        include_empty_bins=False, return_binnumber=False,
                        cosmo=None, z_lens=None, validate_input=True):
    r"""Compute the angular profile of given components

    We assume that the cluster object contains information on the cross and
    tangential shears or ellipticities and angular separation of the source galaxies

    This function can be called in two ways:

    1. Pass explict arguments::

        make_radial_profile([component1, component2], distances, 'radians')

    2. Call it as a method of a GalaxyCluster instance::

        cluster.make_radial_profile('radians', 'radians')

    Parameters
    ----------
    components: list of arrays
        List of arrays to be binned in the radial profile
    angsep: array
        Transvesal distances between the sources and the lens
    angsep_units : str
        Units of the calculated separation of the source galaxies
        Allowed Options = ["radians"]
    bin_units : str
        Units to use for the radial bins of the radial profile
        Allowed Options = ["radians", deg", "arcmin", "arcsec", kpc", "Mpc"]
        (letter case independent)
    bins : array_like, optional
        User defined bins to use for the shear profile. If a list is provided, use that as
        the bin edges. If a scalar is provided, create that many equally spaced bins between
        the minimum and maximum angular separations in bin_units. If nothing is provided,
        default to 10 equally spaced bins.
    components_error: list of arrays, None
        List of errors for input arrays
    error_model : str, optional
        Statistical error model to use for y uncertainties. (letter case independent)
            `ste` - Standard error [=std/sqrt(n) in unweighted computation] (Default).
            `std` - Standard deviation.
    include_empty_bins: bool, optional
        Also include empty bins in the returned table
    return_binnumber: bool, optional
        Also returns the indices of the bins for each object
    cosmo : CLMM.Cosmology
        CLMM Cosmology object to convert angular separations to physical distances
    z_lens: array, optional
        Redshift of the lens
    validate_input: bool
        Validade each input argument

    Returns
    -------
    profile : GCData
        Output table containing the radius grid points, the profile of the components `p_i`, errors
        `p_i_err` and number of sources.  The errors are defined as the standard errors in each bin.
    binnumber: 1-D ndarray of ints, optional
        Indices of the bins (corresponding to `xbins`) in which each value
        of `xvals` belongs.  Same length as `yvals`.  A binnumber of `i` means the
        corresponding value is between (xbins[i-1], xbins[i]).

    Notes
    -----
    This is an example of a place where the cosmology-dependence can be sequestered to another
    module.
    """
    # pylint: disable-msg=too-many-locals
    if validate_input:
        validate_argument(locals(), 'angsep', 'float_array')
        validate_argument(locals(), 'angsep_units', str)
        validate_argument(locals(), 'bin_units', str)
        validate_argument(locals(), 'include_empty_bins', bool)
        validate_argument(locals(), 'return_binnumber', bool)
        validate_argument(locals(), 'z_lens', 'float_array', none_ok=True)
        comp_dict = {f'components[{i}]': comp for i, comp in enumerate(components)}
        arguments_consistency(components, names=comp_dict.keys(), prefix='Input components')
        for component in comp_dict:
            validate_argument(comp_dict, component, 'float_array')
    # Check to see if we need to do a unit conversion
    if angsep_units is not bin_units:
        source_seps = convert_units(angsep, angsep_units, bin_units,
                                    redshift=z_lens, cosmo=cosmo)
    else:
        source_seps = angsep
    # Make bins if they are not provided
    if not hasattr(bins, '__len__'):
        bins = make_bins(np.min(source_seps), np.max(source_seps), bins)
    # Create output table
    profile_table = GCData([bins[:-1], np.zeros(len(bins)-1), bins[1:]],
                           names=('radius_min', 'radius', 'radius_max'),
                           meta={'bin_units': bin_units},  # Add metadata
                           )
    # Compute the binned averages and associated errors
    for i, component in enumerate(components):
        r_avg, comp_avg, comp_err, nsrc, binnumber = compute_radial_averages(
            source_seps, component, xbins=bins,
            yerr=None if components_error is None else components_error[i])
        profile_table[f'p_{i}'] = comp_avg
        profile_table[f'p_{i}_err'] = comp_err
    profile_table['radius'] = r_avg
    profile_table['n_src'] = nsrc
    # return empty bins?
    if not include_empty_bins:
        profile_table = profile_table[nsrc > 1]
    if return_binnumber:
        return profile_table, binnumber
    return profile_table
