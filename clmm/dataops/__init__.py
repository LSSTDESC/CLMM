"""Functions to compute polar/azimuthal averages in radial bins"""
import warnings
import numpy as np
import scipy
from scipy.stats import binned_statistic
from astropy.coordinates import SkyCoord
from astropy import units as u
from ..gcdata import GCData
from ..utils import (
    compute_radial_averages,
    make_bins,
    convert_units,
    arguments_consistency,
    validate_argument,
    _validate_ra,
    _validate_dec,
    _validate_is_deltasigma_sigma_c,
)
from ..redshift import (
    _integ_pzfuncs,
    compute_for_good_redshifts,
)
from ..theory import compute_critical_surface_density_eff


def compute_tangential_and_cross_components(
    ra_lens,
    dec_lens,
    ra_source,
    dec_source,
    shear1,
    shear2,
    geometry="curve",
    is_deltasigma=False,
    sigma_c=None,
    include_quadrupole=False,
    phi_major=None,
    ra_mem=None,
    dec_mem=None,
    weight_mem=None,
    validate_input=True,
):
    r"""Computes tangential- and cross- components for shear or ellipticity

    To do so, we need the right ascension and declination of the lens and of all of the sources.
    We also need the two shape components of all of the sources.

    These quantities can be handed to `compute_tangential_and_cross_components` in two ways

    1. Pass in each as parameters::

        compute_tangential_and_cross_components(ra_lens, dec_lens, ra_source, dec_source,
        shape_component1, shape_component2)

    2. As a method of `GalaxyCluster`::

        cluster.compute_tangential_and_cross_components()

    The angular separation between the source and the lens, :math:`\theta`, and the azimuthal
    position of the source relative to the lens, :math:`\phi`, are computed within the function
    and the angular separation is returned.

    In the flat sky approximation, these angles are calculated using (_lens: lens,
    _source: source, RA is from right to left)

    .. math::
        \theta^2 = & \left(\delta_s-\delta_l\right)^2+
        \left(\alpha_l-\alpha_s\right)^2\cos^2(\delta_l)\\
        \tan\phi = & \frac{\delta_s-\delta_l}{\left(\alpha_l-\alpha_s\right)\cos(\delta_l)}

    The tangential, :math:`g_t`, and cross, :math:`g_x`, ellipticity/shear components are
    calculated using the two ellipticity/shear components :math:`g_1` and :math:`g_2` of the
    source galaxies, following Eq.7 and Eq.8 in Schrabback et al. (2018), arXiv:1611:03866 which
    is consistent with arXiv:0509252

    .. math::
        g_t =& -\left( g_1\cos\left(2\phi\right)+g_2\sin\left(2\phi\right)\right)\\
        g_x =& g_1 \sin\left(2\phi\right)-g_2\cos\left(2\phi\right)

    The quadrupole ellipticity/shear components, :math:`g_4theta' and :math:`g_const', 
    are also calculated using the two ellipticity/shear components :math:`g_1' and :math:`g_2' of the
    source galaxies, following Eq.31 and Eq.34 in Shin et al. (2018), arXiv:1705.11167. 

    .. math::
        g_4theta =& \left( g_1\cos\left(4\phi\right)+g_2\sin\left(4\phi\right)\right)\\
        g_const =& g_1

    Finally, if  the critical surface density (:math:`\Sigma_\text{crit}`) is provided, an estimate
    of the excess surface density :math:`\widehat{\Delta\Sigma}` is obtained from

    .. math::
        \widehat{\Delta\Sigma_{t,x}} = g_{t,x} \times \Sigma_\text{crit}(z_l, z_{\text{src}})

    If :math:`g_{t,x}` correspond to the shear, the above expression is an accurate. However, if
    :math:`g_{t,x}` correspond to ellipticities or reduced shear, this expression only gives an
    estimate :math:`\widehat{\Delta\Sigma_{t,x}}`, valid only in the weak lensing regime.

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
        It requires `sigma_c` argument. Results in units of :math:`M_\odot\ Mpc^{-2}`
    sigma_c : None, array_like
        Critical (effective) surface density in units of :math:`M_\odot\ Mpc^{-2}`.
        Used only when is_deltasigma=True.
    include_quadrupole: bool
        If `True`, the quadrupole shear components (g_4theta, g_const; Shin+2018) are calculated
        instead of g_t and g_x
    phi_major : float, optional
        the direction of the major axis of the input cluster in the unit of radian. 
        only needed when `include_quadrupole` is `True`.
        Users could choose to provide ra_mem, dec_mem and weight_mem instead of this quantity.
    ra_mem : array, optional
        right ascentions of the member galaxies of the input cluster,
        to calculate the direction of the major axis of the cluster.
        Only needed when `include_quadrupole` is `True` and `phi_major` is not provided.
    dec_mem : array, optional
        declinations of the member galaxies of the input cluster,
        to calculate the direction of the major axis of the cluster.
        Only needed when `include_quadrupole` is `True` and `phi_major` is not provided.
    weight_mem : array, optional
        weights for the member galaxies to be applied when calcualting 
        the major axis of the input cluster out of ra_mem and dec_mem. 
        It could be `1` (no weights) or `membership probability (p_mem)` or any user choice. 
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
    IF include_quadrupole:
        4theta_component: array_like
            4theta shear component (or assimilated quantity) for each source galaxy
        const_component: array_like
            constant shear component (or assimilated quantity) for each source galaxy
    """
    # pylint: disable-msg=too-many-locals
    # Note: we make these quantities to be np.array so that a name is not passed from astropy
    # columns
    if validate_input:
        _validate_ra(locals(), "ra_source", True)
        _validate_dec(locals(), "dec_source", True)
        _validate_ra(locals(), "ra_lens", True)
        _validate_dec(locals(), "dec_lens", True)
        validate_argument(locals(), "shear1", "float_array")
        validate_argument(locals(), "shear2", "float_array")
        validate_argument(locals(), "geometry", str)
        validate_argument(locals(), "is_deltasigma", bool)
        validate_argument(locals(), "include_quadrupole", bool)
        validate_argument(locals(), "sigma_c", "float_array", none_ok=True)
        ra_source_, dec_source_, shear1_, shear2_ = arguments_consistency(
            [ra_source, dec_source, shear1, shear2],
            names=("Ra", "Dec", "Shear1", "Shear2"),
            prefix="Tangential- and Cross- shape components sources",
        )
        _validate_is_deltasigma_sigma_c(is_deltasigma, sigma_c)
        if phi_major is not None:
            validate_argument(locals(), "phi_major", float)
        else:
            validate_argument(locals(), "ra_mem", "float_array")
            validate_argument(locals(), "dec_mem", "float_array")
            validate_argument(locals(), "weight_mem", "float_array")
    elif np.iterable(ra_source):
        ra_source_, dec_source_, shear1_, shear2_ = (
            np.array(col) for col in [ra_source, dec_source, shear1, shear2]
        )
    else:
        ra_source_, dec_source_, shear1_, shear2_ = (
            ra_source,
            dec_source,
            shear1,
            shear2,
        )
    # Compute the lensing angles
    if geometry == "flat":
        angsep, phi = _compute_lensing_angles_flatsky(ra_lens, dec_lens, ra_source_, dec_source_)
    elif geometry == "curve":
        angsep, phi = _compute_lensing_angles_astropy(ra_lens, dec_lens, ra_source_, dec_source_)
    else:
        raise NotImplementedError(f"Sky geometry {geometry} is not currently supported")
    # Compute the tangential and cross shears
    tangential_comp = _compute_tangential_shear(shear1_, shear2_, phi)
    cross_comp = _compute_cross_shear(shear1_, shear2_, phi)
    # If the is_deltasigma flag is True, multiply the results by Sigma_crit.
    if sigma_c is not None:
        _sigma_c_arr = np.array(sigma_c)
        tangential_comp *= _sigma_c_arr
        cross_comp *= _sigma_c_arr
    if include_quadrupole:
        if phi_major is not None:
            phi_major_ = phi_major
        else:
            phi_major_ = _calculate_major_axis(ra_lens, dec_lens, ra_mem, dec_mem, weight_mem)
        rotated_phi = phi - phi_major_
        shear_vector = shear1_ + shear2_*1.j
        rotated_shear_vector = shear_vector * np.exp(-2.j*phi_major_)
        rotated_shear1, rotated_shear2 = np.real(rotated_shear_vector), np.imag(rotated_shear_vector)
        # Compute the quadrupole shear components
        four_theta_comp = _compute_4theta_shear(rotated_shear1, rotated_shear2, rotated_phi)
        const_comp = rotated_shear1
        # If the is_deltasigma flag is True, multiply the results by Sigma_crit.
        if sigma_c is not None:
            four_theta_comp *= _sigma_c_arr
            const_comp *= _sigma_c_arr
    if include_quadrupole:
        return angsep, tangential_comp, cross_comp, four_theta_comp, const_comp
    else:
        return angsep, tangential_comp, cross_comp

def compute_background_probability(
    z_lens, z_src=None, use_pdz=False, pzpdf=None, pzbins=None, validate_input=True
):
    r"""Probability for being a background galaxy, defined by:

        .. math::
            P(z_s > z_l) = \int_{z_l}^{+\infty} dz_s \; p_{\text{photoz}}(z_s),

    when the photometric probability density functions (:math:`p_{\text{photoz}}(z_s)`) are
    provided. In the case of true redshifts, it returns 1 if :math:`z_s > z_l` else returns 0.


    Parameters
    ----------
    z_lens: float
        Redshift of the lens.
    z_src: array, optional
        Redshift of the source. Used only if pzpdf=pzbins=None.

    Returns
    -------
    p_background : array
        Probability for being a background galaxy
    """
    if validate_input:
        validate_argument(locals(), "z_lens", float, argmin=0, eqmin=True)
        validate_argument(locals(), "z_src", "float_array", argmin=0, eqmin=True, none_ok=True)

    if use_pdz is False:
        if z_src is None:
            raise ValueError("z_src must be provided.")
        p_background = np.array(z_src > z_lens, dtype=float)
    else:
        if pzpdf is None or pzbins is None:
            raise ValueError("pzbins must be provided with pzpdf.")
        p_background = _integ_pzfuncs(pzpdf, pzbins, z_lens)

    return p_background


def compute_galaxy_weights(
    use_shape_noise=False,
    shape_component1=None,
    shape_component2=None,
    use_shape_error=False,
    shape_component1_err=None,
    shape_component2_err=None,
    is_deltasigma=False,
    sigma_c=None,
    validate_input=True,
):
    r"""Computes the individual lens-source pair weights

    The weights :math:`w_{ls}` express as : :math:`w_{ls} = w_{ls, \text{geo}} \times w_{ls,
    \text{shape}}`, following E. S. Sheldon et al. (2003), arXiv:astro-ph/0312036:

    1. If computed for shear, the geometrical weights :math:`w_{ls, \text{geo}}` are equal to 1. If
    computed for :math:`\Delta \Sigma`, it depends on lens and source redshift information via the
    critical surface density. This component can be expressed as:

        .. math::
            w_{ls, \text{geo}} = \Sigma_\text{crit}(z_l, z_{\text{src}})^{-2}\;.

        when only redshift point estimates are provided, or as:


        .. math::
            w_{ls, \text{geo}} = \Sigma_\text{crit}^\text{eff}(z_l, z_{\text{src}})^{-2}
            = \left[\int_{\delta + z_l} dz_s \; p_{\text{photoz}}(z_s)
            \Sigma_\text{crit}(z_l, z_s)^{-1}\right]^2

        when the redshift pdf of each source, :math:`p_{\text{photoz}}(z_s)`, is known.

    2. The shape weight :math:`w_{ls,{\text{shape}}}` depends on shapenoise and/or shape
    measurement errors

        .. math::
            w_{ls, \text{shape}}^{-1} = \sigma_{\text{shapenoise}}^2 +
            \sigma_{\text{measurement}}^2


    Parameters
    ----------
    is_deltasigma: bool
        If `False`, weights are computed for shear, else weights are computed for
        :math:`\Delta \Sigma`.
    sigma_c : None, array_like
        Critical (effective) surface density in units of :math:`M_\odot\ Mpc^{-2}`.
        Used only when is_deltasigma=True.
    use_shape_noise: bool
        If `True` shape noise is included in the weight computation. It then requires
        `shape_componenet{1,2}` to be provided. Default: False.
    shape_component1: array_like
        The measured shear (or reduced shear or ellipticity) of the source galaxies,
        used if `use_shapenoise=True`
    shape_component2: array_like
        The measured shear (or reduced shear or ellipticity) of the source galaxies,
        used if `use_shapenoise=True`
    use_shape_error: bool
        If `True` shape errors are included in the weight computation. It then requires
        shape_component{1,2}_err` to be provided. Default: False.
    shape_component1_err: array_like
        The measurement error on the 1st-component of ellipticity of the source galaxies,
        used if `use_shape_error=True`
    shape_component2_err: array_like
        The measurement error on the 2nd-component of ellipticity of the source galaxies,
        used if `use_shape_error=True`
    validate_input: bool
        Validade each input argument

    Returns
    -------
    w_ls: array_like
        Individual lens source pair weights
    """
    if validate_input:
        validate_argument(locals(), "sigma_c", "float_array", none_ok=True)
        validate_argument(locals(), "shape_component1", "float_array", none_ok=True)
        validate_argument(locals(), "shape_component2", "float_array", none_ok=True)
        validate_argument(locals(), "shape_component1_err", "float_array", none_ok=True)
        validate_argument(locals(), "shape_component2_err", "float_array", none_ok=True)
        validate_argument(locals(), "use_shape_noise", bool)
        validate_argument(locals(), "use_shape_error", bool)
        arguments_consistency(
            [shape_component1, shape_component2],
            names=("shape_component1", "shape_component2"),
            prefix="Shape components sources",
        )
        _validate_is_deltasigma_sigma_c(is_deltasigma, sigma_c)

    # computing w_ls_geo
    w_ls_geo = 1.0
    if sigma_c is not None:
        w_ls_geo /= np.array(sigma_c) ** 2

    # computing w_ls_shape
    err_e2 = 0

    if use_shape_noise:
        if shape_component1 is None or shape_component2 is None:
            raise ValueError(
                "With the shape noise option, the source shapes "
                "`shape_component_{1,2}` must be specified"
            )
        err_e2 += np.std(shape_component1) ** 2 + np.std(shape_component2) ** 2
    if use_shape_error:
        if shape_component1_err is None or shape_component2_err is None:
            raise ValueError(
                "With the shape error option, the source shapes errors"
                "`shape_component_err{1,2}` must be specified"
            )
        err_e2 += shape_component1_err**2
        err_e2 += shape_component2_err**2

    if hasattr(err_e2, "__len__"):
        w_ls_shape = np.ones(len(err_e2))
        w_ls_shape[err_e2 > 0] = 1.0 / err_e2[err_e2 > 0]
    else:
        w_ls_shape = 1.0 / err_e2 if err_e2 > 0 else 1.0

    w_ls = w_ls_shape * w_ls_geo

    return w_ls


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
    delta_ra = np.radians(ra_source_list - ra_lens)
    # Put angles between -pi and pi
    delta_ra -= np.round(delta_ra / (2.0 * np.pi)) * 2.0 * np.pi
    deltax = delta_ra * np.cos(np.radians(dec_lens))
    deltay = np.radians(dec_source_list - dec_lens)
    # Ensure that abs(delta ra) < pi
    # deltax[deltax >= np.pi] = deltax[deltax >= np.pi]-2.*np.pi
    # deltax[deltax < -np.pi] = deltax[deltax < -np.pi]+2.*np.pi
    angsep = np.sqrt(deltax**2 + deltay**2)
    phi = np.arctan2(deltay, -deltax)
    # Forcing phi to be zero everytime angsep is zero. This is necessary due to arctan2 features.
    if np.iterable(phi):
        phi[angsep == 0.0] = 0.0
    else:
        phi = 0.0 if angsep == 0.0 else phi
    if np.any(angsep > np.pi / 180.0):
        warnings.warn("Using the flat-sky approximation with separations >1 deg may be inaccurate")
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
    sk_lens = SkyCoord(ra_lens * u.deg, dec_lens * u.deg, frame="icrs")
    sk_src = SkyCoord(ra_source_list * u.deg, dec_source_list * u.deg, frame="icrs")
    angsep, phi = sk_lens.separation(sk_src).rad, sk_lens.position_angle(sk_src).rad
    # Transformations for phi to have same orientation as _compute_lensing_angles_flatsky
    phi += 0.5 * np.pi
    if np.iterable(phi):
        phi[phi > np.pi] -= 2 * np.pi
        phi[angsep == 0] = 0
    else:
        phi -= 2 * np.pi if phi > np.pi else 0
        phi = 0 if angsep == 0 else phi
    return angsep, phi

def _calculate_major_axis(ra_lens_, dec_lens_, ra_mem_, dec_mem_, weight_mem_):
    r"""Compute the major axis of a given cluster from the distribution of
    its member galaxies using the position second moments. 

    The computation is done according to Eq. 5-10 of Shin et al. 2018, arXiv:1705.11167

    Current implementation assumes the +RA direction is aligned with +x direction. 

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    sk_lens = SkyCoord(ra_lens_ * u.deg, dec_lens_ * u.deg, frame="icrs")
    sk_mem = SkyCoord(ra_mem_ * u.deg, dec_mem_ * u.deg, frame="icrs")
    position_angle_mem = sk_lens.position_angle(sk_mem).radian + np.pi/2.
    separation_mem = sk_lens.separation(sk_mem).degree
    x_mem = separation_mem * np.cos(position_angle_mem)
    y_mem = separation_mem * np.sin(position_angle_mem)
    distance_weight_mem = 1./(separation_mem**2)
    weight_total_mem = weight_mem_ * distance_weight_mem
    sum_weight_total_mem = np.sum(weight_total_mem)
    # Calcualte second moments of the member galaxies
    Ixx = np.sum(x_mem**2 * weight_total_mem)/sum_weight_total_mem
    Iyy = np.sum(y_mem**2 * weight_total_mem)/sum_weight_total_mem
    Ixy = np.sum(x_mem * y_mem * weight_total_mem)/sum_weight_total_mem
    # Transformation of the second moments to the direction of the major axis
    phi_major = np.arctan2(2.*Ixy , Ixx-Iyy)/2.
    return phi_major



def _compute_tangential_shear(shear1, shear2, phi):
    r"""Compute the tangential shear given the two shears and azimuthal positions for
    a single source or list of sources.

    We compute the tangential shear following Eq. 7 of Schrabback et al. 2018, arXiv:1611:03866

    .. math::
        g_t = -\left( g_1\cos\left(2\phi\right)+g_2\sin\left(2\phi\right)\right)

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    return -(shear1 * np.cos(2.0 * phi) + shear2 * np.sin(2.0 * phi))


def _compute_cross_shear(shear1, shear2, phi):
    r"""Compute the cross shear given the two shears and azimuthal position for a single
    source of list of sources.

    We compute the cross shear following Eq. 8 of Schrabback et al. 2018, arXiv:1611:03866
    also checked arxiv 0509252

    .. math::
        g_x = g_1 \sin\left(2\phi\right)-g_2\cos\left(2\phi\right)

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    return shear1 * np.sin(2.0 * phi) - shear2 * np.cos(2.0 * phi)

def _compute_4theta_shear(shear1, shear2, phi):
    r"""Compute the 4theta component of the quadrupole shear given the two shear components and
    azimuthal positions for a single source or list of sources.

    We compute the tangential shear following Eq. 31 of Shin et al. 2018, arXiv:1705.11167

    .. math::
        g_4theta = g_1\cos\left(4\phi\right)+g_2\sin\left(4\phi\right)

    Note that here `\phi` is the position angle of the source galaxies 
    with respect to the major axis of the cluster.
    Also, `shear1` and `shear2` are measured in the coordinate where
    the +x axis is aligned with the major axis of the cluster.

    For extended descriptions of parameters, see `compute_shear()' documentation.
    """
    return shear1 * np.cos(4.0 * phi) + shear2 * np.sin(4.0 * phi)

def make_radial_profile(
    components,
    angsep,
    angsep_units,
    bin_units,
    bins=10,
    components_error=None,
    error_model="ste",
    include_empty_bins=False,
    return_binnumber=False,
    cosmo=None,
    z_lens=None,
    validate_input=True,
    weights=None,
):
    r"""Compute the angular profile of given components

    We assume that the cluster object contains information on the cross and
    tangential shears or ellipticities and angular separation of the source galaxies

    This function can be called in two ways:

    1. Pass explict arguments::

        make_radial_profile([component1, component2], distances, 'radians')

    2. Call it as a method of a GalaxyCluster instance and specify `bin_units`:

        cluster.make_radial_profile('Mpc')

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
    weights: array-like, optional
        Array of individual galaxy weights. If specified, the radial binned profile is
        computed using a weighted average

    Returns
    -------
    profile : GCData
        Output table containing the radius grid points, the profile of the components `p_i`,
        errors `p_i_err` and number of sources. The errors are defined as the standard errors in
        each bin.
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
        validate_argument(locals(), "angsep", "float_array")
        validate_argument(locals(), "angsep_units", str)
        validate_argument(locals(), "bin_units", str)
        validate_argument(locals(), "include_empty_bins", bool)
        validate_argument(locals(), "return_binnumber", bool)
        validate_argument(locals(), "z_lens", "float_array", none_ok=True)
        comp_dict = {f"components[{i}]": comp for i, comp in enumerate(components)}
        arguments_consistency(components, names=comp_dict.keys(), prefix="Input components")
        for component in comp_dict:
            validate_argument(comp_dict, component, "float_array")
    # Check to see if we need to do a unit conversion
    if angsep_units is not bin_units:
        source_seps = convert_units(angsep, angsep_units, bin_units, redshift=z_lens, cosmo=cosmo)
    else:
        source_seps = angsep
    # Make bins if they are not provided
    if not hasattr(bins, "__len__"):
        bins = make_bins(np.min(source_seps), np.max(source_seps), bins)
    # Create output table
    profile_table = GCData(
        [bins[:-1], np.zeros(len(bins) - 1), bins[1:]],
        names=("radius_min", "radius", "radius_max"),
        meta={"bin_units": bin_units},  # Add metadata
    )
    # Compute the binned averages and associated errors
    for i, component in enumerate(components):
        r_avg, comp_avg, comp_err, nsrc, binnumber, wts_sum = compute_radial_averages(
            source_seps,
            component,
            xbins=bins,
            yerr=None if components_error is None else components_error[i],
            error_model=error_model,
            weights=weights,
        )
        profile_table[f"p_{i}"] = comp_avg
        profile_table[f"p_{i}_err"] = comp_err
    profile_table["radius"] = r_avg
    profile_table["n_src"] = nsrc
    profile_table["weights_sum"] = wts_sum
    # return empty bins?
    if not include_empty_bins:
        profile_table = profile_table[nsrc > 1]
    if return_binnumber:
        return profile_table, binnumber
    return profile_table


def make_stacked_radial_profile(angsep, weights, components):
    """Compute stacked profile, and mean separation distances.

    Parameters
    ----------
    angsep: 2d array
        Transvesal distances corresponding to each object with shape `n_obj, n_rad_bins`.
    weights: 2d array
        Weights corresponding to each objects with shape `n_obj, n_rad_bins`.
    components: list of 2d arrays
        List of 2d properties of each array to be stacked with shape
        `n_components, n_obj, n_rad_bins`.

    Returns
    -------
    staked_angsep: array
        Mean transversal distance in each radial bin.
    stacked_components: list of arrays
        List of stacked components.
    """
    staked_angsep = np.average(angsep, axis=0, weights=None)
    stacked_components = [
        np.average(component, axis=0, weights=weights) for component in components
    ]
    return staked_angsep, stacked_components

def measure_Delta_Sigma_const_triaxiality(w, gamma1, Sigma_crit) :
    """Measure Delta sigma const quadrupole value for individual galaxies.

    Parameters
    ----------
    w: float or array
        Weight computed for each galaxy by the descrition as given in Shin et al. (https://doi.org/10.1093/mnras/stx3366)
    gamma1: float or array
        Shear component gamma 1
    Sigma_crit: float
        Critical surface mass density in M_{sun}/Mpc^{2}

    Returns
    -------
    dsconst_data: The measured "4theta" quadrupole component from the input shear measurements for each galaxy
    """
    dsconst_data = w * Sigma_crit * gamma1 / w
    return dsconst_data

def measure_Delta_Sigma_4theta_triaxiality(w1, w2, gamma1, gamma2, theta, Sigma_crit) :
    """Measure Delta sigma 4theta quadrupole value for individual galaxies.

    Parameters
    ----------
    w1: float or array
        Weight computed for each galaxy by the descrition as given in Shin et al. (https://doi.org/10.1093/mnras/stx3366)
    w2: float or array
        Weight computed for each galaxy by the descrition as given in Shin et al. (https://doi.org/10.1093/mnras/stx3366)
    gamma1: float or array
        Shear component gamma 1
    gamma2: float or array
        Shear component gamma 2
    theta: float or array
        Position Angle of galaxy measured counter-clockwise from the gamma1 direction
    Sigma_crit: float
        Critical surface mass density in M_{sun}/Mpc^{2}

    Returns
    -------
    ds4theta_data: The measured "const" quadrupole component from the input shear measurements for each galaxy
    """
    ds4theta_data = Sigma_crit * (w1*gamma1/np.cos(4*theta) + w2*gamma2/np.sin(4*theta)) / (w1 + w2)
    return ds4theta_data

def measure_weights_triaxiality(Sigma_crit, theta, Sigma_shape=0.0001, Sigma_meas=0) :
    """Measure weight values for quadrupole measurement of individual galaxies. 
    Using Equations 35, 32, 33 from Shin et al. 2018 (https://doi.org/10.1093/mnras/stx3366)

    Parameters
    ----------
    Sigma_crit: float
        Critical surface mass density in M_{sun}/Mpc^{2}
    theta: float or array
        Position Angle of galaxy measured counter-clockwise from the gamma1 direction
    Sigma_shape: Float, optional
        Scatter in the shape measurement from shape noise
    Sigma_meas: Float, optional
        Measurement error for shapes

    Returns
    -------
    w, w1, w2: float or array
        Weights for each galaxy
    """
	
    w  = 1 / (Sigma_crit**2 * (Sigma_shape**2 + Sigma_meas**2))
    w1 = np.cos(4*theta)**2 * w
    w2 = np.sin(4*theta)**2 * w
    return w, w1, w2

def make_binned_estimators_triaxiality(gamma1, gamma2, x_arcsec, y_arcsec, z_cluster, z_source, cosmo, monopole_bins=15,
                                      quadrupole_bins=15, r_min=0.3, r_max=2.5):
    """
    Makes radially binned profiles for Monopole, Quadrupole (4theta and const) as described in Shin et al. 2018
    Parameters
    ----------
    gamma1: float or array
        Shear component gamma 1
    gamma2: float or array
        Shear component gamma 2
    x_arcsec: float or array
        X position in arcsec when cluster center is at 0.0 and x axis is assumed to be along the major axis of the cluster
    y_arcsec: float or array
        Y position in arcsec when cluster center is at 0.0
    z_cluster: float
        Redshift of lens cluster
    cosmo: clmm.cosmology.Cosmology object
        CLMM Cosmology object 
    monopole_bins: float, optional
        Number of bins for monopole lensing profile
    quadrupole_bins: float, optional
        Number of bins for quadrupole lensing profiles (4theta, const)
    r_min: Float, optional
        Minimum radial distance to measure monopole and quadrupole shear profiles in Mpc (Default is 0.3 Mpc)
    r_max: Float, optional
        Maximum radial distance to measure monopole and quadrupole shear profiles in Mpc (Default is 2.5 Mpc)

    Returns
    -------
    ds_mono: float or array
        Binned monopole lensing profile
    ds_mono_err: float or array
        Uncertainty for the binned monopole lensing shear profile
    r_mono: float or array
        Bin centers for the monopole lensing shear profile
    ds_4theta: float or array
        Binned binned quadrupole 4-theta lensing shear profile
    ds_4theta_err: float or array
        Uncertainty for the binned quadrupole 4-theta lensing shear profile
    ds_const: float or array
        Binned binned quadrupole const lensing shear profile
    ds_const_err: float or array
        Uncertainty for the binned quadrupole const lensing shear profile
    r_quad: float or array
        Bin centers for the quadrupole lensing shear profiles (4theta and const)
    """
    
    sigma_crit = cosmo.eval_sigma_crit(z_cluster,z_source)
    r = np.sqrt((x_arcsec**2 + y_arcsec**2)) #In arcsecs
    theta = np.arctan2(y_arcsec, x_arcsec)
    r_mpc = r*cosmo.eval_da(z_cluster) * np.pi/180.0 * 1/3600 #In Mpc
    
    w, w1, w2 = measure_weights_triaxiality(sigma_crit, theta)
    DS4theta = measure_Delta_Sigma_4theta_triaxiality(w1, w2, gamma1, gamma2, theta, sigma_crit)
    DSconst = measure_Delta_Sigma_const_triaxiality(w, gamma1, sigma_crit)
    
    #Binning for Quadrupole measurements DS4theta and DSconst
    bins=quadrupole_bins
    r_min = r_min
    r_max = r_max
    
    bin_edges = np.logspace(np.log10(r_min), np.log10(r_max), bins)
    N_i = []
    for i in np.arange(bins-1):
        N_i.append(len(r_mpc[(r_mpc > bin_edges[i]) & (r_mpc < bin_edges[i+1])]))
    N_i=np.array(N_i)


    result = binned_statistic(r_mpc, gamma1, statistic='mean', bins=bin_edges)
    gamma1_i = result.statistic
    res = binned_statistic(r_mpc, gamma2, statistic='mean', bins=bin_edges)
    gamma2_i = res.statistic
    res = binned_statistic(r_mpc, DS4theta, statistic='mean', bins=bin_edges)
    DS4theta_i_err = binned_statistic(r_mpc, DS4theta, statistic='std', bins=bin_edges).statistic/np.sqrt(N_i)
    DS4theta_i = res.statistic
    res = binned_statistic(r_mpc, DSconst, statistic='mean', bins=bin_edges)
    DSconst_i = res.statistic
    DSconst_i_err = binned_statistic(r_mpc, DSconst, statistic='std', bins=bin_edges).statistic/np.sqrt(N_i)
    r_i = bin_edges
    
    #Binning for Monopole Measurements:
    bins_mono=monopole_bins
    bin_edges = np.logspace(np.log10(r_min), np.log10(r_max), bins_mono)
    N_i = []
    for i in np.arange(bins_mono-1):
        N_i.append(len(r_mpc[(r_mpc > bin_edges[i]) & (r_mpc < bin_edges[i+1])]))
    N_i=np.array(N_i)

    r_mono = bin_edges
    res = binned_statistic(r_mpc, -gamma1*np.cos(2*theta)-gamma2*np.sin(2*theta), statistic='mean', bins=bin_edges)
    gammat_mono = res.statistic
    ds_mono_err = binned_statistic(r_mpc, -gamma1*np.cos(2*theta)-gamma2*np.sin(2*theta), statistic='std',
                                   bins=bin_edges).statistic/np.sqrt(N_i)*sigma_crit
    ds_mono = gammat_mono*sigma_crit
        
    # SAFEGUARD AGAINST BINS WITH NANs and 0.0s
    
    ind = np.invert(np.isnan(ds_mono) | np.isnan(ds_mono_err))
    ds_mono = ds_mono[ind]
    ds_mono_err = ds_mono_err[ind]
    r_mono = np.sqrt(r_mono[:-1]*r_mono[1:])[ind]
    ind = (ds_mono!= 0.0) & (ds_mono_err!= 0.0)
    ds_mono = ds_mono[ind]
    ds_mono_err = np.abs(ds_mono_err[ind]) 
    r_mono = r_mono[ind] 
    
    ind = np.invert(np.isnan(DS4theta_i) | np.isnan(DS4theta_i_err) | np.isnan(DSconst_i) | np.isnan(DSconst_i_err))
    ds_4theta = DS4theta_i[ind]
    ds_4theta_err = np.abs(DS4theta_i_err[ind])
    ds_const = DSconst_i[ind] 
    ds_const_err = np.abs(DSconst_i_err[ind])
    r_quad = np.sqrt(r_i[:-1]*r_i[1:])[ind]
    
    return ds_mono,ds_mono_err,r_mono,ds_4theta,ds_4theta_err,ds_const,ds_const_err,r_quad

