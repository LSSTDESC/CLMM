"""Data operation for polar/azimuthal averages in radial bins and weights"""

import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from ..gcdata import GCData
from ..utils import (
    _integ_pzfuncs,
    _validate_coordinate_system,
    _validate_dec,
    _validate_include_quadrupole_phi_major,
    _validate_is_deltasigma_sigma_c,
    _validate_ra,
    arguments_consistency,
    compute_radial_averages,
    convert_units,
    make_bins,
    validate_argument,
)


def compute_tangential_and_cross_components(
    ra_lens,
    dec_lens,
    ra_source,
    dec_source,
    shear1,
    shear2,
    coordinate_system=None,
    geometry="curve",
    is_deltasigma=False,
    sigma_c=None,
    include_quadrupole=False,
    phi_major=None,
    info_mem=None,
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

    The quadrupole ellipticity/shear components, :math:`g_4theta` and :math:`g_const`, 
    are also calculated using the two ellipticity/shear components :math:`g_1` and :math:`g_2` 
    of the source galaxies, following Eq.31 and Eq.34 in Shin et al. (2018), arXiv:1705.11167. 

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
    coordinate_system: str, optional
        Coordinate system of the ellipticity components. Must be either `celestial` or
        `euclidean`. See https://doi.org/10.48550/arXiv.1407.7676 section 5.1 for more details.
        Default is `euclidean`.
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
    phi_major : float, optional
        the direction of the major axis of the input cluster in the unit of radian. 
        only needed when `include_quadrupole` is `True`.
        This quantity is always in Euclidean coordinates, for celestial coordinates only set
        the coordinate_system=`celestial`.
        Users could choose to provide ra_mem, dec_mem and weight_mem instead of this quantity.
    info_mem : list of array, optional
        RAs, DECs and weights of the member galaxies of the input cluster,
        to calcualte the direction of the major axis of the cluster.
        Only needed when `include_quadrupole` is `True` and `phi_major` is not provided.
        The shape must be in `[ra_mem,dec_mem,weight_mem]`, where each element is an array.
        The weights could be `1` (no weights) or 
        `membership probability (p_mem)` or any user choice. 
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
    if coordinate_system is None:
        warnings.warn("coordinate_system not set, defaulting to 'euclidean'")
        coordinate_system = "euclidean"

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
        _validate_coordinate_system(locals(), "coordinate_system")
        ra_source_, dec_source_, shear1_, shear2_ = arguments_consistency(
            [ra_source, dec_source, shear1, shear2],
            names=("Ra", "Dec", "Shear1", "Shear2"),
            prefix="Tangential- and Cross- shape components sources",
        )
        _validate_is_deltasigma_sigma_c(is_deltasigma, sigma_c)
        _validate_include_quadrupole_phi_major(include_quadrupole, phi_major, info_mem)
        validate_argument(locals(), "phi_major", float, none_ok=True)
        validate_argument(locals(), "info_mem", list, none_ok=True)
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
        angsep, phi = _compute_lensing_angles_flatsky(
            ra_lens, dec_lens, ra_source_, dec_source_, coordinate_system=coordinate_system
        )
    elif geometry == "curve":
        angsep, phi = _compute_lensing_angles_astropy(
            ra_lens, dec_lens, ra_source_, dec_source_, coordinate_system=coordinate_system
        )
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
        if (phi_major is None) and (info_mem is None):
            raise ValueError("Either phi_major or (ra_mem, dec_mem, weight_mem) should be provided")
        phi_major_ = phi_major
        if phi_major_ is None:
            # info_mem=[ra_mem,dec_mem,weight_mem]
            phi_major_ = calculate_major_axis(
                ra_lens, dec_lens, info_mem[0], info_mem[1], info_mem[2]
            )
        if coordinate_system == "celestial":
            phi_major_ = np.pi - phi_major_
        rotated_phi = phi - phi_major_
        rotated_shear1, rotated_shear2 = _rotate_shear(shear1_, shear2_, phi_major_)
        # Compute the quadrupole shear components
        four_theta_comp = _compute_4theta_shear(rotated_shear1, rotated_shear2, rotated_phi)
        const_comp = rotated_shear1
        # If the is_deltasigma flag is True, multiply the results by Sigma_crit.
        if sigma_c is not None:
            four_theta_comp *= _sigma_c_arr
            const_comp *= _sigma_c_arr
        return angsep, tangential_comp, cross_comp, four_theta_comp, const_comp
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


def _compute_lensing_angles_flatsky(
    ra_lens, dec_lens, ra_source_list, dec_source_list, coordinate_system="euclidean"
):
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
    coordinate_system: str, optional
        Coordinate system of the ellipticity components. Must be either 'celestial' or
        euclidean'. See https://doi.org/10.48550/arXiv.1407.7676 section 5.1 for more details.
        Default is 'euclidean'.

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
    if coordinate_system == "celestial":
        phi = np.pi - phi
    if np.any(angsep > np.pi / 180.0):
        warnings.warn("Using the flat-sky approximation with separations >1 deg may be inaccurate")
    return angsep, phi


def _compute_lensing_angles_astropy(
    ra_lens, dec_lens, ra_source_list, dec_source_list, coordinate_system="euclidean"
):
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
    coordinate_system: str, optional
        Coordinate system of the ellipticity components. Must be either 'celestial' or
        euclidean'. See https://doi.org/10.48550/arXiv.1407.7676 section 5.1 for more details.
        Default is 'euclidean'.

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
    if coordinate_system == "celestial":
        phi = np.pi - phi
    return angsep, phi


def calculate_major_axis(ra_lens_, dec_lens_, ra_mem_, dec_mem_, weight_mem_):
    r"""Compute the major axis of a given cluster from the distribution of
    its member galaxies using the position second moments.

    The computation is done according to Eq. 5-10 of Shin et al. 2018, arXiv:1705.11167

    Current implementation assumes the +RA direction is aligned with +x direction.

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    ind_bcg = (np.array(ra_mem_) == ra_lens_) * (np.array(dec_mem_) == dec_lens_)
    sk_lens = SkyCoord(ra_lens_ * u.deg, dec_lens_ * u.deg, frame="icrs")
    sk_mem = SkyCoord(
        np.array(ra_mem_)[~ind_bcg] * u.deg, np.array(dec_mem_)[~ind_bcg] * u.deg, frame="icrs"
    )
    position_angle_mem = np.array(sk_lens.position_angle(sk_mem).radian + np.pi / 2.0)
    separation_mem = np.array(sk_lens.separation(sk_mem).degree)
    x_mem = separation_mem * np.cos(position_angle_mem)
    y_mem = separation_mem * np.sin(position_angle_mem)
    distance_weight_mem = 1.0 / (separation_mem**2)
    weight_total_mem = np.array(weight_mem_) * distance_weight_mem
    sum_weight_total_mem = np.sum(weight_total_mem)
    # Calcualte second moments of the member galaxies
    ixx = np.sum(x_mem**2 * weight_total_mem) / sum_weight_total_mem
    iyy = np.sum(y_mem**2 * weight_total_mem) / sum_weight_total_mem
    ixy = np.sum(x_mem * y_mem * weight_total_mem) / sum_weight_total_mem
    # Transformation of the second moments to the direction of the major axis
    phi_major = np.arctan2(2.0 * ixy, ixx - iyy) / 2.0
    return phi_major


def _rotate_shear(shear1, shear2, phi_major):
    r"""Rotate shear components into the coordinate where +x axis is aligned with
    the cluster major axis.

    .. math::
        g_rotated = g * \exp\left(-2i\phi\right)

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    shear_vec = shear1 + shear2 * 1.0j
    rotated_shear_vec = shear_vec * np.exp(-2.0j * phi_major)
    return np.real(rotated_shear_vec), np.imag(rotated_shear_vec)


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
    coordinate_system=None,
    empty_bins_value=np.nan,
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
    coordinate_system: str, optional
        Coordinate system of the ellipticity components. Must be either 'celestial' or
        euclidean'. See https://doi.org/10.48550/arXiv.1407.7676 section 5.1 for more details.
        Default is 'euclidean'.
    empty_bins_value: float, None
        Values to be assigned to empty bins.

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
    if not coordinate_system:
        warnings.warn("coordinate_system not set, defaulting to 'euclidean'")
        coordinate_system = "euclidean"
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
        _validate_coordinate_system(locals(), "coordinate_system")
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
        meta={"bin_units": bin_units, "coordinate_system": coordinate_system},  # Add metadata
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
        profile_table = profile_table[nsrc >= 1]
    else:
        for col in (
            "radius",
            *(_col for i in range(len(components)) for _col in (f"p_{i}", f"p_{i}_err")),
        ):
            profile_table[col][nsrc < 1] = empty_bins_value
    if return_binnumber:
        return profile_table, binnumber
    return profile_table


def not_nan_average(values, axis=0, weights=None):
    """Computes averages using only not nan values

    Parameters
    ----------
    values: nd array
        Values to be averaged
    axis: int
        axis to make the average on
    weighs: nd array
        Weights, must have same shape as values

    Returns
    -------
    array
        Averaged values
    """
    _values = np.copy(values)
    _weights = np.ones(_values.shape) if weights is None else np.copy(weights)
    _values[np.isnan(values)] = 0
    _weights[np.isnan(values)] = 0
    return np.average(_values, axis=axis, weights=_weights)


def make_stacked_radial_profile(angsep, weights, components):
    """Compute stacked profile, and mean separation distances.

    Parameters
    ----------
    angsep: 2d array
        Transversal distances corresponding to each object with shape `n_obj, n_rad_bins`.
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
    # rm nan values
    staked_angsep = not_nan_average(angsep, axis=0, weights=None)
    stacked_components = [
        not_nan_average(component, axis=0, weights=weights) for component in components
    ]
    return staked_angsep, stacked_components
