"""Functions to compute polar/azimuthal averages in radial bins"""
# try: # 7481794
#     import pyccl as ccl
# except ImportError:
#     pass
import math
import warnings
import numpy as np
from .. gcdata import GCData
from . .utils import compute_radial_averages, make_bins, convert_units, arguments_consistency
from .. theory import get_critical_surface_density


def compute_tangential_and_cross_components(
                ra_lens, dec_lens, ra_source, dec_source,
                shear1, shear2, geometry='flat',
                is_deltasigma=False, cosmo=None,
                z_lens=None, z_source=None, sigma_c=None):
    r"""Computes tangential- and cross- components for shear or ellipticity

    To do so, we need the right ascension and declination of the lens and of
    all of the sources. We also need the two shape components of all of the sources.

    These quantities can be handed to `tangential_and_cross_components` in two ways

    1. Pass in each as parameters::

        compute_tangential_and_cross_components(ra_lens, dec_lens, ra_source, dec_source, shape_component1, shape_component2)

    2. As a method of `GalaxyCluster`::

        cluster.tangential_and_cross_components()

    The angular separation between the source and the lens, :math:`\theta`, and the azimuthal
    position of the source relative to the lens, :math:`\phi`, are computed within the function
    and the angular separation is returned.

    In the flat sky approximation, these angles are calculated using (_lens: lens, _source: source,
    RA is from right to left)

    .. math::

        \theta^2 = & \left(\delta_s-\delta_l\right)^2+
        \left(\alpha_l-\alpha_s\right)^2\cos^2(\delta_l)\\
        \tan\phi = & \frac{\delta_s-\delta_l}{\left(\alpha_l-\alpha_s\right)\cos(\delta_l)}

    The tangential, :math:`g_t`, and cross, :math:`g_x`, ellipticity/shear components are calculated using the two
    ellipticity/shear components :math:`g_1` and :math:`g_2` of the source galaxies, following Eq.7 and Eq.8
    in Schrabback et al. (2018), arXiv:1611:03866
    which is consistent with arXiv:0509252

    .. math::

        g_t =& -\left( g_1\cos\left(2\phi\right)-g_2\sin\left(2\phi\right)\right)\\
        g_x =& g_1 \sin\left(2\phi\right)-g_2\cos\left(2\phi\right)

    Finally, and if requested by the user throught the `is_deltasigma` flag, an estimate of the excess surface density :math:`\widehat{\Delta\Sigma}` is obtained from

    .. math::

        \widehat{\Delta\Sigma_{t,x}} = g_{t,x} \times \Sigma_c(cosmo, z_L, z_{\rm src})

    where :math:`\Sigma_c` is the critical surface density that depends on the cosmology and on the lens and source redshifts. If :math:`g_{t,x}` correspond to the shear, the above expression is an accurate. However, if :math:`g_{t,x}` correspond to ellipticities or reduced shear, this expression only gives an estimate :math:`\widehat{\Delta\Sigma_{t,x}}`, valid only in the weal lensing regime.

    Parameters
    ----------
    ra_lens: float
        Right ascension of the lensing cluster
    dec_lens: float
        Declination of the lensing cluster
    ra_source: array
        Right ascensions of each source galaxy
    dec_source: array
        Declinations of each source galaxy
    shear1: array
        The measured shear (or reduced shear or ellipticity) of the source galaxies
    shear2: array
        The measured shear (or reduced shear or ellipticity) of the source galaxies
    geometry: str, optional
        Sky geometry to compute angular separation.
        Flat is currently the only supported option.
    is_deltasigma: bool
        If `True`, the tangential and cross components returned are multiplied by Sigma_crit. Results in units of :math:`M_\odot\ Mpc^{-2}`
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

    Returns
    -------
    angsep: array_like
        Angular separation between lens and each source galaxy in radians
    tangential_component: array_like
        Tangential shear (or assimilated quantity) for each source galaxy
    cross_component: array_like
        Cross shear (or assimilated quantity) for each source galaxy
    """
    # Note: we make these quantities to be np.array so that a name is not passed from astropy columns
    ra_source_, dec_source_, shear1_, shear2_ = arguments_consistency([ra_source, dec_source, shear1, shear2],
                                                                names=('Ra', 'Dec', 'Shear1', 'Shear2'),
                                                                prefix='Tangential- and Cross- shape components sources')
    # Compute the lensing angles
    if geometry == 'flat':
        angsep, phi = _compute_lensing_angles_flatsky(ra_lens, dec_lens,
                                                      ra_source_, dec_source_)
    else:
        raise NotImplementedError(f"Sky geometry {geometry} is not currently supported")
    # Compute the tangential and cross shears
    tangential_comp = _compute_tangential_shear(shear1_, shear2_, phi)
    cross_comp = _compute_cross_shear(shear1_, shear2_, phi)
    # If the is_deltasigma flag is True, multiply the results by Sigma_crit.
    if is_deltasigma:
        if sigma_c is None:
            # Need to verify that cosmology and redshifts are provided
            if any(t_ is None for t_ in (z_lens, z_source, cosmo)):
                raise TypeError('To compute DeltaSigma, please provide a i) cosmology, ii) redshift of lens and sources')
            sigma_c = get_critical_surface_density(cosmo, z_lens, z_source)
        tangential_comp *= sigma_c
        cross_comp *= sigma_c
    return angsep, tangential_comp, cross_comp



def _compute_lensing_angles_flatsky(ra_lens, dec_lens, ra_source_list, dec_source_list):
    r"""Compute the angular separation between the lens and the source and the azimuthal
    angle from the lens to the source in radians.

    In the flat sky approximation, these angles are calculated using
    .. math::
        \theta = \sqrt{\left(\delta_s-\delta_l\right)^2+
        \left(\alpha_l-\alpha_s\right)^2\cos^2(\delta_l)}

        \tan\phi = \frac{\delta_s-\delta_l}{\left(\alpha_l-\alpha_s\right)\cos(\delta_l)}

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    if not -360. <= ra_lens <= 360.:
        raise ValueError(f"ra = {ra_lens} of lens if out of domain")
    if not -90. <= dec_lens <= 90.:
        raise ValueError(f"dec = {dec_lens} of lens if out of domain")
    if not all(-360. <= x_ <= 360. for x_ in ra_source_list):
        raise ValueError("Cluster has an invalid ra in source catalog")
    if not all(-90. <= x_ <= 90 for x_ in dec_source_list):
        raise ValueError("Cluster has an invalid dec in the source catalog")
    # Put angles between -pi and pi
    r2pi = lambda x: x-np.round(x/(2.0*math.pi))*2.0*math.pi
    deltax = r2pi(np.radians(ra_source_list-ra_lens))*math.cos(math.radians(dec_lens))
    deltay = np.radians(dec_source_list-dec_lens)
    # Ensure that abs(delta ra) < pi
    #deltax[deltax >= np.pi] = deltax[deltax >= np.pi]-2.*np.pi
    #deltax[deltax < -np.pi] = deltax[deltax < -np.pi]+2.*np.pi
    angsep = np.sqrt(deltax**2+deltay**2)
    phi = np.arctan2(deltay, -deltax)
    # Forcing phi to be zero everytime angsep is zero. This is necessary due to arctan2 features (it returns ).
    phi[angsep==0.0] = 0.0
    if np.any(angsep > np.pi/180.):
        warnings.warn("Using the flat-sky approximation with separations >1 deg may be inaccurate")
    return angsep, phi


def _compute_tangential_shear(shear1, shear2, phi):
    r"""Compute the tangential shear given the two shears and azimuthal positions for
    a single source or list of sources.

    We compute the tangential shear following Eq. 7 of Schrabback et al. 2018, arXiv:1611:03866
    .. math::
        g_t = -\left( g_1\cos\left(2\phi\right)-g_2\sin\left(2\phi\right)\right)

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
                        bins=10, include_empty_bins=False,
                        return_binnumber=False,
                        cosmo=None, z_lens=None):
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
        Units to use for the radial bins of the shear profile
        Allowed Options = ["radians", deg", "arcmin", "arcsec", kpc", "Mpc"]
    bins : array_like, optional
        User defined bins to use for the shear profile. If a list is provided, use that as
        the bin edges. If a scalar is provided, create that many equally spaced bins between
        the minimum and maximum angular separations in bin_units. If nothing is provided,
        default to 10 equally spaced bins.
    include_empty_bins: bool, optional
        Also include empty bins in the returned table
    gal_ids_in_bins: bool, optional
        Also include the list of galaxies ID belonging to each bin in the returned table
    return_binnumber: bool, optional
        Also returns the indices of the bins for each object
    cosmo: dict, optional
        Cosmology parameters to convert angular separations to physical distances
    z_lens: array, optional
        Redshift of the lens

    Returns
    -------
    profile : GCData
        Output table containing the radius grid points, the profile of the components `p_i`, errors `p_i_err` and number of sources.
        The errors are defined as the standard errors in each bin.
    binnumber: 1-D ndarray of ints, optional
        Indices of the bins (corresponding to `xbins`) in which each value
        of `xvals` belongs.  Same length as `yvals`.  A binnumber of `i` means the
        corresponding value is between (xbins[i-1], xbins[i]).

    Notes
    -----
    This is an example of a place where the cosmology-dependence can be sequestered to another module.
    """
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
                           meta={'bin_units' : bin_units}, # Add metadata
                          )
    # Compute the binned averages and associated errors
    for i, component in enumerate(components):
        r_avg, comp_avg, comp_err, nsrc, binnumber = compute_radial_averages(
            source_seps, component, xbins=bins, error_model='std/sqrt_n')
        profile_table[f'p_{i}'] = comp_avg
        profile_table[f'p_{i}_err'] = comp_err
    profile_table['radius'] = r_avg
    profile_table['n_src'] = nsrc
    # return empty bins?
    if not include_empty_bins:
        profile_table = profile_table[nsrc>1]
    if return_binnumber:
        return profile_table, binnumber
    return profile_table
