"""Functions to compute polar/azimuthal averages in radial bins"""
# try: # 7481794
#     import pyccl as ccl
# except ImportError:
#     pass
import math
import warnings
import numpy as np
from .gcdata import GCData
from .utils import compute_radial_averages, make_bins, convert_units
from .galaxycluster import GalaxyCluster
from .modeling import get_critical_surface_density

# def _astropy_to_CCL_cosmo_object(astropy_cosmology_object): # 7481794
#     """Generates a ccl cosmology object from an GCR or astropy cosmology object.
#
#     Adapted from https://github.com/LSSTDESC/CLMM/blob/issue/111/model-definition/clmm/modeling.py
#     """
#     apy_cosmo = astropy_cosmology_object
#     ccl_cosmo = ccl.Cosmology(Omega_c=(apy_cosmo.Odm0-apy_cosmo.Ob0), Omega_b=apy_cosmo.Ob0,
#                               h=apy_cosmo.h, n_s=apy_cosmo.n_s, sigma8=apy_cosmo.sigma8,
#                               Omega_k=apy_cosmo.Ok0)
#
#     return ccl_cosmo


def compute_tangential_and_cross_components(cluster=None,
                  shape_component1='e1', shape_component2='e2',
                  tan_component='et', cross_component='ex',
                  ra_lens=None, dec_lens=None, ra_source=None,
                  dec_source=None, shear1=None, shear2=None, geometry='flat',
                  add_to_cluster=True,  is_deltasigma=False, cosmo=None):
    r"""Computes tangential- and cross- components for shear or ellipticity

    To do so, we need the right ascension and declination of the lens and of
    all of the sources. We also need the two shape components of all of the sources.

    These quantities can be handed to `tangential_and_cross_components` in three ways

    1. Pass in each as parameters::

        compute_tangential_and_cross_components(ra_lens, dec_lens, ra_source, dec_source, shape_component1, shape_component2)

    2. Given a `GalaxyCluster` object::

        compute_tangential_and_cross_components(cluster)

    3. As a method of `GalaxyCluster`::

        cluster.tangential_and_cross_components()

    The angular separation between the source and the lens, :math:`\theta`, and the azimuthal
    position of the source relative to the lens, :math:`\phi`, are computed within the function
    and the angular separation is returned.

    In the flat sky approximation, these angles are calculated using (_lens: lens, _source: source,
    RA is from right to left)

    .. math::

        \theta^2 = & \left(\delta_s - \delta_l\right)^2 +
        \left(\alpha_l-\alpha_s\right)^2\cos^2(\delta_l)\\
        \tan\phi = & \frac{\delta_s - \delta_l}{\left(\alpha_l - \alpha_s\right)\cos(\delta_l)}

    The tangential, :math:`g_t`, and cross, :math:`g_x`, ellipticity/shear components are calculated using the two
    ellipticity/shear components :math:`g_1` and :math:`g_2` of the source galaxies, following Eq.7 and Eq.8
    in Schrabback et al. (2018), arXiv:1611:03866
    which is consistent with arXiv:0509252

    .. math::

        g_t =& -\left( g_1\cos\left(2\phi\right) - g_2\sin\left(2\phi\right)\right)\\
        g_x =& g_1 \sin\left(2\phi\right) - g_2\cos\left(2\phi\right)

    Finally, and if requested by the user throught the `is_deltasigma` flag, an estimate of the excess surface density :math:`\widehat{\Delta\Sigma}` is obtained from

    .. math::

        \widehat{\Delta\Sigma_{t,x}} = g_{t,x} \times \Sigma_c(cosmo, z_L, z_{\rm src})

    where :math:`\Sigma_c` is the critical surface density that depends on the cosmology and on the lens and source redshifts. If :math:`g_{t,x}` correspond to the shear, the above expression is an accurate. However, if :math:`g_{t,x}` correspond to ellipticities or reduced shear, this expression only gives an estimate :math:`\widehat{\Delta\Sigma_{t,x}}`, valid only in the weal lensing regime.

    Parameters
    ----------
    cluster: GalaxyCluster, optional
        Instance of `GalaxyCluster()` and must contain right ascension and declinations of both
        the lens and sources and the two shear components all of the sources. If this
        object is specified, right ascension, declination, and shear or ellipticity inputs are ignored.
    shape_component1: string, optional
        Name of the column in the `galcat` astropy table of the cluster object that contains
        the shape or shear measurement along the first axis. Default: `e1`
    shape_component1: string, optional
        Name of the column in the `galcat` astropy table of the cluster object that contains
        the shape or shear measurement along the second axis. Default: `e2`
    tan_component: string, optional
        Name of the column to be added to the `galcat` astropy table that will contain the
        tangential component computed from columns `shape_component1` and `shape_component2`.
        Default: `et`
    cross_component: string, optional
        Name of the column to be added to the `galcat` astropy table that will contain the
        cross component computed from columns `shape_component1` and `shape_component2`.
        Default: `ex`
    ra_lens: float, optional
        Right ascension of the lensing cluster
    dec_lens: float, optional
        Declination of the lensing cluster
    ra_source: array, optional
        Right ascensions of each source galaxy
    dec_source: array, optional
        Declinations of each source galaxy
    shear1: array, optional
        The measured shear (or reduced shear or ellipticity) of the source galaxies
    shear2: array, optional
        The measured shear (or reduced shear or ellipticity) of the source galaxies
    geometry: str, optional
        Sky geometry to compute angular separation.
        Flat is currently the only supported option.
    add_to_cluster: bool
        If `True` and a cluster was input, add the computed shears to the `GalaxyCluster` object
    is_deltasigma: bool
        If `True`, the tangential and cross components returned are multiplied by Sigma_crit. Results in units of :math:`h\ M_\odot\ pc^{-2}`
    cosmo: astropy cosmology object
        Specifying a cosmology is required if `is_deltasigma` is True

    Returns
    -------
    angsep: array_like
        Angular separation between lens and each source galaxy in radians
    tangential_component: array_like
        Tangential shear (or assimilated quantity) for each source galaxy
    cross_component: array_like
        Cross shear (or assimilated quantity) for each source galaxy
    """

    if cluster is None:
        add_to_cluster=False
    
    if cluster is not None:
        required_cols = ['ra', 'dec', shape_component1, shape_component2]
        if not all([t_ in cluster.galcat.columns for t_ in required_cols]):
            raise TypeError('GalaxyCluster\'s galaxy catalog missing required columns.' +\
                            'Do you mean to first convert column names?')

        ra_lens, dec_lens = cluster.ra, cluster.dec
        ra_source, dec_source = cluster.galcat['ra'], cluster.galcat['dec']
        shear1, shear2 = cluster.galcat[shape_component1], cluster.galcat[shape_component2]


    # If a cluster object is not specified, we require all of these inputs
    elif any(t_ is None for t_ in (ra_lens, dec_lens, ra_source, dec_source,
                                   shear1, shear2)):
        raise TypeError('To compute tangential- and cross- shape components, please provide a GalaxyCluster object or ra and dec' +\
                        'of lens and sources and shears or ellipticities of the sources.')

    # If there is only 1 source, make sure everything is a scalar
    if all(not hasattr(t_, '__len__') for t_ in [ra_source, dec_source, shear1, shear2]):
        pass
    # Otherwise, check that the length of all of the inputs match
    elif not all(len(t_) == len(ra_source) for t_ in [dec_source, shear1, shear2]):
        raise TypeError('To compute the tangential- and cross- shape components you should supply the same number of source' +\
                        'positions and shear or ellipticity.')

    # Compute the lensing angles
    if geometry == 'flat':
        angsep, phi = _compute_lensing_angles_flatsky(ra_lens, dec_lens, ra_source,
                                                      dec_source)
    else:
        raise NotImplementedError(f"Sky geometry {geometry} is not currently supported")

    # Compute the tangential and cross shears
    tangential_comp = _compute_tangential_shear(shear1, shear2, phi)
    cross_comp = _compute_cross_shear(shear1, shear2, phi)

    # If the is_deltasigma flag is True, multiply the results by Sigma_crit.
    # Need to verify that cosmology and redshifts are provided
    if is_deltasigma:
        if cluster is not None:
            if 'z' not in cluster.galcat.columns:
                raise TypeError('GalaxyCluster\'s galaxy catalog missing the redshift column.' +\
                                'Cannot compute DeltaSigma')
            z_lens = cluster.z
            z_source = cluster.galcat['z']
            if cosmo is None:
                raise TypeError('To compute DeltaSigma, please provide a cosmology')

        elif any(t_ is None for t_ in (z_lens,z_source,cosmo)):
            raise TypeError('To compute DeltaSigma, please provide a i) cosmology, ii) redshift of lens and sources')

        Sigma_c = get_critical_surface_density(cosmo, z_lens, z_source)
        tangential_comp *= Sigma_c
        cross_comp *= Sigma_c

    if add_to_cluster:
        cluster.galcat['theta'] = angsep
        cluster.galcat[tan_component] = tangential_comp
        cluster.galcat[cross_component] = cross_comp
        if is_deltasigma:
            # also save Sigma_c as new column as it is often
            # used in the weighing scheme when stacking data
            cluster.galcat['sigma_c'] = Sigma_c

    return np.array(angsep), np.array(tangential_comp), np.array(cross_comp)


def _compute_lensing_angles_flatsky(ra_lens, dec_lens, ra_source_list, dec_source_list):
    r"""Compute the angular separation between the lens and the source and the azimuthal
    angle from the lens to the source in radians.

    In the flat sky approximation, these angles are calculated using
    .. math::
        \theta = \sqrt{\left(\delta_s - \delta_l\right)^2 +
        \left(\alpha_l-\alpha_s\right)^2\cos^2(\delta_l)}

        \tan\phi = \frac{\delta_s - \delta_l}{\left(\alpha_l - \alpha_s\right)\cos(\delta_l)}

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
    r2pi = lambda x: x - np.round(x/(2.0*math.pi))*2.0*math.pi

    deltax = r2pi (np.radians(ra_source_list - ra_lens)) * math.cos(math.radians(dec_lens))
    deltay = np.radians(dec_source_list - dec_lens)

    # Ensure that abs(delta ra) < pi
    #deltax[deltax >= np.pi] = deltax[deltax >= np.pi] - 2.*np.pi
    #deltax[deltax < -np.pi] = deltax[deltax < -np.pi] + 2.*np.pi

    angsep = np.sqrt(deltax**2 + deltay**2)
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
        g_t = -\left( g_1\cos\left(2\phi\right) - g_2\sin\left(2\phi\right)\right)

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    return - (shear1 * np.cos(2.*phi) + shear2 * np.sin(2.*phi))


def _compute_cross_shear(shear1, shear2, phi):
    r"""Compute the cross shear given the two shears and azimuthal position for a single
    source of list of sources.

    We compute the cross shear following Eq. 8 of Schrabback et al. 2018, arXiv:1611:03866
    also checked arxiv 0509252
    .. math::
        g_x = g_1 \sin\left(2\phi\right) - g_2\cos\left(2\phi\right)

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    return shear1 * np.sin(2.*phi) - shear2 * np.cos(2.*phi)


def make_binned_profile(cluster,
                       angsep_units, bin_units, bins=10, cosmo=None,
                       tan_component_in='et', cross_component_in='ex',
                       tan_component_out='gt', cross_component_out='gx', table_name='profile',
                       add_to_cluster=True, include_empty_bins=False, gal_ids_in_bins=False):
    r"""Compute the shear or ellipticity profile of the cluster

    We assume that the cluster object contains information on the cross and
    tangential shears or ellipticities and angular separation of the source galaxies

    This function can be called in two ways using an instance of GalaxyCluster

    1. Pass an instance of GalaxyCluster into the function::

        make_shear_profile(cluster, 'radians', 'radians')

    2. Call it as a method of a GalaxyCluster instance::

        cluster.make_shear_profile('radians', 'radians')

    Parameters
    ----------
    cluster : GalaxyCluster
        Instance of GalaxyCluster that contains the cross and tangential shears or ellipticities of
        each source galaxy in its `galcat`
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
    cosmo: dict, optional
        Cosmology parameters to convert angular separations to physical distances
    tan_component_in: string, optional
        Name of the column in the `galcat` astropy table of the `cluster` object that contains
        the tangential component to be binned. Default: 'et'
    tan_component_out: string, optional
        Name of the column in the `profile` table of the `cluster` object that will contain
        the binned profile of the tangential component. Default: 'gx'
    cross_component_in: string, optional
        Name of the column in the `galcat` astropy table of the `cluster` object that contains
        the cross component to be binned. Default: 'ex'
    cross_component_out: string, optional
        Name of the column in the `profile` table of the `cluster` object that will contain
        the  binned profile of the cross component. Default: 'gx'
    add_to_cluster: bool, optional
        Attach the profile to the cluster object as `cluster.profile`
    include_empty_bins: bool, optional
        Also include empty bins in the returned table
    gal_ids_in_bins: bool, optional
        Also include the list of galaxies ID belonging to each bin in the returned table

    Returns
    -------
    profile : GCData
        Output table containing the radius grid points, the tangential and cross shear profiles
        on that grid, and the errors in the two shear profiles. The errors are defined as the
        standard errors in each bin.

    Notes
    -----
    This is an example of a place where the cosmology-dependence can be sequestered to another module.
    """
    if not all([t_ in cluster.galcat.columns for t_ in (tan_component_in, cross_component_in, 'theta')]):
        raise TypeError('Shear or ellipticity information is missing!  Galaxy catalog must have tangential' +\
                        'and cross shears (gt,gx) or ellipticities (et,ex). Run compute_tangential_and_cross_components first.')
    if 'z' not in cluster.galcat.columns:
        raise TypeError('Missing galaxy redshifts!')

    # Check to see if we need to do a unit conversion
    if angsep_units is not bin_units:
        source_seps = convert_units(cluster.galcat['theta'], angsep_units, bin_units,
                                    redshift=cluster.z, cosmo=cosmo)
    else:
        source_seps = cluster.galcat['theta']

    # Make bins if they are not provided
    if not hasattr(bins, '__len__'):
        bins = make_bins(np.min(source_seps), np.max(source_seps), bins)

    # Compute the binned averages and associated errors
    r_avg, gt_avg, gt_err, nsrc, binnumber = compute_radial_averages(
        source_seps, cluster.galcat[tan_component_in].data, xbins=bins, error_model='std/sqrt_n')
    r_avg, gx_avg, gx_err, _, _ = compute_radial_averages(
        source_seps, cluster.galcat[cross_component_in].data, xbins=bins, error_model='std/sqrt_n')
    r_avg, z_avg, z_err, _, _ = compute_radial_averages(
        source_seps, cluster.galcat['z'].data, xbins=bins, error_model='std/sqrt_n')

    # Make out table
    profile_table = GCData([bins[:-1], r_avg, bins[1:], gt_avg, gt_err, gx_avg, gx_err,
                            z_avg, z_err, nsrc],
                            names=('radius_min', 'radius', 'radius_max',
                                   tan_component_out, tan_component_out+'_err',
                                   cross_component_out, cross_component_out+'_err',
                                   'z', 'z_err', 'n_src'),
                            meta={'cosmo':cosmo, 'bin_units':bin_units}, # Add metadata
                            )
    # add galaxy IDs
    if gal_ids_in_bins:
        if 'id' not in cluster.galcat.columns:
            raise TypeError('Missing galaxy IDs!')
        profile_table['gal_id'] = [list(cluster.galcat['id'][binnumber==i+1])
                                    for i in np.arange(len(r_avg))]

    # return empty bins?
    if not include_empty_bins:
        profile_table = profile_table[profile_table['n_src'] > 1]

    if add_to_cluster:
        setattr(cluster, table_name, profile_table)

    return profile_table


# Monkey patch functions onto Galaxy Cluster object
GalaxyCluster.compute_tangential_and_cross_components = compute_tangential_and_cross_components
GalaxyCluster.make_binned_profile = make_binned_profile
