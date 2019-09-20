"""@file polaraveraging.py
Functions to compute polar/azimuthal averages in radial bins
"""

# try: # 7481794
#     import pyccl as ccl
# except ImportError:
#     pass
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy import units as u


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


def compute_shear(cluster=None, ra_lens=None, dec_lens=None, ra_source_list=None,
                  dec_source_list=None, shear1=None, shear2=None, geometry='flat',
                  add_to_cluster=True):
    r"""Computes tangential shear, cross shear, and angular separation

    To compute the shear, we need the right ascension and declination of the len and of
    all of the sources. We also need the two shears or ellipticities of all of the sources.

    These quantities can be handed to `compute_shear` in three ways
    1. Pass in each as parameters
        `compute_shear(ra_lens, dec_lens, ra_source_list, dec_source_list, shear1, shear2)`
    2. Given a `GalaxyCluster()` object
        `compute_shear(cluster)`
    3. As a method of `GalaxyCluster()`
        `cluster.compute_shear()`

    The angular separation between the source and the lens, :math:`\theta`, and the azimuthal
    position of the source relative to the lens, :math:`\phi`, are computed within and the
    angular separation is returned.

    In the flat sky approximation, these angles are calculated using
    .. math::
        \theta = \sqrt{\left(\delta_s - \delta_l\right)^2 +
        \left(\alpha_l-\alpha_s\right)^2\cos^2(\delta_l)}

        \tan\phi = \frac{\delta_s - \delta_l}{\left(\alpha_l - \alpha_s\right)\cos(\delta_l)}

    The tangential, :math:`g_t`, and cross, :math:`g_x`, shears are calculated using the two
    measured ellipticities or shears of the source galaxies following
    Schrabback et al. 2018, arXiv:1611:03866
    .. math::
        g_t = -\left( g_1\cos\left(2\phi\right) + g_2\sin\left(2\phi\right)\right)

        g_x = -g_1 \sin\left(2\phi\right) + g_2\cos\left(2\phi\right)


    Parameters
    ----------
    cluster: GalaxyCluster, optional
        Instance of `GalaxyCluster()` and must contain right ascension and declinations of both
        the lens and sources and the two shears or ellipticities of all of the sources. If this
        object is specified, right ascension, declination, and shear inputs are ignored.
    ra_lens, dec_lens: float, optional
        Right ascension and declination of the lensing cluster
    ra_source_list, dec_source_list: array_like, optional
        Right ascensions and declinations of each source galaxy
    shear1, shear2: array_like, optional
        The measured shears or ellipticities of the source galaxies
    geometry: str, optional
        Sky geometry to compute angular separation.
        Flat is currently the only supported option.
    add_to_cluster: bool
        If `True` and a cluster was input, add the computed shears to the `GalaxyCluster` object

    Returns
    -------
    angsep: array_like
        Angular separation between lens and each source galaxy in radians
    tangential_shear: array_like
        Tangential shear for each source galaxy
    cross_shear: array_like
        Cross shear for each source galaxy
    """
    if cluster is not None:
        required_cols = ['ra', 'dec', 'e1', 'e2']
        if not all([t_ in cluster.galcat.columns for t_ in required_cols]):
            raise TypeError('GalaxyCluster\'s galaxy catalog missing required columns.')

        ra_lens, dec_lens = cluster.ra, cluster.dec
        ra_source_list, dec_source_list = cluster.galcat['ra'], cluster.galcat['dec']
        shear1, shear2 = cluster.galcat['e1'], cluster.galcat['e2']

    # If a cluster object is not specified, we require all of these inputs
    elif any(t_ is None for t_ in (ra_lens, dec_lens, ra_source_list, dec_source_list,
                                   shear1, shear2)):
        raise TypeError('To compute shear, please provide a GalaxyCluster object or ra and dec of\
                         lens and sources and both shears or ellipticities of the sources.')

    # If there is only 1 source, make sure everything is a scalar
    if all(not hasattr(t_, '__len__') for t_ in [ra_source_list, dec_source_list, shear1, shear2]):
        pass
    # Otherwise, check that the length of all of the inputs match
    elif not all(len(t_) == len(ra_source_list) for t_ in [dec_source_list, shear1, shear2]):
        raise TypeError("To compute the shear you should supply the same number of source\
                         positions and shear.")

    # Compute the lensing angles
    if geometry == 'flat':
        angsep, phi = _compute_lensing_angles_flatsky(ra_lens, dec_lens, ra_source_list,
                                                      dec_source_list)
    else:
        raise NotImplementedError("Sky geometry {} is not currently supported".format(geometry))

    # Compute the tangential and cross shears
    tangential_shear = _compute_tangential_shear(shear1, shear2, phi)
    cross_shear = _compute_cross_shear(shear1, shear2, phi)

    if add_to_cluster:
        cluster.galcat['theta'] = angsep
        cluster.galcat['gt'] = tangential_shear
        cluster.galcat['gx'] = cross_shear

    return angsep, tangential_shear, cross_shear


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
        raise ValueError("ra = {} of lens if out of domain".format(ra_lens))
    if not -90. <= dec_lens <= 90.:
        raise ValueError("dec = {} of lens if out of domain".format(dec_lens))
    if not all(-360. <= x_ <= 360. for x_ in ra_source_list):
        raise ValueError("Cluster has an invalid ra in source catalog")
    if not all(-90. <= x_ <= 90 for x_ in dec_source_list):
        raise ValueError("Cluster has an invalid dec in the source catalog")

    deltax = np.radians(ra_source_list - ra_lens) * math.cos(math.radians(dec_lens))
    deltay = np.radians(dec_source_list - dec_lens)

    # Ensure that abs(delta ra) < pi
    deltax[deltax >= np.pi] = deltax[deltax >= np.pi] - 2.*np.pi
    deltax[deltax < -np.pi] = deltax[deltax < -np.pi] + 2.*np.pi

    angsep = np.sqrt(deltax**2 + deltay**2)
    phi = np.arctan2(deltay, -deltax)

    if np.any(angsep > np.pi/180.):
        warnings.warn("Using the flat-sky approximation with separations >1 deg may be inaccurate")

    return angsep, phi


def _compute_tangential_shear(shear1, shear2, phi):
    r"""Compute the tangential shear given the two shears and azimuthal position for
    a single source or list of sources.

    We compute the cross shear following Schrabback et al. 2018, arXiv:1611:03866
    .. math::
        g_t = -\left( g_1\cos\left(2\phi\right) + g_2\sin\left(2\phi\right)\right)

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    return -(shear1*np.cos(2.*phi) + shear2*np.sin(2.*phi))


def _compute_cross_shear(shear1, shear2, phi):
    r"""Compute the cross shear given the two shears and azimuthal position for a single
    source of list of sources.

    We compute the cross shear following Schrabback et al. 2018, arXiv:1611:03866
    .. math::
        g_x = -g_1 \sin\left(2\phi\right) + g_2\cos\left(2\phi\right)

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    return -shear1*np.sin(2.*phi) + shear2*np.cos(2.*phi)











def make_shear_profile(cluster, radius_unit, bins=None, cosmo=None, add_to_cluster=True):
    """Computes shear profile of the cluster

    Parameters
    ----------
    cluster: GalaxyCluster object
        GalaxyCluster object with galaxies
    radius_unit:
        Radial unit of the profile, one of
        ["rad", deg", "arcmin", "arcsec", kpc", "Mpc"]
    bins: array_like, float
        User defined n_bins + 1 dimensional array of bins, if 'None',
        the default is 10 equally spaced radial bins
    cosmo:
        Cosmology object
    add_to_cluster: bool
        Adds the outputs to cluster.profile

    Returns
    -------
    profile_table: astropy Table
        Table with r_profile, gt profile (and error) and
        gx profile (and error)

    Note
    ----
    Currently, the radial_units are not saved in the profile_table.
    We have to add it somehow.
    """
    if not ('gt' in cluster.galcat.columns and 'gx' in cluster.galcat.columns
            and 'theta' in cluster.galcat.columns):
        raise TypeError('shear information is missing in galaxy must have tangential and cross\
                         shears (gt,gx). Run compute_shear first!')
    radial_values = _theta_units_conversion(cluster.galcat['theta'], radius_unit, z_cl=cluster.z,
                                            cosmo=cosmo)
    r_avg, gt_avg, gt_std = _compute_radial_averages(radial_values, cluster.galcat['gt'].data,
                                                     bins=bins)
    r_avg, gx_avg, gx_std = _compute_radial_averages(radial_values, cluster.galcat['gx'].data,
                                                     bins=bins)
    profile_table = Table([r_avg, gt_avg, gt_std, gx_avg, gx_avg],
                          names=('radius', 'gt', 'gt_err', 'gx', 'gx_err'))
    if add_to_cluster:
        cluster.profile = profile_table
        cluster.profile_radius_unit = radius_unit
    return profile_table



def _theta_units_conversion(theta, unit, z_cl=None, cosmo=None):
    """Converts theta from radian to whatever units specified in units

    Parameters
    ----------
    theta : float
        We assume the input unit is radian. Theta is angular seperation between source galaxies and
        the cluster center in 2D image.
    unit : string
        Output unit you would like to convert to
        Supported units are : "rad", deg", "arcmin", "arcsec", kpc", "Mpc".
    z_cl :  float
	Cluster redshift, needed to convert angle to physical distances.
    cosmo : object
	Cosmology object to convert angle to physical distances. Can be from astropy or ccl.

    Returns
    -------
    radius : float
        Theta in the converted unit you want to.

    Notes
    -----
    This stuff below is left over, replace it above and remove this.
    units: one of ["rad", deg", "arcmin", "arcsec", kpc", "Mpc"]
    cosmo : cosmo object
    z_cl : cluster redshift
    """
    theta = theta * u.rad

    units_bank = {
        "rad": u.rad,
        "deg": u.deg,
        "arcmin": u.arcmin,
        "arcsec": u.arcsec,
        "kpc": u.kpc,
        "Mpc": u.Mpc,
        }

    if unit in units_bank:
        unit_ = units_bank[unit]
        if unit[1:] == "pc":
            if isinstance(cosmo, astropy.cosmology.core.FlatLambdaCDM): # astropy cosmology type
                Da = cosmo.angular_diameter_distance(z_cl).to(unit_).value
            # elif isinstance(cosmo, ccl.core.Cosmology): # astropy cosmology type
            #     Da = ccl.comoving_angular_distance(cosmo, 1/(1+z_cl)) / (1+z_cl) * u.Mpc.to(unit_)
            else:
                raise ValueError("cosmo object (%s) not an astropy or ccl cosmology"%str(cosmo))
            return theta.value * Da
        else:
            return theta.to(unit_).value
    else:
        raise ValueError("unit (%s) not in %s"%(unit, str(units_bank.keys())))
    if z_cl is None:
        raise ValueError("To compute physical units, z_cl must not be None")

# def make_bins(rmin, rmax, n_bins=10, log_bins=False):
#     """Define equal sized bins with an array of n_bins+1 bin edges
#
#     Parameters
#     ----------
#     rmin, rmax,: float
#         minimum and and maximum range of data (any units)
#     n_bins: float
#         number of bins you want to create
#     log_bins: bool
#         set to 'True' equal sized bins in log space
#
#     Returns
#     -------
#     binedges: array_like, float
#         n_bins+1 dimensional array that defines bin edges
#     """
#     if rmax < rmin:
#         raise ValueError("rmax should be larger than rmin")
#     if n_bins <= 0:
#         raise ValueError("n_bins must be > 0")
#     if type(log_bins) != bool:
#         raise TypeError("log_bins must be type bool")
#     if type(n_bins) != int:
#         raise TypeError("You need an integer number of bins")
#
#     if log_bins == True:
#         rmin = np.log(rmin)
#         rmax = np.log(rmax)
#         logbinedges = np.linspace(rmin, rmax, n_bins+1, endpoint=True)
#         binedges = np.exp(logbinedges)
#     else:
#         binedges = np.linspace(rmin, rmax, n_bins+1, endpoint=True)
#
#     return binedges


def _compute_radial_averages(radius, g, bins=None):
    """Returns astropy table containing shear profile of either tangential or cross shear

    Parameters
    ----------
    radius: array_like, float
        Distance (physical or angular) between source galaxy to cluster center
    g: array_like, float
        Either tangential or cross shear (g_t or g_x)
    bins: array_like, float
        User defined n_bins + 1 dimensional array of bins
        If 'None', the default is 10 equally spaced radial bins

    Returns
    -------
    r_profile: array_like, float
        Centers of radial bins
    g_profile: array_like, float
        Average shears per bin
    gerr_profile: array_like, float
        Standard deviation of shear per bin
    """
    if not isinstance(radius, (np.ndarray)):
        raise TypeError("radius must be an array")
    if not isinstance(g, (np.ndarray)):
        raise TypeError("g must be an array")
    if len(radius) != len(g):
        raise TypeError("radius and g must be arrays of the same length")
    if np.any(bins) == None:
        nbins = 10
        bins = np.linspace(np.min(radius), np.max(radius), nbins)

    g_profile = np.zeros(len(bins) - 1)
    gerr_profile = np.zeros(len(bins) - 1)
    r_profile = np.zeros(len(bins) - 1)

    if np.amax(radius) > np.amax(bins):
        warnings.warn("Maximum radius is not within range of bins")
    if np.amin(radius) < np.amin(bins):
        warnings.warn("Minimum radius is not within the range of bins")

    for i in range(len(bins)-1):
        cond = (radius >= bins[i]) & (radius < bins[i+1])
        index = np.where(cond)[0]
        r_profile[i] = np.average(radius[index])
        g_profile[i] = np.average(g[index])
        if len(index) != 0:
            gerr_profile[i] = np.std(g[index]) / np.sqrt(float(len(index)))
        else:
            gerr_profile[i] = np.nan

    return r_profile, g_profile, gerr_profile


def plot_profiles(cluster, r_units=None):
    """Plot shear profiles for validation

    Parameters
    ----------
    cluster: GalaxyCluster object
        GalaxyCluster object with galaxies
    """
    prof = cluster.profile
    if r_units is not None:
        if cluster.profile['radius'].unit is not None:
            warning.warn(('r_units provided (%s) differ from r_units in galcat table (%s) using\
                            user defined')%(r_units, cluster.profile['radius'].unit))
        else:
            r_units = cluster.profile['radius'].unit
    return _plot_profiles(*[cluster.profile[c] for c in ('radius', 'gt', 'gt_err', 'gx', 'gx_err')],
                          r_unit=cluster.profile_radius_unit)


def _plot_profiles(r, gt, gterr, gx=None, gxerr=None, r_unit=""):
    """Plot shear profiles for validation

    Parameters
    ----------
    r: array_like, float
        radius
    gt: array_like, float
        tangential shear
    gterr: array_like, float
        error on tangential shear
    gx: array_like, float
        cross shear
    gxerr: array_like, float
        error on cross shear
    r_unit: string
	unit of radius

    """
    fig, ax = plt.subplots()
    ax.plot(r, gt, 'bo-', label="tangential shear")
    ax.errorbar(r, gt, gterr, label=None)

    try:
        plt.plot(r, gx, 'ro-', label="cross shear")
        plt.errorbar(r, gx, gxerr, label=None)
    except:
        pass

    ax.legend()
    if r_unit is not None:
        ax.set_xlabel("r [%s]"%r_unit)
    else:
        ax.set_xlabel("r")

    ax.set_ylabel('$\\gamma$')

    return(fig, ax)

# Monkey patch functions onto Galaxy Cluster object
from .galaxycluster import GalaxyCluster
GalaxyCluster.compute_shear = compute_shear
