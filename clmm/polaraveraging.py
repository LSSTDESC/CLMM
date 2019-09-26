"""Functions to compute polar/azimuthal averages in radial bins"""
# try: # 7481794
#     import pyccl as ccl
# except ImportError:
#     pass
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
#123123 
#import clmm.utils
#123123 

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

    To compute the shear, we need the right ascension and declination of the lens and of
    all of the sources. We also need the two shear components of all of the sources.

    These quantities can be handed to `compute_shear` in three ways

    1. Pass in each as parameters::

        compute_shear(ra_lens, dec_lens, ra_source_list, dec_source_list, shear1, shear2)

    2. Given a `GalaxyCluster` object::

        compute_shear(cluster)

    3. As a method of `GalaxyCluster`::

        cluster.compute_shear()

    The angular separation between the source and the lens, :math:`\theta`, and the azimuthal
    position of the source relative to the lens, :math:`\phi`, are computed within the function and the
    angular separation is returned.

    In the flat sky approximation, these angles are calculated using

    .. math::

        \theta^2 = & \left(\delta_s - \delta_l\right)^2 +
        \left(\alpha_l-\alpha_s\right)^2\cos^2(\delta_l)\\
        \tan\phi = & \frac{\delta_s - \delta_l}{\left(\alpha_l - \alpha_s\right)\cos(\delta_l)}

    The tangential, :math:`g_t`, and cross, :math:`g_x`, shears are calculated using the two
    shear components :math:`g_1` and :math:`g_2` of the source galaxies, following Eq.7 and Eq.8
    in Schrabback et al. (2018), arXiv:1611:03866

    .. math::

        g_t =& -\left( g_1\cos\left(2\phi\right) + g_2\sin\left(2\phi\right)\right)\\
        g_x =& -g_1 \sin\left(2\phi\right) + g_2\cos\left(2\phi\right)


    Parameters
    ----------
    cluster: GalaxyCluster, optional
        Instance of `GalaxyCluster()` and must contain right ascension and declinations of both
        the lens and sources and the two shear components all of the sources. If this
        object is specified, right ascension, declination, and shear inputs are ignored.
    ra_lens: float, optional
        Right ascension of the lensing cluster
    dec_lens: float, optional
        Declination of the lensing cluster
    ra_source_list: array_like, optional
        Right ascensions of each source galaxy
    dec_source_list: array_like, optional
        Declinations of each source galaxy
    shear1: array_like, optional
        The measured shear of the source galaxies
    shear2: array_like, optional
        The measured shear of the source galaxies
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
    r"""Compute the tangential shear given the two shears and azimuthal positions for
    a single source or list of sources.

    We compute the cross shear following Eq. 7 of Schrabback et al. 2018, arXiv:1611:03866
    .. math::
        g_t = -\left( g_1\cos\left(2\phi\right) + g_2\sin\left(2\phi\right)\right)

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    return - (shear1 * np.cos(2.*phi) + shear2 * np.sin(2.*phi))


def _compute_cross_shear(shear1, shear2, phi):
    r"""Compute the cross shear given the two shears and azimuthal position for a single
    source of list of sources.

    We compute the cross shear following Eq. 8 of Schrabback et al. 2018, arXiv:1611:03866
    .. math::
        g_x = -g_1 \sin\left(2\phi\right) + g_2\cos\left(2\phi\right)

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    return - shear1 * np.sin(2.*phi) + shear2 * np.cos(2.*phi)


def make_shear_profile(cluster, angsep_units, bin_units, bins=10, cosmo=None,
                       add_to_cluster=True):
    r"""Compute the shear profile of the cluster

    We assume that the cluster object contains information on the cross and
    tangential shears and angular separation of the source galaxies

    This function can be called in two ways using an instance of GalaxyCluster

    1. Pass an instance of GalaxyCluster into the function::

        make_shear_profile(cluster, 'radians', 'radians')

    2. Call it as a method of a GalaxyCluster instance::

        cluster.make_shear_profile('radians', 'radians')

    Parameters
    ----------
    cluster : GalaxyCluster
        Instance of GalaxyCluster that contains the cross and tangential shears of
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
    add_to_cluster: bool, optional
        Attach the profile to the cluster object as `cluster.profile`

    Returns
    -------
    profile : astropy.table.Table
        Output table containing the radius grid points, the tangential and cross shear profiles
        on that grid, and the errors in the two shear profiles. The errors are defined as the standard errors
        in each bin.
    """
    if not all([t_ in cluster.galcat.columns for t_ in ('gt', 'gx', 'theta')]):
        raise TypeError('Shear information is missing in galaxy catalog must have tangential\
                        and cross shears (gt,gx). Run compute_shear first!')

#123123 
    # Check to see if we need to do a unit conversion
    if angsep_units is not bin_units:
        source_seps = _theta_units_conversion(cluster.galcat['theta'], angsep_units, bin_units,
                                              z_cl=cluster.z, cosmo=cosmo)
    else:
        source_seps = cluster.galcat['theta']
#123123 

#123123    
    # Make bins if they are not provided
    if not hasattr(bins, '__len__'):
        bins = make_bins(np.min(source_seps), np.max(source_seps), bins)
#123123

#123123
    # Compute the binned average shears and associated errors
    r_avg, gt_avg, gt_std, gt_counts = _compute_radial_averages(source_seps, cluster.galcat['gt'].data,
                                                     bins=bins)
    r_avg, gx_avg, gx_std, gx_counts = _compute_radial_averages(source_seps, cluster.galcat['gx'].data,
                                                    bins=bins)
#123123
    gt_err = gt_std / gt_counts
    gx_err = gx_std / gx_counts
    
    profile_table = Table([bins[:-1], r_avg, bins[1:], gt_avg, gt_err, gx_avg, gx_err],
                          names=('radius_min', 'radius', 'radius_max', 'gt', 'gt_err',
                                 'gx', 'gx_err'))

    if add_to_cluster:
        cluster.profile = profile_table
        cluster.profile_bin_units = bin_units

    return profile_table

#123123
def _compute_radial_averages(distances, measurements, bins):
    """Given a list of distances, measurements and bins, sort into bins

    Parameters
    ----------
    distance : array_like
        Distance from origin to measurement
    measurements : array_like
        Measurements corresponding to each distance
    bins: array_like
        Bin edges to sort distance

    Returns
    -------
    r_profile : array_like
        Centers of radial bins
    y_profile : array_like
        Average of measurements in distance bin
    yerr_profile: array_like
        Standard deviation of measurements in distance bin
    """
    # if not isinstance(distance, (np.ndarray)):
    #     raise TypeError("Distances must be an array")
    # if not isinstance(measurements, (np.ndarray)):
    #     raise TypeError("Measurements must be an array")
    # if not isinstance(bins, (np.ndarray)):
    #     raise TypeError("Bins must be an array")
    # if len(radius) != len(g):
    #     raise TypeError("Must have a measurement at each distance!")

    # nbins = len(bins)-1
    # r_profile = np.zeros(nbins)
    # y_profile = np.zeros(nbins)
    # yerr_profile = np.zeros(nbins)

    # if np.amax(distance) > np.amax(bins):
    #     warnings.warn("Maximum distance is not within range of bins")
    # if np.amin(distance) < np.amin(bins):
    #     warnings.warn("Minimum distance is not within the range of bins")

    # for i in range(nbins):
    #     index = np.where((radius >= bins[i]) & (radius < bins[i+1]))[0]
    #     if len(index) == 0:
    #         r_profile[i] = np.nan
    #         y_profile[i] = np.nan
    #         yerr_profile[i] = np.nan
    #     else:
    #         r_profile[i] = np.average(distances[index])
    #         y_profile[i] = np.average(measurements[index])
    #         yerr_profile[i] = np.std(measurements[index])/np.sqrt(float(len(index)))
    
    r_profile, _, _ = binned_statistic(distances, distances, statistic='mean', bins=bins)
    y_profile, _, _ = binned_statistic(distances, measurements, statistic='mean', bins=bins)
    yerr_profile, _, _ = binned_statistic(distances, measurements, statistic='std', bins=bins)
    counts_profile, _, _ = binned_statistic(distances, measurements, statistic='count', bins=bins)

    return r_profile, y_profile, yerr_profile, counts_profile
#123123

#123123 
def make_bins(rmin, rmax, n_bins=10, log10_bins=False, method='equal'):
    """Define the bins edges corresponding to the binning of data using user-defined method and parameters.
     The binning method currently supported returns equaly sized bins. 

     Parameters
     ----------
     rmin, rmax,: float
         minimum and maximum bin edges wanted (any units).
     n_bins: float
         number of bins you want to create, default to 10.
     log10_bins: bool
         set to 'True' for binning in log space of base 10, default to False, 
     method : str
         binning method used, 'equal' is currently the only supported option.

     Returns
     -------
     binedges: array_like, float
         n_bins+1 dimensional array that defines bin edges
    """
    if (rmax < rmin):
        raise ValueError("rmax should be larger than rmin")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")
    if type(log10_bins) != bool:
        raise TypeError("log_bins must be type bool")
    if type(n_bins) != int:
        raise TypeError("You need an integer number of bins")

    if method == 'equal':
        if log10_bins == True:
            binedges = np.logspace(np.log10(rmin), np.log10(rmax), n_bins+1, endpoint=True)
        else:
            binedges = np.linspace(rmin, rmax, n_bins+1, endpoint=True)
    else:
        raise NotImplementedError("Binning method '{}' is not currently supported".format(method))
        
    return binedges
#123123 

#123123 
def _theta_units_conversion(source_seps, input_units, output_units, z_cl=None, cosmo=None):
    """Convert source separations from input_units to output_units

    Parameters
    ----------
    source_seps : array_like
        Separation between the lens and each source galaxy on the sky
    input_units : str
        Units of the input source_seps
    output_units : str
        Units to convert source_seps to
        Options = ["rad", deg", "arcmin", "arcsec", kpc", "Mpc"]
    z_cl :  float, optional
	Cluster redshift. Required to convert to physical distances.
    cosmo : astropy.cosmology.core.FlatLambdaCDM, optional
        Cosmology object. Required to convert to physical distances.

    Returns
    -------
    new_radii : array_like 
        Source-lens separation in output_units.
    """
    from astropy import units as u
    units_bank = {"radians": u.rad, "deg": u.deg, "arcmin": u.arcmin, "arcsec": u.arcsec,
                  "kpc": u.kpc, "Mpc": u.Mpc}

    # Check to make sure both the input_units and output_units are supported
    if not input_units in units_bank:
        raise ValueError("Input units{} for separation not supported".format(input_units))
    if not output_units in units_bank:
        raise ValueError("Output units{} for separation not supported".format(output_units))

    # Set input_units on source_seps
    source_seps = source_seps*units_bank[input_units]

    # Convert to output units and return
    if 'pc' in output_units:
        if z_cl is None or cosmo is None:
            raise ValueError("Cluster redshift and cosmology object required to convert to\
                              physical units")
        out_units_obj = units_bank[output_units]
        angular_diameter_distance = cosmo.angular_diameter_distance(z_cl).to(out_units_obj).value
        # if isinstance(cosmo, astropy.cosmology.core.FlatLambdaCDM): # astropy cosmology type
        #     Da = cosmo.angular_diameter_distance(z_cl).to(unit_).value
        # elif isinstance(cosmo, ccl.core.Cosmology): # astropy cosmology type # 7481794
        #     Da = ccl.comoving_angular_distance(cosmo, 1/(1+z_cl)) / (1+z_cl) * u.Mpc.to(unit_)
        # else:
        #     raise ValueError("cosmo object (%s) not an astropy or ccl cosmology"%str(cosmo))
        return source_seps.value*angular_diameter_distance

    else:
        return source_seps.to(units_bank[output_units]).value
#123123 
    
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
GalaxyCluster.make_shear_profile = make_shear_profile
