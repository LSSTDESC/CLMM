"""@file polaraveraging.py
Functions to compute polar/azimuthal averages in radial bins
"""
try:
    import pyccl as ccl
except:
    pass
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy import units as u
import math


##############################################################################################
def _astropy_to_CCL_cosmo_object(astropy_cosmology_object) :
#ALLOWS TO USE EITHER ASTROPY OR CCL FOR COSMO OBJECT, MAYBE THIS FUNCTION SOULD NOT BE HERE
#adapted from https://github.com/LSSTDESC/CLMM/blob/issue/111/model-definition/clmm/modeling.py
    ''' 
    Generates a ccl cosmology object from an GCR or astropy cosmology object.  
    '''
    apy_cosmo = astropy_cosmology_object
    ccl_cosmo = ccl.Cosmology(Omega_c=(apy_cosmo.Odm0-apy_cosmo.Ob0),
                  Omega_b=apy_cosmo.Ob0,
                  h=apy_cosmo.h,
                  n_s=apy_cosmo.n_s,
                  sigma8=apy_cosmo.sigma8,
                  Omega_k=apy_cosmo.Ok0)
    
    return ccl_cosmo
##############################################################################################
#### Wrapper functions #######################################################################
##############################################################################################

def compute_shear(cluster, geometry="flat", add_to_cluster=True):
    """Computs tangential and cross shear along 
         with radius in radians
    Parameters
    ----------
    cluster: GalaxyCluster object
        GalaxyCluster object with galaxies
    geometry: str ('flat', 'curve')
        Geometry to be used in the computation of theta, phi
    add_to_cluster: bool
        Adds the outputs to cluster.galcat
    Returns
    -------
    gt: float vector
        tangential shear
    gx: float vector
        cross shear
    theta: float vector
        radius in radians
    """
    if not ('e1' in cluster.galcat.columns  
        and 'e2' in cluster.galcat.columns):
        raise TypeError('shear information is missing in galaxy, ',
                        'must have (e1, e2) or (gamma1, gamma2, kappa)')
    theta, gt , gx = _compute_shear(cluster.ra, cluster.dec, 
        cluster.galcat['ra'], cluster.galcat['dec'], 
        cluster.galcat['e1'], cluster.galcat['e2'], 
        sky=geometry)
    if add_to_cluster:
        cluster.galcat['theta'] = theta
        cluster.galcat['gt'] = gt
        cluster.galcat['gx'] = gx
    return theta, gt, gx

def make_shear_profile(cluster, radial_units, bins=None,
                        cosmo=None, cosmo_object_type="astropy",
                        add_to_cluster=True):
    """ Computes shear profile of the cluster

    Parameters
    ----------
    cluster: GalaxyCluster object
        GalaxyCluster object with galaxies
    radial_units:
        Radial units of the profile, one of 
        ["rad", deg", "arcmin", "arcsec", kpc", "Mpc"]
    bins: array_like, float
        User defined n_bins + 1 dimensional array of bins, if 'None',
        the default is 10 equally spaced radial bins
    cosmo:
        Cosmology object 
    cosmo_object_type : str
        Keywords that can be either "ccl" or "astropy" 
    add_to_cluster: bool
        Adds the outputs to cluster.profile

    Returns
    -------
    profile_table: astropy Table
        Table with r_profile, gt profile (and error) and
        gx profile (and error)
    """
    if not ('gt' in cluster.galcat.columns  
        and 'gx' in cluster.galcat.columns
        and 'theta' in cluster.galcat.columns):
        raise TypeError('shear information is missing in galaxy, ',
                        'must have tangential and cross shears (gt,gx).',
                        'Run compute_shear first!')
    radial_values = _theta_units_conversion(cluster.galcat['theta'],
                                        radial_units, z_cl=cluster.z,
                                        cosmo_object_type=cosmo_object_type)
    r_avg, gt_avg, gt_std = _compute_radial_averages(radial_values, cluster.galcat['gt'])
    r_avg, gx_avg, gx_std = _compute_radial_averages(radial_values, cluster.galcat['gx'])
    profile_table = Table([r_avg, gt_avg, gt_std, gx_avg, gx_avg],
        names = ('radius', 'gt', 'gt_err', 'gx', 'gx_err'))
    if add_to_cluster:
        cluster.profile = profile_table
    return profile_table

def plot_profiles(cluster):
    """Plot shear profiles for validation

    Parameters
    ----------
    cluster: GalaxyCluster object
        GalaxyCluster object with galaxies
    """
    prof = cluster.profile
    return _plot_profiles(*[cluster.profile[c] for c in
            ('radius', 'gt', 'gt_err', 'gx', 'gx_err')],
            r_units=cluster.profile['radius'].unit)

# Maybe these functions should be here instead of __init__
#GalaxyCluster.compute_shear = compute_shear
#GalaxyCluster.make_shear_profile = make_shear_profile
#GalaxyCluster.plot_profiles = plot_profiles

##############################################################################################
#### Internal functions ######################################################################
##############################################################################################

def _compute_theta_phi(ra_l, dec_l, ra_s, dec_s, sky="flat"):
    """Returns the characteristic angles of the lens system
    
    Add extended description

    Parameters
    ----------
    ra_l, dec_l : float 
        ra and dec of lens in decimal degrees
    ra_s, dec_s : array_like, float
        ra and dec of source in decimal degrees
    sky : str, optional
        'flat' uses the flat sky approximation (default) and 'curved' uses exact angles

    Returns
    -------
    theta : array_like, float
        Angular separation on the sky in radians
    phi : array_like, float
        Angle in radians, (can we do better)
    """
    if not -360. < ra_l < 360.:
        raise ValueError("ra = %f of lens if out of domain"%ra_l)
    if not -90. < dec_l < 90.:
        raise ValueError("dec = %f of lens if out of domain"%dec_l)
    if not np.array([-360. < x_ < 360. for x_ in ra_s]).all():
        raise ValueError("Object has an invalid ra in source catalog")
    if not np.array([-90. < x_ < 90 for x_ in dec_s]).all():
        raise ValueError("Object has an invalid dec in the source catalog")

    deg_to_rad = np.pi/180.

    if sky == "flat":
        dx = (ra_s - ra_l)*deg_to_rad * math.cos(dec_l*deg_to_rad)
        dy = (dec_s - dec_l)*deg_to_rad
        ## make sure absolute value of all RA differences are < 180 deg:
        ## subtract 360 deg from RA angles > 180 deg
        dx[dx>np.pi] = dx[dx>np.pi] - 2.*np.pi
        ## add 360 deg to RA angles < -180 deg
        dx[dx<-np.pi] = dx[dx<-np.pi] + 2.*np.pi 
        theta =  np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, -dx)
        
    elif sky == "curved":
        raise ValueError("Curved sky functionality not yet supported!")
        # coord_l = SkyCoord(ra_l*u.deg,dec_l*u.deg)
        # coord_s = SkyCoord(ra_s*u.deg,dec_s*u.deg)
        # theta = coord_l.separation(coord_s).to(u.rad).value
    else:
        raise ValueError("Sky option %s not supported!"%sky)

    return theta, phi


def _compute_g_t(g1, g2, phi):
    r"""Computes the tangential shear for each source in the galaxy catalog

    Add extended description

    Parameters
    ----------
    g1, g2 : array_like, float
        Ellipticity or shear for each source in the galaxy catalog
    phi: array_like, float
        As defined in comput_theta_phi (readdress this one)

    Returns
    -------
    g_t : array_like, float
        tangential shear (need not be reduced shear)

    Notes
    -----
    g_t = - (g_1 * \cos(2\phi) + g_2 * \sin(2\phi)) [cf. eqs. 7-8 of Schrabback et al. 2018, arXiv:1611.03866]
    """
    g_t = - (g1*np.cos(2*phi) + g2*np.sin(2*phi))
    return g_t


def _compute_g_x(g1, g2, phi):
    r"""Computes cross shear for each source in galaxy catalog
    
    Parameters
    ----------
    g1, g2,: array_like, float
        ra and dec of the lens (l) and source (s)  in decimal degrees
    phi: array_like, float
        As defined in comput_theta_phi

    Returns
    -------
    gx: array_like, float
        cross shear

    Notes
    -----
    Computes the cross shear for each source in the catalog as:
    g_x = - g_1 * \sin(2\phi) + g_2 * \cos(2\phi)    [cf. eqs. 7-8 of Schrabback et al. 2018, arXiv:1611.03866]
    """ 
    g_x = - g1 * np.sin(2*phi) + g2 *np.cos(2*phi)
    return g_x


def _compute_shear(ra_l, dec_l, ra_s, dec_s, g1, g2, sky="flat"):
    r"""Wrapper that returns tangential and cross shear along with radius in radians
    
    Parameters
    ----------
    ra_l, dec_l: float 
        ra and dec of lens in decimal degrees
    ra_s, dec_s: array_like, float
        ra and dec of source in decimal degrees
    g1, g2: array_like, float
        shears or ellipticities from galaxy table
    sky: str, optional
        'flat' uses the flat sky approximation (default) and 'curved' uses exact angles

    Returns
    -------
    gt: array_like, float
        tangential shear
    gx: array_like, float
        cross shear
    theta: array_like, float
        Angular separation between lens and sources

    Notes
    -----
    Computes the cross shear for each source in the galaxy catalog as:
    g_x = - g_1 * \sin(2\phi) + g_2 * \cos(2\phi)
    g_t = - (g_1 * \cos(2\phi) + g_2 * \sin(2\phi)) [cf. eqs. 7-8 of Schrabback et al. 2018, arXiv:1611.03866]
    """ 
    theta, phi = _compute_theta_phi(ra_l, dec_l, ra_s, dec_s, sky = sky)
    g_t = _compute_g_t(g1,g2,phi)
    g_x = _compute_g_x(g1,g2,phi)
    return theta, g_t, g_x

def _theta_units_conversion(theta, units, z_cl=None, cosmo=None,
                                        cosmo_object_type="astropy"):
    
    """
    Converts theta from radian to whatever units specified in units
    units: one of ["rad", deg", "arcmin", "arcsec", kpc", "Mpc"]
    cosmo : cosmo object 
    z_cl : cluster redshift
    cosmo_object_type : string keywords that can be either "ccl" or "astropy" 
    """
    
    theta = theta * u.rad
    
    if units == "rad":
        radius = theta.value
        
    if units == "deg":
        radius = theta.to(u.deg).value
        
    if units == "arcmin":
        radius = theta.to(u.arcmin).value
        
    if units == "arcsec":
        radius = theta.to(u.arcsec).value 
    
    if cosmo_object_type == "astropy" and units == "Mpc":
        radius = theta.value * cosmo.angular_diameter_distance(z_cl).to(u.Mpc).value

    if cosmo_object_type == "astropy" and units == "kpc":
        radius = theta.value * cosmo.angular_diameter_distance(z_cl).to(u.kpc).value
        
    if cosmo_object_type == "ccl" and units == "Mpc":
        radius = theta.value * cosmo.comoving_angular_distance(cosmo_ccl, 1/(1+z_cl)) / (1+z_cl) * u.Mpc.to(u.Mpc)
        
    if cosmo_object_type == "ccl" and units == "kpc":
        radius = theta.value * cosmo.comoving_angular_distance(cosmo_ccl, 1/(1+z_cl)) / (1+z_cl) * u.Mpc.to(u.kpc)
        
    return radius

def make_bins(rmin, rmax, n_bins=10, log_bins=False):
    """Define equal sized bins with an array of n_bins+1 bin edges
    
    Parameters
    ----------
    rmin, rmax,: float
        minimum and and maximum range of data (any units)
    n_bins: float
        number of bins you want to create
    log_bins: bool
        set to 'True' equal sized bins in log space

    Returns
    -------
    binedges: array_like, float
        n_bins+1 dimensional array that defines bin edges
    """

    if rmax<rmin:
        raise ValueError("rmax should be larger than rmin")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")
    if type(log_bins)!=bool:
        raise TypeError("log_bins must be type bool")
    if type(n_bins)!=int:
        raise TypeError("You need an integer number of bins")
    
    if log_bins==True:
        rmin = np.log(rmin)
        rmax = np.log(rmax)
        logbinedges = np.linspace(rmin, rmax, n_bins+1, endpoint=True)
        binedges = np.exp(logbinedges)
    else:
        binedges = np.linspace(rmin, rmax, n_bins+1, endpoint=True)
            
    return binedges


def _compute_radial_averages(radius, g, bins=None):
    """Returns astropy table containing shear profile of either tangential or cross shear

    Parameters
    ----------
    radius: array_like, float
        Distance (physical or angular) between source galaxy to cluster center
    g: array_like, float
        Either tangential or cross shear (g_t or g_x)
    bins: array_like, float
        User defined n_bins + 1 dimensional array of bins, if 'None', the default is 10 equally spaced radial bins

    Returns
    -------
    r_profile: array_like, float
        Centers of radial bins
    g_profile: array_like, float
        Average shears per bin
    gerr_profile: array_like, float
        Standard deviation of shear per bin
    """

    if type(radius) != np.ndarray:
        raise TypeError("radius must be an array")
    if type(g) != np.ndarray:
        raise TypeError("g must be an array")
    if len(radius) != len(g):
        raise TypeError("radius and g must be arrays of the same length")
    if np.amax(radius) >= np.amax(bins):
        raise ValueError("maxium radius must be within range of bins")
    if np.amin(radius) < np.amin(bins):
        raise ValueError("Minimum radius must be within the range of bins")
    if len(bins) < 2:
        raise TypeError("you need to define at least one bin")
    
    if np.any(bins) == None:
        nbins = 10
        bins = np.linspace(np.min(radius), np.max(radius), nbins)

    g_profile = np.zeros(len(bins) - 1)
    gerr_profile = np.zeros(len(bins) - 1)
    r_profile =  np.zeros(len(bins) - 1)

    for i in range(len(bins)-1):
        cond = (radius>= bins[i]) & (radius < bins[i+1])
        index = np.where(cond)[0]
        r_profile[i] = np.average(radius[index])
        g_profile[i] = np.average(g[index])
        if len(index) != 0:
            gerr_profile[i] = np.std(g[index]) / np.sqrt(float(len(index)))
        else:
            gerr_profile[i] = np.nan

    return r_profile, g_profile, gerr_profile


def _plot_profiles(r, gt, gterr, gx=None, gxerr=None, r_units=""):
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
    ax.set_xlabel("r [%s]"%r_units)
    ax.set_ylabel('$\\gamma$')

    return(fig, ax)
