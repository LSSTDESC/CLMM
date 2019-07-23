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


def _astropy_to_CCL_cosmo_object(astropy_cosmology_object) :
#ALLOWS TO USE EITHER ASTROPY OR CCL FOR COSMO OBJECT, MAYBE THIS FUNCTION SOULD NOT BE HERE
#adapted from https://github.com/LSSTDESC/CLMM/blob/issue/111/model-definition/clmm/modeling.py
    ''' 
    Generates a ccl cosmology object from an GCR or astropy cosmology object.  
    '''
    apy_cosmo = astropy_cosmology_object
    ccl_cosmo = ccl.Cosmology(Omega_c=apy_cosmo.Odm0,
                  Omega_b=apy_cosmo.Ob0,
                  h=apy_cosmo.h,
                  n_s=apy_cosmo.n_s,
                  sigma8=apy_cosmo.sigma8,
                  Omega_k=apy_cosmo.Ok0)
    
    return ccl_cosmo



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
    dx = (ra_s-ra_l)*u.deg.to(u.rad) * np.cos(dec_l *u.deg.to(u.rad))             
    dy = (dec_s - dec_l)*u.deg.to(u.rad)                 
    phi = np.arctan2(dy, -dx)     
    
    if sky == "curved":
        coord_l = SkyCoord(ra_l*u.deg,dec_l*u.deg)
        coord_s = SkyCoord(ra_s*u.deg,dec_s*u.deg)
        theta = coord_l.separation(coord_s).to(u.rad).value

    else:                     
        theta =  np.sqrt(dx**2 + dy**2)

    return theta, phi


def _compute_g_t(g1, g2, phi):
    """Computes the tangential shear for each source in the galaxy catalog

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
    """Computes cross shear for each source in galaxy catalog
    
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
    """Wrapper that returns tangential and cross shear along with radius in radians
    
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


def _make_bins(rmin, rmax, n_bins=10, log_bins=False):
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
    if log_bins==True:
        rmin = np.log(rmin)
        rmax = np.log(rmax)
        logbinedges = np.linspace(rmin, rmax, n_bins+1, endpoint=True)
        binedges = np.exp(logbinedges)
    else:
        binedges = np.linspace(rmin, rmax, n_bins+1, endpoint=True)
            
    return binedges


def _make_shear_profile(radius, g, bins=None):
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
    ax.errorbar(r, gt, gterr)
    
    if type(gx) is np.ndarray:
        plt.plot(r, gx, 'ro-', label="cross shear")
        plt.errorbar(r, gx, gxerr)

    ax.legend()
    ax.set_xlabel("r [%s]"%r_units)
    ax.set_ylabel('$\\gamma$')

    return(fig, ax)
