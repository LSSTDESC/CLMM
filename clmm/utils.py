"""General data- and model- and GCData- independent utility functions that are used in multiple modules"""
import math
import numpy as np
from scipy.stats import binned_statistic
import warnings

from .hybrid import _convert_rad_to_mpc

def _get_a_from_z(redshift):
    """ Convert redshift to scale factor

    Parameters
    ----------
    redshift : array_like
        Redshift

    Returns
    -------
    scale_factor : array_like
        Scale factor
    """
    redshift = np.array(redshift)
    if np.any(redshift < 0.0):
        raise ValueError(f"Cannot convert negative redshift to scale factor")
    return 1. / (1. + redshift)


def _get_z_from_a(scale_factor):
    """ Convert scale factor to redshift

    Parameters
    ----------
    scale_factor : array_like
        Scale factor

    Returns
    -------
    redshift : array_like
        Redshift
    """
    scale_factor = np.array(scale_factor)
    if np.any(scale_factor > 1.0):
        raise ValueError(f"Cannot convert invalid scale factor a > 1 to redshift")
    return 1. / scale_factor - 1.


def compute_radial_averages(xvals, yvals, xbins, error_model='std/sqrt_n'):
    """ Given a list of xvalss, yvals and bins, sort into bins

    Parameters
    ----------
    xvals : array_like
        Values to be binned
    yvals : array_like
        Values to compute statistics on
    xbins: array_like
        Bin edges to sort into
    error_model : str, optional
        Error model to use for y uncertainties.
        std/sqrt_n - Standard Deviation/sqrt(Counts) (Default)
        std - Standard deviation

    Returns
    -------
    meanx : array_like
        Mean x value in each bin
    meany : array_like
        Mean y value in each bin
    yerr : array_like
        Error on the mean y value in each bin. Specified by error_model
    n : array_like
        Number of objects in each bin
    """
    meanx, xbins, binnumber = binned_statistic(xvals, xvals, statistic='mean', bins=xbins)[:3]
    meany = binned_statistic(xvals, yvals, statistic='mean', bins=xbins)[0]
    # number of objects
    n = np.histogram(xvals, xbins)[0]
    n_zero = n==0

    if error_model == 'std':
        yerr = binned_statistic(xvals, yvals, statistic='std', bins=xbins)[0]
    elif error_model == 'std/sqrt_n':
        yerr = binned_statistic(xvals, yvals, statistic='std', bins=xbins)[0]
        sqrt_n = np.sqrt(binned_statistic(xvals, yvals, statistic='count', bins=xbins)[0])
        sqrt_n[n_zero] = 1.0
        yerr = yerr/sqrt_n
    else:
        raise ValueError(f"{error_model} not supported err model for binned stats")

    meanx[n_zero] = 0
    meany[n_zero] = 0
    yerr[n_zero]  = 0

    return meanx, meany, yerr, n, binnumber


def make_bins(rmin, rmax, nbins=10, method='evenwidth', source_seps=None):
    """ Define bin edges

    Parameters
    ----------
    rmin : float
        Minimum bin edges wanted
    rmax : float
        Maximum bin edges wanted
    nbins : float
        Number of bins you want to create, default to 10.
    method : str, optional
        Binning method to use
        'evenwidth' - Default, evenly spaced bins between rmin and rmax
        'evenlog10width' - Logspaced bins with even width in log10 between rmin and rmax
        'equaloccupation' - Bins with equal occupation numbers
    source_seps : array-like
        Radial distance of source separations

    Returns
    -------
    binedges: array_like, float
        n_bins+1 dimensional array that defines bin edges
    """
    if (rmin > rmax) or (rmin < 0.0) or (rmax < 0.0):
        raise ValueError(f"Invalid bin endpoints in make_bins, {rmin} {rmax}")
    if (nbins <= 0) or not isinstance(nbins, int):
        raise ValueError(f"Invalid nbins={nbins}. Must be integer greater than 0.")

    if method == 'evenwidth':
        binedges = np.linspace(rmin, rmax, nbins+1, endpoint=True)
    elif method == 'evenlog10width':
        binedges = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1, endpoint=True)
    elif method == 'equaloccupation':
        if source_seps is None:
            raise ValueError(f"Binning method '{method}' requires source separations array")
        # by default, keep all galaxies
        mask = np.full(len(source_seps), True)
        if rmin is not None or rmax is not None:
        # Need to filter source_seps to only keep galaxies in the [rmin, rmax]
            if rmin is None: rmin = np.min(source_seps)
            if rmax is None: rmax = np.max(source_seps)
            mask = (np.array(source_seps)>=rmin)*(np.array(source_seps)<=rmax)
        binedges = np.percentile(source_seps[mask], tuple(np.linspace(0,100,nbins+1, endpoint=True)))
    else:
        raise ValueError(f"Binning method '{method}' is not currently supported")

    return binedges


def convert_shapes_to_epsilon(shape_1, shape_2, shape_definition='epsilon', kappa=0):
    """ Given shapes and their definition, convert them to epsilon ellipticities or reduced shears, which can be used in GalaxyCluster.galcat
    Definitions used here based on Bartelmann & Schneider 2001 (https://arxiv.org/pdf/astro-ph/9912508.pdf):
    axis ratio (q) and position angle (phi) (Not implemented)
    epsilon = (1-q/(1+q) exp(2i phi)
    chi = (1-q^2/(1+q^2) exp(2i phi)
    shear (gamma)
    reduced_shear (g) = gamma/(1-kappa)
    convergence (kappa)


    Parameters
    ==========
    shape_1 : array_like
        Input shapes or shears along principal axis (g1 or e1)
    shape_2 : array_like
        Input shapes or shears along secondary axis (g2 or e2)
    shape_definition : str
        Definition of the input shapes, can be ellipticities 'epsilon' or 'chi' or shears 'shear' or 'reduced_shear'
    kappa : array_like
        Convergence for transforming to a reduced shear. Default is 0

    Returns
    =======
    epsilon_1 : array_like
        Epsilon ellipticity along principal axis (epsilon1)
    epsilon_2 : array_like
        Epsilon ellipticity along secondary axis (epsilon2)
    """

    if shape_definition=='epsilon' or shape_definition=='reduced_shear':
        return shape_1, shape_2
    elif shape_definition=='chi':
        chi_to_eps_conversion = 1./(1.+(1-(shape_1**2 + shape_2**2))**0.5)
        return shape_1*chi_to_eps_conversion, shape_2*chi_to_eps_conversion
    elif shape_definition=='shear':
        return shape_1/(1.-kappa), shape_2/(1.-kappa)

    else:
        raise TypeError("Please choose epsilon, chi, shear, reduced_shear")


def build_ellipticities(q11, q22, q12):
    """ Build ellipticties from second moments. See, e.g., Schneider et al. (2006)

    Parameters
    ==========
    q11 : float or array
        Second brightness moment tensor, component (1,1)
    q22 : float or array
        Second brightness moment tensor, component (2,2)
    q12 :  float or array
        Second brightness moment tensor, component (1,2)

    Returns
    =======
    x1, x2 : float or array
        Ellipticities using the "chi definition"
    e1, e2 : float or array
        Ellipticities using the "epsilon definition"
    """

    x1, x2 = (q11-q22)/(q11+q22), (2*q12)/(q11+q22)
    e1, e2 = (q11-q22)/(q11+q22+2*np.sqrt(q11*q22-q12*q12)), (2*q12)/(q11+q22+2*np.sqrt(q11*q22-q12*q12))
    return x1,x2, e1,e2


def compute_lensed_ellipticity(ellipticity1_true, ellipticity2_true, shear1, shear2, convergence):
    r""" Compute lensed ellipticities from the intrinsic ellipticities, shear and convergence.
    Following Schneider et al. (2006)

    .. math::
        \epsilon^{\rm lensed}=\epsilon^{\rm lensed}_1+i\epsilon^{\rm lensed}_2=\frac{\epsilon^{\rm true}+g}{1+g^\ast\epsilon^{\rm true}},

    where, the complex reduced shear :math:`g` is obtained from the shear :math:`\gamma=\gamma_1+i\gamma_2`
    and convergence :math:`\kappa` as :math:`g = \gamma/(1-\kappa)`, and the complex intrinsic ellipticity
    is :math:`\epsilon^{\rm true}=\epsilon^{\rm true}_1+i\epsilon^{\rm true}_2`


    Parameters
    ==========
    ellipticity1_true : float or array
        Intrinsic ellipticity of the sources along the principal axis
    ellipticity2_true : float or array
        Intrinsic ellipticity of the sources along the second axis
    shear1 :  float or array
        Shear component along the principal axis at the source location
    shear2 :  float or array
        Shear component along the second axis at the source location
    convergence :  float or array
        Convergence at the source location
    Returns
    =======
    e1, e2 : float or array
        Lensed ellipicity along both reference axes.
    """

    shear = shear1 + shear2*1j # shear (as a complex number)
    ellipticity_true = ellipticity1_true + ellipticity2_true*1j # intrinsic ellipticity (as a complex number)
    reduced_shear = shear / (1.0 - convergence) # reduced shear
    e = (ellipticity_true + reduced_shear) / (1.0 + reduced_shear.conjugate()*ellipticity_true) # lensed ellipticity
    return np.real(e), np.imag(e)


def _compute_tangential_shear(shear1, shear2, phi):#project_tangential
    r"""Compute the tangential shear given the two shears and azimuthal positions for
    a single source or list of sources.

    We compute the tangential shear following Eq. 7 of Schrabback et al. 2018, arXiv:1611:03866
    .. math::
        g_t = -\left( g_1\cos\left(2\phi\right) - g_2\sin\left(2\phi\right)\right)

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    return - (shear1 * np.cos(2.*phi) + shear2 * np.sin(2.*phi))


def _compute_cross_shear(shear1, shear2, phi):#project_cross
    r"""Compute the cross shear given the two shears and azimuthal position for a single
    source of list of sources.

    We compute the cross shear following Eq. 8 of Schrabback et al. 2018, arXiv:1611:03866
    also checked arxiv 0509252
    .. math::
        g_x = g_1 \sin\left(2\phi\right) - g_2\cos\left(2\phi\right)

    For extended descriptions of parameters, see `compute_shear()` documentation.
    """
    return shear1 * np.sin(2.*phi) - shear2 * np.cos(2.*phi)


def get_reduced_shear_from_convergence(shear, convergence):
    """ Calculates reduced shear from shear and convergence

    Parameters
    ----------
    shear : array_like
        Shear
    convergence : array_like
        Convergence

    Returns:
    reduced_shear : array_like
        Reduced shear
    """
    shear, convergence = np.array(shear), np.array(convergence)
    reduced_shear = shear / (1. - convergence)
    return reduced_shear


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
