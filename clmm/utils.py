"""General utility functions that are used in multiple modules"""
import numpy as np
from scipy.stats import binned_statistic
from astropy import units as u


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
    meanx, xbins = binned_statistic(xvals, xvals, statistic='mean', bins=xbins)[:2]
    meany = binned_statistic(xvals, yvals, statistic='mean', bins=xbins)[0]
    # number of objects
    n = np.histogram(xvals, xbins)[0]

    if error_model == 'std':
        yerr = binned_statistic(xvals, yvals, statistic='std', bins=xbins)[0]
    elif error_model == 'std/sqrt_n':
        yerr = binned_statistic(xvals, yvals, statistic='std', bins=xbins)[0]
        yerr = yerr/np.sqrt(binned_statistic(xvals, yvals, statistic='count', bins=xbins)[0])
    else:
        raise ValueError(f"{error_model} not supported err model for binned stats")

    return meanx, meany, yerr, n


def make_bins(rmin, rmax, nbins=10, method='evenwidth'):
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
    else:
        raise ValueError(f"Binning method '{method}' is not currently supported")

    return binedges


def convert_units(dist1, unit1, unit2, redshift=None, cosmo=None):
    """ Convenience wrapper to convert between a combination of angular and physical units.

    Supported units: radians, degrees, arcmin, arcsec, Mpc, kpc, pc

    To convert between angular and physical units you must provide both
    a redshift and a cosmology object.

    Parameters
    ----------
    dist1 : array_like
        Input distances
    unit1 : str
        Unit for the input distances
    unit2 : str
        Unit for the output distances
    redshift : float
        Redshift used to convert between angular and physical units
    cosmo : astropy.cosmology
        Astropy cosmology object to compute angular diameter distance to
        convert between physical and angular units

    Returns
    -------
    dist2: array_like
        Input distances converted to unit2
    """
    angular_bank = {"radians": u.rad, "degrees": u.deg, "arcmin": u.arcmin, "arcsec": u.arcsec}
    physical_bank = {"pc": u.pc, "kpc": u.kpc, "Mpc": u.Mpc}
    units_bank = {**angular_bank, **physical_bank}

    # Some error checking
    if unit1 not in units_bank:
        raise ValueError(f"Input units ({unit1}) not supported")
    if unit2 not in units_bank:
        raise ValueError(f"Output units ({unit2}) not supported")

    # Try automated astropy unit conversion
    try:
        return (dist1 * units_bank[unit1]).to(units_bank[unit2]).value

    # Otherwise do manual conversion
    except u.UnitConversionError:
        # Make sure that we were passed a redshift and cosmology
        if redshift is None or cosmo is None:
            raise TypeError("Redshift and cosmology must be specified to convert units")

        # Redshift must be greater than zero for this approx
        if not redshift > 0.0:
            raise ValueError("Redshift must be greater than 0.")

        # Convert angular to physical
        if (unit1 in angular_bank) and (unit2 in physical_bank):
            dist1_rad = (dist1 * units_bank[unit1]).to(u.rad).value
            dist1_mpc = _convert_rad_to_mpc(dist1_rad, redshift, cosmo, do_inverse=False)
            return (dist1_mpc * u.Mpc).to(units_bank[unit2]).value

        # Otherwise physical to angular
        dist1_mpc = (dist1 * units_bank[unit1]).to(u.Mpc).value
        dist1_rad = _convert_rad_to_mpc(dist1_mpc, redshift, cosmo, do_inverse=True)
        return (dist1_rad * u.rad).to(units_bank[unit2]).value


def _convert_rad_to_mpc(dist1, redshift, cosmo, do_inverse=False):
    r""" Convert between radians and Mpc using the small angle approximation
    and :math:`d = D_A \theta`.

    Parameters
    ==========
    dist1 : array_like
        Input distances
    redshift : float
        Redshift used to convert between angular and physical units
    cosmo : astropy.cosmology
        Astropy cosmology object to compute angular diameter distance to
        convert between physical and angular units
    do_inverse : bool
        If true, converts Mpc to radians

    Returns
    =======
    dist2 : array_like
        Converted distances
    """
    d_a = cosmo.angular_diameter_distance(redshift).to('Mpc').value
    if do_inverse:
        return dist1 / d_a
    return dist1 * d_a


def convert_shapes_to_epsilon(shape_1,shape_2, shape_definition='epsilon',kappa=0):
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
        return shape_1,shape_2
    elif shape_definition=='chi':
        chi_to_eps_conversion = 1./(1.+(1-(shape_1**2 + shape_2**2))**0.5)
        return shape_1*chi_to_eps_conversion,shape_2*chi_to_eps_conversion
    elif shape_definition=='shear':
        return shape_1/(1.-kappa), shape_2/(1.-kappa)
    
    else:
        raise TypeError("Please choose epsilon, chi, shear, reduced_shear")