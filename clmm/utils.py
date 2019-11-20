"""General utility functions that are used in multiple modules"""
import numpy as np
from scipy.stats import binned_statistic
from astropy import units as u


def compute_radial_averages(xvals, yvals, xbins, error_model='std/n'):
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
        std/n - Standard Deviation/Counts (Default)
        std - Standard deviation

    Returns
    -------
    meanx : array_like
        Mean x value in each bin
    meany : array_like
        Mean y value in each bin
    yerr : array_like
        Error on the mean y value in each bin. Specified by error_model
    """
    meanx = binned_statistic(xvals, xvals, statistic='mean', bins=xbins)[0]
    meany = binned_statistic(xvals, yvals, statistic='mean', bins=xbins)[0]

    if error_model == 'std':
        yerr = binned_statistic(xvals, yvals, statistic='std', bins=xbins)[0]
    elif error_model == 'std/n':
        yerr = binned_statistic(xvals, yvals, statistic='std', bins=xbins)[0]
        yerr = yerr/binned_statistic(xvals, yvals, statistic='count', bins=xbins)[0]
    else:
        raise ValueError(f"{error_model} not supported err model for binned stats")

    return meanx, meany, yerr


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
