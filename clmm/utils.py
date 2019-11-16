"""General utility functions that are used in multiple modules"""
import numpy as np
from scipy.stats import binned_statistic
from astropy import units as u


def compute_radial_averages(distances, measurements, bins, error_model='std/n'):
    """ Given a list of distances, measurements and bins, sort into bins

    Parameters
    ----------
    distance : array_like
        Distance from origin to measurement
    measurements : array_like
        Measurements corresponding to each distance
    bins: array_like
        Bin edges to sort distance
    error_model : str, optional
        Error model to use for y uncertainties.
        std/n - Standard Deviation/Counts (Default)
        std - Standard deviation

    Returns
    -------
    r_profile : array_like
        Centers of radial bins
    y_profile : array_like
        Average of measurements in distance bin
    yerr_profile : array_like
        Standard deviation of measurements in distance bin
    """
    r_profile, _, _ = binned_statistic(distances, distances, statistic='mean', bins=bins)
    y_profile, _, _ = binned_statistic(distances, measurements, statistic='mean', bins=bins)

    if error_model == 'std':
        yerr_profile, _, _ = binned_statistic(distances, measurements, statistic='std', bins=bins)
    elif error_model == 'std/n':
        yerr_profile, _, _ = binned_statistic(distances, measurements, statistic='std', bins=bins)
        counts, _, _ = binned_statistic(distances, measurements, statistic='count', bins=bins)
        yerr_profile = yerr_profile/counts
    else:
        raise ValueError("{} not supported err model for binned stats".format(error_model))

    return r_profile, y_profile, yerr_profile


def make_bins(rmin, rmax, n_bins=10, log10_bins=False, method='equal'):
    """ Define bin edges

    Parameters
    ----------
    rmin : float
        Minimum bin edges wanted
    rmax : float
        Maximum bin edges wanted
    n_bins : float
        Number of bins you want to create, default to 10.
    log10_bins : bool
        Bin in logspace rather than linear space
    method : str
        Binning method used, 'equal' is currently the only supported option.

    Returns
    -------
    binedges: array_like, float
        n_bins+1 dimensional array that defines bin edges
    """
    if rmax < rmin:
        raise ValueError("rmax should be larger than rmin")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")
    if isinstance(log10_bins, bool):
        raise TypeError("log10_bins must be type bool")
    if isinstance(n_bins, int):
        raise TypeError("You need an integer number of bins")

    if method == 'equal':
        if log10_bins:
            binedges = np.logspace(np.log10(rmin), np.log10(rmax), n_bins+1, endpoint=True)
        else:
            binedges = np.linspace(rmin, rmax, n_bins+1, endpoint=True)
    else:
        raise NotImplementedError("Binning method '{}' is not currently supported".format(method))

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
        raise ValueError("Input units ({}) not supported".format(unit1))
    if unit2 not in units_bank:
        raise ValueError("Output units ({}) not supported".format(unit2))

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
