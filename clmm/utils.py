"""General utility functions that are used in multiple modules"""
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy import units as u


def _compute_radial_averages(distances, measurements, bins, error_model='std/n'):
    """Given a list of distances, measurements and bins, sort into bins

    Parameters
    ----------
    distance : array_like
        Distance from origin to measurement
    measurements : array_like
        Measurements corresponding to each distance
    bins: array_like
        Bin edges to sort distance
    error_model: str, optional
        Error model to use for y uncertainties.
        std/n - Standard Deviation/Counts (Default)
        std - Standard deviation

    Returns
    -------
    r_profile : array_like
        Centers of radial bins
    y_profile : array_like
        Average of measurements in distance bin
    yerr_profile: array_like
        Standard deviation of measurements in distance bin
    counts_profile: array_like
        Number of objects in the bin
    """
    r_profile, _, _ = binned_statistic(distances, distances, statistic='mean', bins=bins)
    y_profile, _, _ = binned_statistic(distances, measurements, statistic='mean', bins=bins)
    counts_profile, _, _ = binned_statistic(distances, measurements, statistic='count', bins=bins)

    if error_model is 'std':
        yerr_profile, _, _ = binned_statistic(distances, measurements, statistic='std', bins=bins)
    elif error_model == 'std/n':
        yerr_profile, _, _ = binned_statistic(distances, measurements, statistic='std', bins=bins)
        yerr_profile = yerr_profile/counts_profile

    return r_profile, y_profile, yerr_profile, counts_profile


def make_bins(rmin, rmax, n_bins=10, log10_bins=False, method='equal'):
    """Define bin edges

    Parameters
    ----------
    rmin: float
        Minimum bin edges wanted
    rmax: float
        Maximum bin edges wanted
    n_bins: float
        Number of bins you want to create, default to 10.
    log10_bins: bool
        Bin in logspace rather than linear space
    method : str
        Binning method used, 'equal' is currently the only supported option.

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


def convert_units(dist1, unit1, unit2, redshift=None, cosmo=None):
    """Convenience wrapper to convert between a combination of angular and physical units
    """
    angular_bank = {"radians": u.rad, "deg": u.deg, "arcmin": u.arcmin, "arcsec": u.arcsec}
    physical_bank = {"pc": u.pc, "kpc": u.kpc, "Mpc": u.Mpc}

    if not unit1 in {**angular_bank, **physical_bank}:
        raise ValueError("Input units ({}) not supported".format(unit1))
    if not unit2 in {**angular_bank, **physical_bank}:
        raise ValueError("Output units ({}) not supported".format(unit2))

    # If both input and output are angular, use angular function
    if (unit1 in angular_bank) and (unit2 in angular_bank):
        return _convert_angular_units(dist1, unit1, unit2)
    else:
        raise NotImplementedError("OMEGALUL")


    # Finish implementing. This should probably have 3 cases that are treated diff
    # 1. angular to angular(Easy)
    # 2. physical to physical (Easy)
    # 3. angular to physical (or vice versa)
    # Case 3 should should call both internal functions to convert to a standard
    # i.e. arcmin to Mpc should first convert arcmin to radians and then radians to physical





def _convert_angular_units(dist1, unit1, unit2):
    """Convert a distance measure in angular units to different angular units

    Can convert between degrees, arcmin, arcsec, and radians

    TODO: These transformations are all trivial and should just be done with
    constants from constants.py

    Parameters
    ----------
    dist1: array_like
        Input distances
    unit1: str
        Unit for the input distances
    unit2: str
        Unit for the output distances

    Returns
    -------
    dist2: array_like
        Input distances converted to unit2
    """
    # factors_to_degrees = {"radians": RAD_TO_DEG, "arcmin": ARCMIN_TO_DEG, "arcsec": ARCSEC_TO_DEG}
    factors_to_degrees = {"degrees": 1.0, "radians": 180./np.pi, "arcmin": 1./60., "arcsec": 1./3600.}
    dist_degrees = dist1 * factors_to_degrees[unit1]
    return dist_degrees / factors_to_degrees[unit2]


def convert_physical_units(dist1, unit1, unit2, redshift, cosmo):
    pass



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


