"""General utility functions that are used in multiple modules"""
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table


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
    """Define the bins edges corresponding to the binning of data using user-defined method and
    parameters. 
    
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


