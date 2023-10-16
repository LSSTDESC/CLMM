"""General utility functions that are used in multiple modules"""
import warnings
import numpy as np
from scipy.stats import binned_statistic
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


def compute_weighted_bin_sum(xvals, yvals, xbins, weights):
    """Add `yvals * weights` in `xbins`.

    Parameters
    ----------
    xvals : array_like
        Values to be binned
    yvals : array_like
        Values to compute statistics on
    xbins: array_like
        Bin edges to sort into
    weights: array_like, None, optional
        Weights for sum.

    Returns
    -------
    numpy.ndarray
        Sum of `yvals * weights` in `xbins`.
    """
    return binned_statistic(xvals, yvals * weights, statistic="sum", bins=xbins)[0]


def compute_radial_averages(xvals, yvals, xbins, yerr=None, error_model="ste", weights=None):
    """Given a list of `xvals`, `yvals` and `xbins`, sort into bins. If `xvals` or `yvals`
    contain non-finite values, these are filtered.

    Parameters
    ----------
    xvals : array_like
        Values to be binned
    yvals : array_like
        Values to compute statistics on
    xbins: array_like
        Bin edges to sort into
    yerr : array_like, None, optional
        Errors of `yvals`. Default: None
    error_model : str, optional
        Statistical error model to use for y uncertainties. (letter case independent)

            * 'ste' - Standard error [=std/sqrt(n) in unweighted computation] (Default).
            * 'std' - Standard deviation.

    weights: array_like, None, optional
        Weights for averages. Default: None


    Returns
    -------
    mean_x : numpy.ndarray
        Mean x value in each bin
    mean_y : numpy.ndarray
        Mean y value in each bin
    err_y: numpy.ndarray
        Error on the mean y value in each bin. Specified by `error_model`
    num_objects : numpy.ndarray
        Number of objects in each bin
    binnumber: 1-D ndarray of ints
        Indices of the bins (corresponding to `xbins`) in which each value
        of `xvals` belongs.  Same length as `yvals`.  A binnumber of `i` means the
        corresponding value is between (xbins[i-1], xbins[i]).
    wts_sum: numpy.ndarray
        Sum of individual weights in each bin.
    """
    # make case independent
    error_model = error_model.lower()
    # binned_statics throus an error in case of non-finite values, so filtering those out
    filt = np.isfinite(xvals) * np.isfinite(yvals)
    xfilt, yfilt = np.array(xvals)[filt], np.array(yvals)[filt]
    # normalize weights (and computers binnumber)
    wts = np.ones(xfilt.size) if weights is None else np.array(weights, dtype=float)[filt]
    wts_sum, binnumber = binned_statistic(xfilt, wts, statistic="sum", bins=xbins)[:3:2]
    objs_in_bins = (binnumber > 0) * (binnumber <= wts_sum.size)  # mask for binnumber in range
    wts[objs_in_bins] *= 1.0 / wts_sum[binnumber[objs_in_bins] - 1]  # norm weights in each bin
    # means
    mean_x = compute_weighted_bin_sum(xfilt, xfilt, xbins, wts)
    mean_y = compute_weighted_bin_sum(xfilt, yfilt, xbins, wts)
    # errors
    data_yerr2 = (
        0
        if yerr is None
        else compute_weighted_bin_sum(xfilt, np.array(yerr)[filt] ** 2, xbins, wts**2)
    )
    stat_yerr2 = compute_weighted_bin_sum(xfilt, yfilt**2, xbins, wts) - mean_y**2
    if error_model == "ste":
        # sum(wts^2)=1/n for not weighted
        stat_yerr2 *= compute_weighted_bin_sum(xfilt, wts, xbins, wts)
    elif error_model != "std":
        raise ValueError(f"{error_model} not supported err model for binned stats")
    err_y = np.sqrt(stat_yerr2 + data_yerr2)
    # number of objects
    num_objects = np.histogram(xfilt, xbins)[0]
    return mean_x, mean_y, err_y, num_objects, binnumber, wts_sum


def make_bins(rmin, rmax, nbins=10, method="evenwidth", source_seps=None):
    """Define bin edges

    Parameters
    ----------
    rmin : float, None
        Minimum bin edges wanted. If None, min(`source_seps`) is used.
    rmax : float, None
        Maximum bin edges wanted. If None, max(`source_seps`) is used.
    nbins : float, optional
        Number of bins you want to create, default to 10.
    method : str, optional
        Binning method to use (letter case independent):

            * 'evenwidth' - Default, evenly spaced bins between `rmin` and `rmax`
            * 'evenlog10width' - Logspaced bins with even width in log10 between `rmin` and `rmax`
            * 'equaloccupation' - Bins with equal occupation numbers

    source_seps : array_like, None, optional
        Radial distance of source separations. Needed if `method='equaloccupation'`. Default: None

    Returns
    -------
    binedges: numpy.ndarray
        array with `nbins` +1 elements that defines bin edges
    """
    # make case independent
    method = method.lower()
    if method == "equaloccupation":
        if source_seps is None:
            raise ValueError(f"Binning method '{method}' requires source separations array")
        seps = np.array(source_seps)
        rmin = seps.min() if rmin is None else rmin
        rmax = seps.max() if rmax is None else rmax
        # Need to filter source_seps to only keep galaxies in the [rmin, rmax] with a mask
        mask = (seps >= rmin) * (seps <= rmax)
        binedges = np.percentile(seps[mask], tuple(np.linspace(0, 100, nbins + 1, endpoint=True)))
    else:
        # Check consistency
        if (rmin > rmax) or (rmin < 0.0) or (rmax < 0.0):
            raise ValueError(f"Invalid bin endpoints in make_bins, {rmin} {rmax}")
        if (nbins <= 0) or not isinstance(nbins, int):
            raise ValueError(f"Invalid nbins={nbins}. Must be integer greater than 0.")

        if method == "evenwidth":
            binedges = np.linspace(rmin, rmax, nbins + 1, endpoint=True)
        elif method == "evenlog10width":
            binedges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1, endpoint=True)
        else:
            raise ValueError(f"Binning method '{method}' is not currently supported")

    return binedges


def _draw_random_points_from_distribution(xmin, xmax, nobj, dist_func, xstep=0.001):
    """Draw random points with a given distribution.

    Uses a sampling technique found in Numerical Recipes in C, Chap 7.2: Transformation Method.

    Parameters
    ----------
    xmin : float
        The minimum source redshift allowed.
    xmax : float, optional
        If source redshifts are drawn, the maximum source redshift
    nobj : float
        Number of galaxies to generate
    dist_func : function
        Function of the required distribution
    xstep : float, optional
        Size of the step to interpolate the culmulative distribution. Default: 0.001

    Returns
    -------
    numpy.ndarray
        Random points with dist_func distribution
    """
    steps = int((xmax - xmin) / xstep) + 1
    xdomain = np.linspace(xmin, xmax, steps)
    # Cumulative probability function of the redshift distribution
    # probdist = np.vectorize(lambda zmax: integrate.quad(dist_func, xmin, zmax)[0])(xdomain)
    probdist = dist_func(xdomain, is_cdf=True) - dist_func(xmin, is_cdf=True)
    # Get random values for probdist
    uniform_deviate = np.random.uniform(probdist.min(), probdist.max(), nobj)
    return interp1d(probdist, xdomain, kind="linear")(uniform_deviate)


def _draw_random_points_from_tab_distribution(x_tab, pdf_tab, nobj=1, xmin=None, xmax=None):
    """Draw random points from a tabulated distribution.

    Parameters
    ----------
    x_tab : array-like
        Values for which the tabulated pdf is provided
    pdf_tab : array-like
        Value of the pdf at the x_tab locations
    nobj : int, optional
        Number of random samples to generate. Default is 1.
    xmin : float, optional
        Lower bound to draw redshift. Default is the min(x_tab)
    xmax : float, optional
        Upper bound to draw redshift. Default is the max(x_tab)

    Returns
    -------
    samples : numpy.ndarray
        Random points following the pdf_tab distribution
    """
    x_tab = np.array(x_tab)
    pdf_tab = np.array(pdf_tab)
    # cdf = np.array([simps(pdf_tab[:j], x_tab[:j]) for j in range(1, len(x_tab)+1)])
    cdf = cumulative_trapezoid(pdf_tab, x_tab, initial=0)
    # Normalise it
    cdf /= cdf.max()
    cdf_xmin, cdf_xmax = 0.0, 1.0
    # Interpolate cdf at xmin and xmax
    if xmin or xmax:
        cdf_interp = interp1d(x_tab, cdf, kind="linear")
        if xmin is not None:
            if xmin < x_tab.min():
                warnings.warn(
                    "`xmin` is less than the minimum value of `x_tab`. "
                    + f"Using min(x_tab)={x_tab.min()} instead."
                )
            else:
                cdf_xmin = cdf_interp(xmin)
        if xmax is not None:
            if xmax > x_tab.max():
                warnings.warn(
                    "`xmax` is greater than the maximum value of `x_tab`. "
                    + f"Using max(x_tab)={x_tab.max()} instead."
                )
            else:
                cdf_xmax = cdf_interp(xmax)
    # Interpolate the inverse CDF
    inv_cdf = interp1d(cdf, x_tab, kind="linear", bounds_error=False, fill_value=0.0)
    # Finally generate sample from uniform distribution and
    # get the corresponding samples
    samples = inv_cdf(np.random.random(nobj) * (cdf_xmax - cdf_xmin) + cdf_xmin)
    return samples


def gaussian(value, mean, scatter):
    """Normal distribution.

    Parameters
    ----------
    value : array-like
        Values for which to evaluate gaussian.
    mean : float
        Mean value of normal distribution
    scatter : float
        Scatter of normal distribution

    Returns
    -------
    numpy.ndarray
        Gaussian values at `value`
    """
    return np.exp(-0.5 * (value - mean) ** 2 / scatter**2) / np.sqrt(2 * np.pi * scatter**2)
