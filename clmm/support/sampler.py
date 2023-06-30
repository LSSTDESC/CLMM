"""@file.py sampler.py
Functions for sampling (output either peak or full distribution)
"""

from scipy import optimize as spo


def sciopt(model_to_shear_profile, logm_0, **kwargs):
    r"""Uses scipy optimize minimize to output the peak

    Parameters
    ----------
    model_to_shear_profile : callable
        The objective function to be minimized.
        ``model_to_shear_profile(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    logm_0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    kwargs :
        Other optional keyword arguments passed to `minimize`.
        Please check the `scipy` documentaion for information on default values and methods.

    Returns
    -------
    x : array
    The solution of the optimization

    """
    return spo.minimize(model_to_shear_profile, logm_0, **kwargs).x


def basinhopping(model_to_shear_profile, logm_0, **kwargs):
    r"""Uses basinhopping, a scipy global optimization function, to find the minimum.

    Parameters
    ----------
    model_to_shear_profile : callable
        The objective function to be minimized.
        ``model_to_shear_profile(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    logm_0 : array_like
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    kwargs :
        Other optional keyword arguments passed to `basinhopping`.
        Please check the `scipy` documentaion for information on default values and methods.

    Returns
    -------
    x : array
    The solution of the optimization

    """
    return spo.basinhopping(model_to_shear_profile, logm_0, **kwargs).x


def scicurve_fit(profile_model, radius, profile, err_profile, absolute_sigma=True, **kwargs):
    r"""
    Uses scipy.optimize.curve_fit to find best fit parameters

    Parameters
    ----------
    profile_model : callable
        The model function, f(x, ...). It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    radius : array_like or object
        The independent variable where the data is measured.
        Should usually be an M-length sequence or an (k,M)-shaped array for
        functions with k predictors, but can actually be any object.
    profile : array_like
        The dependent data, a length M array - nominally ``f(xdata, ...)``.
    err_profile : M-length sequence or MxM array
        Determines the uncertainty in `ydata`. If we define residuals as
        ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`
        depends on its number of dimensions:
        - A 1-D `sigma` should contain values of standard deviations of
        errors in `ydata`. In this case, the optimized function is
        ``chisq = sum((r / sigma) ** 2)``.
        - A 2-D `sigma` should contain the covariance matrix of
        errors in `ydata`. In this case, the optimized function is
        ``chisq = r.T @ inv(sigma) @ r``.
    absolute_sigma : bool, optional
        If True (default), `sigma` is used in an absolute sense and the estimated parameter
        covariance `pcov` reflects these absolute values.
        If False, only the relative magnitudes of the `sigma` values matter.
        The returned parameter covariance matrix `pcov` is based on scaling
        `sigma` by a constant factor. This constant is set by demanding that the
        reduced `chisq` for the optimal parameters `popt` when using the
        *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
        match the sample variance of the residuals after the fit. Default is True.
        Mathematically,
        ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``

    kwargs :
        Other optional keyword arguments passed to `curve_fit`
        Please check the `scipy` documentaion for information on default values and methods.

    Returns
    -------
    p : list
        contains :
            p[0] : array
                Optimal values for the parameters so that the sum of the squared
                residuals of ``f(xdata, *popt) - ydata`` is minimized.
            p[1] : 2-D array
                The estimated covariance of popt. The diagonals provide the variance
                of the parameter estimate. To compute one standard deviation errors
                on the parameters use ``perr = np.sqrt(np.diag(pcov))``.
                How the `sigma` parameter affects the estimated covariance
                depends on `absolute_sigma` argument, as described above.

    """
    return spo.curve_fit(
        profile_model, radius, profile, sigma=err_profile, absolute_sigma=absolute_sigma, **kwargs
    )


samplers = {"minimize": sciopt, "basinhopping": basinhopping}

fitters = {
    "curve_fit": scicurve_fit,
}
