"""General utility functions that are used in multiple modules"""
import warnings
import numpy as np
from astropy import units as u
from scipy.stats import binned_statistic
from scipy.integrate import quad, cumulative_trapezoid, simps
from scipy.interpolate import interp1d
from .constants import Constants as const
from . import z_distributions as zdist


def compute_nfw_boost(rvals, rs=1000, b0=0.1) :
    """ Given a list of `rvals`, and optional `rs` and `b0`, return the corresponding boost factor
    at each rval

    Parameters
    ----------
    rvals : array_like
        radii
    rs : float, optional
        scale radius for NFW in same units as rvals (default 2000 kpc)
    b0 : float, optional

    Returns
    -------
    boost_factors : numpy.ndarray

    """

    x = np.array(rvals)/rs

    def _calc_finternal(x) :

        radicand = x**2-1

        finternal = (-1j*np.log((1+np.lib.scimath.sqrt(radicand)*1j)
                                /(1-np.lib.scimath.sqrt(radicand)*1j))
                     /(2*np.lib.scimath.sqrt(radicand)))

        return np.nan_to_num(finternal, copy=False, nan=1.0).real

    return 1.+b0 * (1-_calc_finternal(x)) / (x**2-1)


def compute_powerlaw_boost(rvals, rs=1000, b0=0.1, alpha=-1.0) :
    """  Given a list of `rvals`, and optional `rs` and `b0`, and `alpha`,
    return the corresponding boost factor at each `rval`

    Parameters
    ----------
    rvals : array_like
        radii
    rs : float, optional
        scale radius for NFW in same units as rvals (default 2000 kpc)
    b0 : float, optional
        Default: 0.1
    alpha : float, optional
        exponent from Melchior+16. Default: -1.0

    Returns
    -------
    boost_factors : numpy.ndarray

    """

    x = np.array(rvals)/rs

    return 1. + b0 * (x)**alpha


boost_models = {'nfw_boost': compute_nfw_boost,
                'powerlaw_boost': compute_powerlaw_boost}

def correct_sigma_with_boost_values(sigma_vals, boost_factors):
    """ Given a list of boost values and sigma profile, compute corrected sigma

    Parameters
    ----------
    sigma_vals : array_like
        uncorrected sigma with cluster member dilution
    boost_factors : array_like
        Boost values pre-computed

    Returns
    -------
    sigma_corrected : numpy.ndarray
        correted radial profile
    """

    sigma_corrected = np.array(sigma_vals) / np.array(boost_factors)
    return sigma_corrected


def correct_sigma_with_boost_model(rvals, sigma_vals, boost_model='nfw_boost', **boost_model_kw):
    """ Given a boost model and sigma profile, compute corrected sigma

    Parameters
    ----------
    rvals : array_like
        radii
    sigma_vals : array_like
        uncorrected sigma with cluster member dilution
    boost_model : str, optional
        Boost model to use for correcting sigma

            * 'nfw_boost' - NFW profile model (Default)
            * 'powerlaw_boost' - Powerlaw profile

    Returns
    -------
    sigma_corrected : numpy.ndarray
        correted radial profile
    """
    boost_model_func = boost_models[boost_model]
    boost_factors = boost_model_func(rvals, **boost_model_kw)

    sigma_corrected = np.array(sigma_vals) / boost_factors
    return sigma_corrected

def compute_radial_averages(xvals, yvals, xbins, yerr=None, error_model='ste', weights=None):
    """ Given a list of `xvals`, `yvals` and `xbins`, sort into bins. If `xvals` or `yvals`
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
    filt = np.isfinite(xvals)*np.isfinite(yvals)
    x, y = np.array(xvals)[filt], np.array(yvals)[filt]
    # normalize weights (and computers binnumber)
    wts = np.ones(x.size) if weights is None else np.array(weights, dtype=float)[filt]
    wts_sum, binnumber = binned_statistic(x, wts, statistic='sum', bins=xbins)[:3:2]
    objs_in_bins = (binnumber>0)*(binnumber<=wts_sum.size) # mask for binnumber in range
    wts[objs_in_bins] *= 1./wts_sum[binnumber[objs_in_bins]-1] # norm weights in each bin
    weighted_bin_stat = lambda vals: binned_statistic(x, vals*wts, statistic='sum', bins=xbins)[0]
    # means
    mean_x = weighted_bin_stat(x)
    mean_y = weighted_bin_stat(y)
    # errors
    data_yerr2 = 0 if yerr is None else weighted_bin_stat(np.array(yerr)[filt]**2*wts)
    stat_yerr2 = weighted_bin_stat(y**2)-mean_y**2
    if error_model == 'ste':
        stat_yerr2 *= weighted_bin_stat(wts) # sum(wts^2)=1/n for not weighted
    elif error_model != 'std':
        raise ValueError(f"{error_model} not supported err model for binned stats")
    err_y = np.sqrt(stat_yerr2+data_yerr2)
    # number of objects
    num_objects = np.histogram(x, xbins)[0]
    return mean_x, mean_y, err_y, num_objects, binnumber, wts_sum


def make_bins(rmin, rmax, nbins=10, method='evenwidth', source_seps=None):
    """ Define bin edges

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
    if method == 'equaloccupation':
        if source_seps is None:
            raise ValueError(
                f"Binning method '{method}' requires source separations array")
        seps = np.array(source_seps)
        rmin = seps.min() if rmin is None else rmin
        rmax = seps.max() if rmax is None else rmax
        # Need to filter source_seps to only keep galaxies in the [rmin, rmax] with a mask
        mask = (seps >= rmin)*(seps <= rmax)
        binedges = np.percentile(seps[mask], tuple(
        np.linspace(0, 100, nbins+1, endpoint=True)))
    else:
        # Check consistency
        if (rmin > rmax) or (rmin < 0.0) or (rmax < 0.0):
            raise ValueError(f"Invalid bin endpoints in make_bins, {rmin} {rmax}")
        if (nbins <= 0) or not isinstance(nbins, int):
            raise ValueError(
                f"Invalid nbins={nbins}. Must be integer greater than 0.")

        if method == 'evenwidth':
            binedges = np.linspace(rmin, rmax, nbins+1, endpoint=True)
        elif method == 'evenlog10width':
            binedges = np.logspace(np.log10(rmin), np.log10(
                rmax), nbins+1, endpoint=True)
        else:
            raise ValueError(
                f"Binning method '{method}' is not currently supported")

    return binedges


def convert_units(dist1, unit1, unit2, redshift=None, cosmo=None):
    """ Convenience wrapper to convert between a combination of angular and physical units.

    Supported units: radians, degrees, arcmin, arcsec, Mpc, kpc, pc
    (letter case independent)

    To convert between angular and physical units you must provide both
    `redshift` and a CLMM Cosmology object `cosmo`.

    Parameters
    ----------
    dist1 : float, array_like
        Input distances
    unit1 : str
        Unit for the input distances
    unit2 : str
        Unit for the output distances
    redshift : float, None, optional
        Redshift used to convert between angular and physical units. Default: None
    cosmo : clmm.Cosmology, None, optional
        CLMM Cosmology object to compute angular diameter distance to
        convert between physical and angular units. Default: None

    Returns
    -------
    dist2: float, numpy.ndarray
        Input distances converted to unit2
    """
    # make case independent
    unit1, unit2 = unit1.lower(), unit2.lower()
    # Available units
    angular_bank = {"radians": u.rad, "degrees": u.deg,
                    "arcmin": u.arcmin, "arcsec": u.arcsec}
    physical_bank = {"pc": u.pc, "kpc": u.kpc, "mpc": u.Mpc}
    units_bank = {**angular_bank, **physical_bank}
    # Some error checking
    if unit1 not in units_bank:
        raise ValueError(f"Input units ({unit1}) not supported")
    if unit2 not in units_bank:
        raise ValueError(f"Output units ({unit2}) not supported")
    # Try automated astropy unit conversion
    try:
        dist2 = (dist1*units_bank[unit1]).to(units_bank[unit2]).value
    # Otherwise do manual conversion
    except u.UnitConversionError:
        # Make sure that we were passed a redshift and cosmology
        if redshift is None or cosmo is None:
            raise TypeError(
                "Redshift and cosmology must be specified to convert units") \
                from u.UnitConversionError
        # Redshift must be greater than zero for this approx
        if not redshift > 0.0:
            raise ValueError("Redshift must be greater than 0.") from u.UnitConversionError
        # Convert angular to physical
        if (unit1 in angular_bank) and (unit2 in physical_bank):
            dist1_rad = (dist1*units_bank[unit1]).to(u.rad).value
            dist1_mpc = cosmo.rad2mpc(dist1_rad, redshift)
            dist2 = (dist1_mpc*u.Mpc).to(units_bank[unit2]).value
        # Otherwise physical to angular
        else:
            dist1_mpc = (dist1*units_bank[unit1]).to(u.Mpc).value
            dist1_rad = cosmo.mpc2rad(dist1_mpc, redshift)
            dist2 = (dist1_rad*u.rad).to(units_bank[unit2]).value
    return dist2


def convert_shapes_to_epsilon(shape_1, shape_2, shape_definition='epsilon', kappa=0):
    r""" Convert shape components 1 and 2 appropriately to make them estimators of the
    reduced shear once averaged.  The shape 1 and 2 components may correspond to ellipticities
    according the :math:`\epsilon`- or :math:`\chi`-definition, but also to the 1 and 2 components
    of the shear. See Bartelmann & Schneider 2001 for details
    (https://arxiv.org/pdf/astro-ph/9912508.pdf).

    The :math:`\epsilon`-ellipticity is a direct estimator of
    the reduced shear. The shear :math:`\gamma` may be converted to reduced shear :math:`g` if the
    convergence :math:`\kappa` is known. The conversions are given below.

    .. math::
     \epsilon = \frac{\chi}{1+(1-|\chi|^2)^{1/2}}

    .. math::
     g=\frac{\gamma}{1-\kappa}

    - If `shape_definition = 'chi'`, this function returns the corresponding `epsilon`
      ellipticities

    - If `shape_definition = 'shear'`, it returns the corresponding reduced shear, given the
      convergence `kappa`

    - If `shape_definition = 'epsilon'` or `'reduced_shear'`, it returns them as is as no
      conversion is needed.

    Parameters
    ----------
    shape_1 : float, numpy.ndarray
        Input shapes or shears along principal axis (g1 or e1)
    shape_2 : float, numpy.ndarray
        Input shapes or shears along secondary axis (g2 or e2)
    shape_definition : str, optional
        Definition of the input shapes, can be ellipticities 'epsilon' or 'chi' or shears 'shear'
        or 'reduced_shear'. Defaut: 'epsilon'
    kappa : float, numpy.ndarray, optional
        Convergence for transforming to a reduced shear. Default is 0

    Returns
    -------
    epsilon_1 : float, numpy.ndarray
        Epsilon ellipticity (or reduced shear) along principal axis (epsilon1)
    epsilon_2 : float, numpy.ndarray
        Epsilon ellipticity (or reduced shear) along secondary axis (epsilon2)
    """

    if shape_definition in ('epsilon', 'reduced_shear'):
        epsilon_1, epsilon_2 = shape_1, shape_2
    elif shape_definition == 'chi':
        chi_to_eps_conversion = 1./(1.+(1-(shape_1**2+shape_2**2))**0.5)
        epsilon_1, epsilon_2 = shape_1*chi_to_eps_conversion, shape_2*chi_to_eps_conversion
    elif shape_definition == 'shear':
        epsilon_1, epsilon_2 = shape_1/(1.-kappa), shape_2/(1.-kappa)
    else:
        raise TypeError("Please choose epsilon, chi, shear, reduced_shear")
    return epsilon_1, epsilon_2


def build_ellipticities(q11, q22, q12):
    """ Build ellipticties from second moments. See, e.g., Schneider et al. (2006)

    Parameters
    ----------
    q11 : float, numpy.ndarray
        Second brightness moment tensor, component (1,1)
    q22 : float, numpy.ndarray
        Second brightness moment tensor, component (2,2)
    q12 :  float, numpy.ndarray
        Second brightness moment tensor, component (1,2)

    Returns
    -------
    chi1, chi2 : float, numpy.ndarray
        Ellipticities using the "chi definition"
    epsilon1, epsilon2 : float, numpy.ndarray
        Ellipticities using the "epsilon definition"
    """
    norm_x, norm_e = q11+q22, q11+q22+2*np.sqrt(q11*q22-q12*q12)
    chi1, chi2 = (q11-q22)/norm_x, 2*q12/norm_x
    epsilon1, epsilon2 = (q11-q22)/norm_e, 2*q12/norm_e
    return chi1, chi2, epsilon1, epsilon2


def compute_lensed_ellipticity(ellipticity1_true, ellipticity2_true, shear1, shear2, convergence):
    r""" Compute lensed ellipticities from the intrinsic ellipticities, shear and convergence.
    Following Schneider et al. (2006)

    .. math::
        \epsilon^{\text{lensed}}=\epsilon^{\text{lensed}}_1+i\epsilon^{\text{lensed}}_2=
        \frac{\epsilon^{\text{true}}+g}{1+g^\ast\epsilon^{\text{true}}},

    where, the complex reduced shear :math:`g` is obtained from the shear
    :math:`\gamma=\gamma_1+i\gamma_2` and convergence :math:`\kappa` as :math:`g =
    \gamma/(1-\kappa)`, and the complex intrinsic ellipticity is :math:`\epsilon^{\text{
    true}}=\epsilon^{\text{true}}_1+i\epsilon^{\text{true}}_2`

    Parameters
    ----------
    ellipticity1_true : float, numpy.ndarray
        Intrinsic ellipticity of the sources along the principal axis
    ellipticity2_true : float, numpy.ndarray
        Intrinsic ellipticity of the sources along the second axis
    shear1 :  float, numpy.ndarray
        Shear component (not reduced shear) along the principal axis at the source location
    shear2 :  float, numpy.ndarray
        Shear component (not reduced shear) along the 45-degree axis at the source location
    convergence :  float, numpy.ndarray
        Convergence at the source location
    Returns
    -------
    e1, e2 : float, numpy.ndarray
        Lensed ellipicity along both reference axes.
    """
    # shear (as a complex number)
    shear = shear1+shear2*1j
    # intrinsic ellipticity (as a complex number)
    ellipticity_true = ellipticity1_true+ellipticity2_true*1j
    # reduced shear
    reduced_shear = shear/(1.0-convergence)
    # lensed ellipticity
    lensed_ellipticity = (ellipticity_true+reduced_shear) / \
        (1.0+reduced_shear.conjugate()*ellipticity_true)
    return np.real(lensed_ellipticity), np.imag(lensed_ellipticity)


def arguments_consistency(arguments, names=None, prefix=''):
    r"""Make sure all arguments have the same length (or are scalars)

    Parameters
    ----------
    arguments: list, arrays, tuple
        Group of arguments to be checked
    names: list, tuple, None, optional
        Names for each array. Default: None
    prefix: str, optional
        Customized prefix for error message. Default: ''

    Returns
    -------
    list, arrays, tuple
        Group of arguments, converted to numpy arrays if they have length
    """
    sizes = [len(arg) if hasattr(arg, '__len__')
             else None for arg in arguments]
    # check there is a name for each argument
    if names:
        if len(names) != len(arguments):
            raise TypeError(
                f'names (len={len(names)}) must have same length '
                f'as arguments (len={len(arguments)})')
        msg = ', '.join([f'{n}({s})' for n, s in zip(names, sizes)])
    else:
        msg = ', '.join([f'{s}' for s in sizes])
    # check consistency
    if any(sizes):
        # Check that all of the inputs have length and they match
        if not all(sizes) or any([s != sizes[0] for s in sizes[1:]]):
            # make error message
            raise TypeError(f'{prefix} inconsistent sizes: {msg}')
        return tuple(np.array(arg) for arg in arguments)
    return arguments


def _patch_rho_crit_to_cd2018(rho_crit_external):
    r""" Convertion factor for rho_crit of any external modult to
    CODATA 2018+IAU 2015

    rho_crit_external: float
        Critical density of the Universe in units of :math:`M_\odot\ Mpc^{-3}`
    """

    rhocrit_mks = 3.0*100.0*100.0/(8.0*np.pi*const.GNEWT.value)
    rhocrit_cd2018 = (rhocrit_mks*1000.0*1000.0*
        const.PC_TO_METER.value*1.0e6/const.SOLAR_MASS.value)

    return rhocrit_cd2018/rho_crit_external

_valid_types = {
    float: (float, int, np.floating, np.integer),
    int: (int, np.integer),
    'float_array': (float, int, np.floating, np.integer),
    'int_array': (int, np.integer),
    'array': (list, tuple, np.ndarray),
    }

def _is_valid(arg, valid_type):
    r"""Check if argument is of valid type, supports arrays.

    Parameters
    ----------
    arg: any
        Argument to be tested.
    valid_type: str, type
        Valid types for argument, options are object types, list/tuple of types, or:

            * 'int_array' - interger, interger array
            * 'float_array' - float, float array

    Returns
    -------
    valid: bool
        Is argument valid
    """
    if valid_type=='function':
        return callable(arg)
    return (isinstance(arg[0], _valid_types[valid_type])
                if (valid_type in ('int_array', 'float_array') and np.iterable(arg))
                else isinstance(arg, _valid_types.get(valid_type, valid_type)))


def validate_argument(loc, argname, valid_type, none_ok=False, argmin=None, argmax=None,
                      eqmin=False, eqmax=False, shape=None):
    r"""Validate argument type and raise errors.

    Parameters
    ----------
    loc: dict
        Dictionary with all input arguments. Should be locals().
    argname: str
        Name of argument to be tested.
    valid_type: str, type
        Valid types for argument, options are object types, list/tuple of types, or:

            * 'int_array' - interger, interger array
            * 'float_array' - float, float array

    none_ok: bool, optional
        If True, accepts None as a valid type. Default: False
    argmin : int, float, None, optional
        Minimum value allowed. Default: None
    argmax : int, float, None, optional
        Maximum value allowed. Default: None
    eqmin : bool, optional
        If True, accepts min(arg)==argmin. Default: False
    eqmax : bool, optional
        If True, accepts max(arg)==argmax. Default: False
    shape : tuple of ints, None, optional
        Shape of object allowed. Default: None
    """
    var = loc[argname]
    # Check for None
    if none_ok and (var is None):
        return
    # Check for type
    valid = (any(_is_valid(var, types) for types in valid_type)
                if isinstance(valid_type, (list, tuple))
                else _is_valid(var, valid_type))
    if not valid:
        err = f'{argname} must be {valid_type}, received {type(var).__name__}'
        raise TypeError(err)
    # Check min/max
    if any(t is not None for t in (argmin, argmax)):
        try:
            var_array = np.array(var, dtype=float)
        except:
            err = f'{argname} ({type(var).__name__}) cannot be converted to number' \
                  ' for min/max validation.'
            raise TypeError(err)
        if argmin is not None:
            if (var_array.min()<argmin if eqmin else var_array.min()<=argmin):
                err = f'{argname} must be greater than {argmin},' \
                      f' received min({argname}): {var_array.min()}'
                raise ValueError(err)
        if argmax is not None:
            if (var_array.max()>argmax if eqmax else var_array.max()>=argmax):
                err = f'{argname} must be lesser than {argmax},' \
                      f' received max({argname}): {var_array.max()}'
                raise ValueError(err)
    # Check for shape
    if shape is not None :
        if np.shape(var) != shape :
            err = f'{argname} must be of shape {shape},' \
                  f'received shape({argname}): {np.shape(var)}'
            raise ValueError(err)

def _integ_pzfuncs(pzpdf, pzbins, zmin=0., zmax=5, kernel=lambda z: 1., ngrid=1000):
    r"""
    Integrates the product of a photo-z pdf with a given kernel. 
    This function was created to allow for data with different photo-z binnings.

    Parameters
    ----------
    pzpdf : list of arrays
        Photometric probablility density functions of the source galaxies.
    pzbins : list of arrays
        Redshift axis on which the individual photoz pdf is tabulated.
    zmin : float, optional
        Minimum redshift for integration. Default: 0
    zmax : float, optional
        Maximum redshift for integration. Default: 5
    kernel : function, optional
        Function to be integrated with the pdf, must be f(z_array) format.
        Default: kernel(z)=1
    ngrid : int, optional
        Number of points for the interpolation of the redshift pdf.

    Returns
    -------
    numpy.ndarray
        Kernel integrated with the pdf of each galaxy.

    Notes
    -----
        Will be replaced by qp at some point.
    """
    # adding these lines to interpolate CLMM redshift grid for each galaxies
    # to a constant redshift grid for all galaxies. If there is a constant grid for all galaxies
    # these lines are not necessary and z_grid, pz_matrix = pzbins, pzpdf

    if hasattr(pzbins[0], '__len__'):
        # First need to interpolate on a fixed grid
        z_grid = np.linspace(zmin, zmax, ngrid)
        pdf_interp_list = [interp1d(pzbin, pdf, bounds_error=False, fill_value=0.)
                           for pzbin,pdf in zip(pzbins, pzpdf)]
        pz_matrix = np.array([pdf_interp(z_grid) for pdf_interp in pdf_interp_list])
        kernel_matrix = kernel(z_grid)
    else:
        # OK perform the integration directly from the pdf binning common to all galaxies
        mask = (pzbins>=zmin)*(pzbins<=zmax)
        z_grid = pzbins[mask]
        pz_matrix = np.array(pzpdf)[:,mask]
        kernel_matrix = kernel(z_grid)

    return simps(pz_matrix*kernel_matrix, x=z_grid, axis=1)

def compute_for_good_redshifts(function, z1, z2, bad_value, warning_message,
                               z1_arg_name='z1', z2_arg_name='z2', r_proj=None,
                               **kwargs):
    """Computes function only for `z1` < `z2`, the rest is filled with `bad_value`

    Parameters
    ----------
    function: function
        Function to be executed
    z1: float, array_like
        Redshift lower
    z2: float, array_like
        Redshift higher
    bad_value: any
        Value to fill when `z1` >= `z2`
    warning_message: str
        Warning message to be displayed when `z1` >= `z2`
    z1_arg_name: str, optional
        Name of the keyword argument that `z1` is passed to. Default: 'z1'
    z2_arg_name: str, optional
        Name of the keyword argument that `z2` is passed to. Default: 'z2'
    r_proj: float, array_like, optional
        Value to be passed to keyword argument `r_proj` of `function`. Default: None

    Returns
    -------
    Return type of `function`
        Output of `function` with value for `z1` >= `z2` replaced by `bad_value`
    """
    kwargs = {z1_arg_name:locals()['z1'], z2_arg_name:locals()['z2'], **kwargs}

    z_good = np.less(z1, z2)
    if r_proj is not None:
        r_proj = np.array(r_proj)*np.full_like(z_good, True)
        z_good = z_good*r_proj.astype(bool)
        kwargs.update({'r_proj': r_proj[z_good] if np.iterable(r_proj) else r_proj})

    if not np.all(z_good):
        warnings.warn(warning_message, stacklevel=2)
        if np.iterable(z_good):
            res = np.full(z_good.shape, bad_value)
            if np.any(z_good):
                kwargs[z1_arg_name] = np.array(z1)[z_good] if np.iterable(z1) else z1
                kwargs[z2_arg_name] = np.array(z2)[z_good] if np.iterable(z2) else z2
                res[z_good] = function(**kwargs)
        else:
            res = bad_value
    else:
        res = function(**kwargs)
    return res

def compute_beta(z_src, z_cl, cosmo, z_src_info='discrete'):
    r"""Geometric lensing efficicency

    .. math::
        \beta = max(0, D_{a,\ ls}/D_{a,\ s})

    Eq.2 in https://arxiv.org/pdf/1611.03866.pdf

    Parameters
    ----------
    z_src : array_like, float, function
        Information on the background source galaxy redshift(s). Value required depends on
        `z_src_info` (see below).
    z_cl: float
        Galaxy cluster redshift
    cosmo: clmm.Cosmology
        CLMM Cosmology object
    z_src_info : str, optional
        Type of redshift information provided by the `z_src` argument.
        In this function, the only supported option is:

            * 'discrete' (default) : The redshift of sources is provided by `z_src`.
              It can be individual redshifts for each source galaxy when `z_src` is an
              array or all sources are at the same redshift when `z_src` is a float.
    Returns
    -------
    numpy array
        Geometric lensing efficicency
    """
    try:
        len(z_src)
    except TypeError:
        z_src = [z_src]
    beta = [np.heaviside(z_src_i-z_cl, 0) * cosmo.eval_da_z1z2(z_cl, z_src_i) / cosmo.eval_da(z_src_i) for z_src_i in z_src]
    return np.array(beta)

def compute_beta_s(z_src, z_cl, z_inf, cosmo, z_src_info='discrete'):
    r"""Geometric lensing efficicency ratio

    .. math::
        \beta_s = \beta(z_{src})/\beta(z_{inf})

    Parameters
    ----------
    z_src : array_like, float, function
        Information on the background source galaxy redshift(s). Value required depends on
        `z_src_info` (see below).
    z_cl: float
        Galaxy cluster redshift
    z_inf: float
        Redshift at infinity
    cosmo: clmm.Cosmology
        CLMM Cosmology object
    z_src_info : str, optional
        Type of redshift information provided by the `z_src` argument.
        In this function, the only supported option is:

            * 'discrete' (default) : The redshift of sources is provided by `z_src`.
              It can be individual redshifts for each source galaxy when `z_src` is an
              array or all sources are at the same redshift when `z_src` is a float.
    Returns
    -------
    numpy array
        Geometric lensing efficicency ratio
    """
    beta_s = compute_beta(z_src, z_cl, cosmo) / compute_beta(z_inf, z_cl, cosmo)
    return beta_s

def compute_beta_s_func(z_src, z_cl, z_inf, cosmo, z_src_info, func, *args, **kwargs):
    r"""Geometric lensing efficicency ratio times a value of a function

    .. math::
        \beta_{s}\times \text{func} = \beta_s(z_{src}, z_{cl}, z_{inf})
        \times\text{func}(*args,\ **kwargs)

    Parameters
    ----------
    z_src : array_like, float, function
        Information on the background source galaxy redshift(s). Value required depends on
        `z_src_info` (see below).    
    z_cl: float
        Galaxy cluster redshift
    z_inf: float
        Redshift at infinity
    cosmo: clmm.Cosmology
        CLMM Cosmology object
    func: callable
        A scalar function
    z_src_info : str, optional
        Type of redshift information provided by the `z_src` argument.
        In this function, the only supported option is:

            * 'discrete' (default) : The redshift of sources is provided by `z_src`.
              It can be individual redshifts for each source galaxy when `z_src` is an
              array or all sources are at the same redshift when `z_src` is a float.
    *args: positional arguments
        args to be passed to `func`
    **kwargs: keyword arguments
        kwargs to be passed to `func`

    Returns
    -------
    numpy array
        Geometric lensing efficicency ratio for each source
    """
    beta_s = compute_beta(z_src, z_cl, cosmo) / compute_beta(z_inf, z_cl, cosmo)
    beta_s_func = beta_s * func(*args, **kwargs)
    return beta_s_func

def compute_beta_s_mean_from_distribution(z_cl, z_inf, cosmo, zmax=10.0, delta_z_cut=0.1, zmin=None, z_distrib_func=None):
    r"""Mean value of the geometric lensing efficicency

    .. math::
       \left<\beta_s\right> = \frac{\int_{z = z_{min}}^{z_{max}}\beta_s(z)N(z)}
       {\int_{z = z_{min}}^{z_{max}}N(z)}

    Parameters
    ----------
    z_cl: float
        Galaxy cluster redshift
    z_inf: float
        Redshift at infinity
    cosmo: clmm.Cosmology
        CLMM Cosmology object
    zmax: float, optional
        Maximum redshift to be set as the source of the galaxy when performing the sum.
        Default: 10
    delta_z_cut: float, optional
        Redshift interval to be summed with :math:`z_{cl}` to return :math:`z_{min}`.
        This feature is not used if :math:`z_{min}` is provided by the user. Default: 0.1
    zmin: float, None, optional
        Minimum redshift to be set as the source of the galaxy when performing the sum.
        Default: None
    z_distrib_func: one-parameter function, optional
        Redshift distribution function. Default is Chang et al (2013) distribution function.

    Returns
    -------
    float
        Mean value of the geometric lensing efficicency
    """
    if z_distrib_func is None:
        z_distrib_func = zdist.chang2013
    def integrand(z_i):
        return compute_beta_s(z_i, z_cl, z_inf, cosmo) * z_distrib_func(z_i)

    if zmin is None:
        zmin = z_cl + delta_z_cut

    Bs_mean = quad(integrand, zmin, zmax)[0] / quad(z_distrib_func, zmin, zmax)[0]
    return Bs_mean

def compute_beta_s_square_mean_from_distribution(z_cl, z_inf, cosmo, zmax=10.0, delta_z_cut=0.1, zmin=None, z_distrib_func=None):
    r"""Mean square value of the geometric lensing efficicency ratio

    .. math::
       \left<\beta_s^2\right> =\frac{\int_{z = z_{min}}^{z_{max}}\beta_s^2(z)N(z)}
       {\int_{z = z_{min}}^{z_{max}}N(z)}

    Parameters
    ----------
    z_cl: float
        Galaxy cluster redshift
    z_inf: float
        Redshift at infinity
    cosmo: clmm.Cosmology
        CLMM Cosmology object
    zmax: float
        Minimum redshift to be set as the source of the galaxy\
        when performing the sum.
    delta_z_cut: float
        Redshift interval to be summed with $z_cl$ to return\
        $zmin$. This feature is not used if $z_min$ is provided by the user.
    zmin: float, None, optional
        Minimum redshift to be set as the source of the galaxy when performing the sum.
        Default: None
    z_distrib_func: one-parameter function, optional
        Redshift distribution function. Default is Chang et al (2013) distribution function.
    Returns
    -------
    float
        Mean square value of the geometric lensing efficicency ratio.
    """
    if z_distrib_func is None:
        z_distrib_func = zdist.chang2013

    def integrand(z_i):
        return compute_beta_s(z_i, z_cl, z_inf, cosmo)**2 * z_distrib_func(z_i)

    if zmin is None:
        zmin = z_cl + delta_z_cut
    
    Bs_square_mean = quad(integrand, zmin, zmax)[0] / quad(z_distrib_func, zmin, zmax)[0]
    
    return Bs_square_mean   


def compute_beta_s_mean_from_weights(z_src, z_cl, z_inf, cosmo, shape_weights, z_src_info='discrete'):
    r"""Mean square value of the geometric lensing efficicency ratio

    .. math::
       \left<\beta_s\right> =\frac{\sum_i \beta_s(z_i)w_i}
       {\sum_i w_i}

    Parameters
    ----------
    z_src: float, array_like
        Invididual source galaxies redshift.
    z_cl: float
        Galaxy cluster redshift.
    z_inf: float
        Redshift at infinity.
    cosmo: clmm.Cosmology
        CLMM Cosmology object
    shape_weights: float, array_like
        Individual source galaxies shape weights.\
        If not None, the function uses Eq.(13) from\
        https://arxiv.org/pdf/1611.03866.pdf with evenly distributed\
        weights summing to one.
    z_src_info : str, optional
        Type of redshift information provided by the `z_src` argument.
        In this function, the only supported option is:

            * 'discrete' (default) : The redshift of sources is provided by `z_src`.
              It can be individual redshifts for each source galaxy when `z_src` is an
              array or all sources are at the same redshift when `z_src` is a float.
    Returns
    -------
    float
        Mean value of the geometric lensing efficicency ratio.
    """
    try:
        if len(z_src) != len(shape_weights):
            raise ValueError("The source redshifts and the weights array must be the same size.")
    except TypeError:
        z_src = [z_src]
        shape_weights = [shape_weights]
    
    weights_sum = np.sum(shape_weights)
    shape_weights = np.array(shape_weights)
    Bsw = shape_weights * compute_beta_s(z_src, z_cl, z_inf, cosmo)
    Bs_square_mean = np.sum(Bsw) / weights_sum

    return Bs_square_mean

def compute_beta_s_square_mean_from_weights(z_src, z_cl, z_inf, cosmo, shape_weights, z_src_info='discrete'):
    r"""Mean square value of the geometric lensing efficicency ratio

    .. math::
       \left<\beta_s^2\right> =\frac{\sum_i \beta_s^2(z_i)w_i}
       {\sum_i w_i}

    Parameters
    ----------
    z_src: float, array_like
        Invididual source galaxies redshift.
    z_cl: float
        Galaxy cluster redshift.
    z_inf: float
        Redshift at infinity.
    cosmo: clmm.Cosmology
        CLMM Cosmology object
    shape_weights: float, array_like
        Individual source galaxies shape weights.
    z_src_info : str, optional
        Type of redshift information provided by the `z_src` argument.
        In this function, the only supported option is:

            * 'discrete' (default) : The redshift of sources is provided by `z_src`.
              It can be individual redshifts for each source galaxy when `z_src` is an
              array or all sources are at the same redshift when `z_src` is a float.
    Returns
    -------
    float
        Mean square value of the geometric lensing efficicency ratio.
    """
    try:
        if len(z_src) != len(shape_weights):
            raise ValueError("The source redshifts and the weights array must be the same size.")
    except TypeError:
        z_src = [z_src]
        shape_weights = [shape_weights]

    weights_sum = np.sum(shape_weights)
    shape_weights = np.array(shape_weights)
    Bsw = shape_weights * np.square(compute_beta_s(z_src, z_cl, z_inf, cosmo)) 
    Bs_square_mean = np.sum(Bsw) / weights_sum
    
    return Bs_square_mean

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
    steps = int((xmax-xmin)/xstep)+1
    xdomain = np.linspace(xmin, xmax, steps)
    # Cumulative probability function of the redshift distribution
    #probdist = np.vectorize(lambda zmax: integrate.quad(dist_func, xmin, zmax)[0])(xdomain)
    probdist = dist_func(xdomain, is_cdf=True)-dist_func(xmin, is_cdf=True)
    # Get random values for probdist
    uniform_deviate = np.random.uniform(probdist.min(), probdist.max(), nobj)
    return interp1d(probdist, xdomain, kind='linear')(uniform_deviate)

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
    #cdf = np.array([simps(pdf_tab[:j], x_tab[:j]) for j in range(1, len(x_tab)+1)])
    cdf = cumulative_trapezoid(pdf_tab, x_tab, initial=0)
    # Normalise it
    cdf /= cdf.max()
    cdf_xmin, cdf_xmax = 0.0, 1.0
    # Interpolate cdf at xmin and xmax
    if xmin or xmax:
        cdf_interp = interp1d(x_tab, cdf, kind='linear')
        if xmin is not None:
            if xmin<x_tab.min():
                warnings.warn('`xmin` is less than the minimum value of `x_tab`. '+\
                              f'Using min(x_tab)={x_tab.min()} instead.')
            else:
                cdf_xmin = cdf_interp(xmin)
        if xmax is not None:
            if xmax>x_tab.max():
                warnings.warn('`xmax` is greater than the maximum value of `x_tab`. '+\
                              f'Using max(x_tab)={x_tab.max()} instead.')
            else:
                cdf_xmax = cdf_interp(xmax)
    # Interpolate the inverse CDF
    inv_cdf = interp1d(cdf, x_tab, kind='linear', bounds_error=False, fill_value=0.)
    # Finally generate sample from uniform distribution and
    # get the corresponding samples
    samples = inv_cdf(np.random.random(nobj)*(cdf_xmax-cdf_xmin)+cdf_xmin)
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
    return np.exp(-0.5*(value-mean)**2/scatter**2)/np.sqrt(2*np.pi*scatter**2)
