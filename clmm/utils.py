"""General utility functions that are used in multiple modules"""
import numpy as np
from scipy.stats import binned_statistic
from astropy import units as u
from .constants import Constants as const


def compute_radial_averages(xvals, yvals, xbins, error_model='ste', weights=None):
    """ Given a list of xvals, yvals and bins, sort into bins. If xvals or yvals
    contain non-finite values, these are filtered.

    Parameters
    ----------
    xvals : array_like
        Values to be binned
    yvals : array_like
        Values to compute statistics on
    xbins: array_like
        Bin edges to sort into
    error_model : str, optional
        Error model to use for y uncertainties. (letter case independent)
            `ste` - Standard error [=std/sqrt(n) in unweighted computation] (Default).
            `std` - Standard deviation.
    weights: array_like, None
        Weights for averages.

    Returns
    -------
    meanx : array_like
        Mean x value in each bin
    meany : array_like
        Mean y value in each bin
    yerr : array_like
        Error on the mean y value in each bin. Specified by error_model
    num_objects : array_like
        Number of objects in each bin
    binnumber: 1-D ndarray of ints
        Indices of the bins (corresponding to `xbins`) in which each value
        of `xvals` belongs.  Same length as `yvals`.  A binnumber of `i` means the
        corresponding value is between (xbins[i-1], xbins[i]).
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
    wts[objs_in_bins] *= 1./wts_sum[binnumber[objs_in_bins]-1]
    # means
    meanx = binned_statistic(x, x*wts, statistic='sum', bins=xbins)[0]
    meany = binned_statistic(x, y*wts, statistic='sum', bins=xbins)[0]
    # errors
    meany2 = binned_statistic(x, y**2*wts, statistic='sum', bins=xbins)[0]
    yerr = np.sqrt(meany2-meany**2)
    if error_model == 'std':
        pass
    elif error_model == 'ste':
        mean_wts2 = binned_statistic(x, wts**2, statistic='sum', bins=xbins)[0]
        yerr *= np.sqrt(mean_wts2)
    else:
        raise ValueError(f"{error_model} not supported err model for binned stats")
    # number of objects
    num_objects = np.histogram(x, xbins)[0]
    return meanx, meany, yerr, num_objects, binnumber


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
        Binning method to use (letter case independent)
            `evenwidth` - Default, evenly spaced bins between rmin and rmax
            `evenlog10width` - Logspaced bins with even width in log10 between rmin and rmax
            `equaloccupation` - Bins with equal occupation numbers
    source_seps : array_like
        Radial distance of source separations

    Returns
    -------
    binedges: array_like, float
        n_bins+1 dimensional array that defines bin edges
    """
    # make case independent
    method = method.lower()
    # Check consistency
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


def convert_units(dist1, unit1, unit2, redshift=None, cosmo=None):
    """ Convenience wrapper to convert between a combination of angular and physical units.

    Supported units: radians, degrees, arcmin, arcsec, Mpc, kpc, pc
    (letter case independent)

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
    # make case independent
    unit1, unit2 = unit1.lower(), unit2.lower()
    # Available units
    angular_bank = {"radians": u.rad, "degrees": u.deg, "arcmin": u.arcmin, "arcsec": u.arcsec}
    physical_bank = {"pc": u.pc, "kpc": u.kpc, "mpc": u.Mpc}
    units_bank = {**angular_bank, **physical_bank}
    # Some error checking
    if unit1 not in units_bank:
        raise ValueError(f"Input units ({unit1}) not supported")
    if unit2 not in units_bank:
        raise ValueError(f"Output units ({unit2}) not supported")
    # Try automated astropy unit conversion
    try:
        return (dist1*units_bank[unit1]).to(units_bank[unit2]).value
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
            dist1_rad = (dist1*units_bank[unit1]).to(u.rad).value
            dist1_mpc = cosmo.rad2mpc(dist1_rad, redshift)
            return (dist1_mpc*u.Mpc).to(units_bank[unit2]).value
        # Otherwise physical to angular
        else:
            dist1_mpc = (dist1*units_bank[unit1]).to(u.Mpc).value
            dist1_rad = cosmo.mpc2rad(dist1_mpc, redshift)
            return (dist1_rad*u.rad).to(units_bank[unit2]).value


def convert_shapes_to_epsilon(shape_1,shape_2, shape_definition='epsilon',kappa=0):
    r""" Convert shape components 1 and 2 appropriately to make them estimators of the reduced shear once averaged.
    The shape 1 and 2 components may correspond to ellipticities according the :math:`\epsilon`- or :math:`\chi`-definition,
    but also to the 1 and 2 components of the shear. See Bartelmann & Schneider 2001 for details (https://arxiv.org/pdf/astro-ph/9912508.pdf).

    The :math:`\epsilon`-ellipticity is a direct estimator of
    the reduced shear. The shear :math:`\gamma` may be converted to reduced shear :math:`g` if the convergence :math:`\kappa` is known.
    The conversions are given below.

    .. math::
     \epsilon = \frac{\chi}{1+(1-|\chi|^2)^{1/2}}

    .. math::
     g=\frac{\gamma}{1-\kappa}

    - If `shape_definition = 'chi'`, this function returns the corresponding `epsilon` ellipticities

    - If `shape_definition = 'shear'`, it returns the corresponding reduced shear, given the convergence `kappa`

    - If `shape_definition = 'epsilon'` or `'reduced_shear'`, it returns them as is as no conversion is needed.



    Parameters
    ----------
    shape_1 : array_like
        Input shapes or shears along principal axis (g1 or e1)
    shape_2 : array_like
        Input shapes or shears along secondary axis (g2 or e2)
    shape_definition : str
        Definition of the input shapes, can be ellipticities 'epsilon' or 'chi' or shears 'shear' or 'reduced_shear'
    kappa : array_like
        Convergence for transforming to a reduced shear. Default is 0

    Returns
    -------
    epsilon_1 : array_like
        Epsilon ellipticity (or reduced shear) along principal axis (epsilon1)
    epsilon_2 : array_like
        Epsilon ellipticity (or reduced shear) along secondary axis (epsilon2)
    """

    if shape_definition=='epsilon' or shape_definition=='reduced_shear':
        return shape_1,shape_2
    elif shape_definition=='chi':
        chi_to_eps_conversion = 1./(1.+(1-(shape_1**2+shape_2**2))**0.5)
        return shape_1*chi_to_eps_conversion,shape_2*chi_to_eps_conversion
    elif shape_definition=='shear':
        return shape_1/(1.-kappa), shape_2/(1.-kappa)

    else:
        raise TypeError("Please choose epsilon, chi, shear, reduced_shear")


def build_ellipticities(q11,q22,q12):
    """ Build ellipticties from second moments. See, e.g., Schneider et al. (2006)

    Parameters
    ----------
    q11 : float or array
        Second brightness moment tensor, component (1,1)
    q22 : float or array
        Second brightness moment tensor, component (2,2)
    q12 :  float or array
        Second brightness moment tensor, component (1,2)

    Returns
    -------
    x1, x2 : float or array
        Ellipticities using the "chi definition"
    e1, e2 : float or array
        Ellipticities using the "epsilon definition"
    """

    x1,x2 = (q11-q22)/(q11+q22),(2*q12)/(q11+q22)
    e1,e2 = (q11-q22)/(q11+q22+2*np.sqrt(q11*q22-q12*q12)),(2*q12)/(q11+q22+2*np.sqrt(q11*q22-q12*q12))
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
    ----------
    ellipticity1_true : float or array
        Intrinsic ellipticity of the sources along the principal axis
    ellipticity2_true : float or array
        Intrinsic ellipticity of the sources along the second axis
    shear1 :  float or array
        Shear component (not reduced shear) along the principal axis at the source location
    shear2 :  float or array
        Shear component (not reduced shear) along the 45-degree axis at the source location
    convergence :  float or array
        Convergence at the source location
    Returns
    -------
    e1, e2 : float or array
        Lensed ellipicity along both reference axes.
    """

    shear = shear1+shear2*1j # shear (as a complex number)
    ellipticity_true = ellipticity1_true+ellipticity2_true*1j # intrinsic ellipticity (as a complex number)
    reduced_shear = shear/(1.0-convergence) # reduced shear
    e = (ellipticity_true+reduced_shear)/(1.0+reduced_shear.conjugate()*ellipticity_true) # lensed ellipticity
    return np.real(e), np.imag(e)


def arguments_consistency(arguments, names=None, prefix=''):
    r"""Make sure all arguments have the same length (or are scalars)

    Parameters
    ----------
    arguments: list, arrays, tuple
        Group of arguments to be checked
    names: list, tuple
        Names for each array, optional
    prefix: str
        Customized prefix for error message

    Returns
    -------
    list, arrays, tuple
        Group of arguments, converted to numpy arrays if they have length
    """
    sizes = [len(arg) if hasattr(arg, '__len__') else None for arg in arguments]
    # check there is a name for each argument
    if names:
        if len(names)!=len(arguments):
            raise TypeError(f'names (len={len(names)}) must have same length as arguments (len={len(arguments)})')
        msg = ', '.join([f'{n}({s})' for n, s in zip(names, sizes)])
    else:
        msg = ', '.join([f'{s}' for s in sizes])
    # check consistency
    if any(sizes):
        if not all(sizes) or any([s!=sizes[0] for s in sizes[1:]]): # Check that all of the inputs have length and they match
            # make error message
            raise TypeError(f'{prefix} inconsistent sizes: {msg}')
        return tuple(np.array(arg) for arg in (arguments))
    return arguments


def _patch_rho_crit_to_cd2018(rho_crit_external):
    r""" Convertion factor for rho_crit of any external modult to
    CODATA 2018+IAU 2015

    rho_crit_external: float
        Critical density of the Universe in units of :math:`M_\odot\ Mpc^{-3}`
    """

    rhocrit_mks = 3.0*100.0*100.0/(8.0*np.pi*const.GNEWT.value)
    rhocrit_cd2018 = rhocrit_mks*1000.0*1000.0*const.PC_TO_METER.value*1.0e6/const.SOLAR_MASS.value

    return rhocrit_cd2018/rho_crit_external
