"""General utility functions that are used in multiple modules"""
import warnings
import numpy as np
from astropy import units as u
from scipy.stats import binned_statistic
from scipy.special import gamma, gammainc
from scipy.integrate import quad, cumulative_trapezoid, simps
from scipy.interpolate import interp1d
from .constants import Constants as const
import matplotlib.pyplot as plt

from astropy.cosmology import LambdaCDM #update


class profiles:
    """A class to generate 2D covnergence profiles on a regular grid
    """

    def __init__(self,M,c,z_l,z_s,sig=False):
        
        """Initialise the halo and lensing geometry

           Parameters
           ----------
           M: float
               halo mass
           c: float
               halo concentration
           z_l: float
               lens redshift
           z_s: flaot
               source redshift
           delta_sig: bool (optional)  
               If False, return convergence profile, if True, return delta Sigma profile
        """
        
        self.M = M
        self.c = c
        self.z_l = z_l
        self.z_s = z_s
        self.sig = sig
        
        self.cosmology()
        
        return
    
    def cosmology(self, H_0 = 70., Omega_m = 0.3, Omega_Lambda = 0.7, Tcmb0=2.725):
        
        """ Set the cosmology of the object

        Parameters
        ----------
        H_0: float
            Hubble parameter
        Omega_m: float
            Mass density parameter (baryon + dark matter)
        Omega_Lambda: float
            Effective mass density of dark energy
        Tcmb0: float
            cmb temperature at z=0
    
        """

        self.H_0 = H_0
        self.Omega_m = Omega_m
        self.Tcmb0 = Tcmb0
        self.Omega_Lambda = Omega_Lambda
        
        return
        
        
    def get_r_vals(self,nbins,lower_r,upper_r,log=False):
        """Create radial bins for profiles
        
        Parameters
        ----------
        nbins: int
            number of bins
        lower_r: float
            lowest bin value
        upper_r: float
            highest bin value
        log: bool (optional)
            return bins with linear (False) or log (True) separation

        Returns 
        -------
        r_vals: np.ndarray    
            bin edges in units provided, or kpc if none are provided
        """
        #create r values
        self.nbins = nbins
        self.lower_r = lower_r
        self.upper_r = upper_r
        self.log_r_vals = log
        
        
        if self.log_r_vals:
            if self.lower_r == 0:
                raise ValueError('log(lower_r) cannot be evaluated for lower_r = 0')
                
            r_vals = np.logspace(np.log10(self.lower_r),np.log10(self.upper_r),self.nbins+1)
            
        if not self.log_r_vals:
            r_vals = np.linspace(self.lower_r,self.upper_r,self.nbins+1)
        
        #if no units are provided, set units to kpc, otherwise keep input units
        r_vals *= u.dimensionless_unscaled
        if r_vals.unit == u.dimensionless_unscaled:
            r_vals *= u.kpc
            
        self.r_vals = r_vals
        return self.r_vals
    
    def kappa_NFW(self,r_vals=None):
        """Produce a 1D NFW lensing profile

        Parameters
        ----------
        r_vals: np.ndarray (optional) 
           radial bin values for halo lensing profile
        
        Returns
        -------
        nfw: np.ndarray
            1D NFW lensing profile
        """
        
        if np.all(r_vals.value) == None:
            r_vals = self.r_vals
        
        G = const.GNEWT.value * u.N * u.m**2 /u.kg**2
        c = const.CLIGHT.value * u.m / u.s
        Msol = const.SOLAR_MASS.value * u.kg 

        cosmo = LambdaCDM(H0=self.H_0, Om0=self.Omega_m, Ode0=self.Omega_Lambda, Tcmb0=self.Tcmb0)

        r_200 = ( (G * self.M)/(100. * cosmo.H(self.z_l)**2) )**(1./3.)
        r_s = r_200 * self.c
        delta_c = (200./3.) * ( self.c**3 / ( np.log(1+self.c) - ( self.c/(1+self.c) ) ) )
        
        x = r_vals/r_s

        Sigma_crit_coeff = c**2 / (4. * np.pi * G)

        D_s = cosmo.angular_diameter_distance(self.z_s)  
        D_d = cosmo.angular_diameter_distance(self.z_l)  
        D_ds = cosmo.angular_diameter_distance_z1z2(self.z_l,self.z_s)  

        Sigma_crit = Sigma_crit_coeff * D_s / (D_d * D_ds)

        k_s = cosmo.critical_density(self.z_l) * delta_c * r_s

        coeff = 2.*k_s / (x**2 - 1.)

        f_internal = 1. - (2./np.lib.scimath.sqrt(1.-x**2)) * np.arctanh(np.lib.scimath.sqrt((1.-x)/(1.+x)))

        nfw = coeff * f_internal

        if self.sig:
            return nfw.to(Msol / u.Mpc**2).real
        if not self.sig:
            return (nfw.real / Sigma_crit).decompose()

    def make_grid(self,npix,r_max):
        """make 2D grid centered on origin
        
        Parameters
        ----------
        npix: int 
            resolution of grid
        r_max: float 
            width of grid is 2*r_max

        Returns
        -------
        r_2D: np.ndarray
            2D grid centered on origin
        """
        
        npix *= 1j #imaginary so that ogrid knows to interpret npix as array size
        y,x = np.ogrid[-r_max:r_max:npix,-r_max:r_max:npix]
        r_2D = np.hypot(y,x)
        
        return r_2D

    def kappa_NFW_2D(self,npix,r_max):
        """Create 2D NFW profile on grid
        
        Parameters
        ----------
        npix: int
            resolution of grid
        r_max: float
            set width of grid (2*r_max)

        Returns
        -------
        kappa_NFW: np.ndarray
            2D NFW map
        """
        
        self.npix = npix
        self.r_max = r_max
        r_2D = self.make_grid(self.npix,self.r_max)
        
        return self.kappa_NFW(r_2D)



def KaiserSquires(Sigma):
    """Apply the Kaiser Squires algorithm to a 2D convergence field

       Parameters
       ----------
       Sigma: np.ndarray
           convergence map       

       Returns 
       -------
       e1: np.ndarray
           real component of complex shear map
       e2: np.ndarray
           imaginary component of complex shear map
    """
    
    kappa_tilde = np.fft.fft2(Sigma)

    k = np.fft.fftfreq(kappa_tilde.shape[0])

    oper_1  = - 1./(k[:, None]**2 + k[None, :]**2) * (k[:, None]**2 - k[None, :]**2)
    oper_2  = - 2./(k[:, None]**2 + k[None, :]**2) * k[:, None]*k[None, :]

    oper_1[0, 0] = -1
    oper_2[0, 0] = -1

    e1_tilde = oper_1*kappa_tilde
    e2_tilde = oper_2*kappa_tilde

    e1 = np.fft.ifft2(e1_tilde).real
    e2 = np.fft.ifft2(e2_tilde).real

    return e1, e2

def getTangetial(e1, e2, center, dx=10./1000.):
    """Measure the tangential and cross shear maps from the e1 and e2 maps
       Requires a center as an input, about which the tangential and cross maps are calculated

       Parameters
       ----------
       e1: np.ndarray
           real component of complex shear map
       e2: np.ndarray
           imaginary component of complex shear map
       center: list
           x and y coordinates of the center of the map as numpy array indices [center 1, center 2]
      
       Returns
       -------
       et: np.ndarray
           tangentail shear map
       ex: np.ndarray
           cross shear map
       radius: np.ndarray
           radius map
       anlge: np.ndarray
           angle map       
    """

    n = e1.shape[0]

    xx = np.arange(-n/2, n/2)*dx

    XX, YY = np.meshgrid(xx, xx)

    center_1, center_2 = center

    from_cent_1 = XX - center_1
    from_cent_2 = YY - center_2

    angle = -np.sign(from_cent_2)*np.arccos(from_cent_1/np.sqrt(from_cent_1**2+from_cent_2**2))
    radius = np.sqrt(from_cent_1**2+from_cent_2**2)

    angle[np.isnan(angle)] = 0

    et = - e1*np.cos(2*angle) - e2*np.sin(2*angle)
    ex = + e1*np.sin(2*angle) - e2*np.cos(2*angle)

    return et, ex, radius, angle

def getRadial(radius_map,r_bins,angle_map,phi_bins,kappa_map,et_map):
    """Bin a map into radial and angular bins as a 2d histogram
    
    Parameters
    ----------
    radius_map: np.ndarray
        2d radius map
    r_bins: np.ndarray
        1d radial bins
    angle_map: np.ndarray
       2d angle map
    phi_bins: np.ndarray
       1d angular bins
    kappa_map: np.ndarray
       2d convergence map
    et_map: np.ndarray
       2d tangential shear map
            
    Returns
    -------
    kappa_radial: np.ndarray
        1d convergence radial profile
    gammat_radial: np.ndarray
        1d tangential shear radial profile

    """
    _N = np.zeros((len(r_bins)-1, len(phi_bins)-1))
    _K = np.zeros((len(r_bins)-1, len(phi_bins)-1))
    _GT = np.zeros((len(r_bins)-1, len(phi_bins)-1))

    ii_r = np.digitize(radius_map, r_bins) - 1
    ii_phi = np.digitize(angle_map, phi_bins) - 1
    
    for i_r, i_phi, kk, gg in zip(ii_r.flatten(), ii_phi.flatten(), kappa_map.flatten(), et_map.flatten()):

        _N[i_r, i_phi] += 1
        _K[i_r, i_phi] += kk
        _GT[i_r, i_phi] += gg

    kappa_radial = _K/_N
    gammat_radial = _GT/_N
    
    return kappa_radial,gammat_radial



def compute_nfw_boost(rvals, rs=1000, b0=0.1) :
    """ Given a list of rvals, and optional rs and b0, return the corresponding boost factor at each rval

    Parameters
    ----------
    rvals : array_like
        radii
    rs : float (optional)
        scale radius for NFW in same units as rvals (default 2000 kpc)
    b0 : float (optional)

    Returns
    -------
    boost_factors : numpy.ndarray

    """

    x = np.array(rvals)/rs

    def _calc_finternal(x) :

        radicand = x**2-1

        finternal = -1j *  np.log( (1 + np.lib.scimath.sqrt(radicand)*1j) / (1 - np.lib.scimath.sqrt(radicand)*1j) ) / ( 2 * np.lib.scimath.sqrt(radicand) )

        return np.nan_to_num(finternal, copy=False, nan=1.0).real

    return 1. + b0 * (1 - _calc_finternal(x)) / (x**2 - 1)


def compute_powerlaw_boost(rvals, rs=1000, b0=0.1, alpha=-1.0) :
    """  Given a list of rvals, and optional rs and b0, and alpha, return the corresponding boost factor at each rval

    Parameters
    ----------
    rvals : array_like
        radii
    rs : float (optional)
        scale radius for NFW in same units as rvals (default 2000 kpc)
    b0 : float (optional)
    alpha : float (optional)
        exponent from Melchior+16

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
            `nfw_boost` - NFW profile model (Default)
            `powerlaw_boost` - Powerlaw profile

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
    yerr : array_like, None
        Errors of component y
    error_model : str, optional
        Statistical error model to use for y uncertainties. (letter case independent)

            * `ste` - Standard error [=std/sqrt(n) in unweighted computation] (Default).
            * `std` - Standard deviation.

    weights: array_like, None
        Weights for averages.


    Returns
    -------
    mean_x : array_like
        Mean x value in each bin
    mean_y : array_like
        Mean y value in each bin
    err_y: array_like
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
    return mean_x, mean_y, err_y, num_objects, binnumber


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
        Binning method to use (letter case independent):

            * `evenwidth` - Default, evenly spaced bins between rmin and rmax
            * `evenlog10width` - Logspaced bins with even width in log10 between rmin and rmax
            * `equaloccupation` - Bins with equal occupation numbers

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
        raise ValueError(
            f"Invalid nbins={nbins}. Must be integer greater than 0.")

    if method == 'evenwidth':
        binedges = np.linspace(rmin, rmax, nbins+1, endpoint=True)
    elif method == 'evenlog10width':
        binedges = np.logspace(np.log10(rmin), np.log10(
            rmax), nbins+1, endpoint=True)
    elif method == 'equaloccupation':
        if source_seps is None:
            raise ValueError(
                f"Binning method '{method}' requires source separations array")
        # by default, keep all galaxies
        seps = np.array(source_seps)
        mask = np.full(seps.size, True)
        if rmin is not None or rmax is not None:
            # Need to filter source_seps to only keep galaxies in the [rmin, rmax]
            rmin = seps.min() if rmin is None else rmin
            rmax = seps.max() if rmax is None else rmax
            mask = (seps >= rmin)*(seps <= rmax)
        binedges = np.percentile(seps[mask], tuple(
            np.linspace(0, 100, nbins+1, endpoint=True)))
    else:
        raise ValueError(
            f"Binning method '{method}' is not currently supported")

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
    cosmo : CLMM.Cosmology
        CLMM Cosmology object to compute angular diameter distance to
        convert between physical and angular units

    Returns
    -------
    dist2: array_like
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
    r""" Convert shape components 1 and 2 appropriately to make them estimators of the reduced shear
    once averaged.  The shape 1 and 2 components may correspond to ellipticities according the
    :math:`\epsilon`- or :math:`\chi`-definition, but also to the 1 and 2 components of the shear.
    See Bartelmann & Schneider 2001 for details (https://arxiv.org/pdf/astro-ph/9912508.pdf).

    The :math:`\epsilon`-ellipticity is a direct estimator of
    the reduced shear. The shear :math:`\gamma` may be converted to reduced shear :math:`g` if the
    convergence :math:`\kappa` is known. The conversions are given below.

    .. math::
     \epsilon = \frac{\chi}{1+(1-|\chi|^2)^{1/2}}

    .. math::
     g=\frac{\gamma}{1-\kappa}

    - If `shape_definition = 'chi'`, this function returns the corresponding `epsilon` ellipticities

    - If `shape_definition = 'shear'`, it returns the corresponding reduced shear, given the
      convergence `kappa`

    - If `shape_definition = 'epsilon'` or `'reduced_shear'`, it returns them as is as no conversion
      is needed.

    Parameters
    ----------
    shape_1 : array_like
        Input shapes or shears along principal axis (g1 or e1)
    shape_2 : array_like
        Input shapes or shears along secondary axis (g2 or e2)
    shape_definition : str
        Definition of the input shapes, can be ellipticities 'epsilon' or 'chi' or shears 'shear' or
        'reduced_shear'
    kappa : array_like
        Convergence for transforming to a reduced shear. Default is 0

    Returns
    -------
    epsilon_1 : array_like
        Epsilon ellipticity (or reduced shear) along principal axis (epsilon1)
    epsilon_2 : array_like
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
    q11 : float or array
        Second brightness moment tensor, component (1,1)
    q22 : float or array
        Second brightness moment tensor, component (2,2)
    q12 :  float or array
        Second brightness moment tensor, component (1,2)

    Returns
    -------
    chi1, chi2 : float or array
        Ellipticities using the "chi definition"
    epsilon1, epsilon2 : float or array
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
        \epsilon^{\rm lensed}=\epsilon^{\rm lensed}_1+i\epsilon^{\rm lensed}_2=
        \frac{\epsilon^{\rm true}+g}{1+g^\ast\epsilon^{\rm true}},

    where, the complex reduced shear :math:`g` is obtained from the shear
    :math:`\gamma=\gamma_1+i\gamma_2` and convergence :math:`\kappa` as :math:`g =
    \gamma/(1-\kappa)`, and the complex intrinsic ellipticity is :math:`\epsilon^{\rm
    true}=\epsilon^{\rm true}_1+i\epsilon^{\rm true}_2`

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
    names: list, tuple
        Names for each array, optional
    prefix: str
        Customized prefix for error message

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
    'int_array': (int, np.integer)
    }

def _is_valid(arg, valid_type):
    r"""Check if argument is of valid type, supports arrays.

    Parameters
    ----------
    arg: any
        Argument to be tested.
    valid_type: str, type
        Valid types for argument, options are object types, list/tuple of types, or:

            * `int_array` - interger, interger array
            * `float_array` - float, float array

    Returns
    -------
    valid: bool
        Is argument valid
    """
    return (isinstance(arg[0], _valid_types[valid_type])
                if (valid_type in ('int_array', 'float_array') and np.iterable(arg))
                else isinstance(arg, _valid_types.get(valid_type, valid_type)))


def validate_argument(loc, argname, valid_type, none_ok=False, argmin=None, argmax=None,
                      eqmin=False, eqmax=False):
    r"""Validate argument type and raise errors.

    Parameters
    ----------
    loc: dict
        Dictionary with all input arguments. Should be locals().
    argname: str
        Name of argument to be tested.
    valid_type: str, type
        Valid types for argument, options are object types, list/tuple of types, or:

            * `int_array` - interger, interger array
            * `float_array` - float, float array

    none_ok: True
        Accepts None as a valid type.
    argmin (optional) : int, float, None
        Minimum value allowed.
    argmax (optional) : int, float, None
        Maximum value allowed.
    eqmin: bool
        Accepts min(arg)==argmin.
    eqmax: bool
        Accepts max(arg)==argmax.
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

def _integ_pzfuncs(pzpdf, pzbins, zmin=0., zmax=5, kernel=lambda z: 1., is_unique_pzbins=False, ngrid=1000):
    r"""
    Integrates the product of a photo-z pdf with a given kernel.
    This function was created to allow for data with different photo-z binnings.

    Parameters
    ----------
    pzpdf : list of arrays
        Photometric probablility density functions of the source galaxies.
    pzbins : list of arrays
        Redshift axis on which the individual photoz pdf is tabulated.
    zmin : float
        Minimum redshift for integration
    kernel : function, optional
        Function to be integrated with the pdf, must be f(z_array) format.
        Default: kernel(z)=1
    ngrid : int, optional
        Number of points for the interpolation of the redshift pdf.

    Returns
    -------
    array
        Kernel integrated with the pdf of each galaxy.

    Notes
    -----
        Will be replaced by qp at some point.
    """
    # adding these lines to interpolate CLMM redshift grid for each galaxies
    # to a constant redshift grid for all galaxies. If there is a constant grid for all galaxies
    # these lines are not necessary and z_grid, pz_matrix = pzbins, pzpdf

    if is_unique_pzbins==False:
        # First need to interpolate on a fixed grid
        z_grid = np.linspace(zmin, zmax, ngrid)
        pdf_interp_list = [interp1d(pzbin, pdf, bounds_error=False, fill_value=0.) for pzbin,pdf in zip(pzbins, pzpdf)]
        pz_matrix = np.array([pdf_interp(z_grid) for pdf_interp in pdf_interp_list])
        kernel_matrix = kernel(z_grid)
    else:
        # OK perform the integration directly from the pdf binning common to all galaxies
        z_grid = pzbins[0][(pzbins[0]>=zmin)*(pzbins[0]<=zmax)]
        pz_matrix = pzpdf
        kernel_matrix = kernel(z_grid)

    return simps(pz_matrix*kernel_matrix, x=z_grid, axis=1)

def compute_for_good_redshifts(function, z1, z2, bad_value, error_message):
    """Computes function only for z1>z2, the rest is filled with bad_value

    Parameters
    ----------
    function: function
        Function to be executed
    z1: float, array_like
        Redshift lower
    z2: float, array_like
        Redshift higher
    bad_value: any
        Value to be added when z1>=z2
    error_message: str
        Message to be displayed
    """
    z_good = np.less(z1, z2)
    if not np.all(z_good):
        warnings.warn(error_message)
        if np.iterable(z_good):
            res = np.full(z_good.size, bad_value)
            if np.any(z_good):
                res[z_good] = function(
                    np.array(z1)[z_good] if np.iterable(z1) else z1,
                    np.array(z2)[z_good] if np.iterable(z2) else z2)
        else:
            res = bad_value
    else:
        res = function(z1, z2)
    return res

def compute_beta(z_s, z_cl, cosmo):
    r"""Geometric lensing efficicency

    .. math::
        beta = max(0, Dang_ls/Dang_s)

    Eq.2 in https://arxiv.org/pdf/1611.03866.pdf

    Parameters
    ----------
    z_cl: float
            Galaxy cluster redshift
    z_s:  float
            Source galaxy  redshift
    cosmo: Cosmology
        Cosmology object

    Returns
    -------
    float
        Geometric lensing efficicency
    """
    beta = np.heaviside(z_s-z_cl, 0) * cosmo.eval_da_z1z2(z_cl, z_s) / cosmo.eval_da(z_s)
    return beta

def compute_beta_s(z_s, z_cl, z_inf, cosmo):
    r"""Geometric lensing efficicency ratio

    .. math::
        beta_s =beta(z_s)/beta(z_inf)

    Parameters
    ----------
    z_cl: float
            Galaxy cluster redshift
    z_s:  float
            Source galaxy redshift
    z_inf: float
            Redshift at infinity
    cosmo: Cosmology
        Cosmology object

    Returns
    -------
    float
        Geometric lensing efficicency ratio
    """
    beta_s = compute_beta(z_s, z_cl, cosmo) / compute_beta(z_inf, z_cl, cosmo)
    return beta_s

def compute_beta_mean(z_cl, cosmo, zmax=10.0, delta_z_cut=0.1, zmin=None, z_distrib_func=None):
    r"""Mean value of the geometric lensing efficicency

    .. math::
       \left\<beta\right\> =\frac{\sum_{z = z_{min}}^{z = z_{max}}\beta(z)p(z)}{\sum_{z = z_{min}}^{z = z_{max}}p(z)}

    Parameters
    ----------
    z_cl: float
            Galaxy cluster redshift
    z_inf: float
            Redshift at infinity
    z_distrib_func: one-parameter function
            Redshift distribution function. Default is\
            Chang et al (2013) distribution\
            function.
    zmin: float
            Minimum redshift to be set as the source of the galaxy\
             when performing the sum.
    zmax: float
            Maximum redshift to be set as the source of the galaxy\
            when performing the sum.
    delta_z_cut: float
            Redshift interval to be summed with $z_cl$ to return\
            $zmin$. This feature is not used if $z_min$ is provided by the user.
    cosmo: Cosmology
        Cosmology object

    Returns
    -------
    float
        Mean value of the geometric lensing efficicency
    """
    if z_distrib_func == None:
        z_distrib_func = _chang_z_distrib
    def integrand(z_i, z_cl=z_cl, cosmo=cosmo):
        return compute_beta(z_i, z_cl, cosmo) * z_distrib_func(z_i)

    if zmin==None:
        zmin = z_cl + delta_z_cut

    B_mean = quad(integrand, zmin, zmax)[0] / quad(z_distrib_func, zmin, zmax)[0]
    return B_mean

def compute_beta_s_mean(z_cl, z_inf, cosmo, zmax=10.0, delta_z_cut=0.1, zmin=None, z_distrib_func=None):
    r"""Mean value of the geometric lensing efficicency ratio

    .. math::
       \left\<beta_s\right\> =\frac{\sum_{z = z_{min}}^{z = z_{max}}\beta_s(z)p(z)}{\sum_{z = z_{min}}^{z = z_{max}}p(z)}

    Parameters
    ----------
    z_cl: float
            Galaxy cluster redshift
    z_inf: float
            Redshift at infinity
    z_distrib_func: one-parameter function
            Redshift distribution function. Default is\
            Chang et al (2013) distribution\
            function.
    zmin: float
            Minimum redshift to be set as the source of the galaxy\
            when performing the sum.
    zmax: float
            Minimum redshift to be set as the source of the galaxy\
            when performing the sum.
    delta_z_cut: float
            Redshift interval to be summed with $z_cl$ to return\
            $zmin$. This feature is not used if $z_min$ is provided by the user.
    cosmo: Cosmology
        Cosmology object

    Returns
    -------
    float
        Mean value of the geometric lensing efficicency ratio
    """
    if z_distrib_func == None:
        z_distrib_func = _chang_z_distrib

    def integrand(z_i, z_cl=z_cl, z_inf=z_inf, cosmo=cosmo):
        return compute_beta_s(z_i, z_cl, z_inf, cosmo) * z_distrib_func(z_i)

    if zmin==None:
        zmin = z_cl + delta_z_cut
    Bs_mean = quad(integrand, zmin, zmax)[0] / quad(z_distrib_func, zmin, zmax)[0]
    return Bs_mean

def compute_beta_s_square_mean(z_cl, z_inf, cosmo, zmax=10.0, delta_z_cut=0.1, zmin=None, z_distrib_func=None):
    r"""Mean square value of the geometric lensing efficicency ratio

    .. math::
       \left\<beta_s\right\>2 =\frac{\sum_{z = z_{min}}^{z = z_{max}}\beta_s^2(z)p(z)}{\sum_{z = z_{min}}^{z = z_{max}}p(z)}

    Parameters
    ----------
    z_cl: float
            Galaxy cluster redshift
    z_inf: float
            Redshift at infinity
    z_distrib_func: one-parameter function
            Redshift distribution function. Default is\
            Chang et al (2013) distribution\
            function.
    zmin: float
            Minimum redshift to be set as the source of the galaxy\
            when performing the sum.
    zmax: float
            Minimum redshift to be set as the source of the galaxy\
            when performing the sum.
    delta_z_cut: float
            Redshift interval to be summed with $z_cl$ to return\
            $zmin$. This feature is not used if $z_min$ is provided by the user.
    cosmo: Cosmology
        Cosmology object

    Returns
    -------
    float
        Mean square value of the geometric lensing efficicency ratio.
    """
    if z_distrib_func == None:
        z_distrib_func = _chang_z_distrib

    def integrand(z_i, z_cl=z_cl, z_inf=z_inf, cosmo=cosmo):
        return compute_beta_s(z_i, z_cl, z_inf, cosmo)**2 * z_distrib_func(z_i)

    if zmin==None:
        zmin = z_cl + delta_z_cut
    Bs_square_mean = quad(integrand, zmin, zmax)[0] / quad(z_distrib_func, zmin, zmax)[0]
    return Bs_square_mean

def _chang_z_distrib(redshift, is_cdf=False):
    """
    A private function that returns the Chang et al (2013) unnormalized galaxy redshift distribution
    function, with the fiducial set of parameters.

    Parameters
    ----------
    redshift : float
        Galaxy redshift
    is_cdf : bool
        If True, returns cumulative distribution function.

    Returns
    -------
    The value of the distribution at z
    """
    alpha, beta, redshift0 = 1.24, 1.01, 0.51
    if is_cdf:
        return redshift0**(alpha+1)*gammainc((alpha+1)/beta, (redshift/redshift0)**beta)/beta*gamma((alpha+1)/beta)
    else:
        return (redshift**alpha)*np.exp(-(redshift/redshift0)**beta)

def _srd_z_distrib(redshift, is_cdf=False):
    """
    A private function that returns the unnormalized galaxy redshift distribution function used in
    the LSST/DESC Science Requirement Document (arxiv:1809.01669).

    Parameters
    ----------
    redshift : float
        Galaxy redshift
    is_cdf : bool
        If True, returns cumulative distribution function.

    Returns
    -------
    The value of the distribution at z
    """
    alpha, beta, redshift0 = 2., 0.9, 0.28
    if is_cdf:
        return redshift0**(alpha+1)*gammainc((alpha+1)/beta, (redshift/redshift0)**beta)/beta*gamma((alpha+1)/beta)
    else:
        return (redshift**alpha)*np.exp(-(redshift/redshift0)**beta)

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
    xstep : float
        Size of the step to interpolate the culmulative distribution.

    Returns
    -------
    ndarray
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
    xmin : float
        Lower bound to draw redshift. Default is the min(x_tab)
    xmax : float
        Upper bound to draw redshift. Default is the max(x_tab)

    Returns
    -------
    samples : ndarray
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
