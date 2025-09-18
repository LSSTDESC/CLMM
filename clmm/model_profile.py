import numpy as np
from astropy import units as un
import clmm
from clmm.utils.constants import Constants as const
from clmm.dataops import _compute_tangential_shear, _compute_cross_shear


class profiles:
    '''
    Class for computing shear profiles

    Attributes
    ----------

    cosmo : clmm.cosmology.Cosmology object
        CLMM Cosmology object
    mdelta : float
        Galaxy cluster mass in :math:`M_\odot`.
    cdelta : float
        Galaxy cluster concentration
    z_l: float
        Redshift of the lens
    z_s: float
        Redshift of the source
    halo_profile_model : str, optional
        Profile model parameterization (letter case independent):
            * 'nfw' (default)
            * 'einasto' - not in cluster_toolkit
            * 'hernquist' - not in cluster_toolkit
    delta_mdef : int, optional
        Mass overdensity definition; defaults to 200.
    massdef : str, optional
        Profile mass definition, with the following supported options (letter case independent):
            * 'mean' (default)
            * 'critical'
            * 'virial'
    compute_sigma: bool
        True for computing Sigma (surface density) profiles,
        False for computing kappa (convergence) profiles
    '''     
    
    def __init__(self,cosmo,mdelta,cdelta,z_l,z_s,halo_profile_model='nfw',delta_mdef=200,massdef='mean',compute_sigma=False):

        self.mdelta = mdelta  # M_sun
        self.cdelta = cdelta
        self.z_l = z_l
        self.z_s = z_s
        self.halo_profile_model = halo_profile_model
        self.delta_mdef = delta_mdef
        self.massdef = massdef
        self.compute_sigma = compute_sigma
        
        # physical constants with units (cosmological)
        self.const_G = const.GNEWT_SOLAR_MASS* (un.Mpc/const.PC_TO_METER/1e6)**3 /un.M_sun / un.s**2  # Mpc^3/Msun/s^2
        self.const_c = const.CLIGHT *  (un.Mpc/const.PC_TO_METER/1e6)/ un.s    # Mpc/s

        # cosmology
        self.cosmo = cosmo
        return
    
        
    def get_r_vals(self,nbins,lower_r,upper_r,log=False):
        '''
        Create radial bins for profiles
        
        Attributes
        ----------
        nbins: int
            number of bins
        lower_r: float
            lowest bin value
        upper_r: float
            highest bin value
            
        Returns
        ----------
        r_vals: float or array
            bin edges in units provided, or kpc if none are provided
        '''
        
        #create r values
        self.nbins   = nbins
        self.lower_r = lower_r
        self.upper_r = upper_r
        self.log_r_vals = log
        
        if self.log_r_vals:
            if self.lower_r == 0:
                raise ValueError('log(lower_r) cannot be evaluated for lower_r = 0')
                
            r_vals = np.geomspace(self.lower_r,self.upper_r,self.nbins+1)
            
        else:
            r_vals = np.linspace(self.lower_r,self.upper_r,self.nbins+1)
        
        #if no units are provided, set units to kpc, otherwise keep input units
        r_vals *= un.dimensionless_unscaled
        if r_vals.unit == un.dimensionless_unscaled:
            r_vals *= un.kpc
            
        self.r_vals = r_vals
        return self.r_vals
    
    def kappa_or_profile(self,r_vals=None):
        '''
        Compute kappa or surface density profiles
        
        Attributes
        ----------
        r_vals: float or array
            radial bin values for halo lensing profile
            
        Returns
        ----------
        nfw: float or array
            If compute_sigma is True, returns the surface density (Sigma) profile,
            else, returns kappa profiles
        '''
        
        if np.all(r_vals.value) == None:
            r_vals = self.r_vals

        if self.compute_sigma:
            return clmm.compute_surface_density(r_vals.value,self.mdelta.value,self.cdelta,
                                                self.z_l,self.cosmo,self.delta_mdef,
                                                self.halo_profile_model,self.massdef) * un.M_sun/un.Mpc**2 
        else:
            return clmm.compute_convergence(r_vals.value,self.mdelta.value,self.cdelta,
                                            self.z_l,self.z_s,self.cosmo,self.delta_mdef,
                                            self.halo_profile_model,self.massdef) * un.dimensionless_unscaled

            
            
    def make_grid(self,npix,r_max):
        '''
        Make 2D grid centered on origin
        
        Attributes
        ----------
        npix: int
            resolution of grid
        r_max: float
            width of grid is 2*r_max

        Returns
        ---------
        r_2D: array
           2D grid (side=npix)
        '''
        
        npix *= 1j #imaginary so that ogrid knows to interpret npix as array size
        y,x = np.ogrid[-r_max:r_max:npix,-r_max:r_max:npix]
        r_2D = np.hypot(y,x)

        return r_2D

    def kappa_or_profile_2D(self,npix,r_max):
        '''
        Create 2D NFW profile on a grid
        
        Attributes
        ----------
        npix: int
            resolution of grid
        r_max: float
            width of grid is 2*r_max

        Returns
        ---------
        kappa_or_profile: array
            NFW profile (Sigma or kappa) for each point of the 2D grid
        
        '''
        
        self.npix  = npix
        self.r_max = r_max 
        r_2D = self.make_grid(self.npix,self.r_max)

        kappa = np.array([self.kappa_or_profile(r_2D[i]) for i in range(self.npix)])
        return kappa



def KaiserSquires(Sigma):
    '''
    Kaiser & Squires method for computing shear components
    
    Attributes
    ----------
    Sigma: array
        Sigma map
        
    Returns
    ---------
    e1,e2: arrays
        shear maps
    '''

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

def get_Tangential_and_Cross(e1, e2, center, dx=10./1000.):
    '''
    Compute tangential shear profiles
    
    Attributes
    ----------
    e1,e2: arrays
        shear maps
    center: array
        center of the maps
    dx: float
        physcial size of a pixel
        
    Returns
    ---------
    et,ex: arrays
        tangential and cross shear maps aroud the given center
    radius, angle: arrays
        radius and angle maps, used for computing the radial profile
    '''    
    
    n  = e1.shape[0]
    xx = np.arange(-n/2, n/2)*dx
    XX, YY = np.meshgrid(xx, xx)

    center_1, center_2 = center
    from_cent_1 = XX - center_1
    from_cent_2 = YY - center_2

    angle  = -np.sign(from_cent_2) * np.arccos(from_cent_1/np.sqrt(from_cent_1**2 + from_cent_2**2))
    radius = np.sqrt(from_cent_1**2 + from_cent_2**2)

    angle[np.isnan(angle)] = 0

    et = _compute_tangential_shear(e1, e2, angle)
    ex = _compute_cross_shear(e1, e2, angle)

    return et, ex, radius, angle

def get_Radial(radius_map,angle_map,r_bins,phi_bins,kappa_map,et_map):
    '''
     Calculate the radial convergence profile from the kappa map 
     and the radial tangential shear profile from the e_tangential map
    
    Attributes
    ----------
    radius_map: array
        radius map
    angle_map:
        angle map
    r_bins: array
        radial bins 
    phi_bins: array
        angular bins
    kappa_map: array
        convergence map
    et_map: array
        tangential shear map
        
    Returns
    ---------
    kappa_radial: array
        radial convergence profile
    gammat_radial: array
        radial tangential shear profile
    '''    
    
    _N  = np.zeros((len(r_bins)-1, len(phi_bins)-1))
    _K  = np.zeros((len(r_bins)-1, len(phi_bins)-1))
    _GT = np.zeros((len(r_bins)-1, len(phi_bins)-1))

    ii_r   = np.digitize(radius_map, r_bins) - 1
    ii_phi = np.digitize(angle_map, phi_bins) - 1
    
    for i_r, i_phi, kk, gg in zip(ii_r.flatten(), ii_phi.flatten(), kappa_map.flatten(), et_map.flatten()):

        _N[i_r, i_phi] += 1
        _K[i_r, i_phi] += kk
        _GT[i_r, i_phi] += gg

    kappa_radial = _K/_N
    gammat_radial = _GT/_N
    
    return kappa_radial,gammat_radial
