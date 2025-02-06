import numpy as np
import pyccl as ccl
from pyccl import physical_constants as const
from astropy import units as un


class profiles:
    '''
    Class for computing shear profiles

    Attributes
    ----------
    M: float
        mass (units: M_sun)
    c: float
        concentration
    z_l: float
        lens redshift
    z_s: float
        source redshift
    compute_sigma: bool
        True for computing Sigma (surface density) profiles,
        False for computing kappa (convergence) profiles
    '''     
    
    def __init__(self,cosmo,M,c,z_l,z_s,compute_sigma=False):

        self.M = M  # M_sun
        self.c = c
        self.a_l = 1./(1.+z_l)  # ccl works with scale factor instead of redshift
        self.a_s = 1./(1.+z_s)
        self.compute_sigma = compute_sigma
        
        # physical constants with units (cosmological)
        self.const_G = const.GNEWT * (un.Mpc/const.MPC_TO_METER)**3 /(un.M_sun/const.SOLAR_MASS) / un.s**2.  # Mpc^3/Msun/s^2
        self.const_c = const.CLIGHT * (un.Mpc/const.MPC_TO_METER)/ un.s    # Mpc/s

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
    
    def kappa_NFW(self,r_vals=None):
        '''
        Create radial bins for profiles
        
        Attributes
        ----------
        r_vals: float or array
            radial bin values for halo lensing profile
         cosmo : clmm.Cosmology object
            CLMM Cosmology object 
            
        Returns
        ----------
        nfw: float or array
            If compute_sigma is True, returns the surface density (Sigma) profile,
            else, returns kappa profiles
        '''
        
        if np.all(r_vals.value) == None:
            r_vals = self.r_vals


        Hz = self.cosmo.h_over_h0(self.a_l) * self.cosmo.cosmo.params.h * 100 * 1/(const.MPC_TO_METER*1e-3) / un.s   # units: 1/s
        r_200 = (((self.const_G * self.M)/(100. * Hz**2) )**(1./3.))   # units: Mpc
        r_s   = r_200 * self.c 
        delta_c = (200./3.) * ( self.c**3 / ( np.log(1+self.c) - ( self.c/(1+self.c) ) ) )
        
        x = r_vals/r_s
        
        Sigma_crit_coeff = self.const_c**2 / (4. * np.pi * self.const_G)  # units: Msun/Mpc
        
        D_s  = self.cosmo.angular_diameter_distance(self.a_s) *un.Mpc
        D_d  = self.cosmo.angular_diameter_distance(self.a_l) *un.Mpc
        D_ds = self.cosmo.angular_diameter_distance(self.a_l,self.a_s) *un.Mpc
        
        Sigma_crit = Sigma_crit_coeff * D_s / (D_d * D_ds)
        critical_density = self.cosmo.rho_x(self.a_l, 'critical', is_comoving=False) * un.M_sun/un.Mpc**3 
        k_s = critical_density * delta_c * r_s
        
        coeff = 2.*k_s / (x**2 - 1.)
        f_internal = 1. - (2./np.lib.scimath.sqrt(1.-x**2)) * np.arctanh(np.lib.scimath.sqrt((1.-x)/(1.+x)))
    
        nfw = coeff * f_internal.real

        if self.compute_sigma:
            return nfw   # units: Msun/Mpc^2
        else:
            return nfw / Sigma_crit # units: adimensional
            
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

    def kappa_NFW_2D(self,npix,r_max):
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
        kappa_nfw: array
            NFW profile (Sigma or kappa) for each point of the 2D grid
        
        '''
        
        self.npix  = npix
        self.r_max = r_max
        r_2D = self.make_grid(self.npix,self.r_max)
        
        return self.kappa_NFW(r_2D)



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

def getTangential(e1, e2, center, dx=10./1000.):
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

def getRadial(radius_map,angle_map,r_bins,phi_bins,kappa_map,et_map):
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
