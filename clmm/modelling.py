from astropy.cosmology import LambdaCDM
from astropy import constants as const
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt

class profiles:
    
    def __init__(self,M,c,z_l,z_s,sig=False):
        import numpy as np
        from astropy.cosmology import FlatLambdaCDM
        from astropy import constants as const
        from astropy import units as u
        
        self.M = M
        self.c = c
        self.z_l = z_l
        self.z_s = z_s
        self.sig = sig
        
        self.cosmology()
        
        return
    
    def cosmology(self, H_0 = 70., Omega_m = 0.3, Omega_Lambda = 0.7, Tcmb0=2.725):
        
        self.H_0 = H_0
        self.Omega_m = Omega_m
        self.Tcmb0 = Tcmb0
        self.Omega_Lambda = Omega_Lambda
        
        return
        
        
    def get_r_vals(self,nbins,lower_r,upper_r,log=False):
        '''create radial bins for profiles
        nbins: number of bins
        lower_r: lowest bin value
        upper_r: highest bin value
        returns bin edges in units provided, or kpc if none are provided
        '''
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
        '''r_vals: radial bin values for halo lensing profile
           M: halo mass
           c: concentration
           z_l: lens redshift
           z_s: source redshift
           delta_sig: If False, return convergence profile, if True, return delta Sigma profile
        '''
        
        if np.all(r_vals.value) == None:
            r_vals = self.r_vals
        
        cosmo = LambdaCDM(H0=self.H_0, Om0=self.Omega_m, Ode0=self.Omega_Lambda, Tcmb0=self.Tcmb0)

        r_200 = ( (const.G * self.M)/(100. * cosmo.H(self.z_l)**2) )**(1./3.)
        r_s = r_200 * self.c
        delta_c = (200./3.) * ( self.c**3 / ( np.log(1+self.c) - ( self.c/(1+self.c) ) ) )
        
        x = r_vals/r_s

        Sigma_crit_coeff = const.c**2 / (4. * np.pi * const.G)

        D_s = cosmo.angular_diameter_distance(self.z_s)  
        D_d = cosmo.angular_diameter_distance(self.z_l)  
        D_ds = cosmo.angular_diameter_distance_z1z2(self.z_l,self.z_s)  

        Sigma_crit = Sigma_crit_coeff * D_s / (D_d * D_ds)

        k_s = cosmo.critical_density(self.z_l) * delta_c * r_s

        coeff = 2.*k_s / (x**2 - 1.)

        f_internal = 1. - (2./np.lib.scimath.sqrt(1.-x**2)) * np.arctanh(np.lib.scimath.sqrt((1.-x)/(1.+x)))

        nfw = coeff * f_internal

        if self.sig:
            return nfw.to(const.M_sun / u.Mpc**2).real
        if not self.sig:
            return (nfw.real / Sigma_crit).decompose()

    def make_grid(self,npix,r_max):
        '''make 2D grid centered on origin
        npix: resolution of grid
        r_max: width of grid is 2*r_max
        '''
        
        npix *= 1j #imaginary so that ogrid knows to interpret npix as array size
        y,x = np.ogrid[-r_max:r_max:npix,-r_max:r_max:npix]
        r_2D = np.hypot(y,x)
        
        return r_2D

    def kappa_NFW_2D(self,npix,r_max):
        
        '''create 2D NFW profile on grid
        '''
        
        self.npix = npix
        self.r_max = r_max
        r_2D = self.make_grid(self.npix,self.r_max)
        
        return self.kappa_NFW(r_2D)



def KaiserSquires(Sigma):

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


def plot_maps_and_profiles(kappa_map,e1_map,e2_map,et_map,ex_map,
                       kappa_radial,gammat_radial,r_bins):
    
    fig, ax = plt.subplots(2,3,figsize=[9,6])
    ax.ravel()
    
    ax[0,0].imshow(kappa_map,origin = 'lower')
    ax[0,0].set_title(r'$\kappa$')


    ax[0,1].imshow(e1_map,origin = 'lower')
    ax[0,1].set_title(r'$e_1$')


    ax[0,2].imshow(e2_map,origin = 'lower')
    ax[0,2].set_title(r'$e_2$')


    ax[1,0].imshow(et_map,origin = 'lower')
    ax[1,0].set_title(r'$e_t$')


    ax[1,1].imshow(ex_map,origin = 'lower')
    ax[1,1].set_title(r'$e_x$')

    
    
    r_bins_mid = 0.5 * (r_bins[1:] + r_bins[:-1])

    ax[1,2].plot(r_bins_mid, kappa_radial.mean(axis=-1), label=r'$\kappa(r)$')
    ax[1,2].plot(r_bins_mid, gammat_radial.mean(axis=-1), label=r'$\gamma_t(r)$')
    
    plt.legend()
    plt.show()
    
    fig,ax = plt.subplots(1,2)
    
    ax[0].imshow(kappa_radial,origin='lower')
    ax[1].imshow(gammat_radial,origin='lower')
    
    ax[0].set_ylabel('r')
    ax[0].set_xlabel('$\phi$')
    
    ax[1].set_ylabel('r')
    ax[1].set_xlabel('$\phi$')
    
    ax[0].set_title(r'$\kappa$')
    ax[1].set_title(r'$\gamma_t$')

    plt.show()
