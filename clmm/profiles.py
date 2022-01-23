import numpy as np
from scipy.optimize import fsolve

class NFW():
    def __init__(self, Mdelta, c, z, massdef, delta_mdef, cosmo):
        r"""
        Attributes:
        -----------
        Mdelta : float
            halo mass
        c : float
            halo concentration
        z : float
            halo redshift
        massdef : str
            background density required for the SOD definition (critical, mean)
        delta_mdef : float
            overdensity scale (200, 500, etc.)
        cosmo : CLMMCosmology
            Cosmology object
        Returns:
        --------
        Compute specific halo quantities (r_delta, r_s) for the mass profile
        """
        self.Mdelta = Mdelta
        self.c = c
        self.delta_mdef = delta_mdef
        if massdef == 'matter': alpha = cosmo.get_Omega_m(z)
        else : alpha = 1
        rho_critical = cosmo.get_rho_crit(z)#.to(u.Msun / u.Mpc**3).value
        self.rdelta = ((Mdelta * 3) / (alpha * 4 * np.pi * delta_mdef * rho_critical)) ** (1/3)
        self.rs = self.rdelta / self.c

    def _delta_c(self, c):
        return np.log(1 + c) - c/(1 + c)

    def M(self, r3d):
        r"""
        Attributes:
        -----------
        r3d : array
            3d radius from the halo center
        Returns:
        --------
        M_in_r : array
            Mass enclosed in a sphere of radius r
        """
        x = np.array(r3d)/self.rs
        M = self.Mdelta * self._delta_c(x) / self._delta_c(self.c)
        return M

def convert_def(Mdelta1, c1, z, massdef1, delta_mdef1, massdef2, delta_mdef2, cosmo):
    r"""
    Attributes:
    -----------
    Mdelta1: float
        halo mass (1st SOD definition)
    c1: float
        halo concentration (1st SOD definition)
    z: float
        halo redshift
    massdef1: float
        Overdensity scale (1st SOD definition)
    delta_mdef1: str
        background density (1st SOD definition)

    massdef2: float
        Overdensity scale (2nd SOD definition)
    delta_mdef2: str
        background density (2nd SOD definition)
    cosmo_astropy: astropy object
        cosmology
    Returns:
    --------
    M2fit, c2fit: float, float
        mass and concentration for the 2nd halo definition
    """
    cl1 = NFW(Mdelta1, c1, z, massdef1, delta_mdef1, cosmo)

    def f(param):
        Mdelta2, c2 = param    
        cl2 = NFW(Mdelta2, c2, z, massdef2, delta_mdef2, cosmo)
        first_term = Mdelta1 - cl2.M(cl1.rdelta)
        second_term = Mdelta2 - cl1.M(cl2.rdelta)  
        return first_term, second_term
    x0 = [Mdelta1, c1]
    M2fit, c2fit = fsolve(func = f, x0 = x0, maxfev = 1000)
    M2fit, c2fit = fsolve(func = f, x0 = [M2fit, c2fit], maxfev = 100)
    return M2fit, c2fit
