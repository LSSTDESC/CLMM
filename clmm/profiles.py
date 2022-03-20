import numpy as np
from scipy.optimize import fsolve

class NFW():
    def __init__(self, mdelta, cdelta, z, massdef, delta_mdef, cosmo):
        r"""
        Attributes
        ----------
        mdelta : float
            halo mass for the given massdef
        cdelta : float
            halo concentration
        z : float
            halo redshift
        massdef : str
            background density required for the SOD definition (critical, mean)
        delta_mdef : float
            overdensity scale (200, 500, etc.)
        cosmo : CLMMCosmology
            Cosmology object
        """
        self.mdelta = mdelta
        self.cdelta = cdelta
        self.z = z
        self.massdef = massdef
        self.delta_mdef = delta_mdef
        self.cosmo = cosmo

        if massdef == 'mean':
            alpha = cosmo.get_Omega_m(z)
        else :
            alpha = 1

        rho_c = cosmo.get_rho_c(z) #Msun / Mpc**3

        self.rdelta = ((mdelta * 3) / (alpha * 4 * np.pi * delta_mdef * rho_c)) ** (1/3)
        self.rs = self.rdelta / self.cdelta

    def _Delta_c(self, c):
        return np.log(1 + c) - c/(1 + c)

    def M(self, r3d):
        """
        Parameters
        ----------
        r3d : array_like
            3D radius from the halo center

        Returns
        -------
        M : array
            Mass enclosed within a sphere of radius r3d
        """
        x = np.array(r3d)/self.rs
        M = self.mdelta * self._Delta_c(x) / self._Delta_c(self.cdelta)
        return M

    def to_def(self, massdef, delta_mdef):
        """
        Parameters
        ----------
        massdef: float
            Overdensity scale (2nd SOD definition)
        delta_mdef: str
            background density (2nd SOD definition)

        Returns
        -------
        NFW profile: NFW
            NFW object
        """
        mdelta2, cdelta2 = convert_def(self.mdelta, self.cdelta, self.z,\
        self.massdef, self.delta_mdef, massdef, delta_mdef, self.cosmo)
        return NFW(mdelta2, cdelta2, self.z, massdef, delta_mdef, self.cosmo)

def convert_def(mdelta1, cdelta1, z, massdef1, delta_mdef1, massdef2, delta_mdef2, cosmo):
    r"""
    Parameters
    ----------
    mdelta1: float
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
    cosmo : CLMMCosmology
            Cosmology object

    Returns
    -------
    M2fit, c2fit: float, float
        mass and concentration for the 2nd halo definition
    """

    def1 = NFW(mdelta1, cdelta1, z, massdef1, delta_mdef1, cosmo)

    def f(params):
        mdelta2, cdelta2 = params
        def2 = NFW(mdelta2, cdelta2, z, massdef2, delta_mdef2, cosmo)
        return def1.mdelta - def2.M(def1.rdelta), def2.mdelta - def1.M(def2.rdelta)

    mdelta2_fit, cdelta2_fit = fsolve(func = f, x0 = [mdelta1, cdelta1], maxfev = 1000)
    mdelta2_fit, cdelta2_fit = fsolve(func = f, x0 = [mdelta2_fit, cdelta2_fit], maxfev = 100)

    return mdelta2_fit, cdelta2_fit
