""" Patches for cluster_toolkit """
import numpy as np
from .constants import Constants as c

def _patch_comoving_coord_cluster_toolkit_rho_m(omega_m, redshift):
    r""" Make use of proper distances instead of comoving in cluster_toolkit

    cluster_toolkit works in comoving coordinates, so the distances have to be converted
    We are able to make this correction by passing :math:`\Omega_m * (1+z)^3` instead
    of :math:`\Omega_m(z=0)`. It works because `\Omega_m` is only used in cluster_toolkit
    to convert between mass and radius.

    Parameters
    ----------
    omega_m : float
        Mean matter density at z=0 in units of the critical density
    redshift: array_like
        Redshift

    Returns
    -------
    omega_m : float
        Transformed mean matter density
    """
    redshift = np.array(redshift)

    rhocrit_mks = 3.*100.*100./(8.*np.pi*c.GNEWT.value)
    rhocrit_cosmo = rhocrit_mks * 1000. * 1000. * c.PC_TO_METER.value * 1.e6 / c.SOLAR_MASS.value
    rhocrit_cltk = 2.77533742639e+11

    return omega_m * (1.0 + redshift)**3 * (rhocrit_cosmo/rhocrit_cltk)
