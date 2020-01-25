""" Patches for cluster_toolkit """
import numpy as np
from .constants import Constants as c

def _patch_zevolution_cluster_toolkit_rho_m(omega_m, redshift):
    r""" Evolve the matter density, rho_m, in cluster_toolkit with redshift

    We currently use this as a patch to fix `cluster_toolkit`'s z=0 limitation.
    It works by passing :math:`\Omega_m * (1+z)^3` instead of :math:`\Omega_m(z=0)`.
    This density is only used to compute :math:`\rho_m` to convert between mass and
    radius. So we are able to get the redshift evolution of :math:`rho_m` by passing
    in the evolution alongside :math:`\Omega_m`

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
