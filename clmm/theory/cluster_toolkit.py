"""@file cluster_toolkit.py
Modeling using cluster_toolkit
"""
# Functions to model halo profiles
import numpy as np

import cluster_toolkit as ct

from . import func_layer
from . func_layer import *

from .parent_class import CLMModeling

from .. utils import _patch_rho_crit_to_cd2018

from .. cosmology.cluster_toolkit import AstroPyCosmology
Cosmology = AstroPyCosmology

__all__ = ['CTModeling', 'Modeling', 'Cosmology']+func_layer.__all__


def _assert_correct_type_ct(arg):
    """ Convert the argument to a type compatible with cluster_toolkit
    cluster_toolkit does not handle correctly scalar arguments that are
    not float or numpy array and others that contain non-float64 elements.
    It only convert lists to the correct type. To circumvent this we
    pre-convert all arguments going to cluster_toolkit to the appropriated
    types.

    Parameters
    ----------
    arg : array_like or scalar

    Returns
    -------
    scale_factor : array_like
        Scale factor
    """
    if np.isscalar(arg):
        return float(arg)
    return np.array(arg).astype(np.float64, order='C', copy=False)


class CTModeling(CLMModeling):
    r"""Object with functions for halo mass modeling

    Attributes
    ----------
    backend: str
        Name of the backend being used
    massdef : str
        Profile mass definition (`mean`, `critical`, `virial` - letter case independent)
    delta_mdef : int
        Mass overdensity definition.
    halo_profile_model : str
        Profile model parameterization (`nfw`, `einasto`, `hernquist` - letter case independent)
    cosmo: Cosmology
        Cosmology object
    hdpm: Object
        Backend object with halo profiles
    mdef_dict: dict
        Dictionary with the definitions for mass
    hdpm_dict: dict
        Dictionary with the definitions for profile
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, massdef='mean', delta_mdef=200, halo_profile_model='nfw',
                 validate_input=True):
        CLMModeling.__init__(self, validate_input)
        # Update class attributes
        self.backend = 'ct'
        self.mdef_dict = {'mean': 'mean'}
        self.hdpm_dict = {'nfw': 'nfw'}
        self.cosmo_class = AstroPyCosmology
        # Attributes exclusive to this class
        self.cor_factor = _patch_rho_crit_to_cd2018(2.77533742639e+11)
        # Set halo profile and cosmology
        self.set_halo_density_profile(halo_profile_model, massdef, delta_mdef)
        self.set_cosmo(None)

    def _set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        """"set halo density profile"""
        # Update values
        self.halo_profile_model = halo_profile_model
        self.massdef = massdef
        self.delta_mdef = delta_mdef

    def _set_concentration(self, cdelta):
        """" set concentration"""
        self.cdelta = cdelta

    def _set_mass(self, mdelta):
        """" set mass"""
        self.mdelta = mdelta

    def _eval_3d_density(self, r3d, z_cl):
        """"eval 3d density"""
        h = self.cosmo['h']
        Omega_m = self.cosmo.get_E2Omega_m(z_cl)*self.cor_factor
        return ct.density.rho_nfw_at_r(
            _assert_correct_type_ct(r3d)*h, self.mdelta*h,
            self.cdelta, Omega_m, delta=self.delta_mdef)*h**2

    def _eval_surface_density(self, r_proj, z_cl):
        """"eval surface density"""
        h = self.cosmo['h']
        Omega_m = self.cosmo.get_E2Omega_m(z_cl)*self.cor_factor
        return ct.deltasigma.Sigma_nfw_at_R(
            _assert_correct_type_ct(r_proj)*h, self.mdelta*h,
            self.cdelta, Omega_m, delta=self.delta_mdef)*h*1.0e12  # pc**-2 to Mpc**-2

    def _eval_mean_surface_density(self, r_proj, z_cl):
        r''' Computes the mean value of surface density inside radius r_proj

        Parameters
        ----------
        r_proj : array_like
            Projected radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster

        Returns
        -------
        array_like, float
            Excess surface density in units of :math:`M_\odot\ Mpc^{-2}`.

        Note
        ----
        This function just adds eval_surface_density+eval_excess_surface_density
        '''
        return (self.eval_surface_density(r_proj, z_cl)
                +self.eval_excess_surface_density(r_proj, z_cl))

    def _eval_excess_surface_density(self, r_proj, z_cl):
        """"eval excess surface density"""
        if np.min(r_proj) < 1.e-11:
            raise ValueError(
                f"Rmin = {np.min(r_proj):.2e} Mpc!"
                " This value is too small and may cause computational issues.")
        Omega_m = self.cosmo.get_E2Omega_m(z_cl)*self.cor_factor
        h = self.cosmo['h']
        r_proj = _assert_correct_type_ct(r_proj)*h
        # Computing sigma on a larger range than the radial range requested,
        # with at least 1000 points.
        sigma_r_proj = np.logspace(np.log10(np.min(
            r_proj))-1, np.log10(np.max(r_proj))+1, np.max([1000, 10*np.array(r_proj).size]))
        sigma = self.eval_surface_density(
            sigma_r_proj/h, z_cl)/(h*1e12)  # rm norm for ct
        # ^ Note: Let's not use this naming convention when transfering ct to ccl....
        return ct.deltasigma.DeltaSigma_at_R(
            r_proj, sigma_r_proj, sigma, self.mdelta*h,
            self.cdelta, Omega_m, delta=self.delta_mdef)*h*1.0e12  # pc**-2 to Mpc**-2


Modeling = CTModeling
