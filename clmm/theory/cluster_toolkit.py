# Functions to model halo profiles

import cluster_toolkit as ct

import numpy as np
import warnings

from astropy import units
from astropy.cosmology import LambdaCDM, FlatLambdaCDM

from .. constants import Constants as const
from . import func_layer
from . func_layer import *

from .parent_class import CLMModeling

from .. utils import _patch_rho_crit_to_cd2018

from .. cosmology.cluster_toolkit import AstroPyCosmology
Cosmology = AstroPyCosmology

__all__ = ['CTModeling', 'Modeling', 'Cosmology']+func_layer.__all__


def _assert_correct_type_ct(a):
    """ Convert the argument to a type compatible with cluster_toolkit
    cluster_toolkit does not handle correctly scalar arguments that are
    not float or numpy array and others that contain non-float64 elements.
    It only convert lists to the correct type. To circumvent this we
    pre-convert all arguments going to cluster_toolkit to the appropriated
    types.
    Parameters
    ----------
    a : array_like or scalar
    Returns
    -------
    scale_factor : array_like
        Scale factor
    """
    if np.isscalar(a):
        return float(a)
    else:
        return np.array(a).astype(np.float64, order='C', copy=False)


class CTModeling(CLMModeling):

    def __init__(self, massdef='mean', delta_mdef=200, halo_profile_model='nfw'):
        CLMModeling.__init__(self)
        # Update class attributes
        self.backend = 'ct'
        self.mdef_dict = {'mean': 'mean'}
        self.hdpm_dict = {'nfw': 'nfw'}
        # Attributes exclusive to this class
        self.cor_factor = _patch_rho_crit_to_cd2018(2.77533742639e+11)
        # Set halo profile and cosmology
        self.set_halo_density_profile(halo_profile_model, massdef, delta_mdef)
        self.set_cosmo(None)

    def set_cosmo(self, cosmo):
        self._set_cosmo(cosmo, AstroPyCosmology)

    def set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        # Check if choices are supported and put in lower case
        massdef, halo_profile_model = self.validate_definitions(massdef, halo_profile_model)
        # Update values
        self.halo_profile_model = halo_profile_model
        self.massdef = massdef
        self.delta_mdef = delta_mdef

    def _set_concentration(self, cdelta):
        self.cdelta = cdelta

    def _set_mass(self, mdelta):
        self.mdelta = mdelta

    def eval_3d_density(self, r3d, z_cl):
        h = self.cosmo['h']
        Omega_m = self.cosmo.get_E2Omega_m(z_cl)*self.cor_factor
        return ct.density.rho_nfw_at_r(_assert_correct_type_ct(r3d)*h, self.mdelta*h,
                self.cdelta, Omega_m, delta=self.delta_mdef)*h**2

    def eval_surface_density(self, r_proj, z_cl):
        h = self.cosmo['h']
        Omega_m = self.cosmo.get_E2Omega_m(z_cl)*self.cor_factor
        return ct.deltasigma.Sigma_nfw_at_R(_assert_correct_type_ct(r_proj)*h, self.mdelta*h,
                self.cdelta, Omega_m, delta=self.delta_mdef)*h*1.0e12 # pc**-2 to Mpc**-2

    def eval_mean_surface_density(self, r_proj, z_cl):
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
        return self.eval_surface_density(r_proj, z_cl)+self.eval_excess_surface_density(r_proj, z_cl)

    def eval_excess_surface_density(self, r_proj, z_cl):
        if np.min(r_proj)<1.e-11:
            raise ValueError(f"Rmin = {np.min(r_proj):.2e}! This value is too small and may cause computational issues.")
        Omega_m = self.cosmo.get_E2Omega_m(z_cl)*self.cor_factor
        h = self.cosmo['h']
        r_proj = _assert_correct_type_ct(r_proj)*h
        # Computing sigma on a larger range than the radial range requested, with at least 1000 points.
        sigma_r_proj = np.logspace(np.log10(np.min(r_proj))-1, np.log10(np.max(r_proj))+1, np.max([1000,10*np.array(r_proj).size]))
        sigma = self.eval_surface_density(sigma_r_proj/h, z_cl)/(h*1e12) # rm norm for ct
        # ^ Note: Let's not use this naming convention when transfering ct to ccl....
        return ct.deltasigma.DeltaSigma_at_R(r_proj, sigma_r_proj, sigma, self.mdelta*h,
                self.cdelta, Omega_m, delta=self.delta_mdef)*h*1.0e12 # pc**-2 to Mpc**-2

    def eval_tangential_shear(self, r_proj, z_cl, z_src):
        delta_sigma = self.eval_excess_surface_density(r_proj, z_cl)
        sigma_c = self.eval_critical_surface_density(z_cl, z_src)
        return delta_sigma/sigma_c

    def eval_convergence(self, r_proj, z_cl, z_src):
        sigma = self.eval_surface_density(r_proj, z_cl)
        sigma_c = self.eval_critical_surface_density(z_cl, z_src)
        return sigma/sigma_c

    def eval_reduced_tangential_shear(self, r_proj, z_cl, z_src):
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_tangential_shear(r_proj, z_cl, z_src)
        return np.divide(gamma_t, (1-kappa))

    def eval_magnification(self, r_proj, z_cl, z_src):
        # The magnification is computed taking into account just the tangential shear. This is valid for
        # spherically averaged profiles, e.g., NFW and Einasto (by construction the cross shear is zero).
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_tangential_shear(r_proj, z_cl, z_src)
        return 1./((1-kappa)**2-abs(gamma_t)**2)


Modeling = CTModeling
