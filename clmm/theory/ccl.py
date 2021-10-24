"""@file ccl.py
Modeling using CCL
"""
# Functions to model halo profiles

import pyccl as ccl

import numpy as np
from scipy.interpolate import interp1d
import warnings
from packaging import version

from . import func_layer
from . func_layer import *

from .parent_class import CLMModeling

from .. utils import _patch_rho_crit_to_cd2018

from .. cosmology.ccl import CCLCosmology
Cosmology = CCLCosmology

__all__ = ['CCLCLMModeling', 'Modeling', 'Cosmology']+func_layer.__all__


class CCLCLMModeling(CLMModeling):
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

    def __init__(self, massdef='mean', delta_mdef=200, halo_profile_model='nfw'):
        CLMModeling.__init__(self)
        # Update class attributes
        self.backend = 'ccl'
        self.mdef_dict = {
            'mean': 'matter',
            'critical': 'critical',
            'virial': 'critical'}
        self.hdpm_dict = {'nfw': ccl.halos.HaloProfileNFW,
                          'einasto': ccl.halos.HaloProfileEinasto,
                          'hernquist': ccl.halos.HaloProfileHernquist}
        # Uncomment lines below when CCL einasto and hernquist profiles are stable (also add version number)
        #if version.parse(ccl.__version__) >= version.parse('???'):
        # self.hdpm_dict.update({
        #        'einasto': ccl.halos.HaloProfileEinasto,
        #        'hernquist': ccl.halos.HaloProfileHernquist})
        # Attributes exclusive to this class
        self.hdpm_opts = {'nfw': {'truncated': False,
                                  'projected_analytic': True,
                                  'cumul2d_analytic': True},
                          'einasto': {'truncated': False},
                          'hernquist': {'truncated': False}}
        self.mdelta = 0.0
        self.cor_factor = _patch_rho_crit_to_cd2018(ccl.physical_constants.RHO_CRITICAL)

        # Set halo profile and cosmology
        self.set_halo_density_profile(halo_profile_model, massdef, delta_mdef)
        self.set_cosmo(None)

    def set_cosmo(self, cosmo):
        """"set cosmo"""
        self._set_cosmo(cosmo, CCLCosmology)

    def set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        """"set halo density profile"""
        # Check if choices are supported and put in lower case
        massdef, halo_profile_model = self.validate_definitions(
            massdef, halo_profile_model)

        # Check if we have already an instance of the required object, if not create one
        if not ((halo_profile_model==self.halo_profile_model)
                and (massdef == self.massdef)
                and (delta_mdef == self.delta_mdef)):
            self.halo_profile_model = halo_profile_model
            self.massdef = massdef

            cur_cdelta = 0.0
            cur_values = False
            if self.hdpm:
                cur_cdelta = self.conc.c
                cur_values = True

            self.mdef = ccl.halos.MassDef(delta_mdef, self.mdef_dict[massdef])
            self.conc = ccl.halos.ConcentrationConstant(c=4.0, mdef=self.mdef)
            self.mdef.concentration = self.conc
            self.hdpm = self.hdpm_dict[halo_profile_model](
                self.conc, **self.hdpm_opts[halo_profile_model])
            if cur_values:
                self.conc.c = cur_cdelta
        

    def _set_concentration(self, cdelta):
        """" set concentration"""
        self.conc.c = cdelta

    def _set_mass(self, mdelta):
        """" set mass"""
        self.mdelta = mdelta/self.cor_factor

    def eval_3d_density(self, r3d, z_cl):
        """"eval 3d density"""
        a_cl = self.cosmo.get_a_from_z(z_cl)
        dens = self.hdpm.real(
            self.cosmo.be_cosmo, r3d/a_cl, self.mdelta, a_cl, self.mdef)
        
        if self.halo_profile_model == 'einasto':
            alpha = self.hdpm._get_alpha (self.cosmo.be_cosmo, self.mdelta, a_cl, self.mdef)
            print(f"Einasto alpha = {alpha}")

        return dens*self.cor_factor/a_cl**3

    def eval_surface_density(self, r_proj, z_cl):
        a_cl = self.cosmo.get_a_from_z(z_cl)
        if self.halo_profile_model == 'nfw':
            return self.hdpm.projected(self.cosmo.be_cosmo, r_proj/a_cl, self.mdelta, a_cl, self.mdef)*self.cor_factor/a_cl**2
        else:
            rtmp = np.geomspace(np.min(r_proj)/10., np.max(r_proj)*10., 1000)
            tmp = self.hdpm.projected (self.cosmo.be_cosmo, rtmp/a_cl, self.mdelta, a_cl, self.mdef)*self.cor_factor/a_cl**2
            ptf = interp1d(np.log(rtmp), np.log(tmp), bounds_error=False, fill_value=-100)
            return np.exp(ptf(np.log(r_proj)))


    def eval_mean_surface_density(self, r_proj, z_cl):
        a_cl = self.cosmo.get_a_from_z(z_cl)
        if self.halo_profile_model =='nfw':
            return self.hdpm.cumul2d(self.cosmo.be_cosmo, r_proj/a_cl, self.mdelta, self.cosmo.get_a_from_z(z_cl), self.mdef)*self.cor_factor/a_cl**2
        else:
            rtmp = np.geomspace(np.min(r_proj)/10., np.max(r_proj)*10., 1000)
            tmp = self.hdpm.cumul2d (self.cosmo.be_cosmo, rtmp/a_cl, self.mdelta, a_cl, self.mdef)*self.cor_factor/a_cl**2
            ptf = interp1d(np.log(rtmp), np.log(tmp), bounds_error=False, fill_value=-100)
            return np.exp(ptf(np.log(r_proj)))

    def eval_excess_surface_density(self, r_proj, z_cl):
        a_cl = self.cosmo.get_a_from_z(z_cl)
        r_cor = r_proj/a_cl

        if self.halo_profile_model =='nfw':
            return (self.hdpm.cumul2d(self.cosmo.be_cosmo, r_cor, self.mdelta, self.cosmo.get_a_from_z(z_cl), self.mdef)-
                    self.hdpm.projected(self.cosmo.be_cosmo, r_cor, self.mdelta, self.cosmo.get_a_from_z(z_cl), self.mdef))*self.cor_factor/a_cl**2
        else:
            return self.eval_mean_surface_density(r_proj, z_cl) - self.eval_surface_density(r_proj, z_cl)

    def eval_tangential_shear(self, r_proj, z_cl, z_src):
        """"eval tangential shear"""
        sigma_excess = self.eval_excess_surface_density(r_proj, z_cl)
        sigma_crit = self.eval_critical_surface_density(z_cl, z_src)

        return sigma_excess/sigma_crit

    def eval_convergence(self, r_proj, z_cl, z_src):
        """"eval convergence"""
        sigma = self.eval_surface_density(r_proj, z_cl)
        sigma_crit = self.eval_critical_surface_density(z_cl, z_src)

        return sigma/sigma_crit

    def eval_reduced_tangential_shear(self, r_proj, z_cl, z_src):
        """"eval reduced tangential shear"""
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_tangential_shear(r_proj, z_cl, z_src)

        return np.divide(gamma_t, (1-kappa))

    def eval_magnification(self, r_proj, z_cl, z_src):
        """"eval magnification"""
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_tangential_shear(r_proj, z_cl, z_src)

        return 1.0/((1.0-kappa)**2-np.abs(gamma_t)**2)

Modeling = CCLCLMModeling
