# Functions to model halo profiles

import pyccl as ccl

import numpy as np
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

    def __init__(self, massdef='mean', delta_mdef=200, halo_profile_model='nfw'):
        CLMModeling.__init__(self)
        # Update class attributes
        self.backend = 'ccl'
        self.mdef_dict = {
            'mean': 'matter',
            'critical': 'critical',
            'virial': 'critical'}
        self.hdpm_dict = {'nfw': ccl.halos.HaloProfileNFW}
        # Only add the options of einasto and hernquist if CLL version >= 10(?)
        # because results below this version are unstable.
        if version.parse(ccl.__version__) >= version.parse('10'):
            self.hdpm_dict.update({
                'einasto': ccl.halos.HaloProfileEinasto,
                'hernquist': ccl.halos.HaloProfileHernquist})
        # Attributes exclusive to this class
        self.hdpm_opts = {'nfw': {'truncated': False,
                                  'projected_analytic': True,
                                  'cumul2d_analytic': True},
                          'einasto': {},
                          'hernquist': {}}
        self.MDelta = 0.0
        self.cor_factor = _patch_rho_crit_to_cd2018(ccl.physical_constants.RHO_CRITICAL)
        # Set halo profile and cosmology
        self.set_halo_density_profile(halo_profile_model, massdef, delta_mdef)
        self.set_cosmo(None)

    def set_cosmo(self, cosmo):
        self._set_cosmo(cosmo, CCLCosmology)

    def set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        # Check if choices are supported
        self.validate_definitions(massdef, halo_profile_model)

        # Check if we have already an instance of the required object, if not create one
        if not((halo_profile_model == self.halo_profile_model) and (massdef == self.massdef) and (delta_mdef == self.delta_mdef)):
            self.halo_profile_model = halo_profile_model
            self.massdef = massdef

            cur_cdelta = 0.0
            cur_mdelta = 0.0
            cur_values = False
            if self.hdpm:
                cur_cdelta = self.conc.c
                cur_values = True

            self.mdef = ccl.halos.MassDef(delta_mdef, self.mdef_dict[massdef])
            self.conc = ccl.halos.ConcentrationConstant(c=4.0, mdef=self.mdef)
            self.mdef.concentration = self.conc
            self.hdpm = self.hdpm_dict[halo_profile_model](self.conc, **self.hdpm_opts[halo_profile_model])
            if cur_values:
                self.conc.c = cur_cdelta

    def set_concentration(self, cdelta):
        self.conc.c = cdelta

    def set_mass(self, mdelta):
        self.MDelta = mdelta/self.cor_factor

    def eval_density(self, r3d, z_cl):
        a_cl = self.cosmo._get_a_from_z(z_cl)
        return self.hdpm.real(self.cosmo.be_cosmo, r3d/a_cl, self.MDelta, a_cl, self.mdef)*self.cor_factor/a_cl**3

    def eval_sigma(self, r_proj, z_cl):
        a_cl = self.cosmo._get_a_from_z(z_cl)
        return self.hdpm.projected(self.cosmo.be_cosmo, r_proj/a_cl, self.MDelta, a_cl, self.mdef)*self.cor_factor/a_cl**2

    def eval_sigma_mean(self, r_proj, z_cl):
        a_cl = self.cosmo._get_a_from_z(z_cl)
        return self.hdpm.cumul2d(self.cosmo.be_cosmo, r_proj/a_cl, self.MDelta, self.cosmo._get_a_from_z(z_cl), self.mdef)*self.cor_factor/a_cl**2

    def eval_sigma_excess(self, r_proj, z_cl):
        a_cl = self.cosmo._get_a_from_z(z_cl)
        r_cor = r_proj/a_cl

        return (self.hdpm.cumul2d(self.cosmo.be_cosmo, r_cor, self.MDelta, self.cosmo._get_a_from_z(z_cl), self.mdef)-
                self.hdpm.projected(self.cosmo.be_cosmo, r_cor, self.MDelta, self.cosmo._get_a_from_z(z_cl), self.mdef))*self.cor_factor/a_cl**2

    def eval_shear(self, r_proj, z_cl, z_src):
        sigma_excess = self.eval_sigma_excess(r_proj, z_cl)
        sigma_crit = self.eval_sigma_crit(z_cl, z_src)

        return sigma_excess/sigma_crit

    def eval_convergence(self, r_proj, z_cl, z_src):
        sigma = self.eval_sigma(r_proj, z_cl)
        sigma_crit = self.eval_sigma_crit(z_cl, z_src)

        return np.nan_to_num(sigma/sigma_crit, nan=np.nan, posinf=np.inf, neginf=-np.inf)

    def eval_reduced_shear(self, r_proj, z_cl, z_src):
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_shear(r_proj, z_cl, z_src)

        return np.nan_to_num(np.divide(gamma_t, (1-kappa)), nan=np.nan, posinf=np.inf, neginf=-np.inf)

    def eval_magnification(self, r_proj, z_cl, z_src):
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_shear(r_proj, z_cl, z_src)

        return 1.0/((1.0-kappa)**2-np.abs(gamma_t)**2)


Modeling = CCLCLMModeling
