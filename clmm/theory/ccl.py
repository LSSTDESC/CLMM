"""@file ccl.py
Modeling using CCL
"""
# Functions to model halo profiles

import pyccl as ccl

import numpy as np
from scipy.interpolate import interp1d
from packaging.version import parse

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
    mdef: ccl.halos.MassDef, None
        Internal MassDef object
    conc: ccl.halos.ConcentrationConstant, None
        Internal ConcentrationConstant object
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, massdef='mean', delta_mdef=200, halo_profile_model='nfw',
                 validate_input=True):
        CLMModeling.__init__(self, validate_input)
        # Update class attributes
        self.backend = 'ccl'
        self.mdef_dict = {
            'mean': 'matter',
            'critical': 'critical',
            'virial': 'critical'}
        self.hdpm_dict = {'nfw': ccl.halos.HaloProfileNFW,
                          'einasto': ccl.halos.HaloProfileEinasto,
                          'hernquist': ccl.halos.HaloProfileHernquist}
        self.cosmo_class = CCLCosmology
        self.hdpm_opts = {'nfw': {'truncated': False,
                                  'projected_analytic': True,
                                  'cumul2d_analytic': True},
                          'einasto': {'truncated': False},
                          'hernquist': {'truncated': False,
                                        'projected_analytic': True,
                                        'cumul2d_analytic': True}}
        self.cor_factor = _patch_rho_crit_to_cd2018(ccl.physical_constants.RHO_CRITICAL)
        self.__mdelta_cor = 0.0 ## mass with corretion for input
        #self.hdpm_opts['einasto'].update({'alpha': 0.25}) # same as NC default

        # Set halo profile and cosmology
        self.mdef = None
        self.conc = None
        self.set_halo_density_profile(halo_profile_model, massdef, delta_mdef)
        self.set_cosmo(None)


    # Functions implemented by child class


    def _update_halo_density_profile(self):
        """"updates halo density profile with set internal properties"""
        # prepare mdef object
        self.mdef = ccl.halos.MassDef(self.delta_mdef, self.mdef_dict[self.massdef])
        # adjust it for ccl version > 2.7.0
        if (parse(ccl.__version__) > parse('2.7.0')
            and parse(ccl.__version__) < parse('2.7.1.dev10')):
            ccl.UnlockInstance.Funlock(type(self.mdef), "_concentration_init", True)
        # setting concentration (also updates hdpm)
        self.cdelta = self.cdelta if self.hdpm else 4.0 # ccl always needs an input concentration

    def _get_concentration(self):
        """"get concentration"""
        return self.conc.c

    def _get_mass(self):
        """"get mass"""
        return self.__mdelta_cor*self.cor_factor

    def _set_concentration(self, cdelta):
        """"set concentration. Also sets/updates hdpm"""
        self.conc = ccl.halos.ConcentrationConstant(c=cdelta, mdef=self.mdef)
        self.mdef._concentration_init(self.conc)
        self.hdpm = self.hdpm_dict[self.halo_profile_model](
            self.conc, **self.hdpm_opts[self.halo_profile_model])
        self.hdpm.update_precision_fftlog(
            padding_lo_fftlog=1e-4, padding_hi_fftlog=1e3)

    def _set_mass(self, mdelta):
        """" set mass"""
        self.__mdelta_cor = mdelta/self.cor_factor

    def _set_einasto_alpha(self, alpha):
        if alpha is None:
            self.hdpm.update_parameters(alpha='cosmo')
        else:
            self.hdpm.update_parameters(alpha=alpha)

    def _get_einasto_alpha(self, z_cl=None):
        """"get the value of the Einasto slope"""
        if self.hdpm.alpha!='cosmo':
            a_cl = 1 # a_cl does not matter in this case
        else:
            a_cl = self.cosmo.get_a_from_z(z_cl)
        return self.hdpm._get_alpha(self.cosmo.be_cosmo, self.__mdelta_cor, a_cl, self.mdef)

    def _eval_3d_density(self, r3d, z_cl):
        """"eval 3d density"""
        return self._call_ccl_profile_lens(self.hdpm.real, r3d, z_cl, ndim=3)

    def _eval_surface_density(self, r_proj, z_cl):
        """eval surface density"""
        return self._call_ccl_profile_lens(self.hdpm.projected, r_proj, z_cl)

    def _eval_mean_surface_density(self, r_proj, z_cl):
        """"eval mean surface density"""
        return self._call_ccl_profile_lens(self.hdpm.cumul2d, r_proj, z_cl)

    def _eval_excess_surface_density(self, r_proj, z_cl):
        """"eval excess surface density"""
        return self._eval_mean_surface_density(r_proj, z_cl) \
               - self._eval_surface_density(r_proj, z_cl)

    def _eval_convergence_core(self, r_proj, z_cl, z_src):
        """eval convergence"""
        return self._call_ccl_profile_lens_src(self.hdpm.convergence, r_proj, z_cl, z_src)

    def _eval_tangential_shear_core(self, r_proj, z_cl, z_src):
        """eval tangential shear"""
        return self._call_ccl_profile_lens_src(self.hdpm.shear, r_proj, z_cl, z_src)

    def _eval_reduced_tangential_shear_core(self, r_proj, z_cl, z_src):
        """eval reduced tangential shear with all background sources at the same plane"""
        return self._call_ccl_profile_lens_src(self.hdpm.reduced_shear, r_proj, z_cl, z_src)

    def _eval_magnification_core(self, r_proj, z_cl, z_src):
        """eval magnification"""
        return self._call_ccl_profile_lens_src(self.hdpm.magnification, r_proj, z_cl, z_src)

    # Helper functions unique to this class

    def _call_ccl_profile_lens(self, ccl_hdpm_func, radius, z_lens, ndim=2):
        """"call ccl profile functions that depend on the lens only"""
        a_lens = self.cosmo.get_a_from_z(z_lens)

        return ccl_hdpm_func(self.cosmo.be_cosmo, radius/a_lens, self.__mdelta_cor,
                             a_lens, self.mdef)*self.cor_factor/a_lens**ndim

    def _call_ccl_profile_lens_src(self, ccl_hdpm_func, radius, z_lens, z_src):
        """"call ccl profile functions that depend on the lens and the sources"""
        a_lens = self.cosmo.get_a_from_z(z_lens)
        a_src = self.cosmo.get_a_from_z(z_src)

        return ccl_hdpm_func(self.cosmo.be_cosmo, radius/a_lens, self.__mdelta_cor,
                             a_lens, a_src, self.mdef)

Modeling = CCLCLMModeling
