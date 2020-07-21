# Functions to model halo profiles 

import pyccl as ccl

from .cluster_toolkit_patches import _patch_comoving_coord_cluster_toolkit_rho_m

import numpy as np
import warnings

from . import func_layer
from . func_layer import *
from . clmm_modeling import CLMModeling

__all__ = ['CCLCLMModeling', 'Modeling'] + func_layer.__all__

class CCLCLMModeling (CLMModeling):
    def __init__ (self, massdef = 'mean', delta_mdef = 200, halo_profile_model = 'nfw', z_max = 5.0):

        self.mdef_dict = {'mean':      'matter', 
                          'critial':   'critical',
                          'virial':    'critical'}
        self.hdpm_dict = {'nfw':       ccl.halos.HaloProfileNFW, 
                          'einasto':   ccl.halos.HaloProfileEinasto,
                          'hernquist': ccl.halos.HaloProfileHernquist}
        self.hdpm_opts = {'nfw': {'truncated': False, 
                                  'projected_analytic': True, 
                                  'cumul2d_analytic': True},
                          'einasto': {},
                          'hernquist': {}}

        self.halo_profile_model = ''
        self.massdef = ''
        self.delta_mdef = 0
        self.hdpm = None
        self.MDelta = 0.0

        self.set_cosmo_params_dict ({})
        self.set_halo_density_profile (halo_profile_model, massdef, delta_mdef)
        
        self.cor_factor = 2.77533742639e+11 * _patch_comoving_coord_cluster_toolkit_rho_m (1.0, 0.0) / ccl.physical_constants.RHO_CRITICAL

    def set_cosmo_params_dict (self, cosmo_dict):
        h = 0.67
        Omega_c = 0.27
        Omega_b = 0.045
        
        if 'H0' in cosmo_dict:
            h = cosmo_dict['H0'] / 100.0
        if 'Omega_b' in cosmo_dict:
            Omega_b = cosmo_dict['Omega_b']
        if 'Omega_c' in cosmo_dict:
            Omega_c = cosmo_dict['Omega_c']
        
        self.cosmo = ccl.Cosmology (Omega_c = Omega_c, Omega_b = Omega_b, h = h, sigma8 = 0.8, n_s = 0.96, T_CMB = 0.0, Neff = 0.0,
                                    transfer_function='bbks', matter_power_spectrum='linear')
        

    def set_halo_density_profile (self, halo_profile_model = 'nfw', massdef = 'mean', delta_mdef = 200):
        # Check if choices are supported
        if not halo_profile_model in self.hdpm_dict:
            raise ValueError (f"Halo density profile model {halo_profile_model} not currently supported")
        if not massdef in self.mdef_dict:
            raise ValueError (f"Halo density profile mass definition {massdef} not currently supported")

        # Check if we have already an instance of the required object, if not create one
        if not ((halo_profile_model == self.halo_profile_model) and (massdef == self.massdef) and (delta_mdef == self.delta_mdef)):
            self.halo_profile_model = halo_profile_model
            self.massdef = massdef
            
            cur_cdelta = 0.0
            cur_mdelta = 0.0
            cur_values = False
            if self.hdpm:
                cur_cdelta = self.conc.c
                cur_values = True
            
            self.mdef = ccl.halos.MassDef (delta_mdef, self.mdef_dict[massdef])
            self.conc = ccl.halos.ConcentrationConstant (self.mdef)
            self.mdef.concentration = self.conc
            self.hdpm = self.hdpm_dict[halo_profile_model] (self.conc, **self.hdpm_opts[halo_profile_model])
            if cur_values:
                self.conc.c = cur_cdelta

    def set_concentration (self, cdelta):
        self.conc.c = cdelta

    def set_mass (self, mdelta):
        self.MDelta = mdelta / self.cosmo['h'] / self.cor_factor

    def eval_da_z1z2 (self, z1, z2):
        return ccl.angular_diameter_distance (self.cosmo, _get_a_from_z (z1), _get_a_from_z (z2)) * self.cosmo['h'] * 1.0e6

    def eval_sigma_crit (self, z_len, z_src):
        a_len = _get_a_from_z (z_len)
        a_src = np.atleast_1d (_get_a_from_z (z_src))
        cte   = ccl.physical_constants.CLIGHT**2 / (4.0 * np.pi * ccl.physical_constants.GNEWT * ccl.physical_constants.SOLAR_MASS) * ccl.physical_constants.PC_TO_METER

        z_cut = (a_src < a_len)
        if np.isscalar (a_len):
            a_len = np.repeat (a_len, len (a_src))

        res = np.zeros_like (a_src)
        
        if np.any (z_cut):
            Ds  = ccl.angular_diameter_distance (self.cosmo, a_src[z_cut])
            Dl  = ccl.angular_diameter_distance (self.cosmo, a_len)
            Dls = ccl.angular_diameter_distance (self.cosmo, a_len, a_src[z_cut])
        
            res[z_cut] = (cte * Ds / (Dl * Dls)) * self.cor_factor * 1.0e-6 / self.cosmo['h']

        res[~z_cut] = np.Inf

        return np.squeeze (res)

    def eval_density (self, r3d, z_cl):
        a_cl = _get_a_from_z (z_cl)
        return self.hdpm.real (self.cosmo, r3d / (self.cosmo['h'] * a_cl), self.MDelta, a_cl, self.mdef) * self.cor_factor / (self.cosmo['h']**2 * a_cl**3)

    def eval_sigma (self, r_proj, z_cl):
        a_cl = _get_a_from_z (z_cl)
        return self.hdpm.projected (self.cosmo, r_proj / (self.cosmo['h'] * a_cl), self.MDelta, a_cl, self.mdef) * self.cor_factor * 1.0e-12 / (self.cosmo['h'] * a_cl**2)

    def eval_sigma_mean (self, r_proj, z_cl):
        a_cl = _get_a_from_z (z_cl)
        return self.hdpm.cumul2d (self.cosmo, r_proj / (self.cosmo['h'] * a_cl), self.MDelta, _get_a_from_z (z_cl), self.mdef) * self.cor_factor * 1.0e-12 / (self.cosmo['h'] * a_cl**2)

    def eval_sigma_excess (self, r_proj, z_cl):
        a_cl = _get_a_from_z (z_cl)
        r_cor = r_proj / (self.cosmo['h'] * a_cl)
        
        return (self.hdpm.cumul2d (self.cosmo, r_cor, self.MDelta, _get_a_from_z (z_cl), self.mdef) - 
                self.hdpm.projected (self.cosmo, r_cor, self.MDelta, _get_a_from_z (z_cl), self.mdef)) * self.cor_factor * 1.0e-12 / (self.cosmo['h'] * a_cl**2)

    def eval_shear (self, r_proj, z_cl, z_src):
        sigma_excess = self.eval_sigma_excess (r_proj, z_cl) 
        sigma_crit   = self.eval_sigma_crit (z_cl, z_src)
        
        return sigma_excess / sigma_crit

    def eval_convergence (self, r_proj, z_cl, z_src):
        sigma      = self.eval_sigma (r_proj, z_cl) 
        sigma_crit = self.eval_sigma_crit (z_cl, z_src)
        
        return np.nan_to_num (sigma / sigma_crit,  nan=np.nan, posinf=np.inf, neginf=-np.inf)

    def eval_reduced_shear (self, r_proj, z_cl, z_src):
        kappa   = self.eval_convergence (r_proj, z_cl, z_src)
        gamma_t = self.eval_shear (r_proj, z_cl, z_src)
        
        return np.nan_to_num(np.divide(gamma_t , (1 - kappa)),  nan=np.nan, posinf=np.inf, neginf=-np.inf)

    def eval_magnification (self, r_proj, z_cl, z_src):
        kappa   = self.eval_convergence (r_proj, z_cl, z_src)
        gamma_t = self.eval_shear (r_proj, z_cl, z_src)

        return 1.0 / ((1.0 - kappa)**2 - np.abs (gamma_t)**2)


Modeling = CCLCLMModeling

func_layer.gcm = CCLCLMModeling ()

