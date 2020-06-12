""" NumCosmo implementation of CLMModeling """

import gi
gi.require_version('NumCosmo', '1.0')
gi.require_version('NumCosmoMath', '1.0')
from gi.repository import NumCosmo as Nc
from gi.repository import NumCosmoMath as Ncm

from .cluster_toolkit_patches import _patch_zevolution_cluster_toolkit_rho_m

import numpy as np
import warnings

from . import func_layer
from . func_layer import *
from . clmm_modeling import CLMModeling

__all__ = ['NumCosmoCLMModeling'] + func_layer.__all__

class NumCosmoCLMModeling (CLMModeling):
    def __init__ (self, massdef = 'mean', delta_mdef = 200, halo_profile_model = 'nfw', z_max = 5.0):
        Ncm.cfg_init ()
        self.cosmo = Nc.HICosmo.new_from_name (Nc.HICosmo, "NcHICosmoDEXcdm")
        self.cosmo.omega_x2omega_k ()
        self.cosmo.param_set_by_name ("w",      -1.0)
        self.cosmo.param_set_by_name ("Omegak",  0.0)
        self.cosmo.param_set_by_name ("Tgamma0", 0.0)
        
        self.mdef_dict = {'mean':      Nc.HaloDensityProfileMassDef.MEAN, 
                          'critial':   Nc.HaloDensityProfileMassDef.CRITICAL,
                          'virial':    Nc.HaloDensityProfileMassDef.VIRIAL}
        self.hdpm_dict = {'nfw':       Nc.HaloDensityProfileNFW.new, 
                          'einasto':   Nc.HaloDensityProfileEinasto.new,
                          'hernquist': Nc.HaloDensityProfileHernquist.new}
                          
        self.halo_profile_model = ''
        self.massdef = ''
        self.delta_mdef = 0
        self.hdpm = None

        self.set_halo_density_profile (halo_profile_model, massdef, delta_mdef)
        
        self.dist = Nc.Distance.new (z_max)
        self.dist.prepare (self.cosmo)
        
        self.smd = Nc.WLSurfaceMassDensity.new (self.dist)
        self.smd.prepare (self.cosmo)
        
        self.cor_factor = 2.77533742639e+11 * _patch_zevolution_cluster_toolkit_rho_m (1.0, 0.0) / Ncm.C.crit_mass_density_h2_solar_mass_Mpc3 ()

    def set_cosmo_params_dict (self, cosmo_dict):
        if 'H0' in cosmo_dict:
            self.cosmo.param_set_by_name ("H0", cosmo_dict['H0'])
        if 'Omega_b' in cosmo_dict:
            self.cosmo.param_set_by_name ("Omegab", cosmo_dict['Omega_b'])
        if 'Omega_c' in cosmo_dict:
            self.cosmo.param_set_by_name ("Omegac", cosmo_dict['Omega_c'])
        self.dist.prepare_if_needed (self.cosmo)
        self.smd.prepare_if_needed (self.cosmo)

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
                cur_cdelta = self.hdpm.props.cDelta
                cur_mdelta = self.hdpm.props.MDelta
                cur_values = True
            
            self.hdpm = self.hdpm_dict[halo_profile_model] (self.mdef_dict[massdef], delta_mdef)
            if cur_values:
                self.hdpm.props.cDelta = cur_cdelta
                self.hdpm.props.MDelta = cur_mdelta

    def set_concentration (self, cdelta):
        self.hdpm.props.cDelta = cdelta

    def set_mass (self, mdelta):
        self.hdpm.props.MDelta = mdelta / self.cosmo.h () / self.cor_factor

    def eval_da_z1z2 (self, z1, z2):
        h   = self.cosmo.h ()
        fac = self.cosmo.RH_Mpc () * h * 1.0e6
                
        f = lambda zi, zf: self.dist.angular_diameter_z1_z2 (self.cosmo, zi, zf) * fac
        return np.vectorize (f) (z1, z2)

    def eval_sigma_crit (self, z_len, z_src):
        h   = self.cosmo.h ()
        fac = self.cor_factor * 1.0e-12 / h
                
        f = lambda z_len, z_src: self.smd.sigma_critical (self.cosmo, z_src, z_len, z_len) * fac if z_src > z_len else np.inf
        return np.vectorize (f) (z_len, z_src)

    def eval_density (self, r3d, z_cl):
        h   = self.cosmo.h ()
        h2  = h * h
        fac = self.cor_factor / h2
        
        f = lambda r3d, z_cl: self.hdpm.eval_density (self.cosmo, r3d / h, z_cl) * fac
        return np.vectorize (f) (r3d, z_cl)

    def eval_sigma (self, r_proj, z_cl):
        h   = self.cosmo.h ()
        fac = self.cor_factor * 1.0e-12 / h
        
        f = lambda r_proj, z_cl: self.smd.sigma (self.hdpm, self.cosmo, r_proj / h, z_cl) * fac
        return np.vectorize (f) (r_proj, z_cl)

    def eval_sigma_mean (self, r_proj, z_cl):
        h   = self.cosmo.h ()
        fac = self.cor_factor * 1.0e-12 / h
                
        f = lambda r_proj, z_cl: self.smd.sigma_mean (self.hdpm, self.cosmo, r_proj / h, z_cl) * fac
        return np.vectorize (f) (r_proj, z_cl)

    def eval_sigma_excess (self, r_proj, z_cl):
        h   = self.cosmo.h ()
        fac = self.cor_factor * 1.0e-12 / h

        f = lambda r_proj, z_cl: self.smd.sigma_excess (self.hdpm, self.cosmo, r_proj / h, z_cl) * fac
        return np.vectorize (f) (r_proj, z_cl)

    def eval_shear (self, r_proj, z_cl, z_src):
        h   = self.cosmo.h ()

        f = lambda r_proj, z_src, z_cl: self.smd.shear (self.hdpm, self.cosmo, r_proj / h, z_src, z_cl, z_cl) if z_src > z_cl else 0.0
        return np.vectorize (f) (r_proj, z_src, z_cl)

func_layer.gcm = NumCosmoCLMModeling ()

