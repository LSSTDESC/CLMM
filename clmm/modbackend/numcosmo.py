# NumCosmo implementation of CLMModeling

import gi
gi.require_version('NumCosmo', '1.0')
gi.require_version('NumCosmoMath', '1.0')
from gi.repository import NumCosmo as Nc
from gi.repository import NumCosmoMath as Ncm

import math
import numpy as np
import warnings

from . import func_layer
from . func_layer import *
from .. clmm_cosmo import CLMMCosmology
from .. clmm_modeling import CLMModeling

__all__ = ['NumCosmoCLMModeling', 'Modeling', 'Cosmology'] + func_layer.__all__

class NumCosmoCLMModeling (CLMModeling):
    def __init__ (self, massdef = 'mean', delta_mdef = 200, halo_profile_model = 'nfw'):
        Ncm.cfg_init ()
        
        self.backend = 'nc'
        
        self.set_cosmo (None)
        
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
        
    def set_cosmo (self, cosmo):
        if cosmo:
            if not isinstance (cosmo, NumCosmoCosmology):
                raise ValueError (f"Incompatible cosmology object {cosmo}.")
            self.cosmo = cosmo
        else:
            self.cosmo = NumCosmoCosmology ()

        self.smd = Nc.WLSurfaceMassDensity.new (self.cosmo.dist)
        self.smd.prepare_if_needed (self.cosmo.be_cosmo)

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
                cur_log10_mdelta = self.hdpm.props.log10MDelta
                cur_values = True
            
            self.hdpm = self.hdpm_dict[halo_profile_model] (self.mdef_dict[massdef], delta_mdef)
            if cur_values:
                self.hdpm.props.cDelta = cur_cdelta
                self.hdpm.props.log10MDelta = cur_log10_mdelta
                
    def get_mset (self):
        mset = Ncm.MSet.empty_new ()
        mset.set (self.cosmo.be_cosmo)
        mset.set (self.hdpm)
        mset.set (self.smd)
        return mset
        
    def set_mset (self, mset):
        
        self.cosmo.set_be_cosmo (mset.get (Nc.HICosmo.id ()))
    
        self.hdpm  = mset.get (Nc.HaloDensityProfile.id ())
        self.smd   = mset.get (Nc.WLSurfaceMassDensity.id ())

        self.smd.prepare_if_needed (self.cosmo.be_cosmo)
        
    def set_concentration (self, cdelta):
        self.hdpm.props.cDelta = cdelta

    def set_mass (self, mdelta):
        self.hdpm.props.log10MDelta = math.log10 (mdelta)

    def eval_da_z1z2 (self, z1, z2):
        fac = self.cosmo.be_cosmo.RH_Mpc ()
                
        f = lambda zi, zf: self.cosmo.dist.angular_diameter_z1_z2 (self.cosmo.be_cosmo, zi, zf) * fac
        return np.vectorize (f) (z1, z2)

    def eval_sigma_crit (self, z_len, z_src):
    
        self.smd.prepare_if_needed (self.cosmo.be_cosmo)
        f = lambda z_len, z_src: self.smd.sigma_critical (self.cosmo.be_cosmo, z_src, z_len, z_len)
        return np.vectorize (f) (z_len, z_src)

    def eval_density (self, r3d, z_cl):
        
        f = lambda r3d, z_cl: self.hdpm.eval_density (self.cosmo.be_cosmo, r3d, z_cl)
        return np.vectorize (f) (r3d, z_cl)

    def eval_sigma (self, r_proj, z_cl):
        
        self.smd.prepare_if_needed (self.cosmo.be_cosmo)
        f = lambda r_proj, z_cl: self.smd.sigma (self.hdpm, self.cosmo.be_cosmo, r_proj, z_cl)
        return np.vectorize (f) (r_proj, z_cl)

    def eval_sigma_mean (self, r_proj, z_cl):
                
        self.smd.prepare_if_needed (self.cosmo.be_cosmo)
        f = lambda r_proj, z_cl: self.smd.sigma_mean (self.hdpm, self.cosmo.be_cosmo, r_proj, z_cl)
        return np.vectorize (f) (r_proj, z_cl)

    def eval_sigma_excess (self, r_proj, z_cl):

        self.smd.prepare_if_needed (self.cosmo.be_cosmo)
        f = lambda r_proj, z_cl: self.smd.sigma_excess (self.hdpm, self.cosmo.be_cosmo, r_proj, z_cl) 
        return np.vectorize (f) (r_proj, z_cl)

    def eval_shear (self, r_proj, z_cl, z_src):

        self.smd.prepare_if_needed (self.cosmo.be_cosmo)
        f = lambda r_proj, z_src, z_cl: self.smd.shear (self.hdpm, self.cosmo.be_cosmo, r_proj, z_src, z_cl, z_cl)
        return np.vectorize (f) (r_proj, z_src, z_cl)

    def eval_convergence (self, r_proj, z_cl, z_src):

        self.smd.prepare_if_needed (self.cosmo.be_cosmo)
        f = lambda r_proj, z_src, z_cl: self.smd.convergence (self.hdpm, self.cosmo.be_cosmo, r_proj, z_src, z_cl, z_cl)
        return np.vectorize (f) (r_proj, z_src, z_cl)

    def eval_reduced_shear (self, r_proj, z_cl, z_src):
        
        self.smd.prepare_if_needed (self.cosmo.be_cosmo)
        if isinstance(r_proj,(list,np.ndarray)) and isinstance(z_src,(list,np.ndarray)) and len (r_proj) == len (z_src):
            return self.smd.reduced_shear_array_equal (self.hdpm, self.cosmo.be_cosmo, np.atleast_1d (r_proj), 1.0, 1.0, np.atleast_1d (z_src), z_cl, z_cl)
        else:        
            return self.smd.reduced_shear_array (self.hdpm, self.cosmo.be_cosmo, np.atleast_1d (r_proj), 1.0, 1.0, np.atleast_1d (z_src), z_cl, z_cl)

    def eval_magnification (self, r_proj, z_cl, z_src):

        self.smd.prepare_if_needed (self.cosmo.be_cosmo)
        f = lambda r_proj, z_src, z_cl: self.smd.magnification (self.hdpm, self.cosmo.be_cosmo, r_proj, z_src, z_cl, z_cl)
        return np.vectorize (f) (r_proj, z_src, z_cl)

# CLMM Cosmology object - NumCosmo implementation

class NumCosmoCosmology (CLMMCosmology):
    def __init__(self, dist = None, dist_zmax = 15.0, **kwargs):
        
        self.dist = None
        
        super(NumCosmoCosmology, self).__init__ (**kwargs)
        
        self.backend = 'nc'
        
        if dist:
            self.set_dist (dist)
        else:
            self.set_dist (Nc.Distance.new (dist_zmax))
        
    def _init_from_cosmo (self, be_cosmo):
    
        assert isinstance (be_cosmo, Nc.HICosmo)
        self.be_cosmo = be_cosmo
        
    def _init_from_params (self, H0, Omega_b0, Omega_dm0, Omega_k0):

        if not self.be_cosmo:
            self.be_cosmo = Nc.HICosmo.new_from_name (Nc.HICosmo, "NcHICosmoDEXcdm")
            self.be_cosmo.param_set_lower_bound (Nc.HICosmoDESParams.T_GAMMA0, 0.0)        
            self.be_cosmo.omega_x2omega_k ()
            self.be_cosmo.param_set_by_name ("w",      -1.0)
            self.be_cosmo.param_set_by_name ("Tgamma0", 0.0)
        else:
            assert isinstance (cosmo, Nc.HICosmoDEXcdm)
            assert isinstance (cosmo.peek_reparam, Nc.HICosmoDEReparamOk)

        self.be_cosmo.param_set_by_name ("H0",     H0)
        self.be_cosmo.param_set_by_name ("Omegab", Omega_b0)
        self.be_cosmo.param_set_by_name ("Omegac", Omega_dm0)
        self.be_cosmo.param_set_by_name ("Omegak", Omega_k0)
        
    def _set_param (self, key, value):
        if key == "Omega_b0":
            self.be_cosmo.param_set_by_name ("Omegab", value)
        elif key == "Omega_dm0":
            self.be_cosmo.param_set_by_name ("Omegac", value)
        elif key == "Omega_k0":
            self.be_cosmo.param_set_by_name ("Omegak", value)
        elif key == 'h':
            self.be_cosmo.param_set_by_name ("H0",     value * 100.0)
        elif key == 'H0':
            self.be_cosmo.param_set_by_name ("H0",     value)
        else:
            raise ValueError (f"Unsupported parameter {key}")

    def _get_param (self, key):
        if key == "Omega_m0":
            return self.be_cosmo.Omega_m0 ()
        elif key == "Omega_b0":
            return self.be_cosmo.Omega_b0 ()
        elif key == "Omega_dm0":
            return self.be_cosmo.Omega_c0 ()
        elif key == "Omega_k0":
            return self.be_cosmo.Omega_k0 ()
        elif key == 'h':
            return self.be_cosmo.h ()
        elif key == 'H0':
            return self.be_cosmo.H0 ()
        else:
            raise ValueError (f"Unsupported parameter {key}")

    def set_dist (self, dist):

        assert isinstance (dist, Nc.Distance)
        self.dist = dist
        self.dist.prepare_if_needed (self.be_cosmo)

    def get_Omega_m (self, z):
    
        return self.be_cosmo.E2Omega_m (z) / self.be_cosmo.E2 (z)

    def get_E2Omega_m (self, z):
    
        return self.be_cosmo.E2Omega_m (z)

    def eval_da_z1z2 (self, z1, z2):
    
        return np.vectorize (self.dist.angular_diameter_z1_z2) (self.be_cosmo, z1, z2) * self.be_cosmo.RH_Mpc ()

Modeling = NumCosmoCLMModeling
Cosmology = NumCosmoCosmology
