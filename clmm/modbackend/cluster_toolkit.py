# Functions to model halo profiles

import cluster_toolkit as ct

import numpy as np
import warnings

from astropy import units
from astropy.cosmology import LambdaCDM, FlatLambdaCDM

from .. constants import Constants as const
from . import func_layer
from . func_layer import *

from .. clmm_cosmo import CLMMCosmology
from .. clmm_modeling import CLMModeling

from .ccl import CCLCosmology
from .numcosmo import NumCosmoCosmology

__all__ = ['CTModeling', 'Modeling', 'Cosmology']+func_layer.__all__


def _patch_ct_to_cd2018():
    r""" Convertion factor from cluster_toolkit hardcoded rho_crit to
    CODATA 2018+IAU 2015

    """

    rhocrit_mks = 3.0*100.0*100.0/(8.0*np.pi*const.GNEWT.value)
    rhocrit_cd2018 = rhocrit_mks*1000.0*1000.0*const.PC_TO_METER.value*1.0e6/const.SOLAR_MASS.value
    rhocrit_cltk = 2.77533742639e+11

    return rhocrit_cd2018/rhocrit_cltk


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
        self.cor_factor = _patch_ct_to_cd2018()
        # Set halo profile and cosmology
        self.set_halo_density_profile(halo_profile_model, massdef, delta_mdef)
        self.set_cosmo(None)

    def set_cosmo(self, cosmo):
        self._set_cosmo(cosmo, AstroPyCosmology, ('ct', 'nc')) # does not work with CCLCosmology - must fix sig_crit

    def set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        # Check if choices are supported
        self.validate_definitions(massdef, halo_profile_model)
        # Update values
        self.halo_profile_model = halo_profile_model
        self.massdef = massdef
        self.delta_mdef = delta_mdef

    def set_concentration(self, cdelta):
        self.cdelta = cdelta

    def set_mass(self, mdelta):
        self.mdelta = mdelta

    def eval_sigma_crit(self, z_len, z_src):
        if np.any(np.array(z_src)<=z_len):
            warnings.warn(f'Some source redshifts are lower than the cluster redshift. Returning Sigma_crit = np.inf for those galaxies.')
        # Constants
        clight_pc_s = const.CLIGHT_KMS.value*1000./const.PC_TO_METER.value
        gnewt_pc3_msun_s2 = const.GNEWT.value*const.SOLAR_MASS.value/const.PC_TO_METER.value**3

        d_l = self.cosmo.eval_da_z1z2(0, z_len)
        d_s = self.cosmo.eval_da_z1z2(0, z_src)
        d_ls = self.cosmo.eval_da_z1z2(z_len, z_src)

        beta_s = np.maximum(0., d_ls/d_s)
        return clight_pc_s**2/(4.0*np.pi*gnewt_pc3_msun_s2)*1/d_l*np.divide(1., beta_s)*1.0e6

    def eval_density(self, r3d, z_cl):
        h = self.cosmo['h']
        Omega_m = self.cosmo.get_E2Omega_m(z_cl)*self.cor_factor
        return ct.density.rho_nfw_at_r(_assert_correct_type_ct(r3d)*h, self.mdelta*h,
                self.cdelta, Omega_m, delta=self.delta_mdef)*h**2

    def eval_sigma(self, r_proj, z_cl):
        if self.cosmo is None:
            raise ValueError(f"Missing cosmology.")
        h = self.cosmo['h']
        Omega_m = self.cosmo.get_E2Omega_m(z_cl)*self.cor_factor
        return ct.deltasigma.Sigma_nfw_at_R(_assert_correct_type_ct(r_proj)*h, self.mdelta*h,
                self.cdelta, Omega_m, delta=self.delta_mdef)*h*1.0e12 # pc**-2 to Mpc**-2

    def eval_sigma_mean(self, r_proj, z_cl):
        '''

        Note
        ----
        This function just adds eval_sigma+eval_sigma_excess
        '''
        return self.eval_sigma(r_proj, z_cl)+self.eval_sigma_excess(r_proj, z_cl)

    def eval_sigma_excess(self, r_proj, z_cl):
        if np.min(r_proj)<1.e-11:
            raise ValueError(f"Rmin = {np.min(r_proj):.2e} Mpc! This value is too small and may cause computational issues.")
        Omega_m = self.cosmo.get_E2Omega_m(z_cl)*self.cor_factor
        h = self.cosmo['h']
        r_proj = _assert_correct_type_ct(r_proj)*h
        # Computing sigma on a larger range than the radial range requested, with at least 1000 points.
        sigma_r_proj = np.logspace(np.log10(np.min(r_proj))-1, np.log10(np.max(r_proj))+1, np.max([1000,10*np.array(r_proj).size]))
        sigma = self.eval_sigma(sigma_r_proj/h, z_cl)/(h*1e12) # rm norm for ct
        # ^ Note: Let's not use this naming convention when transfering ct to ccl....
        return ct.deltasigma.DeltaSigma_at_R(r_proj, sigma_r_proj, sigma, self.mdelta*h,
                self.cdelta, Omega_m, delta=self.delta_mdef)*h*1.0e12 # pc**-2 to Mpc**-2

    def eval_shear(self, r_proj, z_cl, z_src):
        delta_sigma = self.eval_sigma_excess(r_proj, z_cl)
        sigma_c = self.eval_sigma_crit(z_cl, z_src)
        return np.nan_to_num(delta_sigma/sigma_c, nan=np.nan, posinf=np.inf, neginf=-np.inf)

    def eval_convergence(self, r_proj, z_cl, z_src):
        sigma = self.eval_sigma(r_proj, z_cl)
        sigma_c = self.eval_sigma_crit(z_cl, z_src)
        return np.nan_to_num(sigma/sigma_c, nan=np.nan, posinf=np.inf, neginf=-np.inf)

    def eval_reduced_shear(self, r_proj, z_cl, z_src):
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_shear(r_proj, z_cl, z_src)
        return np.nan_to_num(np.divide(gamma_t, (1-kappa)), nan=np.nan, posinf=np.inf, neginf=-np.inf)

    def eval_magnification(self, r_proj, z_cl, z_src):
        # The magnification is computed taking into account just the tangential shear. This is valid for
        # spherically averaged profiles, e.g., NFW and Einasto (by construction the cross shear is zero).
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_shear(r_proj, z_cl, z_src)
        return 1./((1-kappa)**2-abs(gamma_t)**2)


# CLMM Cosmology object - clustertoolkit+astropy implementation


class AstroPyCosmology(CLMMCosmology):

    def __init__(self, **kwargs):
        super(AstroPyCosmology, self).__init__(**kwargs)

        # this tag will be used to check if the cosmology object is accepted by the modeling
        self.backend = 'ct'

        assert isinstance(self.be_cosmo, LambdaCDM)

    def _init_from_cosmo(self, be_cosmo):

        assert isinstance(be_cosmo, LambdaCDM)
        self.be_cosmo = be_cosmo

    def _init_from_params(self, H0, Omega_b0, Omega_dm0, Omega_k0):

        Om0 = Omega_b0+Omega_dm0
        Ob0 = Omega_b0
        Ode0 = 1.0-Om0-Omega_k0

        self.be_cosmo = LambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, Ode0=Ode0)

    def _set_param(self, key, value):
        raise NotImplementedError("Astropy do not support changing parameters")

    def _get_param(self, key):
        if key == "Omega_m0":
            return self.be_cosmo.Om0
        elif key == "Omega_b0":
            return self.be_cosmo.Ob0
        elif key == "Omega_dm0":
            return self.be_cosmo.Odm0
        elif key == "Omega_k0":
            return self.be_cosmo.Ok0
        elif key == 'h':
            return self.be_cosmo.H0.to_value()/100.0
        elif key == 'H0':
            return self.be_cosmo.H0.to_value()
        else:
            raise ValueError(f"Unsupported parameter {key}")

    def get_Omega_m(self, z):
        return self.be_cosmo.Om(z)

    def get_E2Omega_m(self, z):
        return self.be_cosmo.Om(z)*(self.be_cosmo.H(z)/self.be_cosmo.H0)**2

    def eval_da_z1z2(self, z1, z2):
        return self.be_cosmo.angular_diameter_distance_z1z2(z1, z2).to_value(units.Mpc)


Modeling = CTModeling
Cosmology = AstroPyCosmology
