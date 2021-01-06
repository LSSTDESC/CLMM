# NumCosmo implementation of CLMModeling

import gi
gi.require_version('NumCosmo', '1.0')
gi.require_version('NumCosmoMath', '1.0')
from gi.repository import NumCosmo as Nc
from gi.repository import NumCosmoMath as Ncm

import math
import numpy as np

from .parent_class import CLMMCosmology

__all__ = []


class NumCosmoCosmology(CLMMCosmology):

    def __init__(self, dist=None, dist_zmax=15.0, **kwargs):

        self.dist = None

        super(NumCosmoCosmology, self).__init__(**kwargs)

        # this tag will be used to check if the cosmology object is accepted by the modeling
        self.backend = 'nc'

        if dist:
            self.set_dist(dist)
        else:
            self.set_dist(Nc.Distance.new(dist_zmax))

    def _init_from_cosmo(self, be_cosmo):

        assert isinstance(be_cosmo, Nc.HICosmo)
        assert isinstance(be_cosmo, Nc.HICosmoDEXcdm)
        assert isinstance(be_cosmo.peek_reparam (), Nc.HICosmoDEReparamOk)
        self.be_cosmo = be_cosmo

    def _init_from_params(self, H0, Omega_b0, Omega_dm0, Omega_k0):

        self.be_cosmo = Nc.HICosmo.new_from_name(Nc.HICosmo, "NcHICosmoDEXcdm")
        self.be_cosmo.param_set_lower_bound(Nc.HICosmoDESParams.T_GAMMA0, 0.0)
        self.be_cosmo.omega_x2omega_k()
        self.be_cosmo.param_set_by_name("w", -1.0)
        self.be_cosmo.param_set_by_name("Tgamma0", 0.0)

        self.be_cosmo.param_set_by_name("H0", H0)
        self.be_cosmo.param_set_by_name("Omegab", Omega_b0)
        self.be_cosmo.param_set_by_name("Omegac", Omega_dm0)
        self.be_cosmo.param_set_by_name("Omegak", Omega_k0)

    def _set_param(self, key, value):
        if key == "Omega_b0":
            self.be_cosmo.param_set_by_name("Omegab", value)
        elif key == "Omega_dm0":
            self.be_cosmo.param_set_by_name("Omegac", value)
        elif key == "Omega_k0":
            self.be_cosmo.param_set_by_name("Omegak", value)
        elif key == 'h':
            self.be_cosmo.param_set_by_name("H0", value*100.0)
        elif key == 'H0':
            self.be_cosmo.param_set_by_name("H0", value)
        else:
            raise ValueError(f"Unsupported parameter {key}")

    def _get_param(self, key):
        if key == "Omega_m0":
            return self.be_cosmo.Omega_m0()
        elif key == "Omega_b0":
            return self.be_cosmo.Omega_b0()
        elif key == "Omega_dm0":
            return self.be_cosmo.Omega_c0()
        elif key == "Omega_k0":
            return self.be_cosmo.Omega_k0()
        elif key == 'h':
            return self.be_cosmo.h()
        elif key == 'H0':
            return self.be_cosmo.H0()
        else:
            raise ValueError(f"Unsupported parameter {key}")

    def set_dist(self, dist):

        assert isinstance(dist, Nc.Distance)
        self.dist = dist
        self.dist.prepare_if_needed(self.be_cosmo)

    def get_Omega_m(self, z):

        return self.be_cosmo.E2Omega_m(z)/self.be_cosmo.E2(z)

    def get_E2Omega_m(self, z):

        return self.be_cosmo.E2Omega_m(z)

    def eval_da_z1z2(self, z1, z2):

        return np.vectorize(self.dist.angular_diameter_z1_z2)(self.be_cosmo, z1, z2)*self.be_cosmo.RH_Mpc()

    def eval_sigma_crit(self, z_len, z_src):

        self.smd.prepare_if_needed(self.be_cosmo)
        f = lambda z_len, z_src: self.smd.sigma_critical(self.be_cosmo, z_src, z_len, z_len)
        return np.vectorize(f)(z_len, z_src)
