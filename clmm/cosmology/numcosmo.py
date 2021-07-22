"""@file numcosmo.py
Cosmology using NumCosmo
"""
import numpy as np

import gi
gi.require_version('NumCosmo', '1.0')
gi.require_version('NumCosmoMath', '1.0')
from gi.repository import NumCosmo as Nc

from .parent_class import CLMMCosmology

__all__ = []


class NumCosmoCosmology(CLMMCosmology):
    """
    Cosmology object

    Attributes
    ----------
    backend: str
        Name of back-end used
    be_cosmo: cosmology library
        Cosmology library used in the back-end
    """

    def __init__(self, dist=None, dist_zmax=15.0, **kwargs):

        self.dist = None

        super().__init__(**kwargs)

        # this tag will be used to check if the cosmology object is accepted by the modeling
        self.backend = 'nc'

        if dist:
            self.set_dist(dist)
        else:
            self.set_dist(Nc.Distance.new(dist_zmax))

    def _init_from_cosmo(self, be_cosmo):

        assert isinstance(be_cosmo, Nc.HICosmo)
        assert isinstance(be_cosmo, Nc.HICosmoDECpl)
        assert isinstance(be_cosmo.peek_reparam(), Nc.HICosmoDEReparamOk)
        self.be_cosmo = be_cosmo

    def _init_from_params(self, H0, Omega_b0, Omega_dm0, Omega_k0):

        self.be_cosmo = Nc.HICosmo.new_from_name(
            Nc.HICosmo, "NcHICosmoDECpl{'massnu-length':<1>}")
        self.be_cosmo.omega_x2omega_k()
        self.be_cosmo.param_set_by_name("w0", -1.0)
        self.be_cosmo.param_set_by_name("w1", 0.0)
        self.be_cosmo.param_set_by_name("Tgamma0", 2.7255)
        self.be_cosmo.param_set_by_name("massnu_0", 0.06)
        self.be_cosmo.param_set_by_name("H0", H0)
        self.be_cosmo.param_set_by_name("Omegab", Omega_b0)
        self.be_cosmo.param_set_by_name("Omegac", Omega_dm0)
        self.be_cosmo.param_set_by_name("Omegak", Omega_k0)

        cosmo = self.be_cosmo
        ENnu = 3.046 - 3.0 * \
            cosmo.E2Press_mnu(1.0e10) / (cosmo.E2Omega_g(1.0e10)
                                         * (7.0/8.0*(4.0/11.0)**(4.0/3.0)))

        self.be_cosmo.param_set_by_name("ENnu", ENnu)

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
            value = self.be_cosmo.Omega_m0()
        elif key == "Omega_b0":
            value = self.be_cosmo.Omega_b0()
        elif key == "Omega_dm0":
            value = self.be_cosmo.Omega_c0()
        elif key == "Omega_k0":
            value = self.be_cosmo.Omega_k0()
        elif key == 'h':
            value = self.be_cosmo.h()
        elif key == 'H0':
            value = self.be_cosmo.H0()
        else:
            raise ValueError(f"Unsupported parameter {key}")
        return value

    def set_dist(self, dist):
        r"""Sets distance functions (NumCosmo internal use)
        """
        assert isinstance(dist, Nc.Distance)
        self.dist = dist
        self.dist.prepare_if_needed(self.be_cosmo)

    def get_Omega_m(self, z):

        return self.be_cosmo.E2Omega_m(z)/self.be_cosmo.E2(z)

    def get_E2Omega_m(self, z):

        return self.be_cosmo.E2Omega_m(z)

    def eval_da_z1z2(self, z1, z2):

        return np.vectorize(self.dist.angular_diameter_z1_z2)(
            self.be_cosmo, z1, z2)*self.be_cosmo.RH_Mpc()

    def eval_sigma_crit(self, z_len, z_src):

        self.smd.prepare_if_needed(self.be_cosmo)

        func = lambda z_len, z_src: self.smd.sigma_critical(
            self.be_cosmo, z_src, z_len, z_len)
        return np.vectorize(func)(z_len, z_src)
