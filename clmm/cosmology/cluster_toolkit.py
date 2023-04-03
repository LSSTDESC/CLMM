"""@file cluster_toolkit.py
Cosmology using AstroPy (for cluster_toolkit)
"""
import numpy as np

from astropy import units
from astropy.cosmology import LambdaCDM, FlatLambdaCDM

from ..constants import Constants as const

from .parent_class import CLMMCosmology

__all__ = []


class AstroPyCosmology(CLMMCosmology):
    """
    Cosmology object

    Attributes
    ----------
    backend: str
        Name of back-end used
    be_cosmo: cosmology library
        Cosmology library used in the back-end
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # this tag will be used to check if the cosmology object is accepted by the modeling
        self.backend = "ct"

        assert isinstance(self.be_cosmo, LambdaCDM)

    def _init_from_cosmo(self, be_cosmo):
        assert isinstance(be_cosmo, LambdaCDM)
        self.be_cosmo = be_cosmo

    def _init_from_params(self, H0, Omega_b0, Omega_dm0, Omega_k0):
        # pylint: disable=arguments-differ
        kwargs = {
            "H0": H0,
            "Om0": Omega_b0 + Omega_dm0,
            "Ob0": Omega_b0,
            "Tcmb0": 2.7255,
            "Neff": 3.046,
            "m_nu": ([0.06, 0.0, 0.0] * units.eV),
        }
        self.be_cosmo = FlatLambdaCDM(**kwargs)
        if Omega_k0 != 0.0:
            kwargs["Ode0"] = self.be_cosmo.Ode0 - Omega_k0
            self.be_cosmo = LambdaCDM(**kwargs)

    def _set_param(self, key, value):
        raise NotImplementedError("Astropy do not support changing parameters")

    def _get_param(self, key):
        if key == "Omega_m0":
            value = self.be_cosmo.Om0
        elif key == "Omega_b0":
            value = self.be_cosmo.Ob0
        elif key == "Omega_dm0":
            value = self.be_cosmo.Odm0
        elif key == "Omega_k0":
            value = self.be_cosmo.Ok0
        elif key == "h":
            value = self.be_cosmo.H0.to_value() / 100.0
        elif key == "H0":
            value = self.be_cosmo.H0.to_value()
        else:
            raise ValueError(f"Unsupported parameter {key}")
        return value

    def _get_Omega_m(self, z):
        return self.be_cosmo.Om(z)

    def _get_E2(self, z):
        return (self.be_cosmo.H(z) / self.be_cosmo.H0) ** 2

    def _get_E2Omega_m(self, z):
        return self.be_cosmo.Om(z) * (self.be_cosmo.H(z) / self.be_cosmo.H0) ** 2

    def _get_rho_c(self, z):
        return self.be_cosmo.critical_density(z).to(units.Msun / units.Mpc**3).value

    def _eval_da_z1z2_core(self, z1, z2):
        return self.be_cosmo.angular_diameter_distance_z1z2(z1, z2).to_value(units.Mpc)

    def _eval_sigma_crit_core(self, z_len, z_src):
        # Constants
        clight_pc_s = const.CLIGHT_KMS.value * 1000.0 / const.PC_TO_METER.value
        gnewt_pc3_msun_s2 = (
            const.GNEWT.value * const.SOLAR_MASS.value / const.PC_TO_METER.value**3
        )

        d_l = self._eval_da_z1z2_core(0, z_len)
        d_s = self._eval_da_z1z2_core(0, z_src)
        d_ls = self._eval_da_z1z2_core(z_len, z_src)

        return clight_pc_s**2 / (4.0 * np.pi * gnewt_pc3_msun_s2) * d_s / (d_l * d_ls) * 1.0e6
