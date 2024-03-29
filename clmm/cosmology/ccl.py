"""@file ccl.py
Cosmology using CCL
"""
import numpy as np

import pyccl as ccl

from .parent_class import CLMMCosmology

from ..utils import _patch_rho_crit_to_cd2018

__all__ = []


class CCLCosmology(CLMMCosmology):
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
        self.backend = "ccl"

        assert isinstance(self.be_cosmo, ccl.Cosmology)

        # cor factor for sigma_critical
        self.cor_factor = _patch_rho_crit_to_cd2018(ccl.physical_constants.RHO_CRITICAL)

    def _init_from_cosmo(self, be_cosmo):
        assert isinstance(be_cosmo, ccl.Cosmology)
        self.be_cosmo = be_cosmo

    def _init_from_params(self, H0, Omega_b0, Omega_dm0, Omega_k0):
        # pylint: disable=arguments-differ
        self.be_cosmo = ccl.Cosmology(
            Omega_c=Omega_dm0,
            Omega_b=Omega_b0,
            Omega_k=Omega_k0,
            h=H0 / 100.0,
            sigma8=0.8,
            n_s=0.96,
            T_CMB=2.7255,
            Neff=3.046,
            m_nu=[0.06, 0.0, 0.0],
            transfer_function="eisenstein_hu",
            matter_power_spectrum="linear",
        )

    def _set_param(self, key, value):
        raise NotImplementedError("CCL do not support changing parameters")

    def _get_param(self, key):
        if key == "Omega_m0":
            value = ccl.omega_x(self.be_cosmo, 1.0, "matter")
        elif key == "Omega_b0":
            value = self.be_cosmo["Omega_b"]
        elif key == "Omega_dm0":
            value = self.be_cosmo["Omega_c"]
        elif key == "Omega_k0":
            value = self.be_cosmo["Omega_k"]
        elif key == "h":
            value = self.be_cosmo["h"]
        elif key == "H0":
            value = self.be_cosmo["h"] * 100.0
        else:
            raise ValueError(f"Unsupported parameter {key}")
        return value

    def _get_Omega_m(self, z):
        a = self._get_a_from_z(z)
        return ccl.omega_x(self.be_cosmo, a, "matter")

    def _get_E2(self, z):
        a = self._get_a_from_z(z)
        return (ccl.h_over_h0(self.be_cosmo, a)) ** 2

    def _get_E2Omega_m(self, z):
        a = self._get_a_from_z(z)
        return ccl.omega_x(self.be_cosmo, a, "matter") * (ccl.h_over_h0(self.be_cosmo, a)) ** 2

    def _get_rho_m(self, z):
        # total matter density in physical units [Msun/Mpc3]
        a = self._get_a_from_z(z)
        return ccl.rho_x(self.be_cosmo, a, "matter", is_comoving=False)

    def _get_rho_c(self, z):
        a = self._get_a_from_z(z)
        return ccl.rho_x(self.be_cosmo, a, "critical", is_comoving=False)

    def _eval_da_z1z2_core(self, z1, z2):
        a1 = np.atleast_1d(self._get_a_from_z(z1))
        a2 = np.atleast_1d(self._get_a_from_z(z2))
        if len(a1) == 1 and len(a2) != 1:
            a1 = np.full_like(a2, a1)
        elif len(a2) == 1 and len(a1) != 1:
            a2 = np.full_like(a1, a2)

        da = ccl.angular_diameter_distance(self.be_cosmo, a1, a2)
        res = da if np.iterable(z1) or np.iterable(z2) else da.item()

        return res

    def _eval_sigma_crit_core(self, z_len, z_src):
        a_len = self._get_a_from_z(z_len)
        a_src = self._get_a_from_z(z_src)

        return self.be_cosmo.sigma_critical(a_lens=a_len, a_source=a_src) * self.cor_factor

    def _eval_linear_matter_powerspectrum(self, k_vals, redshift):
        return ccl.linear_matter_power(self.be_cosmo, k_vals, self._get_a_from_z(redshift))
