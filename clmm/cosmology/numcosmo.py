"""@file numcosmo.py
Cosmology using NumCosmo
"""
import numpy as np

import gi

# pylint: disable=wrong-import-position
gi.require_version("NumCosmo", "1.0")
gi.require_version("NumCosmoMath", "1.0")
from gi.repository import NumCosmo as Nc
from gi.repository import NumCosmoMath as Ncm

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

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=abstract-method

    def __init__(self, dist=None, dist_zmax=15.0, **kwargs):
        self.dist = None
        self.smd = None

        super().__init__(**kwargs)

        # this tag will be used to check if the cosmology object is accepted by the modeling
        self.backend = "nc"

        # set kmin/kmax for powerspectrum computations
        self.additional_config["pk_kmin"] = 1.0e-5
        self.additional_config["pk_kmax"] = 1.0
        self.additional_config["ns"] = 0.96
        self.additional_config["sigma8"] = 0.8
        self.additional_config["sigma8_eps"] = 1e-8

        if dist:
            self.set_dist(dist)
        else:
            self.set_dist(Nc.Distance.new(dist_zmax))

        self.smd = Nc.WLSurfaceMassDensity.new(self.dist)
        self.smd.prepare_if_needed(self.be_cosmo)


    def _init_from_cosmo(self, be_cosmo):
        assert isinstance(be_cosmo, Nc.HICosmo)
        assert isinstance(be_cosmo, Nc.HICosmoDECpl)
        assert isinstance(be_cosmo.peek_reparam(), Nc.HICosmoDEReparamOk)
        self.be_cosmo = be_cosmo
        self._update_vec_funcs()

    def _init_from_params(self, H0, Omega_b0, Omega_dm0, Omega_k0):
        # pylint: disable=arguments-differ
        self.be_cosmo = Nc.HICosmoDECpl(massnu_length=1)
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
        en_nu = 3.046 - 3.0 * cosmo.E2Press_mnu(1.0e10) / (
            cosmo.E2Omega_g(1.0e10) * (7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0))
        )

        self.be_cosmo.param_set_by_name("ENnu", en_nu)
        self._update_vec_funcs()

    def _set_param(self, key, value):
        if key == "Omega_b0":
            self.be_cosmo.param_set_by_name("Omegab", value)
        elif key == "Omega_dm0":
            self.be_cosmo.param_set_by_name("Omegac", value)
        elif key == "Omega_k0":
            self.be_cosmo.param_set_by_name("Omegak", value)
        elif key == "h":
            self.be_cosmo.param_set_by_name("H0", value * 100.0)
        elif key == "H0":
            self.be_cosmo.param_set_by_name("H0", value)
        else:
            raise ValueError(f"Unsupported parameter {key}")
        self._update_vec_funcs()

    def _get_param(self, key):
        if key == "Omega_m0":
            value = self.be_cosmo.Omega_m0()
        elif key == "Omega_b0":
            value = self.be_cosmo.Omega_b0()
        elif key == "Omega_dm0":
            value = self.be_cosmo.Omega_c0()
        elif key == "Omega_k0":
            value = self.be_cosmo.Omega_k0()
        elif key == "h":
            value = self.be_cosmo.h()
        elif key == "H0":
            value = self.be_cosmo.H0()
        else:
            raise ValueError(f"Unsupported parameter {key}")
        return value

    def set_dist(self, dist):
        r"""Sets distance functions (NumCosmo internal use)"""
        assert isinstance(dist, Nc.Distance)
        self.dist = dist
        self.dist.prepare_if_needed(self.be_cosmo)

    def _get_Omega_m(self, z):
        return self._get_E2Omega_m(z) / self._get_E2(z)

    def _update_vec_funcs(self):
        """Update all functions that are vectorized"""

        self._get_E2 = np.vectorize(self.be_cosmo.E2)
        self._get_E2Omega_m = np.vectorize(self.be_cosmo.E2Omega_m)
        self._eval_da_z1z2_core = np.vectorize(
            lambda z1, z2: (
                self.dist.angular_diameter_z1_z2(self.be_cosmo, z1, z2) * self.be_cosmo.RH_Mpc()
            )
        )
        self._eval_sigma_crit_core = np.vectorize(
            lambda z_len, z_src: (self.smd.sigma_critical(self.be_cosmo, z_src, z_len, z_len))
        )
        if self.dist is not None:
            self.dist.prepare_if_needed(self.be_cosmo)
        if self.smd is not None:
            self.smd.prepare_if_needed(self.be_cosmo)

    def _get_rho_c(self, z):
        return (
            Ncm.C.crit_mass_density_h2_solar_mass_Mpc3()
            * self._get_param("h") ** 2
            * self._get_E2(z)
        )

    def _get_rho_m(self, z):
        # total matter density in physical units [Msun/Mpc3]
        rho_m = (
            self._get_E2Omega_m(z)
            * Ncm.C.crit_mass_density_h2_solar_mass_Mpc3()
            * self._get_param("h")
            * self._get_param("h")
        )
        return rho_m

    def _eval_linear_matter_powerspectrum(self, k_vals, redshift):
        # Using the EH transfer function as this is the
        # default for the CCL backend as well
        pow_spec = Nc.PowspecMLTransfer.new(Nc.TransferFuncEH.new())
        pow_spec.set_kmin(self.additional_config["pk_kmin"])
        pow_spec.set_kmax(self.additional_config["pk_kmax"])

        # Instead, computing the PS from the CLASS backend of Numcosmo
        # pow_spec  = Nc.PowspecMLCBE.new ()
        # pow_spec.peek_cbe().props.use_ppf = True

        if self.be_cosmo.reion is None:
            reion = Nc.HIReionCamb.new()
            self.be_cosmo.add_submodel(reion)
        if self.be_cosmo.prim is None:
            prim = Nc.HIPrimPowerLaw.new()
            self.be_cosmo.add_submodel(prim)
            # The default CLMM cosmology has ns=0.96 and sigma8=0.8
            # Need to adapt the NC cosmology accordingly
            self.be_cosmo.prim.props.n_SA = self.additional_config["ns"]
            sigma8 = pow_spec.sigma_tophat_R(  # computes sigma8 at z=0
                self.be_cosmo,
                self.additional_config["sigma8_eps"],
                0.0,
                8.0 / self.be_cosmo.h(),
            )
            old_amplitude = np.exp(self.be_cosmo.prim.props.ln10e10ASA)
            self.be_cosmo.prim.props.ln10e10ASA = np.log(
                (self.additional_config["sigma8"] / sigma8) ** 2 * old_amplitude
            )

        pow_spec.prepare(self.be_cosmo)

        return [pow_spec.eval(self.be_cosmo, redshift, k) for k in k_vals]
