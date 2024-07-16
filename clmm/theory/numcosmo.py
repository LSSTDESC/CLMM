"""@file numcosmo.py
NumCosmo implementation of CLMModeling
"""
import math
import numpy as np

import gi
from gi.repository import NumCosmoMath as Ncm
from gi.repository import NumCosmo as Nc

from ..cosmology.numcosmo import NumCosmoCosmology
from .parent_class import CLMModeling

gi.require_version("NumCosmo", "1.0")
gi.require_version("NumCosmoMath", "1.0")


class NumCosmoCLMModeling(CLMModeling):
    r"""Object with functions for halo mass modeling

    Attributes
    ----------
    backend: str
        Name of the backend being used
    massdef : str
        Profile mass definition ("mean", "critical", "virial" - letter case independent)
    delta_mdef : int
        Mass overdensity definition.
    halo_profile_model : str
        Profile model parameterization ("nfw", "einasto", "hernquist" - letter case independent)
    cosmo: Cosmology
        Cosmology object
    hdpm: Object
        Backend object with halo profiles
    mdef_dict: dict
        Dictionary with the definitions for mass
    hdpm_dict: dict
        Dictionary with the definitions for profile
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=abstract-method

    def __init__(
        self,
        massdef="mean",
        delta_mdef=200,
        halo_profile_model="nfw",
        validate_input=True,
    ):
        CLMModeling.__init__(self, validate_input)
        # Update class attributes
        Ncm.cfg_init()
        self.backend = "nc"
        self.mdef_dict = {
            "mean": Nc.HaloDensityProfileMassDef.MEAN,
            "critical": Nc.HaloDensityProfileMassDef.CRITICAL,
            "virial": Nc.HaloDensityProfileMassDef.VIRIAL,
        }
        self.hdpm_dict = {
            "nfw": Nc.HaloDensityProfileNFW.new,
            "einasto": Nc.HaloDensityProfileEinasto.new,
            "hernquist": Nc.HaloDensityProfileHernquist.new,
        }
        self.cosmo_class = NumCosmoCosmology
        # Set halo profile and cosmology
        self.set_halo_density_profile(halo_profile_model, massdef, delta_mdef)
        self.set_cosmo(None)

    # Functions implemented by child class

    def _update_halo_density_profile(self):
        """updates halo density profile with set internal properties"""
        # Makes sure current cdelta/mdelta values are kept
        has_cm_vals = self.hdpm is not None
        if has_cm_vals:
            cdelta = self.cdelta
            log10_mdelta = self.hdpm.props.log10MDelta

        self.hdpm = self.hdpm_dict[self.halo_profile_model](
            self.mdef_dict[self.massdef], self.delta_mdef
        )

        if has_cm_vals:
            self.cdelta = cdelta
            self.hdpm.props.log10MDelta = log10_mdelta

        self._update_vec_funcs()

    def _get_concentration(self):
        """get concentration"""
        return self.hdpm.props.cDelta

    def _get_mass(self):
        """get mass"""
        return 10**self.hdpm.props.log10MDelta

    def _set_concentration(self, cdelta):
        """set concentration"""
        self.hdpm.props.cDelta = cdelta
        self._update_vec_funcs()

    def _set_mass(self, mdelta):
        """set mass"""
        self.hdpm.props.log10MDelta = math.log10(mdelta)
        self._update_vec_funcs()

    def _set_einasto_alpha(self, alpha):
        if alpha is None:
            self.hdpm.props.alpha = 0.25
        else:
            self.hdpm.props.alpha = alpha
        self._update_vec_funcs()

    def _get_einasto_alpha(self, z_cl=None):
        """get the value of the Einasto slope"""
        # Note that z_cl is needed for CCL<2.6 only
        return self.hdpm.props.alpha

    def _eval_reduced_tangential_shear_core(self, r_proj, z_cl, z_src):
        """eval reduced tangential shear considering a single redshift plane
        for background sources"""

        if (
            isinstance(r_proj, (list, np.ndarray))
            and isinstance(z_src, (list, np.ndarray))
            and len(r_proj) == len(z_src)
        ):
            func = self.cosmo.smd.reduced_shear_array_equal
        else:
            func = self.cosmo.smd.reduced_shear_array
        return func(
            self.hdpm,
            self.cosmo.be_cosmo,
            np.atleast_1d(r_proj),
            1.0,
            1.0,
            np.atleast_1d(z_src),
            z_cl,
            z_cl,
        )

    # Functions unique to this class

    def _update_vec_funcs(self):
        """Set/update all functions that are vectorized"""
        self._eval_3d_density = np.vectorize(
            lambda r3d, z_cl: self.hdpm.eval_density(self.cosmo.be_cosmo, r3d, z_cl)
        )
        self._eval_surface_density = np.vectorize(
            lambda r_proj, z_cl: self.cosmo.smd.sigma(self.hdpm, self.cosmo.be_cosmo, r_proj, z_cl)
        )
        self._eval_mean_surface_density = np.vectorize(
            lambda r_proj, z_cl: self.cosmo.smd.sigma_mean(
                self.hdpm, self.cosmo.be_cosmo, r_proj, z_cl
            )
        )
        self._eval_excess_surface_density = np.vectorize(
            lambda r_proj, z_cl: self.cosmo.smd.sigma_excess(
                self.hdpm, self.cosmo.be_cosmo, r_proj, z_cl
            )
        )
        self._eval_tangential_shear_core = np.vectorize(
            lambda r_proj, z_cl, z_src: self.cosmo.smd.shear(
                self.hdpm, self.cosmo.be_cosmo, r_proj, z_src, z_cl, z_cl
            )
        )
        self._eval_convergence_core = np.vectorize(
            lambda r_proj, z_cl, z_src: self.cosmo.smd.convergence(
                self.hdpm, self.cosmo.be_cosmo, r_proj, z_src, z_cl, z_cl
            )
        )
        self._eval_magnification_core = np.vectorize(
            lambda r_proj, z_cl, z_src: self.cosmo.smd.magnification(
                self.hdpm, self.cosmo.be_cosmo, r_proj, z_src, z_cl, z_cl
            )
        )

    def get_mset(self):
        r"""
        Gets a mass set (NumCosmo internal use)
        """
        mset = Ncm.MSet.empty_new()
        mset.set(self.cosmo.be_cosmo)
        mset.set(self.hdpm)
        mset.set(self.cosmo.smd)
        return mset

    def set_mset(self, mset):
        r"""
        Sets a mass set (NumCosmo internal use)
        """
        self.cosmo.set_be_cosmo(mset.get(Nc.HICosmo.id()))

        self.hdpm = mset.get(Nc.HaloDensityProfile.id())
        self.cosmo.smd = mset.get(Nc.WLSurfaceMassDensity.id())
        self.cosmo.smd.prepare_if_needed(self.cosmo.be_cosmo)


Cosmology = NumCosmoCosmology
Modeling = NumCosmoCLMModeling
