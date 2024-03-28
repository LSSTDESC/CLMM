"""@file parent_class.py
CLMMCosmology abstract class
"""
# CLMM Cosmology object abstract superclass
import numpy as np
from ..utils import validate_argument
from ..redshift import compute_for_good_redshifts
from ..constants import Constants as const


class CLMMCosmology:
    """
    Cosmology object superclass for supporting multiple back-end cosmology objects

    Attributes
    ----------
    backend: str
        Name of back-end used
    be_cosmo: cosmology library
        Cosmology library used in the back-end
    validate_input: bool
        Validade each input argument
    additional_config: dict
        Dictionary with additional (implicit) config that will be used by the class.
    """

    def __init__(self, validate_input=True, **kwargs):
        self.backend = None
        self.be_cosmo = None
        self.validate_input = validate_input
        self.additional_config = {}
        self.set_be_cosmo(**kwargs)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._get_param(key)
        raise TypeError(f"input must be str, not {type(key)}")

    def __setitem__(self, key, val):
        if isinstance(key, str):
            self._set_param(key, val)
        else:
            raise TypeError(f"key input must be str, not {type(key)}")

    # 1. Functions to be implemented by children classes

    def _init_from_cosmo(self, be_cosmo):
        raise NotImplementedError

    def _init_from_params(self, **kwargs):
        raise NotImplementedError

    def _set_param(self, key, value):
        raise NotImplementedError

    def _get_param(self, key):
        raise NotImplementedError

    def _get_Omega_m(self, z):
        raise NotImplementedError

    def _get_E2(self, z):
        raise NotImplementedError

    def _get_E2Omega_m(self, z):
        raise NotImplementedError

    def _get_rho_c(self, z):
        raise NotImplementedError

    def _eval_da_z1z2_core(self, z1, z2):
        raise NotImplementedError

    def _eval_sigma_crit_core(self, z_len, z_src):
        raise NotImplementedError

    def _eval_linear_matter_powerspectrum(self, k_vals, redshift):
        raise NotImplementedError

    # 2. Functions that can be used by all subclasses

    def _get_rho_m(self, z):
        rhocrit_cd2018 = (3.0e16 * const.PC_TO_METER.value) / (
            8.0 * np.pi * const.GNEWT.value * const.SOLAR_MASS.value
        )
        return rhocrit_cd2018 * (z + 1) ** 3 * self["Omega_m0"] * self["h"] ** 2

    def _eval_da_z1z2(self, z1, z2):
        warning_msg = "\nSome values of z2 are lower than z1." + "\nda = np.nan for those."
        return compute_for_good_redshifts(
            self._eval_da_z1z2_core, z1, z2, np.nan, warning_message=warning_msg
        )

    def _eval_da(self, z):
        return self._eval_da_z1z2(0.0, z)

    def _get_a_from_z(self, z):
        z = np.array(z)
        return 1.0 / (1.0 + z)

    def _get_z_from_a(self, a):
        a = np.array(a)
        return (1.0 / a) - 1.0

    def _eval_sigma_crit(self, z_len, z_src):
        warning_msg = (
            "\nSome source redshifts are lower than the cluster redshift."
            + "\nSigma_crit = np.inf for those galaxies."
        )
        return compute_for_good_redshifts(
            self._eval_sigma_crit_core,
            z_len,
            z_src,
            np.inf,
            z1_arg_name="z_len",
            z2_arg_name="z_src",
            warning_message=warning_msg,
        )

    # 3. Wrapper functions for input validation

    def get_desc(self):
        """
        Returns the Cosmology description.
        """
        return (
            f"{type(self).__name__}(H0={self['H0']}, Omega_dm0={self['Omega_dm0']}, "
            f"Omega_b0={self['Omega_b0']}, Omega_k0={self['Omega_k0']})"
        )

    def init_from_params(self, H0=67.66, Omega_b0=0.049, Omega_dm0=0.262, Omega_k0=0.0):
        """Set the cosmology from parameters

        Parameters
        ----------
        H0: float
            Hubble parameter.
        Omega_b0: float
            Mass density of baryons today.
        Omega_dm0: float
            Mass density of dark matter only (no baryons) today.
        Omega_k0: float
            Mass density of curvature today.
        """
        if self.validate_input:
            validate_argument(locals(), "H0", float, argmin=0)
            validate_argument(locals(), "Omega_b0", float, argmin=0, eqmin=True)
            validate_argument(locals(), "Omega_dm0", float, argmin=0, eqmin=True)
            validate_argument(locals(), "Omega_k0", float, argmin=0, eqmin=True)
        self._init_from_params(H0=H0, Omega_b0=Omega_b0, Omega_dm0=Omega_dm0, Omega_k0=Omega_k0)

    def set_be_cosmo(self, be_cosmo=None, H0=67.66, Omega_b0=0.049, Omega_dm0=0.262, Omega_k0=0.0):
        """Set the cosmology

        Parameters
        ----------
        be_cosmo: clmm.cosmology.Cosmology object, None
            Input cosmology, used if not None.
        **kwargs
            Individual cosmological parameters, see init_from_params function.
        """
        if be_cosmo:
            self._init_from_cosmo(be_cosmo)
        else:
            self.init_from_params(H0=H0, Omega_b0=Omega_b0, Omega_dm0=Omega_dm0, Omega_k0=Omega_k0)

    def get_Omega_m(self, z):
        r"""Gets the value of the dimensionless matter density

        .. math::
            \Omega_m(z) = \frac{\rho_m(z)}{\rho_\text{crit}(z)}.

        Parameters
        ----------
        z : float, array_like
            Redshift

        Returns
        -------
        Omega_m : float, numpy.ndarray
            Dimensionless matter density, :math:`\Omega_m(z)`

        Notes
        -----
        Need to decide if non-relativist neutrinos will contribute here.
        """
        if self.validate_input:
            validate_argument(locals(), "z", "float_array", argmin=0, eqmin=True)
        return self._get_Omega_m(z=z)

    def get_E2(self, z):
        r"""Gets the value of the hubble parameter (normalized at 0)

        .. math::
            E^2(z) = \frac{H(z)^{2}}{H_{0}^{2}}.

        Parameters
        ----------
        z : float
            Redshift.
        Returns
        -------
        Hubble parameter : float
            :math:`H(z)^{2}/H_{0}^{2}`.
        Notes
        -----
        Need to decide if non-relativist neutrinos will contribute here.
        """
        if self.validate_input:
            validate_argument(locals(), "z", "float_array", argmin=0, eqmin=True)
        return self._get_E2(z=z)

    def get_E2Omega_m(self, z):
        r"""Gets the value of the dimensionless matter density times the Hubble parameter squared
        (normalized at 0)

        .. math::
            \Omega_m(z) = \frac{\rho_m(z)}{\rho_\text{crit}(z)}\frac{H(z)^{2}}{H_{0}^{2}}.

        Parameters
        ----------
        z : float, array_like
            Redshift

        Returns
        -------
        Omega_m : float, numpy.ndarray
            Dimensionless matter density, :math:`\Omega_m(z)\times H(z)^{2}/H_{0}^{2}`

        Notes
        -----
        Need to decide if non-relativist neutrinos will contribute here.
        """
        if self.validate_input:
            validate_argument(locals(), "z", "float_array", argmin=0, eqmin=True)
        return self._get_E2Omega_m(z=z)

    def get_rho_m(self, z):
        r"""Gets physical matter density at a given redshift.

        Parameters
        ----------
        z : float, array_like
            Redshift

        Returns
        -------
        float, numpy.ndarray
            Matter density :math:`M_\odot\ Mpc^{-3}`
        """
        if self.validate_input:
            validate_argument(locals(), "z", "float_array", argmin=0, eqmin=True)
        return self._get_rho_m(z=z)

    def get_rho_c(self, z):
        r"""Gets physical critical density at a given redshift.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            Critical density :math:`M_\odot\ Mpc^{-3}`.
        """
        if self.validate_input:
            validate_argument(locals(), "z", "float_array", argmin=0, eqmin=True)
        return self._get_rho_c(z=z)

    def eval_da_z1z2(self, z1, z2):
        r"""Computes the angular diameter distance between z1 and z2

        .. math::
            d_a(z1, z2) = \frac{c}{H_0}\frac{1}{1+z2}\int_{z1}^{z2}\frac{dz'}{E(z')}.

        Parameters
        ----------
        z1 : float, array_like
            Redshift
        z2 : float, array_like
            Redshift

        Returns
        -------
        float, numpy.ndarray
            Angular diameter distance in units :math:`M\!pc`

        Notes
        -----
        np.nan is returned for z1>z2.
        """
        if self.validate_input:
            validate_argument(locals(), "z1", "float_array", argmin=0, eqmin=True)
            validate_argument(locals(), "z2", "float_array", argmin=0, eqmin=True)
        return self._eval_da_z1z2(z1=z1, z2=z2)

    def eval_da(self, z):
        r"""Computes the angular diameter distance between 0.0 and z

        .. math::
            d_a(z) = \frac{c}{H_0}\frac{1}{1+z}\int_{0}^{z}\frac{dz'}{E(z')}.

        Parameters
        ----------
        z : float, array_like
            Redshift

        Returns
        -------
        float, numpy.ndarray
            Angular diameter distance in units :math:`M\!pc`
        """
        if self.validate_input:
            validate_argument(locals(), "z", "float_array", argmin=0, eqmin=True)
        return self._eval_da(z)

    def eval_da_a1a2(self, a1, a2=1.0):
        r"""This is a function to calculate the angular diameter distance
        between two scale factors.

        .. math::
            d_a(a1, a2) = \frac{c}{H_0}a2\int_{a2}^{a1}\frac{da'}{a'^2E(a')}

        If only a1 is specified, this function returns the angular diameter
        distance from a=1 to a1. If both a1 and a2 are specified, this function
        returns the angular diameter distance between a1 and a2.

        .. math::
            d_a(a) = \frac{c}{H_0}a\int_{a}^{1}\frac{da'}{a'^2E(a')}

        Parameters
        ----------
        a1 : float, array_like
            Scale factor
        a2 : float, array_like, optional
            Scale factor

        Returns
        -------
        float, numpy.ndarray
            Angular diameter distance in units :math:`M\!pc`
        """
        if self.validate_input:
            validate_argument(
                locals(),
                "a1",
                "float_array",
                argmin=0,
                eqmin=True,
                argmax=1,
                eqmax=True,
            )
            validate_argument(
                locals(),
                "a2",
                "float_array",
                argmin=0,
                eqmin=True,
                argmax=1,
                eqmax=True,
            )
        z1 = self.get_z_from_a(a2)
        z2 = self.get_z_from_a(a1)
        return self._eval_da_z1z2(z1, z2)

    def get_a_from_z(self, z):
        """Convert redshift to scale factor

        Parameters
        ----------
        z : float, array_like
            Redshift

        Returns
        -------
        a : float, numpy.ndarray
            Scale factor
        """
        if self.validate_input:
            validate_argument(locals(), "z", "float_array", argmin=0, eqmin=True)
        return self._get_a_from_z(z)

    def get_z_from_a(self, a):
        """Convert scale factor to redshift

        Parameters
        ----------
        a : float, array_like
            Scale factor

        Returns
        -------
        z : float, numpy.ndarray
            Redshift
        """
        if self.validate_input:
            validate_argument(
                locals(), "a", "float_array", argmin=0, eqmin=True, argmax=1, eqmax=True
            )
        return self._get_z_from_a(a)

    def rad2mpc(self, dist1, redshift):
        r"""Convert between radians and Mpc using the small angle approximation
        and :math:`d = D_A \theta`.

        Parameters
        ----------
        dist1 : float, array_like
            Input distances in radians
        redshift : float
            Redshift used to convert between angular and physical units

        Returns
        -------
        dist2 : float, numpy.ndarray
            Distances in Mpc
        """
        if self.validate_input:
            validate_argument(locals(), "dist1", "float_array", argmin=0, eqmin=True)
            validate_argument(locals(), "redshift", float, argmin=0, eqmin=True)
        return dist1 * self.eval_da(redshift)

    def mpc2rad(self, dist1, redshift):
        r"""Convert between radians and Mpc using the small angle approximation
        and :math:`d = D_A \theta`.

        Parameters
        ----------
        dist1 : float, array_like
            Input distances in Mpc
        redshift : float
            Redshift used to convert between angular and physical units

        Returns
        -------
        dist2 : float, numpy.ndarray
            Distances in radians
        """
        if self.validate_input:
            validate_argument(locals(), "dist1", "float_array", argmin=0, eqmin=True)
            validate_argument(locals(), "redshift", float, argmin=0, eqmin=True)
        return dist1 / self.eval_da(redshift)

    def eval_sigma_crit(self, z_len, z_src):
        r"""Computes the critical surface density

        Parameters
        ----------
        z_len : float
            Lens redshift
        z_src : float, array_like
            Background source galaxy redshift(s)

        Returns
        -------
        float, numpy.ndarray
            Cosmology-dependent critical surface density in units of :math:`M_\odot\ Mpc^{-2}`

        Notes
        -----
        np.inf is returned for z_src<z_len.
        """
        if self.validate_input:
            validate_argument(locals(), "z_len", float, argmin=0, eqmin=True)
            validate_argument(locals(), "z_src", "float_array", argmin=0, eqmin=True)
        return self._eval_sigma_crit(z_len=z_len, z_src=z_src)

    def eval_linear_matter_powerspectrum(self, k_vals, redshift):
        r"""Computes the linear matter power spectrum

        Parameters
        ----------
        k_vals : float, array_like
            Wavenumber k [math:`Mpc^{-1}`] values to compute the power spectrum.
        redshift : float
            Redshift to get the power spectrum.

        Returns
        -------
        float, numpy.ndarray
            Linear matter spectrum in units of math:`Mpc^{3}`
        """
        if self.validate_input:
            validate_argument(locals(), "k_vals", "float_array", argmin=0)
            validate_argument(locals(), "redshift", float, argmin=0, eqmin=True)
        return self._eval_linear_matter_powerspectrum(k_vals, redshift)
