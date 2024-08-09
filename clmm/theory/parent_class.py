"""@file parent_class.py
CLMModeling abstract class
"""
# pylint: disable=too-many-lines
import warnings

import numpy as np

# functions for the 2h term
from scipy.integrate import simpson, quad
from scipy.special import jv
from scipy.interpolate import splrep, splev

from .generic import (
    compute_reduced_shear_from_convergence,
    compute_magnification_bias_from_magnification,
    compute_rdelta,
    compute_profile_mass_in_radius,
    convert_profile_mass_concentration,
)
from ..utils import (
    validate_argument,
    compute_beta_s_func,
)
from ..redshift import (
    _integ_pzfuncs,
    compute_for_good_redshifts,
)


warnings.filterwarnings("always", module="(clmm).*")


class CLMModeling:
    r"""Object with functions for halo mass modeling

    Attributes
    ----------
    backend: str
        Name of the backend being used
    mdelta: float
        Mass of the profile, in units of :math:`M_\odot`
    cdelta: float
        Concentration of the profile
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
    validate_input: bool
        Validade each input argument
    cosmo_class: type
        Type of used cosmology objects
    z_inf : float
        The value used as infinite redshift
    """
    # pylint: disable=too-many-instance-attributes
    # The disable below is added to avoid a pylint error where it thinks CLMMCosmlogy
    # has duplicates since both have many NotImplementedError functions
    # description of bug at https://github.com/pylint-dev/pylint/issues/7213
    # pylint: disable=duplicate-code

    def __init__(self, validate_input=True, z_inf=1000):
        self.backend = None

        self.__massdef = ""
        self.__delta_mdef = 0
        self.__halo_profile_model = ""

        self.cosmo = None

        self.hdpm = None
        self.mdef_dict = {}
        self.hdpm_dict = {}

        self.validate_input = validate_input
        self.cosmo_class = lambda *args: None

        self.z_inf = z_inf

    # 1. Object properties

    @property
    def mdelta(self):
        """Mass of cluster"""
        return self._get_mass()

    @property
    def cdelta(self):
        """Concentration of cluster"""
        return self._get_concentration()

    @property
    def massdef(self):
        """Definition for the mass of cluster"""
        return self.__massdef

    @property
    def delta_mdef(self):
        """Number of deltas in mass definition of cluster"""
        return self.__delta_mdef

    @property
    def halo_profile_model(self):
        """Halo profile model"""
        return self.__halo_profile_model

    # 1.a Object properties setter

    @mdelta.setter
    def mdelta(self, mdelta):
        """Set mass of cluster"""
        self.set_mass(mdelta)

    @cdelta.setter
    def cdelta(self, cdelta):
        """Set concentration of cluster"""
        self.set_concentration(cdelta)

    @massdef.setter
    def massdef(self, massdef):
        """Set definition for the mass of cluster"""
        self.set_halo_density_profile(
            halo_profile_model=self.halo_profile_model,
            massdef=massdef,
            delta_mdef=self.delta_mdef,
        )

    @delta_mdef.setter
    def delta_mdef(self, delta_mdef):
        """Set number of deltas in mass definition of cluster"""
        self.set_halo_density_profile(
            halo_profile_model=self.halo_profile_model,
            massdef=self.massdef,
            delta_mdef=delta_mdef,
        )

    @halo_profile_model.setter
    def halo_profile_model(self, halo_profile_model):
        """Set halo profile model"""
        self.set_halo_density_profile(
            halo_profile_model=halo_profile_model,
            massdef=self.massdef,
            delta_mdef=self.delta_mdef,
        )

    # 2. Functions to be implemented by children classes

    def _get_mass(self):
        r"""Gets the value of the :math:`M_\Delta`"""
        raise NotImplementedError

    def _get_concentration(self):
        r"""Gets the value of the concentration"""
        raise NotImplementedError

    def _set_mass(self, mdelta):
        r"""Actually sets the value of the :math:`M_\Delta` (without value check)"""
        raise NotImplementedError

    def _set_concentration(self, cdelta):
        r"""Actuall sets the value of the concentration (without value check)"""
        raise NotImplementedError

    def _update_halo_density_profile(self):
        raise NotImplementedError

    def _set_einasto_alpha(self, alpha):
        r"""Actually sets the value of the :math:`\alpha` parameter for the Einasto profile"""
        raise NotImplementedError

    def _get_einasto_alpha(self, z_cl=None):
        r"""Returns the value of the :math:`\alpha` parameter for the Einasto profile,
        if defined"""
        raise NotImplementedError

    def _set_projected_quad(self, use_projected_quad):
        """Implemented for the CCL backend only"""
        raise NotImplementedError

    def _eval_3d_density(self, r3d, z_cl):
        raise NotImplementedError

    def _eval_surface_density(self, r_proj, z_cl):
        raise NotImplementedError

    def _eval_mean_surface_density(self, r_proj, z_cl):
        raise NotImplementedError

    def _eval_excess_surface_density(self, r_proj, z_cl):
        raise NotImplementedError

    # 3. Functions that can be used by all subclasses

    def _set_cosmo(self, cosmo):
        r"""Sets the cosmology to the internal cosmology object"""
        self.cosmo = cosmo if cosmo is not None else self.cosmo_class()

    def _eval_2halo_term_generic(
        self,
        sph_harm_ord,
        r_proj,
        z_cl,
        halobias=1.0,
        logkbounds=(-5, 5),
        ksteps=1000,
        loglbounds=(0, 6),
        lsteps=500,
    ):
        """eval excess surface density from the 2-halo term"""
        # pylint: disable=protected-access
        da = self.cosmo.eval_da(z_cl)
        rho_m = self.cosmo._get_rho_m(z_cl)

        k_values = np.logspace(logkbounds[0], logkbounds[1], ksteps)
        pk_values = self.cosmo._eval_linear_matter_powerspectrum(k_values, z_cl)
        interp_pk = splrep(k_values, pk_values)
        theta = r_proj / da

        # calculate integral, units [Mpc]**-3
        def __integrand__(l_value, theta):
            k_value = l_value / ((1 + z_cl) * da)
            return l_value * jv(sph_harm_ord, l_value * theta) * splev(k_value, interp_pk)

        l_values = np.logspace(loglbounds[0], loglbounds[1], lsteps)
        kernel = np.array([simpson(__integrand__(l_values, t), x=l_values) for t in theta])
        return halobias * kernel * rho_m / (2 * np.pi * (1 + z_cl) ** 3 * da**2)

    def _eval_surface_density_2h(
        self,
        r_proj,
        z_cl,
        halobias=1.0,
        logkbounds=(-5, 5),
        ksteps=1000,
        loglbounds=(0, 6),
        lsteps=500,
    ):
        """eval surface density from the 2-halo term"""
        return self._eval_2halo_term_generic(
            0, r_proj, z_cl, halobias, logkbounds, ksteps, loglbounds, lsteps
        )

    def _eval_excess_surface_density_2h(
        self,
        r_proj,
        z_cl,
        halobias=1.0,
        logkbounds=(-5, 5),
        ksteps=1000,
        loglbounds=(0, 6),
        lsteps=500,
    ):
        """eval excess surface density from the 2-halo term"""
        return self._eval_2halo_term_generic(
            2, r_proj, z_cl, halobias, logkbounds, ksteps, loglbounds, lsteps
        )

    def _eval_rdelta(self, z_cl):
        return compute_rdelta(self.mdelta, z_cl, self.cosmo, self.massdef, self.delta_mdef)

    def _eval_mass_in_radius(self, r3d, z_cl):
        alpha = self._get_einasto_alpha(z_cl) if self.halo_profile_model == "einasto" else None
        return compute_profile_mass_in_radius(
            r3d,
            z_cl,
            self.cosmo,
            self.mdelta,
            self.cdelta,
            self.massdef,
            self.delta_mdef,
            self.halo_profile_model,
            alpha,
        )

    def _convert_mass_concentration(
        self, z_cl, massdef=None, delta_mdef=None, halo_profile_model=None, alpha=None
    ):
        alpha1 = self._get_einasto_alpha(z_cl) if self.halo_profile_model == "einasto" else None
        return convert_profile_mass_concentration(
            self.mdelta,
            self.cdelta,
            z_cl,
            self.cosmo,
            massdef=self.massdef,
            delta_mdef=self.delta_mdef,
            halo_profile_model=self.halo_profile_model,
            alpha=alpha1,
            massdef2=massdef,
            delta_mdef2=delta_mdef,
            halo_profile_model2=halo_profile_model,
            alpha2=alpha,
        )

    # 3.1. All these functions are for the single plane case

    def _eval_tangential_shear_core(self, r_proj, z_cl, z_src):
        delta_sigma = self.eval_excess_surface_density(r_proj, z_cl)
        sigma_c = self.cosmo.eval_sigma_crit(z_cl, z_src)
        return delta_sigma / sigma_c

    def _eval_convergence_core(self, r_proj, z_cl, z_src):
        sigma = self.eval_surface_density(r_proj, z_cl)
        sigma_c = self.cosmo.eval_sigma_crit(z_cl, z_src)
        return sigma / sigma_c

    def _eval_reduced_tangential_shear_core(self, r_proj, z_cl, z_src):
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_tangential_shear(r_proj, z_cl, z_src)
        return compute_reduced_shear_from_convergence(gamma_t, kappa)

    def _eval_magnification_core(self, r_proj, z_cl, z_src):
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_tangential_shear(r_proj, z_cl, z_src)
        return 1.0 / ((1 - kappa) ** 2 - abs(gamma_t) ** 2)

    def _eval_magnification_bias_core(self, r_proj, z_cl, z_src, alpha):
        magnification = self.eval_magnification(r_proj, z_cl, z_src)
        return compute_magnification_bias_from_magnification(magnification, alpha)

    # 4. Wrapper functions for input validation

    def set_mass(self, mdelta):
        r"""Sets the value of the :math:`M_\Delta`.

        Parameters
        ----------
        mdelta : float
            Galaxy cluster mass :math:`M_\Delta` in units of :math:`M_\odot`

        Notes
        -----
            This is equivalent to doing self.mdelta = mdelta
        """
        if self.validate_input:
            validate_argument(locals(), "mdelta", float, argmin=0)
        self._set_mass(mdelta)

    def set_concentration(self, cdelta):
        r"""Sets the concentration

        Parameters
        ----------
        cdelta: float
            Concentration

        Notes
        -----
            This is equivalent to doing self.cdelta = cdelta
        """
        if self.validate_input:
            validate_argument(locals(), "cdelta", float, argmin=0)
        self._set_concentration(cdelta)

    def set_cosmo(self, cosmo):
        r"""Sets the cosmology to the internal cosmology object

        Parameters
        ----------
        cosmo: clmm.Comology object, None
            CLMM Cosmology object. If is None, creates a new instance of self.cosmo_class().
        """
        if self.validate_input:
            if self.cosmo_class() is None:
                raise NotImplementedError
            validate_argument(locals(), "cosmo", self.cosmo_class, none_ok=True)
        self._set_cosmo(cosmo)
        self.cosmo.validate_input = self.validate_input

    def set_halo_density_profile(self, halo_profile_model="nfw", massdef="mean", delta_mdef=200):
        r"""Sets the definitions for the halo profile

        Parameters
        ----------
        halo_profile_model: str
            Halo mass profile, supported options are 'nfw', 'einasto', 'hernquist'
            (letter case independent)
        massdef: str
            Mass definition, supported options are 'mean', 'critical', 'virial'
            (letter case independent)
        delta_mdef: int
            Overdensity number
        """
        # make case independent
        validate_argument(locals(), "massdef", str)
        validate_argument(locals(), "halo_profile_model", str)
        massdef, halo_profile_model = massdef.lower(), halo_profile_model.lower()

        if self.validate_input:
            validate_argument(locals(), "delta_mdef", int, argmin=0)
            if massdef not in self.mdef_dict:
                raise ValueError(
                    f"Halo density profile mass definition {massdef} not currently supported"
                )
            if halo_profile_model not in self.hdpm_dict:
                raise ValueError(
                    f"Halo density profile model {halo_profile_model} not currently supported"
                )
        # Check if we have already an instance of the required object, if not create one
        if (
            (self.hdpm is None)
            or (self.halo_profile_model != halo_profile_model)
            or (self.massdef != massdef)
            or (self.delta_mdef != delta_mdef)
        ):
            # set internal quantities
            self.__halo_profile_model = halo_profile_model
            self.__massdef = massdef
            self.__delta_mdef = delta_mdef
            # set the profile
            self._update_halo_density_profile()

    def set_einasto_alpha(self, alpha):
        r"""Sets the value of the :math:`\alpha` parameter for the Einasto profile

        Parameters
        ----------
        alpha : float, None
            If None, use the default value of the backend. (0.25 for the NumCosmo backend and a
            cosmology-dependent value for the CCL backend.)
        """
        if self.halo_profile_model != "einasto":
            raise NotImplementedError(
                "The Einasto slope cannot be set "
                "for your combination of profile choice "
                "or modeling backend."
            )
        if self.validate_input:
            validate_argument(locals(), "alpha", float, none_ok=True)
        self._set_einasto_alpha(alpha)

    def get_einasto_alpha(self, z_cl=None):
        r"""Returns the value of the :math:`\alpha` parameter for the Einasto profile, if defined

        Parameters
        ----------
        z_cl : float
            Cluster redshift (required for Einasto with the CCL backend, will be ignored for NC)
        """
        if self.halo_profile_model != "einasto":
            raise ValueError(f"Wrong profile model. Current profile = {self.halo_profile_model}")
        return self._get_einasto_alpha(z_cl)

    def set_projected_quad(self, use_projected_quad):
        r"""Control the use of quad_vec to calculate the surface density profile for
        CCL Einasto profile.

        Parameters
        ----------
        use_projected_quad : bool
            Only available for Einasto profile with CCL as the backend. If True, CCL will use
            quad_vec instead of default FFTLog to calculate the surface density profile.
        """
        if self.halo_profile_model != "einasto" or self.backend != "ccl":
            raise NotImplementedError("This option is only available for the CCL Einasto profile.")
        if self.validate_input:
            validate_argument(locals(), "use_projected_quad", bool)
        self._set_projected_quad(use_projected_quad)

    def eval_3d_density(self, r3d, z_cl, verbose=False):
        r"""Retrieve the 3d density :math:`\rho(r)`.

        Parameters
        ----------
        r3d : array_like, float
            Radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster

        Returns
        -------
        numpy.ndarray, float
            3-dimensional mass density in units of :math:`M_\odot\ Mpc^{-3}`
        """
        if self.validate_input:
            validate_argument(locals(), "r3d", "float_array", argmin=0)
            validate_argument(locals(), "z_cl", "float_array", argmin=0)

        if self.halo_profile_model == "einasto" and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        return self._eval_3d_density(r3d=r3d, z_cl=z_cl)

    def eval_critical_surface_density_eff(self, z_len, pzbins, pzpdf):
        r"""Computes the 'effective critical surface density'

        .. math::
            \langle \Sigma_{\rm crit}^{-1}\rangle^{-1} =
            \left(\int \frac{1}{\Sigma_{\rm crit}(z)} p(z) dz\right)^{-1}

        where :math:`p(z)` is the source photoz probability density function.
        This comes from the maximum likelihood estimator for evaluating a
        :math:`\Delta\Sigma` profile.

        For the standard :math:`\Sigma_{\text{crit}}(z)` definition, use the `eval_sigma_crit`
        method of the CLMM cosmology object.

        Parameters
        ----------
        z_len : float
            Galaxy cluster redshift
        pzbins : array-like
            Bins where the source redshift pdf is defined
        pzpdf : array-like
            Values of the source redshift pdf


        Returns
        -------
        sigma_c : numpy.ndarray, float
            Cosmology-dependent effective critical surface density in units of
            :math:`M_\odot\ Mpc^{-2}`
        """

        if self.validate_input:
            validate_argument(locals(), "z_len", float, argmin=0)

        def inv_sigmac(redshift):
            return 1.0 / self.cosmo.eval_sigma_crit(z_len, redshift)

        return 1.0 / _integ_pzfuncs(pzpdf, pzbins, kernel=inv_sigmac)

    def eval_surface_density(self, r_proj, z_cl, verbose=False):
        r"""Computes the surface mass density

        Parameters
        ----------
        r_proj : array_like
            Projected radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster

        Returns
        -------
        numpy.ndarray, float
            2D projected surface density in units of :math:`M_\odot\ Mpc^{-2}`
        """
        if self.validate_input:
            validate_argument(locals(), "r_proj", "float_array", argmin=0)
            validate_argument(locals(), "z_cl", float, argmin=0)

        if self.halo_profile_model == "einasto" and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        return self._eval_surface_density(r_proj=r_proj, z_cl=z_cl)

    def eval_mean_surface_density(self, r_proj, z_cl, verbose=False):
        r"""Computes the mean value of surface density inside radius `r_proj`

        Parameters
        ----------
        r_proj : array_like
            Projected radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster

        Returns
        -------
        numpy.ndarray, float
            Excess surface density in units of :math:`M_\odot\ Mpc^{-2}`.
        """
        if self.validate_input:
            validate_argument(locals(), "r_proj", "float_array", argmin=0)
            validate_argument(locals(), "z_cl", float, argmin=0)

        if self.halo_profile_model == "einasto" and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        return self._eval_mean_surface_density(r_proj=r_proj, z_cl=z_cl)

    def eval_excess_surface_density(self, r_proj, z_cl, verbose=False):
        r"""Computes the excess surface density

        Parameters
        ----------
        r_proj : array_like
            Projected radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster

        Returns
        -------
        numpy.ndarray, float
            Excess surface density in units of :math:`M_\odot\ Mpc^{-2}`.
        """
        if self.validate_input:
            validate_argument(locals(), "r_proj", "float_array", argmin=0)
            validate_argument(locals(), "z_cl", float, argmin=0)

        if self.halo_profile_model == "einasto" and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        return self._eval_excess_surface_density(r_proj=r_proj, z_cl=z_cl)

    def eval_excess_surface_density_2h(
        self,
        r_proj,
        z_cl,
        halobias=1.0,
        logkbounds=(-5, 5),
        ksteps=1000,
        loglbounds=(0, 6),
        lsteps=500,
    ):
        r"""Computes the 2-halo term excess surface density (CCL and NC backends only)

        Parameters
        ----------
        r_proj : array_like
            Projected radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster
        halobias : float, optional
            Value of the halo bias
        logkbounds : tuple(float, float), shape (2,), optional
           Log10 of the upper and lower bounds for the linear matter power spectrum
        ksteps : int, optional
           Number of steps in k-space
        loglbounds : tuple(float, float), shape (2,), optional
           Log10 of the upper and lower bounds for numerical integration
        lsteps: int, optional
            Number of steps for numerical integration

        Returns
        -------
        numpy.ndarray, float
            Excess surface density from the 2-halo term in units of :math:`M_\odot\ Mpc^{-2}`.
        """

        if self.validate_input:
            validate_argument(locals(), "r_proj", "float_array", argmin=0)
            validate_argument(locals(), "z_cl", float, argmin=0)
            validate_argument(locals(), "halobias", float, argmin=0)
            validate_argument(locals(), "logkbounds", tuple, shape=(2,))
            validate_argument(locals(), "ksteps", int, argmin=1)
            validate_argument(locals(), "loglbounds", tuple, shape=(2,))
            validate_argument(locals(), "lsteps", int, argmin=1)

        if self.backend not in ("ccl", "nc"):
            raise NotImplementedError(
                f"2-halo term not currently supported with the {self.backend} backend. "
                "Use the CCL or NumCosmo backend instead"
            )
        return self._eval_excess_surface_density_2h(
            r_proj, z_cl, halobias, logkbounds, ksteps, loglbounds, lsteps
        )

    def eval_surface_density_2h(
        self,
        r_proj,
        z_cl,
        halobias=1.0,
        logkbounds=(-5, 5),
        ksteps=1000,
        loglbounds=(0, 6),
        lsteps=500,
    ):
        r"""Computes the 2-halo term surface density (CCL and NC backends only)

        Parameters
        ----------
        r_proj : array_like
            Projected radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster
        halobias : float, optional
           Value of the halo bias
        logkbounds : tuple(float,float), shape (2,), optional
           Log10 of the upper and lower bounds for the linear matter power spectrum
        ksteps : int, optional
           Number of steps in k-space
        loglbounds : tuple(float,float), shape (2,), optional
           Log10 of the upper and lower bounds for numerical integration
        lsteps: int, optional
            Number of steps for numerical integration

        Returns
        -------
        numpy.ndarray, float
            Excess surface density from the 2-halo term in units of :math:`M_\odot\ Mpc^{-2}`.
        """

        if self.validate_input:
            validate_argument(locals(), "r_proj", "float_array", argmin=0)
            validate_argument(locals(), "z_cl", float, argmin=0)
            validate_argument(locals(), "halobias", float, argmin=0)
            validate_argument(locals(), "logkbounds", tuple, shape=(2,))
            validate_argument(locals(), "ksteps", int, argmin=1)
            validate_argument(locals(), "loglbounds", tuple, shape=(2,))
            validate_argument(locals(), "lsteps", int, argmin=1)

        if self.backend not in ("ccl", "nc"):
            raise NotImplementedError(
                f"2-halo term not currently supported with the {self.backend} backend. "
                "Use the CCL or NumCosmo backend instead"
            )
        return self._eval_surface_density_2h(
            r_proj, z_cl, halobias, logkbounds, ksteps, loglbounds, lsteps
        )

    def eval_tangential_shear(self, r_proj, z_cl, z_src, z_src_info="discrete", verbose=False):
        r"""Computes the tangential shear

        Parameters
        ----------
        r_proj : array_like
            The projected radial positions in :math:`M\!pc`.
        z_cl : float
            Galaxy cluster redshift
        z_src : array_like, float, function
            Information on the background source galaxy redshift(s). Value required depends on
            `z_src_info` (see below).
        z_src_info : str, optional
            Type of redshift information provided by the `z_src` argument.
            The following supported options are:

                * 'discrete' (default) : The redshift of sources is provided by `z_src`.
                  It can be individual redshifts for each source galaxy when `z_src` is an
                  array or all sources are at the same redshift when `z_src` is a float.

                * 'beta' : The averaged lensing efficiency is provided by `z_src`.
                  `z_src` must be a tuple containing
                  ( :math:`\langle \beta_s \rangle, \langle \beta_s^2 \rangle`),
                  the lensing efficiency and square of the lensing efficiency averaged over
                  the galaxy redshift distribution repectively.

                    .. math::
                        \langle \beta_s \rangle = \left\langle \frac{D_{LS}}{D_S}\frac{D_\infty}
                        {D_{L,\infty}}\right\rangle

                    .. math::
                        \langle \beta_s^2 \rangle = \left\langle \left(\frac{D_{LS}}
                        {D_S}\frac{D_\infty}{D_{L,\infty}}\right)^2 \right\rangle

        verbose : bool, optional
            If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and
            CCL backends.

        Returns
        -------
        numpy.ndarray, float
            tangential shear
        """

        if self.validate_input:
            validate_argument(locals(), "r_proj", "float_array", argmin=0)
            validate_argument(locals(), "z_cl", float, argmin=0)
            validate_argument(locals(), "z_src_info", str)
            self._validate_z_src(locals())

        if self.halo_profile_model == "einasto" and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        # function _validate_z_src already safekeeps from this error:
        # pylint: disable=possibly-used-before-assignment

        if z_src_info == "discrete":
            warning_msg = (
                "\nSome source redshifts are lower than the cluster redshift."
                + "\nShear = 0 for those galaxies."
            )
            gammat = compute_for_good_redshifts(
                self._eval_tangential_shear_core,
                z_cl,
                z_src,
                0.0,
                warning_msg,
                "z_cl",
                "z_src",
                r_proj,
            )
        elif z_src_info == "beta":
            beta_s_mean = z_src[0]
            gammat_inf = self._eval_tangential_shear_core(
                r_proj=r_proj, z_cl=z_cl, z_src=self.z_inf
            )
            gammat = beta_s_mean * gammat_inf

        return gammat

    def eval_convergence(self, r_proj, z_cl, z_src, z_src_info="discrete", verbose=False):
        r"""Computes the mass convergence

        .. math::
            \kappa = \frac{\Sigma}{\Sigma_{crit}}

        or

        .. math::
            \kappa = \kappa_\infty \times <\beta_s>

        Parameters
        ----------
        r_proj : array_like
            The projected radial positions in :math:`M\!pc`.
        z_cl : float
            Galaxy cluster redshift
        z_src : array_like, float, function
            Information on the background source galaxy redshift(s). Value required depends on
            `z_src_info` (see below).
        z_src_info : str, optional
            Type of redshift information provided by the `z_src` argument.
            The following supported options are:

                * 'discrete' (default) : The redshift of sources is provided by `z_src`.
                  It can be individual redshifts for each source galaxy when `z_src` is an
                  array or all sources are at the same redshift when `z_src` is a float.

                * 'beta' : The averaged lensing efficiency is provided by `z_src`.
                  `z_src` must be a tuple containing
                  ( :math:`\langle \beta_s \rangle, \langle \beta_s^2 \rangle`),
                  the lensing efficiency and square of the lensing efficiency averaged over
                  the galaxy redshift distribution repectively.

                    .. math::
                        \langle \beta_s \rangle = \left\langle \frac{D_{LS}}{D_S}\frac{D_\infty}
                        {D_{L,\infty}}\right\rangle

                    .. math::
                        \langle \beta_s^2 \rangle = \left\langle \left(\frac{D_{LS}}
                        {D_S}\frac{D_\infty}{D_{L,\infty}}\right)^2 \right\rangle

        verbose : bool, optional
            If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and
            CCL backends.

        Returns
        -------
        numpy.ndarray, float
            Mass convergence, kappa.
        """
        if self.validate_input:
            validate_argument(locals(), "r_proj", "float_array", argmin=0)
            validate_argument(locals(), "z_cl", float, argmin=0)
            validate_argument(locals(), "z_src_info", str)
            self._validate_z_src(locals())

        if self.halo_profile_model == "einasto" and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        # function _validate_z_src already safekeeps from this error:
        # pylint: disable=possibly-used-before-assignment

        if z_src_info == "discrete":
            warning_msg = (
                "\nSome source redshifts are lower than the cluster redshift."
                + "\nConvergence = 0 for those galaxies."
            )
            kappa = compute_for_good_redshifts(
                self._eval_convergence_core,
                z_cl,
                z_src,
                0.0,
                warning_msg,
                "z_cl",
                "z_src",
                r_proj,
            )
        elif z_src_info == "beta":
            beta_s_mean = z_src[0]
            kappa_inf = self._eval_convergence_core(r_proj=r_proj, z_cl=z_cl, z_src=self.z_inf)
            kappa = beta_s_mean * kappa_inf

        return kappa

    def _pdz_weighted_avg(self, core, pdz_func, r_proj, z_cl, integ_kwargs=None):
        r"""Computes function averaged over PDZ

        Parameters
        ----------
        core : function
            Function to be averaged, must take tangential shear and convergence as input.
        pdz_func : function
            Redshift distribution function. Must be a one dimentional function.
        r_proj : array_like
            The projected radial positions in :math:`M\!pc`.
        z_cl : float
            Galaxy cluster redshift
        integ_kwargs: None, dict
            Extra arguments for the redshift integration (when
            `approx=None, z_src_info='distribution'`). Possible keys are:

                * 'zmin' (None, float) : Minimum redshift to be set as the source of the galaxy
                  when performing the sum. (default=None)
                * 'zmax' (float) : Maximum redshift to be set as the source of the galaxy
                  when performing the sum. (default=10.0)
                * 'delta_z_cut' (float) : Redshift cut so that `zmin` = `z_cl` + `delta_z_cut`.
                  `delta_z_cut` is ignored if `z_min` is already provided. (default=0.1)

        Returns
        -------
        array_like
            Function averaged by pdz, with r_proj dimention.
        """

        def tfunc(z, radius):
            return compute_beta_s_func(
                z,
                z_cl,
                self.z_inf,
                self.cosmo,
                self._eval_tangential_shear_core,
                radius,
                z_cl,
                self.z_inf,
            )

        def kfunc(z, radius):
            return compute_beta_s_func(
                z,
                z_cl,
                self.z_inf,
                self.cosmo,
                self._eval_convergence_core,
                radius,
                z_cl,
                self.z_inf,
            )

        def __integrand__(z, radius):
            return pdz_func(z) * core(tfunc(z, radius), kfunc(z, radius))

        _integ_kwargs = {"zmax": 10.0, "delta_z_cut": 0.1}

        _integ_kwargs.update({} if integ_kwargs is None else integ_kwargs)

        zmax = _integ_kwargs["zmax"]
        delta_z_cut = _integ_kwargs["delta_z_cut"]
        zmin = _integ_kwargs.get("zmin", z_cl + delta_z_cut)

        out = np.array([quad(__integrand__, zmin, zmax, (r))[0] for r in r_proj])
        return out / quad(pdz_func, zmin, zmax)[0]

    def eval_reduced_tangential_shear(
        self,
        r_proj,
        z_cl,
        z_src,
        z_src_info="discrete",
        approx=None,
        integ_kwargs=None,
        verbose=False,
    ):
        r"""Computes the reduced tangential shear

        .. math::
            g_t = \frac{\gamma_t}{1-\kappa}

        Parameters
        ----------
        r_proj : array_like
            The projected radial positions in :math:`M\!pc`.
        z_cl : float
            Galaxy cluster redshift
        z_src : array_like, float, function
            Information on the background source galaxy redshift(s). Value required depends on
            `z_src_info` (see below).
        z_src_info : str, optional
            Type of redshift information provided by the `z_src` argument.
            The following supported options are:

                * 'discrete' (default) : The redshift of sources is provided by `z_src`.
                  It can be individual redshifts for each source galaxy when `z_src` is an
                  array or all sources are at the same redshift when `z_src` is a float
                  (Used for `approx=None`).

                * 'distribution' : A redshift distribution function is provided by `z_src`.
                  `z_src` must be a one dimentional function (Used when `approx=None`).

                * 'beta' : The averaged lensing efficiency is provided by `z_src`.
                  `z_src` must be a tuple containing
                  ( :math:`\langle \beta_s \rangle, \langle \beta_s^2 \rangle`),
                  the lensing efficiency and square of the lensing efficiency averaged over
                  the galaxy redshift distribution repectively.

                    .. math::
                        \langle \beta_s \rangle = \left\langle \frac{D_{LS}}{D_S}\frac{D_\infty}
                        {D_{L,\infty}}\right\rangle

                    .. math::
                        \langle \beta_s^2 \rangle = \left\langle \left(\frac{D_{LS}}
                        {D_S}\frac{D_\infty}{D_{L,\infty}}\right)^2 \right\rangle

        approx : str, optional
            Type of computation to be made for reduced tangential shears, options are:

                * None (default): Requires `z_src_info` to be 'discrete' or 'distribution'.
                  If `z_src_info='discrete'`, full computation is made for each
                  `r_proj, z_src` pair individually. If `z_src_info='distribution'`, reduced
                  tangential shear at each value of `r_proj` is calculated as

                  .. math::
                      g_t
                      =\left<\frac{\beta_s\gamma_{\infty}}{1-\beta_s\kappa_{\infty}}\right>
                      =\frac{\int_{z_{min}}^{z_{max}}\frac{\beta_s(z)\gamma_{\infty}}
                      {1-\beta_s(z)\kappa_{\infty}}N(z)\text{d}z}
                      {\int_{z_{min}}^{z_{max}} N(z)\text{d}z}

                * 'order1' : Same approach as in Weighing the Giants - III (equation 6 in
                  Applegate et al. 2014; https://arxiv.org/abs/1208.0605). `z_src_info` must be
                  'beta':

                  .. math::
                      g_t\approx\frac{\left<\beta_s\right>\gamma_{\infty}}
                      {1-\left<\beta_s\right>\kappa_{\infty}}

                * 'order2' : Same approach as in Cluster Mass Calibration at High
                  Redshift (equation 12 in Schrabback et al. 2017;
                  https://arxiv.org/abs/1611.03866).
                  `z_src_info` must be 'beta':

                  .. math::
                      g_t\approx\frac{\left<\beta_s\right>\gamma_{\infty}}
                      {1-\left<\beta_s\right>\kappa_{\infty}}
                      \left(1+\left(\frac{\left<\beta_s^2\right>}
                      {\left<\beta_s\right>^2}-1\right)\left<\beta_s\right>\kappa_{\infty}\right)

        integ_kwargs: None, dict
            Extra arguments for the redshift integration (when
            `approx=None, z_src_info='distribution'`). Possible keys are:

                * 'zmin' (None, float) : Minimum redshift to be set as the source of the galaxy
                  when performing the sum. (default=None)
                * 'zmax' (float) : Maximum redshift to be set as the source of the galaxy
                  when performing the sum. (default=10.0)
                * 'delta_z_cut' (float) : Redshift cut so that `zmin` = `z_cl` + `delta_z_cut`.
                  `delta_z_cut` is ignored if `z_min` is already provided. (default=0.1)

        verbose : bool, optional
            If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and
            CCL backends.

        Returns
        -------
        gt : numpy.ndarray, float
            Reduced tangential shear

        Notes
        -----
        Need to figure out if we want to raise exceptions rather than errors here?
        """
        if self.validate_input:
            validate_argument(locals(), "r_proj", "float_array", argmin=0)
            validate_argument(locals(), "z_cl", float, argmin=0)
            validate_argument(locals(), "z_src_info", str)
            validate_argument(locals(), "approx", str, none_ok=True)
            self._validate_approx_z_src_info(locals())
            self._validate_z_src(locals())

        if self.halo_profile_model == "einasto" and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        # functions _validate_z_src, _validate_approx_z_src_info already safekeeps from this error:
        # pylint: disable=possibly-used-before-assignment

        if approx is None:
            if z_src_info == "distribution":
                gt = self._pdz_weighted_avg(
                    lambda gammat, kappa: gammat / (1 - kappa),
                    z_src,
                    r_proj,
                    z_cl,
                    integ_kwargs=integ_kwargs,
                )
            elif z_src_info == "discrete":
                warning_msg = (
                    "\nSome source redshifts are lower than the cluster redshift."
                    + "\nReduced_shear = 0 for those galaxies."
                )
                gt = compute_for_good_redshifts(
                    self._eval_reduced_tangential_shear_core,
                    z_cl,
                    z_src,
                    0.0,
                    warning_msg,
                    "z_cl",
                    "z_src",
                    r_proj,
                )
        elif approx in ("order1", "order2"):
            beta_s_mean = z_src[0]

            gammat_inf = self._eval_tangential_shear_core(r_proj, z_cl, z_src=self.z_inf)
            kappa_inf = self._eval_convergence_core(r_proj, z_cl, z_src=self.z_inf)

            gt = beta_s_mean * gammat_inf / (1.0 - beta_s_mean * kappa_inf)

            if approx == "order2":
                beta_s_square_mean = z_src[1]
                gt *= (
                    1.0
                    + (beta_s_square_mean / (beta_s_mean * beta_s_mean) - 1.0)
                    * beta_s_mean
                    * kappa_inf
                )

        return gt

    def eval_magnification(
        self,
        r_proj,
        z_cl,
        z_src,
        z_src_info="discrete",
        approx=None,
        verbose=False,
        integ_kwargs=None,
    ):
        r"""Computes the magnification

        .. math::
            \mu = \frac{1}{(1-\kappa)^2-|\gamma_t|^2}

        Parameters
        ----------
        r_proj : array_like
            The projected radial positions in :math:`M\!pc`.
        z_cl : float
            Galaxy cluster redshift
        z_src : array_like, float, function
            Information on the background source galaxy redshift(s). Value required depends on
            `z_src_info` (see below).
        z_src_info : str, optional
            Type of redshift information provided by the `z_src` argument.
            The following supported options are:

                * 'discrete' (default) : The redshift of sources is provided by `z_src`.
                  It can be individual redshifts for each source galaxy when `z_src` is an
                  array or all sources are at the same redshift when `z_src` is a float
                  (Used for `approx=None`).

                * 'distribution' : A redshift distribution function is provided by `z_src`.
                  `z_src` must be a one dimentional function (Used when `approx=None`).

                * 'beta' : The averaged lensing efficiency is provided by `z_src`.
                  `z_src` must be a tuple containing
                  ( :math:`\langle \beta_s \rangle, \langle \beta_s^2 \rangle`),
                  the lensing efficiency and square of the lensing efficiency averaged over
                  the galaxy redshift distribution repectively.

                    .. math::
                        \langle \beta_s \rangle = \left\langle \frac{D_{LS}}{D_S}\frac{D_\infty}
                        {D_{L,\infty}}\right\rangle

                    .. math::
                        \langle \beta_s^2 \rangle = \left\langle \left(\frac{D_{LS}}
                        {D_S}\frac{D_\infty}{D_{L,\infty}}\right)^2 \right\rangle

        approx : str, optional
            Type of computation to be made for magnifications, options are:

                * None (default): Requires `z_src_info` to be 'discrete' or 'distribution'.
                  If `z_src_info='discrete'`, full computation is made for each
                  `r_proj, z_src` pair individually. If `z_src_info='distribution'`, magnification
                  at each value of `r_proj` is calculated as

                  .. math::
                      \mu
                      =\left<\frac{1}{\left(1-\beta_s\kappa_{\infty}\right)^2
                      -\left(\beta_s\gamma_{\infty}\right)^2}\right>
                      =\frac{\int_{z_{min}}^{z_{max}}\frac{N(z)\text{d}z}
                      {\left(1-\beta_s(z)\kappa_{\infty}\right)^2
                      -\left(\beta_s(z)\gamma_{\infty}\right)^2}}
                      {\int_{z_{min}}^{z_{max}} N(z)\text{d}z}

                * 'order1' : Uses the weak lensing approximation of the magnification with up to
                  first-order terms in :math:`\kappa_{\infty}` or :math:`\gamma_{\infty}`
                  (`z_src_info` must be 'beta'):

                  .. math::
                      \mu \approx 1 + 2 \left<\beta_s\right>\kappa_{\infty}

                * 'order2' : Uses the weak lensing approximation of the magnification with up to
                  second-order terms in :math:`\kappa_{\infty}` or :math:`\gamma_{\infty}`
                  (`z_src_info` must be 'beta'):

                  .. math::
                      \mu \approx 1 + 2 \left<\beta_s\right>\kappa_{\infty}
                      + 3 \left<\beta_s^2\right>\kappa_{\infty}^2
                      + \left<\beta_s^2\right>\gamma_{\infty}^2

        integ_kwargs: None, dict
            Extra arguments for the redshift integration (when
            `approx=None, z_src_info='distribution'`). Possible keys are:

                * 'zmin' (None, float) : Minimum redshift to be set as the source of the galaxy
                  when performing the sum. (default=None)
                * 'zmax' (float) : Maximum redshift to be set as the source of the galaxy
                  when performing the sum. (default=10.0)
                * 'delta_z_cut' (float) : Redshift cut so that `zmin` = `z_cl` + `delta_z_cut`.
                  `delta_z_cut` is ignored if `z_min` is already provided. (default=0.1)

        verbose : bool, optional
            If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and
            CCL backends.

        Returns
        -------
        mu : numpy.ndarray, float
            magnification, mu.

        """
        if self.validate_input:
            validate_argument(locals(), "r_proj", "float_array", argmin=0)
            validate_argument(locals(), "z_cl", float, argmin=0)
            validate_argument(locals(), "z_src_info", str)
            validate_argument(locals(), "approx", str, none_ok=True)
            self._validate_approx_z_src_info(locals())
            self._validate_z_src(locals())

        if self.halo_profile_model == "einasto" and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        # functions _validate_z_src, _validate_approx_z_src_info already safekeeps from this error:
        # pylint: disable=possibly-used-before-assignment

        if approx is None:
            if z_src_info == "distribution":
                mu = self._pdz_weighted_avg(
                    lambda gammat, kappa: 1 / ((1 - kappa) ** 2 - gammat**2),
                    z_src,
                    r_proj,
                    z_cl,
                    integ_kwargs=integ_kwargs,
                )
            elif z_src_info == "discrete":
                warning_msg = (
                    "\nSome source redshifts are lower than the cluster redshift."
                    + "\nMagnification = 1 for those galaxies."
                )
                mu = compute_for_good_redshifts(
                    self._eval_magnification_core,
                    z_cl,
                    z_src,
                    1.0,
                    warning_msg,
                    "z_cl",
                    "z_src",
                    r_proj,
                )
        elif approx in ("order1", "order2"):
            beta_s_mean = z_src[0]

            kappa_inf = self._eval_convergence_core(r_proj, z_cl, z_src=self.z_inf)
            gammat_inf = self._eval_tangential_shear_core(r_proj, z_cl, z_src=self.z_inf)

            mu = 1 + 2 * beta_s_mean * kappa_inf

            if approx == "order2":
                beta_s_square_mean = z_src[1]
                # Taylor expansion with up to second-order terms
                mu += 3 * beta_s_square_mean * kappa_inf**2 + beta_s_square_mean * gammat_inf**2

        return mu

    def eval_magnification_bias(
        self,
        r_proj,
        z_cl,
        z_src,
        alpha,
        z_src_info="discrete",
        approx=None,
        integ_kwargs=None,
        verbose=False,
    ):
        r"""Computes the magnification bias

        .. math::
            \mu^{\alpha - 1}

        Parameters
        ----------
        r_proj : array_like
            The projected radial positions in :math:`M\!pc`.
        z_cl : float
            Galaxy cluster redshift
        z_src : array_like, float, function
            Information on the background source galaxy redshift(s). Value required depends on
            `z_src_info` (see below).
        alpha : float
            Slope of the cummulative number count of background sources at a given magnitude
        z_src_info : str, optional
            Type of redshift information provided by the `z_src` argument.
            The following supported options are:

                * 'discrete' (default) : The redshift of sources is provided by `z_src`.
                  It can be individual redshifts for each source galaxy when `z_src` is an
                  array or all sources are at the same redshift when `z_src` is a float
                  (Used for `approx=None`).

                * 'distribution' : A redshift distribution function is provided by `z_src`.
                  `z_src` must be a one dimentional function (Used when `approx=None`).

                * 'beta' : The averaged lensing efficiency is provided by `z_src`.
                  `z_src` must be a tuple containing
                  ( :math:`\langle \beta_s \rangle, \langle \beta_s^2 \rangle`),
                  the lensing efficiency and square of the lensing efficiency averaged over
                  the galaxy redshift distribution repectively.

                    .. math::
                        \langle \beta_s \rangle = \left\langle \frac{D_{LS}}{D_S}\frac{D_\infty}
                        {D_{L,\infty}}\right\rangle

                    .. math::
                        \langle \beta_s^2 \rangle = \left\langle \left(\frac{D_{LS}}
                        {D_S}\frac{D_\infty}{D_{L,\infty}}\right)^2 \right\rangle

        approx : str, optional
            Type of computation to be made for magnification biases, options are:

                * None (default): Requires `z_src_info` to be 'discrete' or 'distribution'.
                  If `z_src_info='discrete'`, full computation is made for each
                  `r_proj, z_src` pair individually. If `z_src_info='distribution'`, magnification
                  bias at each value of `r_proj` is calculated as

                  .. math::
                      \mu^{\alpha-1}
                      &=\left(\left<\frac{1}{\left(1-\beta_s\kappa_{\infty}\right)^2
                      -\left(\beta_s\gamma_{\infty}\right)^2}\right>\right)^{\alpha-1}
                      \\\\
                      &=\frac{\int_{z_{min}}^{z_{max}}\frac{N(z)\text{d}z}
                      {\left(\left(1-\beta_s(z)\kappa_{\infty}\right)^2
                      -\left(\beta_s(z)\gamma_{\infty}\right)^2\right)^{\alpha-1}}}
                      {\int_{z_{min}}^{z_{max}} N(z)\text{d}z}

                * 'order1' : Uses the weak lensing approximation of the magnification bias with up
                  to first-order terms in :math:`\kappa_{\infty}` or :math:`\gamma_{\infty}`
                  (`z_src_info` must be 'beta'):

                  .. math::
                      \mu^{\alpha-1} \approx
                      1 + \left(\alpha-1\right)\left(2 \left<\beta_s\right>\kappa_{\infty}\right)

                * 'order2' : Uses the weak lensing approximation of the magnification bias with up
                  to second-order terms in :math:`\kappa_{\infty}` or :math:`\gamma_{\infty}`
                  (`z_src_info` must be 'beta'):

                  .. math::
                      \mu^{\alpha-1} \approx
                      1 &+ \left(\alpha-1\right)\left(2 \left<\beta_s\right>\kappa_{\infty}\right)
                      \\\\
                      &+ \left(\alpha-1\right)\left(\left<\beta_s^2\right>\gamma_{\infty}^2\right)
                      \\\\
                      &+ \left(2\alpha-1\right)\left(\alpha-1\right)
                      \left(\left<\beta_s^2\right>\kappa_{\infty}^2\right)

        integ_kwargs: None, dict
            Extra arguments for the redshift integration (when
            `approx=None, z_src_info='distribution'`). Possible keys are:

                * 'zmin' (None, float) : Minimum redshift to be set as the source of the galaxy
                  when performing the sum. (default=None)
                * 'zmax' (float) : Maximum redshift to be set as the source of the galaxy
                  when performing the sum. (default=10.0)
                * 'delta_z_cut' (float) : Redshift cut so that `zmin` = `z_cl` + `delta_z_cut`.
                  `delta_z_cut` is ignored if `z_min` is already provided. (default=0.1)

        verbose : bool, optional
            If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and
            CCL backends.

        Returns
        -------
        mu_bias : numpy.ndarray, float
            magnification bias.

        """
        if self.validate_input:
            validate_argument(locals(), "r_proj", "float_array", argmin=0)
            validate_argument(locals(), "z_cl", float, argmin=0)
            validate_argument(locals(), "z_src_info", str)
            validate_argument(locals(), "alpha", "float_array")
            validate_argument(locals(), "approx", str, none_ok=True)
            self._validate_approx_z_src_info(locals())
            self._validate_z_src(locals())

        if self.halo_profile_model == "einasto" and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        # functions _validate_z_src, _validate_approx_z_src_info already safekeeps from this error:
        # pylint: disable=possibly-used-before-assignment

        if approx is None:
            # z_src (float or array) is redshift
            if z_src_info == "distribution":
                mu_bias = self._pdz_weighted_avg(
                    lambda gammat, kappa: 1 / ((1 - kappa) ** 2 - gammat**2) ** (alpha - 1),
                    z_src,
                    r_proj,
                    z_cl,
                    integ_kwargs=integ_kwargs,
                )
            elif z_src_info == "discrete":
                warning_msg = (
                    "\nSome source redshifts are lower than the cluster redshift."
                    + "\nMagnification bias = 1 for those galaxies."
                )
                mu_bias = compute_for_good_redshifts(
                    self._eval_magnification_bias_core,
                    z_cl,
                    z_src,
                    1.0,
                    warning_msg,
                    "z_cl",
                    "z_src",
                    r_proj,
                    alpha=alpha,
                )

        elif approx in ("order1", "order2"):
            beta_s_mean = z_src[0]

            kappa_inf = self._eval_convergence_core(r_proj, z_cl, z_src=self.z_inf)
            gammat_inf = self._eval_tangential_shear_core(r_proj, z_cl, z_src=self.z_inf)

            mu_bias = 1 + (alpha - 1) * (2 * beta_s_mean * kappa_inf)

            if approx == "order2":
                beta_s_square_mean = z_src[1]
                # Taylor expansion with up to second-order terms
                mu_bias += (alpha - 1) * (beta_s_square_mean * gammat_inf**2) + (
                    2 * alpha - 1
                ) * (alpha - 1) * beta_s_square_mean * kappa_inf**2

        return mu_bias

    def eval_rdelta(self, z_cl):
        r"""Retrieves the radius for mdelta

        .. math::
            r_\Delta=\left(\frac{3 M_\Delta}{4 \pi \Delta \rho_{bkg}(z)}\right)^{1/3}

        Parameters
        ----------
        z_cl: float
            Redshift of the cluster

        Returns
        -------
        float
            Radius in :math:`M\!pc`.
        """
        if self.validate_input:
            validate_argument(locals(), "z_cl", float, argmin=0)
        return self._eval_rdelta(z_cl)

    def eval_mass_in_radius(self, r3d, z_cl, verbose=False):
        r"""Computes the mass inside a given radius of the profile.
        The mass is calculated as

        .. math::
            M(<\text{r3d}) = M_{\Delta}\;
            \frac{f\left(\frac{\text{r3d}}{r_{\Delta}/c_{\Delta}}\right)}{f(c_{\Delta})},

        where :math:`f(x)` for the different models are

        NFW:

        .. math::
            \quad \ln(1+x)-\frac{x}{1+x}

        Einasto: (:math:`\gamma` is the lower incomplete gamma function)

        .. math::
            \gamma(\frac{3}{\alpha}, \frac{2}{\alpha}x^{\alpha})

        Hernquist:

        .. math::
            \left(\frac{x}{1+x}\right)^2

        Parameters
        ----------
        r3d : array_like, float
            Radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster

        Returns
        -------
        numpy.ndarray, float
            Mass in units of :math:`M_\odot`
        """
        if self.validate_input:
            validate_argument(locals(), "r3d", "float_array", argmin=0)
            validate_argument(locals(), "z_cl", float, argmin=0)

        if self.halo_profile_model == "einasto" and verbose:
            print(f"Einasto alpha (in) = {self._get_einasto_alpha(z_cl=z_cl)}")

        return self._eval_mass_in_radius(r3d, z_cl)

    def convert_mass_concentration(
        self,
        z_cl,
        massdef=None,
        delta_mdef=None,
        halo_profile_model=None,
        alpha=None,
        verbose=False,
    ):
        r"""Converts current mass and concentration to the values for a different model.

        Parameters
        ----------
        z_cl: float
            Redshift of the cluster
        massdef : str, None
            Profile mass definition to convert to ("mean", "critical", "virial").
            If None, same value of current model is used.
        delta_mdef : int, None
            Mass overdensity definition to convert to.
            If None, same value of current model is used.
        halo_profile_model : str, None
            Profile model parameterization to convert to ("nfw", "einasto", "hernquist").
            If None, same value of current model is used.
        alpha : float, None
            Einasto slope to convert to when `halo_profile_model='einasto'`.
            If None, same value of current model is used.

        Returns
        -------
        float
            Mass of different model in units of :math:`M_\odot`.
        float
            Concentration of different model.
        """
        if self.validate_input:
            validate_argument(locals(), "z_cl", float, argmin=0)
            validate_argument(locals(), "massdef", str, none_ok=True)
            validate_argument(locals(), "delta_mdef", int, argmin=0, none_ok=True)
            validate_argument(locals(), "halo_profile_model", str, none_ok=True)
            validate_argument(locals(), "alpha", "float_array", none_ok=True)

        if self.halo_profile_model == "einasto" and verbose:
            print(f"Einasto alpha (in) = {self._get_einasto_alpha(z_cl=z_cl)}")

        if (
            halo_profile_model == "einasto"
            or (self.halo_profile_model == "einasto" and halo_profile_model is None)
        ) and verbose:
            print(
                "Einasto alpha (out) = "
                f"{self._get_einasto_alpha(z_cl=z_cl) if alpha is None else alpha}"
            )

        return self._convert_mass_concentration(
            z_cl, massdef, delta_mdef, halo_profile_model, alpha
        )

    def _validate_z_src(self, loc_dict):
        r"""Validation for z_src according to z_src_info. The conditions are:

            * z_src_info='discrete' : z_src must be array or float.
            * z_src_info='distribution' : z_src must be a one dimentional function.
            * z_src_info='beta' : z_src must be a tuple containing
              ( :math:`\langle \beta_s \rangle, \langle \beta_s^2 \rangle`).

        Parameters
        ----------
        locals_dict: dict
            Should be the call locals()
        """
        if loc_dict["z_src_info"] == "discrete":
            validate_argument(loc_dict, "z_src", "float_array", argmin=0)
        elif loc_dict["z_src_info"] == "distribution":
            validate_argument(loc_dict, "z_src", "function", none_ok=False)
            integ_kwargs = {} if loc_dict["integ_kwargs"] is None else loc_dict["integ_kwargs"]
            _def_keys = ["zmin", "zmax", "delta_z_cut"]
            if any(key not in _def_keys for key in integ_kwargs):
                raise KeyError(
                    f"integ_kwargs must contain only {_def_keys} keys, "
                    f" {integ_kwargs.keys()} provided."
                )
        elif loc_dict["z_src_info"] == "beta":
            validate_argument(loc_dict, "z_src", "array")
            beta_info = {
                "beta_s_mean": loc_dict["z_src"][0],
                "beta_s_square_mean": loc_dict["z_src"][1],
            }
            validate_argument(beta_info, "beta_s_mean", "float_array")
            validate_argument(beta_info, "beta_s_square_mean", "float_array")
        else:
            raise ValueError(f"Unsupported z_src_info (='{loc_dict['z_src_info']}')")

    def _validate_approx_z_src_info(self, loc_dict):
        r"""Validation for compatility between approx and z_src_info. The conditions are:

            * approx=None: z_src_info must be 'discrete' or 'distribution'
            * approx='order1' or 'order2': z_src_info must be 'beta'
            * approx=other: raises error

        Parameters
        ----------
        locals_dict: dict
            Should be the call locals()
        """
        # check compatility between approx and z_src_info
        z_src_info, approx = loc_dict["z_src_info"], loc_dict["approx"]
        if approx is None:
            if z_src_info not in ("discrete", "distribution"):
                raise ValueError(
                    "approx=None requires z_src_info='discrete' or 'distribution',"
                    f" z_src_info='{z_src_info}' was provided."
                )
        elif approx in ("order1", "order2"):
            if z_src_info != "beta":
                raise ValueError(
                    f"approx='{approx}' requires z_src_info='beta', "
                    f"z_src_info='{z_src_info}' was provided."
                )
        else:
            raise ValueError(f"Unsupported approx (='{approx}')")
