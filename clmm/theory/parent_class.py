"""@file parent_class.py
CLMModeling abstract class
"""
import numpy as np

# functions for the 2h term
from scipy.integrate import simps
from scipy.special import jv
from scipy.interpolate import interp1d

from .generic import compute_reduced_shear_from_convergence
import warnings
from .generic import (compute_reduced_shear_from_convergence,
                      compute_magnification_bias_from_magnification,
                      compute_rdelta, compute_profile_mass_in_radius,
                      convert_profile_mass_concentration)
from ..utils import validate_argument, compute_beta_s_mean, compute_beta_s_square_mean


class CLMModeling:
    r"""Object with functions for halo mass modeling

    Attributes
    ----------
    backend: str
        Name of the backend being used
    massdef : str
        Profile mass definition (`mean`, `critical`, `virial` - letter case independent)
    delta_mdef : int
        Mass overdensity definition.
    halo_profile_model : str
        Profile model parameterization (`nfw`, `einasto`, `hernquist` - letter case independent)
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
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, validate_input=True):
        self.backend = None

        self.massdef = ''
        self.delta_mdef = 0
        self.halo_profile_model = ''
        self.mdelta = 0.
        self.cdelta = 0.

        self.cosmo = None

        self.hdpm = None
        self.mdef_dict = {}
        self.hdpm_dict = {}

        self.validate_input = validate_input
        self.cosmo_class = None


    def set_cosmo(self, cosmo):
        r""" Sets the cosmology to the internal cosmology object

        Parameters
        ----------
        cosmo: clmm.Comology object, None
            CLMM Cosmology object. If is None, creates a new instance of self.cosmo_class().
        """
        if self.validate_input:
            if self.cosmo_class is None:
                raise NotImplementedError
            validate_argument(locals(), 'cosmo', self.cosmo_class, none_ok=True)
        self._set_cosmo(cosmo)
        self.cosmo.validate_input = self.validate_input

    def _set_cosmo(self, cosmo):
        r""" Sets the cosmology to the internal cosmology object"""
        self.cosmo = cosmo if cosmo is not None else self.cosmo_class()

    def set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        r""" Sets the definitions for the halo profile

        Parameters
        ----------
        halo_profile_model: str
            Halo mass profile, current options are 'nfw' (letter case independent)
        massdef: str
            Mass definition, current options are 'mean' (letter case independent)
        delta_mdef: int
            Overdensity number
        """
        # make case independent
        massdef, halo_profile_model = massdef.lower(), halo_profile_model.lower()
        if self.validate_input:
            validate_argument(locals(), 'massdef', str)
            validate_argument(locals(), 'halo_profile_model', str)
            validate_argument(locals(), 'delta_mdef', int, argmin=0)
            if not massdef in self.mdef_dict:
                raise ValueError(
                    f"Halo density profile mass definition {massdef} not currently supported")
            if not halo_profile_model in self.hdpm_dict:
                raise ValueError(
                    f"Halo density profile model {halo_profile_model} not currently supported")
        self._set_halo_density_profile(halo_profile_model=halo_profile_model,
                                       massdef=massdef, delta_mdef=delta_mdef)
        self.halo_profile_model = halo_profile_model
        self.massdef = massdef
        self.delta_mdef = delta_mdef

    def _set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        raise NotImplementedError

    def set_mass(self, mdelta):
        r""" Sets the value of the :math:`M_\Delta`

        Parameters
        ----------
        mdelta : float
            Galaxy cluster mass :math:`M_\Delta` in units of :math:`M_\odot`
        """
        if self.validate_input:
            validate_argument(locals(), 'mdelta', float, argmin=0)
        self._set_mass(mdelta)
        self.mdelta = mdelta

    def _set_mass(self, mdelta):
        r""" Actually sets the value of the :math:`M_\Delta` (without value check)"""
        raise NotImplementedError

    def set_einasto_alpha(self, alpha):
        r""" Sets the value of the :math:`\alpha` parameter for the Einasto profile

        Parameters
        ----------
        alpha : float
        """
        if self.halo_profile_model!='einasto' or self.backend!='nc':
            raise NotImplementedError("The Einasto slope cannot be set for your combination of profile choice or modeling backend.")
        else:
            if self.validate_input:
                validate_argument(locals(), 'alpha', float)
            self._set_einasto_alpha(alpha)

    def _set_einasto_alpha(self, alpha):
        r""" Actually sets the value of the :math:`\alpha` parameter for the Einasto profile"""
        raise NotImplementedError

    def get_einasto_alpha(self, z_cl=None):
        r""" Returns the value of the :math:`\alpha` parameter for the Einasto profile, if defined

        Parameters
        ----------
        z_cl : float
            Cluster redshift (required for Einasto with the CCL backend, will be ignored for NC)
        """
        if self.halo_profile_model!='einasto':
            raise ValueError(f"Wrong profile model. Current profile = {self.halo_profile_model}")
        else:
            return self._get_einasto_alpha(z_cl)

    def _get_einasto_alpha(self, z_cl=None):
        r""" Returns the value of the :math:`\alpha` parameter for the Einasto profile, if defined"""
        raise NotImplementedError

    def set_concentration(self, cdelta):
        r""" Sets the concentration

        Parameters
        ----------
        cdelta: float
            Concentration
        """
        if self.validate_input:
            validate_argument(locals(), 'cdelta', float, argmin=0)
        self._set_concentration(cdelta)
        self.cdelta = cdelta

    def _set_concentration(self, cdelta):
        r""" Actuall sets the value of the concentration (without value check)"""
        raise NotImplementedError

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
        array_like, float
            3-dimensional mass density in units of :math:`M_\odot\ Mpc^{-3}`
        """
        if self.validate_input:
            validate_argument(locals(), 'r3d', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', 'float_array', argmin=0)

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        return self._eval_3d_density(r3d=r3d, z_cl=z_cl)

    def _eval_3d_density(self, r3d, z_cl):
        raise NotImplementedError

    def eval_critical_surface_density(self, z_len, z_src):
        r"""Computes the critical surface density

        Parameters
        ----------
        z_len : float
            Lens redshift
        z_src : array_like, float
            Background source galaxy redshift(s)

        Returns
        -------
        float
            Cosmology-dependent critical surface density in units of :math:`M_\odot\ Mpc^{-2}`
        """
        if self.validate_input:
            validate_argument(locals(), 'z_len', float, argmin=0)
            validate_argument(locals(), 'z_src', 'float_array', argmin=0)
        return self._eval_critical_surface_density(z_len=z_len, z_src=z_src)

    def _eval_critical_surface_density(self, z_len, z_src):
        return self.cosmo.eval_sigma_crit(z_len, z_src)

    def eval_surface_density(self, r_proj, z_cl, verbose=False):
        r""" Computes the surface mass density

        Parameters
        ----------
        r_proj : array_like
            Projected radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster

        Returns
        -------
        array_like, float
            2D projected surface density in units of :math:`M_\odot\ Mpc^{-2}`
        """
        if self.validate_input:
            validate_argument(locals(), 'r_proj', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        return self._eval_surface_density(r_proj=r_proj, z_cl=z_cl)


    def _eval_surface_density(self, r_proj, z_cl):
        raise NotImplementedError

    def eval_mean_surface_density(self, r_proj, z_cl, verbose=False):
        r""" Computes the mean value of surface density inside radius r_proj

        Parameters
        ----------
        r_proj : array_like
            Projected radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster

        Returns
        -------
        array_like, float
            Excess surface density in units of :math:`M_\odot\ Mpc^{-2}`.
        """
        if self.validate_input:
            validate_argument(locals(), 'r_proj', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        return self._eval_mean_surface_density(r_proj=r_proj, z_cl=z_cl)

    def _eval_mean_surface_density(self, r_proj, z_cl):
        raise NotImplementedError

    def eval_excess_surface_density(self, r_proj, z_cl, verbose=False):
        r""" Computes the excess surface density

        Parameters
        ----------
        r_proj : array_like
            Projected radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster

        Returns
        -------
        array_like, float
            Excess surface density in units of :math:`M_\odot\ Mpc^{-2}`.
        """
        if self.validate_input:
            validate_argument(locals(), 'r_proj', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        return self._eval_excess_surface_density(r_proj=r_proj, z_cl=z_cl)

    def _eval_excess_surface_density(self, r_proj, z_cl):
        raise NotImplementedError

    def eval_excess_surface_density_2h(self, r_proj, z_cl, halobias=1., lsteps=500):
        r""" Computes the 2-halo term excess surface density (CCL backend only)

        Parameters
        ----------
        r_proj : array_like
            Projected radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster
        halobias : float, optional
            Value of the halo bias
        lsteps: int, optional
            Number of steps for numerical integration

        Returns
        -------
        array_like, float
            Excess surface density from the 2-halo term in units of :math:`M_\odot\ Mpc^{-2}`.
        """

        if self.validate_input:
            validate_argument(locals(), 'r_proj', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)
            validate_argument(locals(), 'lsteps', int, argmin=1)
            validate_argument(locals(), 'halobias', float, argmin=0)


        if self.backend not in ('ccl', 'nc'):
            raise NotImplementedError(
                f"2-halo term not currently supported with the {self.backend} backend. "
                "Use the CCL or NumCosmo backend instead")
        else:
            return self._eval_excess_surface_density_2h(r_proj, z_cl, halobias=halobias, lsteps=lsteps)

    def _eval_excess_surface_density_2h(self, r_proj, z_cl, halobias=1.,lsteps=500):
        """"eval excess surface density from the 2-halo term"""
        da = self.cosmo.eval_da(z_cl)
        rho_m = self.cosmo._get_rho_m(z_cl)

        kk = np.logspace(-5.,5.,1000)
        pk = self.cosmo._eval_linear_matter_powerspectrum(kk, z_cl)
        interp_pk = interp1d(kk, pk, kind='cubic')
        theta = r_proj / da

        # calculate integral, units [Mpc]**-3
        def __integrand__( l , theta ):
            k = l / ((1 + z_cl) * da)
            return l * jv( 2 , l * theta ) * interp_pk( k )

        ll = np.logspace( 0 , 6 , lsteps )
        val = np.array( [ simps( __integrand__( ll , t ) , ll ) for t in theta ] )
        return halobias * val * rho_m / ( 2 * np.pi  * ( 1 + z_cl )**3 * da**2 )

    def eval_surface_density_2h(self, r_proj, z_cl, halobias=1., lsteps=500):
        r""" Computes the 2-halo term surface density (CCL backend only)

        Parameters
        ----------
        r_proj : array_like
            Projected radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster
        halobias : float, optional
           Value of the halo bias
        lsteps: int, optional
            Number of steps for numerical integration

        Returns
        -------
        array_like, float
            Excess surface density from the 2-halo term in units of :math:`M_\odot\ Mpc^{-2}`.
        """

        if self.validate_input:
            validate_argument(locals(), 'r_proj', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)
            validate_argument(locals(), 'lsteps', int, argmin=1)
            validate_argument(locals(), 'halobias', float, argmin=0)

        if self.backend not in ('ccl', 'nc'):
            raise NotImplementedError(
                f"2-halo term not currently supported with the {self.backend} backend. "
                "Use the CCL or NumCosmo backend instead")
        else:
            return self._eval_surface_density_2h(r_proj, z_cl, halobias=halobias, lsteps=lsteps)

    def _eval_surface_density_2h(self, r_proj, z_cl, halobias=1., lsteps=500):
        """"eval surface density from the 2-halo term"""
        da = self.cosmo.eval_da(z_cl)
        rho_m = self.cosmo._get_rho_m(z_cl)

        kk = np.logspace(-5.,5.,1000)
        pk = self.cosmo._eval_linear_matter_powerspectrum(kk, z_cl)
        interp_pk = interp1d(kk, pk, kind='cubic')
        theta = r_proj / da

        # calculate integral, units [Mpc]**-3
        def __integrand__( l , theta ):
            k = l / ((1 + z_cl) * da)
            return l * jv( 0 , l * theta ) * interp_pk( k )

        ll = np.logspace( 0 , 6 , lsteps )
        val = np.array( [ simps( __integrand__( ll , t ) , ll ) for t in theta ] )
        return halobias * val * rho_m / ( 2 * np.pi  * ( 1 + z_cl )**3 * da**2 )

    def eval_tangential_shear(self, r_proj, z_cl, z_src, verbose=False):
        r"""Computes the tangential shear

        Parameters
        ----------
        r_proj : array_like
            The projected radial positions in :math:`M\!pc`.
        z_cl : float
            Galaxy cluster redshift
        z_src : array_like, float
            Background source galaxy redshift(s)

        Returns
        -------
        array_like, float
            tangential shear
        """
        if self.validate_input:
            validate_argument(locals(), 'r_proj', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)
            validate_argument(locals(), 'z_src', 'float_array', argmin=0)

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        return self._eval_tangential_shear(r_proj=r_proj, z_cl=z_cl, z_src=z_src)

    def _eval_tangential_shear(self, r_proj, z_cl, z_src):
        delta_sigma = self.eval_excess_surface_density(r_proj, z_cl)
        sigma_c = self.eval_critical_surface_density(z_cl, z_src)
        return delta_sigma/sigma_c

    def eval_convergence(self, r_proj, z_cl, z_src, verbose=False):
        r"""Computes the mass convergence

        .. math::
            \kappa = \frac{\Sigma}{\Sigma_{crit}}

        or

        .. math::
            \kappa = \kappa_\infty \times \beta_s

        Parameters
        ----------
        r_proj : array_like
            The projected radial positions in :math:`M\!pc`.
        z_cl : float
            Galaxy cluster redshift
        z_src : array_like, float
            Background source galaxy redshift(s)

        Returns
        -------
        array_like, float
            Mass convergence, kappa.
        """
        if self.validate_input:
            validate_argument(locals(), 'r_proj', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)
            validate_argument(locals(), 'z_src', 'float_array', argmin=0)

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        return self._eval_convergence(r_proj=r_proj, z_cl=z_cl, z_src=z_src)

    def _eval_convergence(self, r_proj, z_cl, z_src, verbose=False):
        sigma = self.eval_surface_density(r_proj, z_cl, verbose=verbose)
        sigma_c = self.eval_critical_surface_density(z_cl, z_src)
        return sigma/sigma_c

    def eval_reduced_tangential_shear(self, r_proj, z_cl, z_src, z_src_model='single_plane',
                                      beta_s_mean=None, beta_s_square_mean=None, z_distrib_func=None, verbose=False):
        r"""Computes the reduced tangential shear :math:`g_t = \frac{\gamma_t}{1-\kappa}`.

        Parameters
        ----------
        r_proj : array_like
            The projected radial positions in :math:`M\!pc`.
        z_cl : float
            Galaxy cluster redshift
        z_src : array_like, float
            Background source galaxy redshift(s)
        z_src_model : str, optional
            Source redshift model, with the following supported options:

                * `single_plane` (default): all sources at one redshift (if `z_source` is a float)    or known individual source galaxy redshifts (if `z_source` is an array and `r_proj` is a float).
                * `applegate14`: use the equation (6) in Weighing the Giants - III (Applegate et al. 2014; https://arxiv.org/abs/1208.0605) to evaluate tangential reduced shear.
                * `schrabback18`: use the equation (12) in Cluster Mass Calibration at High Redshift (Schrabback et al. 2017; https://arxiv.org/abs/1611.03866) to evaluate tangential reduced shear.

        z_distrib_func: one-parameter function
            Redshift distribution function. This function is used to compute the beta values if they are not provided. The default is the Chang et al (2013) distribution function.

        beta_s_mean: array_like, float, optional
            Lensing efficiency averaged over the galaxy redshift distribution. If not provided, it will be computed using the default redshift distribution or the one given by the user.

                .. math::
                    \langle \beta_s \rangle = \left\langle \frac{D_{LS}}{D_S}\frac{D_\infty}{D_{L,\infty}}\right\rangle

        beta_s_square_mean: array_like, float, optional
            Square of the lensing efficiency averaged over the galaxy redshift distribution. If not provided, it will be computed using the default redshift distribution or the one given by the user.

                .. math::
                    \langle \beta_s^2 \rangle = \left\langle \left(\frac{D_{LS}}{D_S}\frac{D_\infty}{D_{L,\infty}}\right)^2 \right\rangle

        Returns
        -------
        gt : array_like, float
            Reduced tangential shear

        Notes
        -----
        Need to figure out if we want to raise exceptions rather than errors here?
        """
        if self.validate_input:
            validate_argument(locals(), 'r_proj', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)
            validate_argument(locals(), 'z_src', 'float_array', argmin=0)

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        if z_src_model == 'single_plane':
            gt = self._eval_reduced_tangential_shear_sp(r_proj, z_cl, z_src)

        elif z_src_model == 'applegate14':
            z_source = 1000. #np.inf # INF or a very large number
            z_inf = z_source
            if beta_s_mean is None or beta_s_square_mean is None:
                beta_s_mean = compute_beta_s_mean (z_cl, z_inf, self.cosmo, z_distrib_func=z_distrib_func)
                beta_s_square_mean = compute_beta_s_square_mean (z_cl, z_inf, self.cosmo, z_distrib_func=z_distrib_func)
            gammat = self._eval_tangential_shear(r_proj, z_cl, z_source)
            kappa = self._eval_convergence(r_proj, z_cl, z_source)
            gt = beta_s_mean * gammat / (1. - beta_s_square_mean / beta_s_mean * kappa)

        elif z_src_model == 'schrabback18':
            z_source = 1000. #np.inf # INF or a very large number
            z_inf = z_source
            if beta_s_mean is None or beta_s_square_mean is None:
                beta_s_mean = compute_beta_s_mean (z_cl, z_inf, self.cosmo, z_distrib_func=z_distrib_func)
                beta_s_square_mean = compute_beta_s_square_mean (z_cl, z_inf, self.cosmo, z_distrib_func=z_distrib_func)
            gammat = self._eval_tangential_shear(r_proj, z_cl, z_source)
            kappa = self._eval_convergence(r_proj, z_cl, z_source)
            gt = (1. + (beta_s_square_mean / (beta_s_mean * beta_s_mean) - 1.) * beta_s_mean * kappa) * (beta_s_mean * gammat / (1. - beta_s_mean * kappa))

        else:
            raise ValueError("Unsupported z_src_model")
        return gt

    def _eval_reduced_tangential_shear_sp(self, r_proj, z_cl, z_src):
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_tangential_shear(r_proj, z_cl, z_src)
        return compute_reduced_shear_from_convergence(gamma_t, kappa)

    def eval_magnification(self, r_proj, z_cl, z_src, verbose=False):
        r"""Computes the magnification

        .. math::
            \mu = \frac{1}{(1-\kappa)^2-|\gamma_t|^2}

        Parameters
        ----------
        r_proj : array_like
            The projected radial positions in :math:`M\!pc`.
        z_cl : float
            Galaxy cluster redshift
        z_src : array_like, float
            Background source galaxy redshift(s)

        Returns
        -------
        mu : array_like, float
            magnification, mu.

        Notes
        -----
        The magnification is computed taking into account just the tangential
        shear. This is valid for spherically averaged profiles, e.g., NFW and
        Einasto (by construction the cross shear is zero).
        """
        if self.validate_input:
            validate_argument(locals(), 'r_proj', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)
            validate_argument(locals(), 'z_src', 'float_array', argmin=0)

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        return self._eval_magnification(r_proj=r_proj, z_cl=z_cl, z_src=z_src)

    def _eval_magnification(self, r_proj, z_cl, z_src):
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_tangential_shear(r_proj, z_cl, z_src)
        return 1./((1-kappa)**2-abs(gamma_t)**2)

    def eval_magnification_bias(self, r_proj, z_cl, z_src, alpha):
        r"""Computes the magnification bias

        .. math::
            \mu^{\alpha - 1}

        Parameters
        ----------
        r_proj : array_like
            The projected radial positions in :math:`M\!pc`.
        z_cl : float
            Galaxy cluster redshift
        z_src : array_like, float
            Background source galaxy redshift(s)
        alpha : float
            Slope of the cummulative number count of background sources at a given magnitude

        Returns
        -------
        mu_bias : array_like, float
            magnification bias.

        Notes
        -----
        The magnification is computed taking into account just the tangential
        shear. This is valid for spherically averaged profiles, e.g., NFW and
        Einasto (by construction the cross shear is zero).
        """
        if self.validate_input:
            validate_argument(locals(), 'r_proj', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)
            validate_argument(locals(), 'z_src', 'float_array', argmin=0)
            validate_argument(locals(), 'alpha', 'float_array')
        return self._eval_magnification_bias(r_proj=r_proj, z_cl=z_cl, z_src=z_src, alpha=alpha)

    def _eval_magnification_bias(self, r_proj, z_cl, z_src, alpha):
        magnification = self.eval_magnification(r_proj, z_cl, z_src)
        return compute_magnification_bias_from_magnification(magnification, alpha)

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
            validate_argument(locals(), 'z_cl', float, argmin=0)
        return self._eval_rdelta(z_cl)

    def _eval_rdelta(self, z_cl):
        return compute_rdelta(self.mdelta, z_cl, self.cosmo, self.massdef, self.delta_mdef)

    def eval_mass_in_radius(self, r3d, z_cl, verbose=False):
        r"""Computes the mass inside a given radius of the profile.

        Parameters
        ----------
        r3d : array_like, float
            Radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster

        Returns
        -------
        array_like, float
            Mass in units of :math:`M_\odot`
        """
        if self.validate_input:
            validate_argument(locals(), 'r3d', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha (in) = {self._get_einasto_alpha(z_cl=z_cl)}")

        return self._eval_mass_in_radius(r3d, z_cl)

    def _eval_mass_in_radius(self, r3d, z_cl):
        alpha = self._get_einasto_alpha(z_cl) if self.halo_profile_model=='einasto' else None
        return compute_profile_mass_in_radius(
            r3d, z_cl, self.cosmo, self.mdelta, self.cdelta,
            self.massdef, self.delta_mdef, self.halo_profile_model, alpha)

    def convert_mass_concentration(self, z_cl, massdef=None, delta_mdef=None,
                                   halo_profile_model=None, alpha=None, verbose=False):
        r"""Converts current mass and concentration to the values for a different model.

        Parameters
        ----------
        z_cl: float
            Redshift of the cluster
        massdef : str, None
            Profile mass definition to convert to (`mean`, `critical`, `virial`).
            If None, same value of current model is used.
        delta_mdef : int, None
            Mass overdensity definition to convert to.
            If None, same value of current model is used.
        halo_profile_model : str, None
            Profile model parameterization to convert to (`nfw`, `einasto`, `hernquist`).
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
            validate_argument(locals(), 'z_cl', float, argmin=0)
            validate_argument(locals(), 'massdef', str, none_ok=True)
            validate_argument(locals(), 'delta_mdef', int, argmin=0, none_ok=True)
            validate_argument(locals(), 'halo_profile_model', str, none_ok=True)
            validate_argument(locals(), 'alpha', 'float_array', none_ok=True)

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha (in) = {self._get_einasto_alpha(z_cl=z_cl)}")

        if (halo_profile_model=='einasto'
                    or (self.halo_profile_model=='einasto' and halo_profile_model==None))\
                and verbose:
            print("Einasto alpha (out) = "
                 f"{self._get_einasto_alpha(z_cl=z_cl) if alpha is None else alpha}")

        return self._convert_mass_concentration(z_cl, massdef, delta_mdef,
                                                halo_profile_model, alpha)

    def _convert_mass_concentration(self, z_cl, massdef=None, delta_mdef=None,
                                    halo_profile_model=None, alpha=None):
        alpha1 = self._get_einasto_alpha(z_cl) if self.halo_profile_model=='einasto' else None
        return convert_profile_mass_concentration(
            self.mdelta, self.cdelta, z_cl, self.cosmo,
            massdef=self.massdef, delta_mdef=self.delta_mdef,
            halo_profile_model=self.halo_profile_model, alpha=alpha1,
            massdef2=massdef, delta_mdef2=delta_mdef,
            halo_profile_model2=halo_profile_model, alpha2=alpha)
