"""@file parent_class.py
CLMModeling abstract class
"""
import numpy as np
<<<<<<< HEAD
from .generic import *

=======

from .generic import *

>>>>>>> 5e7a05adc161c678c6aa66f22e9f13b8dbed1bbc

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
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.backend = None

        self.massdef = ''
        self.delta_mdef = 0
        self.halo_profile_model = ''

        self.cosmo = None

        self.hdpm = None
        self.mdef_dict = {}
        self.hdpm_dict = {}

    def validate_definitions(self, massdef, halo_profile_model):
        r""" Makes sure values of `massdef` and `halo_profile_model` are in the supported options
        and fixes casing (code works with lowercase).
        Parameters
        ----------
        massdef: str, optional
            Profile mass definition.
        halo_profile_model: str, optional
            Profile model parameterization.
        Returns
        -------
        massdef: str
            Lowercase profile mass definition.
        halo_profile_model: str
            Lowercase profile model parameterization.
        """
        # make case independent
        massdef, halo_profile_model = massdef.lower(), halo_profile_model.lower()
        if not massdef in self.mdef_dict:
            raise ValueError(
                f"Halo density profile mass definition {massdef} not currently supported")
        if not halo_profile_model in self.hdpm_dict:
            raise ValueError(
                f"Halo density profile model {halo_profile_model} not currently supported")
        return massdef, halo_profile_model

    def set_cosmo(self, cosmo):
        r""" Sets the cosmology to the internal cosmology object
        Parameters
        ----------
        cosmo: clmm.Comology
            CLMM Cosmology object
        """
        raise NotImplementedError

    def _set_cosmo(self, cosmo, cosmo_out_class):
        r""" Sets the cosmology to the internal cosmology object
        Parameters
        ----------
        cosmo: clmm.Comology object, None
            CLMM Cosmology object. If is None, creates a new instance of cosmo_out_class().
        cosmo_out_class: clmm.modbackend Cosmology class
            Cosmology Output for the output object.
        """
        if cosmo is not None:
            if not isinstance(cosmo, cosmo_out_class):
                raise ValueError(
                    f'Cosmo input ({type(cosmo)}) must be a {cosmo_out_class} object.')
            self.cosmo = cosmo
        else:
            self.cosmo = cosmo_out_class()

    def set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        r""" Sets the definitios for the halo profile
        Parameters
        ----------
        halo_profile_model: str
            Halo mass profile, current options are 'nfw' (letter case independent)
        massdef: str
            Mass definition, current options are 'mean' (letter case independent)
        delta_mdef: int
            Overdensity number
        """
        raise NotImplementedError

    def set_mass(self, mdelta):
        r""" Sets the value of the :math:`M_\Delta`
        Parameters
        ----------
        mdelta : float
            Galaxy cluster mass :math:`M_\Delta` in units of :math:`M_\odot`
        """
        self._validate_input(
            mdelta, 0, "min(mdelta) = %s! This value is not accepted.")
        self._set_mass(mdelta)

    def _set_mass(self, mdelta):
        r""" Actually sets the value of the :math:`M_\Delta` (without value check)"""
        raise NotImplementedError

    def set_concentration(self, cdelta):
        r""" Sets the concentration
        Parameters
        ----------
        cdelta: float
            Concentration
        """
        self._validate_input(
            cdelta, 0, "min(cdelta) = %s! This value is not accepted.")
        self._set_concentration(cdelta)

    def _set_concentration(self, cdelta):
        r""" Actuall sets the value of the concentration (without value check)"""
        raise NotImplementedError

    def _validate_input(self, in_val, vmin, err_msg='value %s <= vmin'):
        r'''Raises error if input value<=vmin
        Parameters
        ----------
        radius: array, float
            Input radius
        '''
        in_min = np.min(in_val)
        if in_min <= vmin:
            raise ValueError(err_msg % str(in_min))

    def _check_input_radius(self, radius):
        r'''Raises error if input radius is not positive
        Parameters
        ----------
        radius: array, float
            Input radius
        '''
        self._validate_input(
            radius, 0, "min(R) = %s Mpc! This value is not accepted.")

    def eval_3d_density(self, r3d, z_cl):
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
        if z_len <= 0:
            raise ValueError('Redshift for lens <= 0.')
        if np.min(z_src) <= 0:
            raise ValueError(
                'Some source redshifts are <=0. Please check your inputs.')
        return self.cosmo.eval_sigma_crit(z_len, z_src)

    def eval_surface_density(self, r_proj, z_cl):
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
        raise NotImplementedError

    def eval_mean_surface_density(self, r_proj, z_cl):
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
        raise NotImplementedError

    def eval_excess_surface_density(self, r_proj, z_cl):
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
        raise NotImplementedError

    def eval_tangential_shear(self, r_proj, z_cl, z_src):
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
        delta_sigma = self.eval_excess_surface_density(r_proj, z_cl)
        sigma_c = self.eval_critical_surface_density(z_cl, z_src)
        return delta_sigma/sigma_c

    def eval_convergence(self, r_proj, z_cl, z_src):
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
        kappa : array_like, float
            Mass convergence, kappa.
        Notes
        -----
        Need to figure out if we want to raise exceptions rather than errors here?
        """
        sigma = self.eval_surface_density(r_proj, z_cl)
        sigma_c = self.eval_critical_surface_density(z_cl, z_src)
        return sigma/sigma_c

    def eval_reduced_tangential_shear(self, r_proj, z_cl, z_src):
        r"""Computes the reduced tangential shear :math:`g_t = \frac{\gamma_t}{1-\kappa}`.
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
        gt : array_like, float
            Reduced tangential shear
        Notes
        -----
        Need to figure out if we want to raise exceptions rather than errors here?
        """
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_tangential_shear(r_proj, z_cl, z_src)
        return compute_reduced_shear_from_convergence(gamma_t, kappa)

    def eval_magnification(self, r_proj, z_cl, z_src):
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
        magnification = self.eval_magnification(r_proj, z_cl, z_src)
        return compute_magnification_bias_from_magnification(magnification, alpha)
<<<<<<< HEAD
=======

>>>>>>> 5e7a05adc161c678c6aa66f22e9f13b8dbed1bbc
