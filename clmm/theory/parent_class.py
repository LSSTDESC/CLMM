"""@file parent_class.py
CLMModeling abstract class
"""
import numpy as np

# functions for the 2h term
from scipy.integrate import simps, quad
from scipy.special import jv
from scipy.interpolate import interp1d

from .generic import (compute_reduced_shear_from_convergence,
                      compute_magnification_bias_from_magnification,
                      compute_rdelta, compute_profile_mass_in_radius,
                      convert_profile_mass_concentration)
from ..utils import validate_argument, _integ_pzfuncs, compute_beta_s_mean, compute_beta_s_square_mean, compute_beta_s_func


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

        self.__massdef = ''
        self.__delta_mdef = 0
        self.__halo_profile_model = ''

        self.cosmo = None

        self.hdpm = None
        self.mdef_dict = {}
        self.hdpm_dict = {}

        self.validate_input = validate_input
        self.cosmo_class = None


    # 1. Object properties


    @property
    def mdelta(self):
        return self._get_mass()

    @property
    def cdelta(self):
        return self._get_concentration()

    @property
    def massdef(self):
        return self.__massdef

    @property
    def delta_mdef(self):
        return self.__delta_mdef

    @property
    def halo_profile_model(self):
        return self.__halo_profile_model


    # 1.a Object properties setter


    @mdelta.setter
    def mdelta(self, mdelta):
        self.set_mass(mdelta)

    @cdelta.setter
    def cdelta(self, cdelta):
        self.set_concentration(cdelta)

    @massdef.setter
    def massdef(self, massdef):
        self.set_halo_density_profile(
            halo_profile_model=self.halo_profile_model, massdef=massdef,
            delta_mdef=self.delta_mdef)

    @delta_mdef.setter
    def delta_mdef(self, delta_mdef):
        self.set_halo_density_profile(
            halo_profile_model=self.halo_profile_model, massdef=self.massdef,
            delta_mdef=delta_mdef)

    @halo_profile_model.setter
    def halo_profile_model(self, halo_profile_model):
        self.set_halo_density_profile(
            halo_profile_model=halo_profile_model, massdef=self.massdef,
            delta_mdef=self.delta_mdef)


    # 2. Functions to be implemented by children classes


    def _get_mass(self):
        r""" Gets the value of the :math:`M_\Delta`"""
        raise NotImplementedError

    def _get_concentration(self):
        r""" Gets the value of the concentration"""
        raise NotImplementedError

    def _set_mass(self, mdelta):
        r""" Actually sets the value of the :math:`M_\Delta` (without value check)"""
        raise NotImplementedError

    def _set_concentration(self, cdelta):
        r""" Actuall sets the value of the concentration (without value check)"""
        raise NotImplementedError

    def _set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        raise NotImplementedError

    def _set_einasto_alpha(self, alpha):
        r""" Actually sets the value of the :math:`\alpha` parameter for the Einasto profile"""
        raise NotImplementedError

    def _get_einasto_alpha(self, z_cl=None):
        r""" Returns the value of the :math:`\alpha` parameter for the Einasto profile, if defined"""
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


    def _eval_critical_surface_density(self, z_len, z_src):
        return self.cosmo.eval_sigma_crit(z_len, z_src)

    def _set_cosmo(self, cosmo):
        r""" Sets the cosmology to the internal cosmology object"""
        self.cosmo = cosmo if cosmo is not None else self.cosmo_class()

    def _eval_excess_surface_density_2h(self, r_proj, z_cl, halobias=1., lsteps=500):
        """"eval excess surface density from the 2-halo term"""
        da = self.cosmo.eval_da(z_cl)
        rho_m = self.cosmo._get_rho_m(z_cl)

        kk = np.logspace(-5., 5., 1000)
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

    def _eval_surface_density_2h(self, r_proj, z_cl, halobias=1., lsteps=500):
        """"eval surface density from the 2-halo term"""
        da = self.cosmo.eval_da(z_cl)
        rho_m = self.cosmo._get_rho_m(z_cl)

        kk = np.logspace(-5., 5., 1000)
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

    def _eval_tangential_shear(self, r_proj, z_cl, z_src):
        delta_sigma = self.eval_excess_surface_density(r_proj, z_cl)
        sigma_c = self.eval_critical_surface_density(z_cl, z_src)
        return delta_sigma/sigma_c

    def _eval_convergence(self, r_proj, z_cl, z_src, verbose=False):
        sigma = self.eval_surface_density(r_proj, z_cl, verbose=verbose)
        sigma_c = self.eval_critical_surface_density(z_cl, z_src)
        return sigma/sigma_c

    def _eval_reduced_tangential_shear_sp(self, r_proj, z_cl, z_src):
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_tangential_shear(r_proj, z_cl, z_src)
        return compute_reduced_shear_from_convergence(gamma_t, kappa)

    def _eval_magnification(self, r_proj, z_cl, z_src):
        kappa = self.eval_convergence(r_proj, z_cl, z_src)
        gamma_t = self.eval_tangential_shear(r_proj, z_cl, z_src)
        return 1./((1-kappa)**2-abs(gamma_t)**2)

    def _eval_magnification_bias(self, r_proj, z_cl, z_src, alpha):
        magnification = self.eval_magnification(r_proj, z_cl, z_src)
        return compute_magnification_bias_from_magnification(magnification, alpha)

    def _eval_rdelta(self, z_cl):
        return compute_rdelta(self.mdelta, z_cl, self.cosmo, self.massdef, self.delta_mdef)

    def _eval_mass_in_radius(self, r3d, z_cl):
        alpha = self._get_einasto_alpha(z_cl) if self.halo_profile_model=='einasto' else None
        return compute_profile_mass_in_radius(
            r3d, z_cl, self.cosmo, self.mdelta, self.cdelta,
            self.massdef, self.delta_mdef, self.halo_profile_model, alpha)

    def _convert_mass_concentration(self, z_cl, massdef=None, delta_mdef=None,
                                    halo_profile_model=None, alpha=None):
        alpha1 = self._get_einasto_alpha(z_cl) if self.halo_profile_model=='einasto' else None
        return convert_profile_mass_concentration(
            self.mdelta, self.cdelta, z_cl, self.cosmo,
            massdef=self.massdef, delta_mdef=self.delta_mdef,
            halo_profile_model=self.halo_profile_model, alpha=alpha1,
            massdef2=massdef, delta_mdef2=delta_mdef,
            halo_profile_model2=halo_profile_model, alpha2=alpha)


    # 4. Wrapper functions for input validation


    def set_mass(self, mdelta):
        r""" Sets the value of the :math:`M_\Delta`.

        Parameters
        ----------
        mdelta : float
            Galaxy cluster mass :math:`M_\Delta` in units of :math:`M_\odot`

        Notes
        -----
            This is equivalent to doing self.mdelta = mdelta
        """
        if self.validate_input:
            validate_argument(locals(), 'mdelta', float, argmin=0)
        self._set_mass(mdelta)

    def set_concentration(self, cdelta):
        r""" Sets the concentration

        Parameters
        ----------
        cdelta: float
            Concentration

        Notes
        -----
            This is equivalent to doing self.cdelta = cdelta
        """
        if self.validate_input:
            validate_argument(locals(), 'cdelta', float, argmin=0)
        self._set_concentration(cdelta)

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
        # set the profile
        self._set_halo_density_profile(halo_profile_model=halo_profile_model,
                                       massdef=massdef, delta_mdef=delta_mdef)

        # set internal quantities
        self.__halo_profile_model = halo_profile_model
        self.__massdef = massdef
        self.__delta_mdef = delta_mdef

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

    def eval_critical_surface_density(self, z_len, z_src=None, use_pdz=False, pzbins=None, pzpdf=None):
        r"""Computes either

        the critical surface density if `use_pdz=False`

        .. math::
            \Sigma_{\rm crit} = \frac{c^2}{4\pi G} \frac{D_s}{D_LD_{LS}}

        or

        the 'effective critical surface density' if `use_pdz=True`

        .. math::
            \langle \Sigma_{\rm crit}^{-1}\rangle^{-1} = \left(\int \frac{1}{\Sigma_{\rm crit}(z)} p(z) dz\right)^{-1}

        where :math:`p(z)` is the source photoz probability density function.
        This comes from the maximum likelihood estimator for evaluating a :math:`\Delta\Sigma` profile.


        Parameters
        ----------
        z_len : float
            Galaxy cluster redshift
        z_src : array_like, float
            Background source galaxy redshift(s)
        use_pdz : bool
            Flag to use the photoz pdf. If `False` (default), `sigma_c` is computed using the source redshift point estimates `z_source`.
            If `True`, `sigma_c` is computed as 1/<1/Sigma_crit>, where the average is performed using
            the individual galaxy redshift pdf. In that case, the `pzbins` and `pzpdf` should be specified.

        pzbins : array-like
            Bins where the source redshift pdf is defined
        pzpdf : array-like
            Values of the source redshift pdf
        validate_input: bool
            Validade each input argument


        Returns
        -------
        sigma_c : array_like, float
            Cosmology-dependent (effective) critical surface density in units of :math:`M_\odot\ Mpc^{-2}`
    """

        if self.validate_input:
            validate_argument(locals(), 'z_len', float, argmin=0)
            validate_argument(locals(), 'z_src', 'float_array', argmin=0, none_ok=True)

        if use_pdz is False:
            return self._eval_critical_surface_density(z_len=z_len, z_src=z_src)
        else:
            if pzbins is None or pzpdf is None:
                raise ValueError('Redshift bins and source redshift pdf must be provided when use_pdz is True')
            else:
                def inv_sigmac(redshift):
                    return 1./self._eval_critical_surface_density(z_len=z_len, z_src=redshift)
                return 1./_integ_pzfuncs(pzpdf, pzbins, kernel=inv_sigmac, is_unique_pzbins=np.all(pzbins==pzbins[0]))

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

    def eval_tangential_shear(self, r_proj, z_cl, z_src, z_src_info='discrete', beta_kwargs=None,
                              verbose=False):
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
            Type of redshift information provided, it describes z_src.
            The following supported options are:

                * 'discrete' (default) : The redshift of sources is provided by `z_src`.
                  It can be individual redshifts for each source galaxy when `z_source` is an
                  arrayor all sources are at the same redshift when `z_source` is a float.

                * 'distribution' : A redshift distribution function is provided by `z_src`.
                  `z_src` must be a one dimentional function.

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

        beta_kwargs: None, dict
            Extra arguments for the `compute_beta_s_mean, compute_beta_s_square_mean` functions.
            Only used if `z_src_info='distribution'`. Possible keys are:

                * 'zmin' (None, float) : Minimum redshift to be set as the source of the galaxy
                  when performing the sum. (default=None)
                * 'zmax' (float) : Maximum redshift to be set as the source of the galaxy
                  when performing the sum. (default=10.0)
                * 'delta_z_cut' (float) : Redshift interval to be summed with $z_cl$ to return
                  $zmin$. This feature is not used if $z_min$ is provided. (default=0.1)

        verbose : bool, optional
            If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and
            CCL backends.

        Returns
        -------
        array_like, float
            tangential shear
        """
        if self.validate_input:
            validate_argument(locals(), 'r_proj', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)
            validate_argument(locals(), 'z_src_info', str)
            self._validate_z_src(locals())

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        if z_src_info=='discrete':
            gammat = self._eval_tangential_shear(r_proj=r_proj, z_cl=z_cl, z_src=z_src)
        elif z_src_info in ('distribution', 'beta'):
            z_inf = 1000. #np.inf # INF or a very large number
            gammat_inf = self._eval_tangential_shear(r_proj=r_proj, z_cl=z_cl, z_src=z_inf)
            if z_src_info=='beta':
                # z_src (tuple) is (beta_s_mean, beta_s_square_mean)
                beta_s_mean, beta_s_square_mean = z_src
            elif z_src_info=='distribution':
                # z_src (function) if PDZ
                beta_kwargs = {} if beta_kwargs is None else beta_kwargs
                beta_s_mean = compute_beta_s_mean(z_cl, z_inf, self.cosmo, z_distrib_func=z_src,
                                                  **beta_kwargs)
                beta_s_square_mean = compute_beta_s_square_mean(z_cl, z_inf, self.cosmo,
                                                                z_distrib_func=z_src,
                                                                **beta_kwargs)
            gammat = beta_s_mean * gammat_inf
        else:
            raise ValueError(f"Unsupported z_src_info (='{z_src_info}')")

        return gammat

    def eval_convergence(self, r_proj, z_cl, z_src, z_src_info='discrete', beta_kwargs=None, verbose=False):

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
            Type of redshift information provided, it describes z_src.
            The following supported options are:

                * 'discrete' (default) : The redshift of sources is provided by `z_src`.
                  It can be individual redshifts for each source galaxy when `z_source` is an
                  array or all sources are at the same redshift when `z_source` is a float.

                * 'distribution' : A redshift distribution function is provided by `z_src`.
                  `z_src` must be a one dimentional function.

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

        beta_kwargs: None, dict
            Extra arguments for the `compute_beta_s_mean, compute_beta_s_square_mean` functions.
            Only used if `z_src_info='distribution'`. Possible keys are:

                * 'zmin' (None, float) : Minimum redshift to be set as the source of the galaxy
                  when performing the sum. (default=None)
                * 'zmax' (float) : Maximum redshift to be set as the source of the galaxy
                  when performing the sum. (default=10.0)
                * 'delta_z_cut' (float) : Redshift interval to be summed with $z_cl$ to return
                  $zmin$. This feature is not used if $z_min$ is provided. (default=0.1)

        verbose : bool, optional
            If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and
            CCL backends.

        Returns
        -------
        array_like, float
            Mass convergence, kappa.
        """
        if self.validate_input:
            validate_argument(locals(), 'r_proj', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)
            validate_argument(locals(), 'z_src_info', str)
            self._validate_z_src(locals())

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        if z_src_info=='discrete':
            kappa = self._eval_convergence(r_proj=r_proj, z_cl=z_cl, z_src=z_src)
        elif z_src_info in ('distribution', 'beta'):
            z_inf = 1000. #np.inf # INF or a very large number
            kappa_inf = self._eval_convergence(r_proj=r_proj, z_cl=z_cl, z_src=z_inf)
            if z_src_info=='beta':
                # z_src (tuple) is (beta_s_mean, beta_s_square_mean)
                beta_s_mean, beta_s_square_mean = z_src
            elif z_src_info=='distribution':
                # z_src (function) if PDZ
                beta_kwargs = {} if beta_kwargs is None else beta_kwargs
                beta_s_mean = compute_beta_s_mean(z_cl, z_inf, self.cosmo, z_distrib_func=z_src,
                                                  **beta_kwargs)
                beta_s_square_mean = compute_beta_s_square_mean(z_cl, z_inf, self.cosmo,
                                                                z_distrib_func=z_src,
                                                                **beta_kwargs)

            kappa = beta_s_mean * kappa_inf
        else:
            raise ValueError(f"Unsupported z_src_info (='{z_src_info}')")

        return kappa


    def eval_reduced_tangential_shear(self, r_proj, z_cl, z_src, z_src_info='discrete',
                                      approx=None, beta_kwargs=None, verbose=False):
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
            Type of redshift information provided, it describes z_src.
            The following supported options are:

                * 'discrete' (default) : The redshift of sources is provided by `z_src`.
                  It can be individual redshifts for each source galaxy when `z_source` is an
                  array or all sources are at the same redshift when `z_source` is a float.

                * 'distribution' : A redshift distribution function is provided by `z_src`.
                  `z_src` must be a one dimentional function.

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
            Type of computation to be made for reduced shears, options are:

                * None (default): Full computation is made for each `r_proj, z_src` pair
                  individually. It requires `z_src_info` to be 'discrete' or 'distribution'.

                * 'applegate14' : Uses the approach from Weighing the Giants - III (equation 6 in
                  Applegate et al. 2014; https://arxiv.org/abs/1208.0605). `z_src_info` must be
                  either 'beta', or 'distribution' (that will be used to compute
                  :math:`\langle \beta_s \rangle, \langle \beta_s^2 \rangle`)

                * 'schrabback18' : Uses the approach from Cluster Mass Calibration at High
                  Redshift (equation 12 in Schrabback et al. 2017;
                  https://arxiv.org/abs/1611.03866).
                  `z_src_info` must be either 'beta', or 'distribution' (that will be used
                  to compute :math:`\langle \beta_s \rangle, \langle \beta_s^2 \rangle`)

        beta_kwargs: None, dict
            Extra arguments for the `compute_beta_s_mean, compute_beta_s_square_mean` functions.
            Only used if `z_src_info='distribution'`. Possible keys are:

                * 'zmin' (None, float) : Minimum redshift to be set as the source of the galaxy
                  when performing the sum. (default=None)
                * 'zmax' (float) : Maximum redshift to be set as the source of the galaxy
                  when performing the sum. (default=10.0)
                * 'delta_z_cut' (float) : Redshift interval to be summed with $z_cl$ to return
                  $zmin$. This feature is not used if $z_min$ is provided. (default=0.1)

        verbose : bool, optional
            If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and
            CCL backends.

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
            validate_argument(locals(), 'z_src_info', str)
            validate_argument(locals(), 'approx', str, none_ok=True)
            self._validate_z_src(locals())

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        if approx is None:
            if z_src_info=='distribution':
                z_inf = 1000. #np.inf # INF or a very large number

                def integrand(z_i, r_i):
                    return z_src(z_i)*compute_beta_s_func(z_i, z_cl, z_inf, self.cosmo,
                                                           self._eval_tangential_shear,
                                                           r_i, z_cl, z_inf)\
                    /(1-compute_beta_s_func(z_i, z_cl, z_inf, self.cosmo,
                                            self._eval_convergence,
                                            r_i, z_cl, z_inf))
                kwargs = {'zmax': 10.0, 'delta_z_cut': 0.1, 'zmin': None} if beta_kwargs is None\
                else beta_kwargs
                zmax = kwargs['zmax']
                delta_z_cut = kwargs['delta_z_cut']
                zmin = z_cl+delta_z_cut if kwargs['zmin'] is None else kwargs['zmin']

                gt = np.zeros_like(r_proj)
                for i, r in enumerate(r_proj):
                    gt[i] = quad(integrand, zmin, zmax, r)[0]
                # Normalize
                gt /= quad(z_src, zmin, zmax)[0]
            elif z_src_info=='discrete':
                gt = self._eval_reduced_tangential_shear_sp(r_proj, z_cl, z_src)
            else:
                raise ValueError(
                    "approx=None requires z_src_info='discrete' or 'distribution',"
                    f"z_src_info='{z_src_info}' was provided.")

        elif approx in ('applegate14', 'schrabback18'):
            z_inf = 1000. #np.inf # INF or a very large number
            if z_src_info=='beta':
                # z_src (tuple) is (beta_s_mean, beta_s_square_mean)
                beta_s_mean, beta_s_square_mean = z_src
            elif z_src_info=='distribution':
                beta_kwargs = {} if beta_kwargs is None else beta_kwargs
                beta_s_mean = compute_beta_s_mean(z_cl, z_inf, self.cosmo, z_distrib_func=z_src,
                                                  **beta_kwargs)
                beta_s_square_mean = compute_beta_s_square_mean(z_cl, z_inf, self.cosmo,
                                                                z_distrib_func=z_src,
                                                                **beta_kwargs)
            else:
                raise ValueError(
                    f"approx='{approx}' requires z_src_info='distribution' or 'beta', "
                    f"z_src_info='{z_src_info}' was provided.")

            gammat_inf = self._eval_tangential_shear(r_proj, z_cl, z_src=z_inf)
            kappa_inf = self._eval_convergence(r_proj, z_cl, z_src=z_inf)

            if approx == 'applegate14':
                gt = beta_s_mean * gammat_inf / (1. - beta_s_square_mean / beta_s_mean * kappa_inf)
            elif approx == 'schrabback18':
                gt = (1. + (beta_s_square_mean / (beta_s_mean * beta_s_mean) - 1.) \
                           * beta_s_mean * kappa_inf ) \
                      * (beta_s_mean * gammat_inf / (1. - beta_s_mean * kappa_inf))
        else:
            raise ValueError(f"Unsupported approx (='{approx}')")

        return gt

    def eval_magnification(self, r_proj, z_cl, z_src, z_src_info='discrete',
                           approx=None, beta_kwargs=None, verbose=False):
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
            Type of redshift information provided, it describes z_src.
            The following supported options are:

                * 'discrete' (default) : The redshift of sources is provided by `z_src`.
                  It can be individual redshifts for each source galaxy when `z_source` is an
                  arrayor all sources are at the same redshift when `z_source` is a float.

                * 'distribution' : A redshift distribution function is provided by `z_src`.
                  `z_src` must be a one dimentional function.

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
            Type of computation to be made for reduced shears, options are:

                * None (default): Full computation is made for each `r_proj, z_src` pair
                  individually. It requires `z_src_info` to be 'discrete' or 'distribution'.

                * 'weak lensing' : Uses the weak lensing approximation of the magnification
                  :math:`\mu \approx 1 + 2 \kappa`. `z_src_info` must be either 'beta', or
                  'distribution' (that will be used to compute :math:`\langle \beta_s \rangle`)

        beta_kwargs: None, dict
            Extra arguments for the `compute_beta_s_mean, compute_beta_s_square_mean` functions.
            Only used if `z_src_info='distribution'`. Possible keys are:

                * 'zmin' (None, float) : Minimum redshift to be set as the source of the galaxy
                  when performing the sum. (default=None)
                * 'zmax' (float) : Maximum redshift to be set as the source of the galaxy
                  when performing the sum. (default=10.0)
                * 'delta_z_cut' (float) : Redshift interval to be summed with $z_cl$ to return
                  $zmin$. This feature is not used if $z_min$ is provided. (default=0.1)

        verbose : bool, optional
            If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and
            CCL backends.

        Returns
        -------
        mu : array_like, float
            magnification, mu.

        """
        if self.validate_input:
            validate_argument(locals(), 'r_proj', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)
            validate_argument(locals(), 'z_src_info', str)
            validate_argument(locals(), 'approx', str, none_ok=True)
            self._validate_z_src(locals())

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        if approx is None:
            if z_src_info=='distribution':
                z_inf = 1000. #np.inf # INF or a very large number

                def integrand(z_i, r_i):
                    return z_src(z_i)/((1-compute_beta_s_func(z_i, z_cl, z_inf, self.cosmo,
                                                              self._eval_convergence,
                                                              r_i, z_cl, z_inf))**2\
                                       -(compute_beta_s_func(z_i, z_cl, z_inf, self.cosmo,
                                                             self._eval_tangential_shear,
                                                             r_i, z_cl, z_inf))**2)

                kwargs = {'zmax': 10.0, 'delta_z_cut': 0.1, 'zmin': None} if beta_kwargs is None\
                else beta_kwargs
                zmax = kwargs['zmax']
                delta_z_cut = kwargs['delta_z_cut']
                zmin = z_cl+delta_z_cut if kwargs['zmin'] is None else kwargs['zmin']

                mu = np.zeros_like(r_proj)
                for i, r in enumerate(r_proj):
                    mu[i] = quad(integrand, zmin, zmax, (r))[0]
                # Normalize
                mu /= quad(z_src, zmin, zmax)[0]
            elif z_src_info=='discrete':
                mu = self._eval_magnification(r_proj=r_proj, z_cl=z_cl, z_src=z_src)
            else:
                raise ValueError(
                    "approx=None requires z_src_info='discrete' or 'distribution',"
                    f"z_src_info='{z_src_info}' was provided.")

        elif approx == 'weak lensing':
            z_inf = 1000. #np.inf # INF or a very large number
            kappa_inf = self._eval_convergence(r_proj, z_cl, z_src=z_inf)
            gamma_inf = self._eval_tangential_shear(r_proj, z_cl, z_src=z_inf)
            if z_src_info=='beta':
                # z_src (tuple) is (beta_s_mean, beta_s_square_mean)
                beta_s_mean, beta_s_square_mean = z_src
            elif z_src_info=='distribution':
                beta_kwargs = {} if beta_kwargs is None else beta_kwargs
                beta_s_mean = compute_beta_s_mean(z_cl, z_inf, self.cosmo, z_distrib_func=z_src,
                                                  **beta_kwargs)
                beta_s_square_mean = compute_beta_s_square_mean(z_cl, z_inf, self.cosmo,
                                                                z_distrib_func=z_src,
                                                                **beta_kwargs)
            else:
                raise ValueError(
                    f"approx='{approx}' requires z_src_info='distribution' or 'beta', "
                    f"z_src_info='{z_src_info}' was provided.")

            #mu = 1 + 2*beta_s_mean*kappa_inf

            mu = 1 / ((1 - beta_s_mean*kappa_inf)**2 - (beta_s_mean*gamma_inf)**2)
            # correction terms
            # exact
            mu *= 1-(beta_s_mean*kappa_inf-beta_s_mean*gamma_inf)
            mu *= 1-(beta_s_mean*kappa_inf+beta_s_mean*gamma_inf)
            # Taylor expansion with optimized prefactor of cross terms
            mu *= 1+(beta_s_mean*kappa_inf-beta_s_mean*gamma_inf)\
                   +beta_s_square_mean*kappa_inf**2+beta_s_square_mean*gamma_inf**2\
                   +4*beta_s_square_mean*kappa_inf*gamma_inf
            mu *= 1+(beta_s_mean*kappa_inf+beta_s_mean*gamma_inf)\
                   +beta_s_square_mean*kappa_inf**2+beta_s_square_mean*gamma_inf**2\
                   -4*beta_s_square_mean*kappa_inf*gamma_inf

        else:
            raise ValueError(f"Unsupported approx (='{approx}')")
        return mu

    def eval_magnification_bias(self, r_proj, z_cl, z_src, alpha, z_src_info='discrete',
                                approx=None, beta_kwargs=None, verbose=False):
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
            Type of redshift information provided, it describes z_src.
            The following supported options are:

                * 'discrete' (default) : The redshift of sources is provided by `z_src`.
                  It can be individual redshifts for each source galaxy when `z_source` is an
                  array or all sources are at the same redshift when `z_source` is a float.

                * 'distribution' : A redshift distribution function is provided by `z_src`.
                  `z_src` must be a one dimentional function.

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
            Type of computation to be made for reduced shears, options are:

                * None (default): Full computation is made for each `r_proj, z_src` pair
                  individually. It requires `z_src_info` to be 'discrete' or 'distribution'.

                * 'weak lensing' : Uses the weak lensing approximation of the magnification bias
                  :math:`\mu \approx 1 + 2 \kappa \left(\alpha - 1 \right)`. `z_src_info` must be
                  either 'beta', or 'distribution' (that will be used to compute
                  :math:`\langle \beta_s \rangle`)

        beta_kwargs: None, dict
            Extra arguments for the `compute_beta_s_mean, compute_beta_s_square_mean` functions.
            Only used if `z_src_info='distribution'`. Possible keys are:

                * 'zmin' (None, float) : Minimum redshift to be set as the source of the galaxy
                  when performing the sum. (default=None)
                * 'zmax' (float) : Maximum redshift to be set as the source of the galaxy
                  when performing the sum. (default=10.0)
                * 'delta_z_cut' (float) : Redshift interval to be summed with $z_cl$ to return
                  $zmin$. This feature is not used if $z_min$ is provided. (default=0.1)

        verbose : bool, optional
            If True, the Einasto slope (alpha_ein) is printed out. Only availble for the NC and
            CCL backends.

        Returns
        -------
        mu_bias : array_like, float
            magnification bias.

        """
        if self.validate_input:
            validate_argument(locals(), 'r_proj', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)
            validate_argument(locals(), 'z_src_info', str)
            validate_argument(locals(), 'alpha', 'float_array')
            validate_argument(locals(), 'approx', str, none_ok=True)
            self._validate_z_src(locals())

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha = {self._get_einasto_alpha(z_cl=z_cl)}")

        if approx is None:
            # z_src (float or array) is redshift
            if z_src_info=='distribution':
                z_inf = 1000. #np.inf # INF or a very large number

                def integrand(z_i, r_i):
                    return z_src(z_i)/((1-compute_beta_s_func(z_i, z_cl, z_inf, self.cosmo,
                                                              self._eval_convergence,
                                                              r_i, z_cl, z_inf))**2\
                                       -(compute_beta_s_func(z_i, z_cl, z_inf, self.cosmo,
                                                             self._eval_tangential_shear,
                                                             r_i, z_cl, z_inf))**2)**(alpha-1)

                kwargs = {'zmax': 10.0, 'delta_z_cut': 0.1, 'zmin': None} if beta_kwargs is None\
                else beta_kwargs
                zmax = kwargs['zmax']
                delta_z_cut = kwargs['delta_z_cut']
                zmin = z_cl+delta_z_cut if kwargs['zmin'] is None else kwargs['zmin']

                mu_bias = np.zeros_like(r_proj)
                for i, r in enumerate(r_proj):
                    mu_bias[i] = quad(integrand, zmin, zmax, (r))[0]
                # Normalize
                mu_bias /= quad(z_src, zmin, zmax)[0]
            elif z_src_info=='discrete':
                mu_bias = self._eval_magnification_bias(
                    r_proj=r_proj, z_cl=z_cl, z_src=z_src, alpha=alpha)
            else:
                raise ValueError(
                    "approx=None requires z_src_info='discrete' or 'distribution',"
                    f"z_src_info='{z_src_info}' was provided.")

        elif approx == 'weak lensing':
            z_inf = 1000. #np.inf # INF or a very large number
            kappa_inf = self._eval_convergence(r_proj, z_cl, z_src=z_inf)
            gamma_inf = self._eval_tangential_shear(r_proj, z_cl, z_src=z_inf)
            if z_src_info=='beta':
                # z_src (tuple) is (beta_s_mean, beta_s_square_mean)
                beta_s_mean, beta_s_square_mean = z_src
            elif z_src_info=='distribution':
                beta_kwargs = {} if beta_kwargs is None else beta_kwargs
                beta_s_mean = compute_beta_s_mean(z_cl, z_inf, self.cosmo, z_distrib_func=z_src,
                                                  **beta_kwargs)
                beta_s_square_mean = compute_beta_s_square_mean(z_cl, z_inf, self.cosmo,
                                                                z_distrib_func=z_src,
                                                                **beta_kwargs)
            else:
                raise ValueError(
                    f"approx='{approx}' requires z_src_info='distribution' or 'beta', "
                    f"z_src_info='{z_src_info}' was provided.")

            #mu_bias = 1 + 2*beta_s_mean*kappa_inf*(alpha-1)

            mu_bias = (1/((1 - beta_s_mean*kappa_inf)**2 - (beta_s_mean*gamma_inf)**2))**(alpha-1)
            # correction terms
            # exact
            mu_bias *= (1-(beta_s_mean*kappa_inf-beta_s_mean*gamma_inf))**(alpha-1)
            mu_bias *= (1-(beta_s_mean*kappa_inf+beta_s_mean*gamma_inf))**(alpha-1)
            # Taylor expansion with optimized prefactor of cross terms
            mu_bias *= (1+(beta_s_mean*kappa_inf-beta_s_mean*gamma_inf)\
                   +beta_s_square_mean*kappa_inf**2+beta_s_square_mean*gamma_inf**2\
                   +4*beta_s_square_mean*kappa_inf*gamma_inf)**(alpha-1)
            mu_bias *= (1+(beta_s_mean*kappa_inf+beta_s_mean*gamma_inf)\
                   +beta_s_square_mean*kappa_inf**2+beta_s_square_mean*gamma_inf**2\
                   -4*beta_s_square_mean*kappa_inf*gamma_inf)**(alpha-1)

        else:
            raise ValueError(f"Unsupported approx (='{approx}')")

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
            validate_argument(locals(), 'z_cl', float, argmin=0)
        return self._eval_rdelta(z_cl)

    def eval_mass_in_radius(self, r3d, z_cl, verbose=False):
        r"""Computes the mass inside a given radius of the profile.
        The mass is calculated as

        .. math::
            M(<\text{r3d}) = M_{\Delta}\;\frac{f\left(\frac{\text{r3d}}{r_{\Delta}/c_{\Delta}}\right)}{f(c_{\Delta})},

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
        array_like, float
            Mass in units of :math:`M_\odot`
        """
        if self.validate_input:
            validate_argument(locals(), 'r3d', 'float_array', argmin=0)
            validate_argument(locals(), 'z_cl', float, argmin=0)

        if self.halo_profile_model=='einasto' and verbose:
            print(f"Einasto alpha (in) = {self._get_einasto_alpha(z_cl=z_cl)}")

        return self._eval_mass_in_radius(r3d, z_cl)

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

    def _validate_z_src(self, loc_dict):
        r"""Validation for z_src

        Parameters
        ----------
        locals_dict: dict
            Should be the call locals()
        """
        if loc_dict['z_src_info']=='discrete':
            validate_argument(loc_dict, 'z_src', 'float_array', argmin=0)
        elif loc_dict['z_src_info']=='distribution':
            validate_argument(loc_dict, 'z_src', 'function', none_ok=False)
            beta_kwargs = {} if loc_dict['beta_kwargs'] is None else loc_dict['beta_kwargs']
            _def_keys = ['zmin', 'zmax', 'delta_z_cut']
            if any(key not in _def_keys for key in beta_kwargs):
                raise KeyError(f'beta_kwargs must contain only {_def_keys} keys, '
                               f' {beta_kwargs.keys()} provided.')
        elif loc_dict['z_src_info']=='beta':
            validate_argument(loc_dict, 'z_src', 'array')
            beta_info = {'beta_s_mean':loc_dict['z_src'][0],
                         'beta_s_square_mean':loc_dict['z_src'][1]}
            validate_argument(beta_info, 'beta_s_mean', 'float_array')
            validate_argument(beta_info, 'beta_s_square_mean', 'float_array')
