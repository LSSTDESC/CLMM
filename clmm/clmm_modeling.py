# CLMModeling abstract class
import numpy as np
import warnings


class CLMModeling:
    r"""Object with functions for halo mass modeling

    Attributes
    ----------
    backend: str
        Name of the backend being used
    massdef : str
        Profile mass definition (`mean`, `critical`, `virial`)
    delta_mdef : int
        Mass overdensity definition.
    halo_profile_model : str
        Profile model parameterization (`nfw`, `einasto`, `hernquist`)
    cosmo: Cosmology
        Cosmology object
    hdpm: Object
        Backend object with halo profiles
    mdef_dict: dict
        Dictionary with the definitions for mass
    hdpm_dict: dict
        Dictionary with the definitions for profile
    """

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
        if not massdef in self.mdef_dict:
            raise ValueError(f"Halo density profile mass definition {massdef} not currently supported")
        if not halo_profile_model in self.hdpm_dict:
            raise ValueError(f"Halo density profile model {halo_profile_model} not currently supported")

    def set_cosmo(self, cosmo):
        r""" Sets the cosmology to the internal cosmology object

        Parameters
        ----------
        cosmo: clmm.Comology
            CLMM Cosmology object
        """
        raise NotImplementedError

    def _set_cosmo(self, cosmo, CosmoOutput, valid_cosmo=None):
        r""" Sets the cosmology to the internal cosmology object

        Parameters
        ----------
        cosmo: clmm.Comology object, None
            CLMM Cosmology object. If is None, creates a new instance of CosmoOutput().
        CosmoOutput: clmm.modbackend Cosmology class
            Cosmology Output for the output object.
        valid_cosmo: clmm.Comology, tuple, None
            Accepted type (or list of accepted types) of cosmology objects.
            If is None, uses CosmoOutput.
        """
        if valid_cosmo is None:
            valid_cosmo = CosmoOutput
        if cosmo:
            if not isinstance(cosmo, valid_cosmo):
                warnings.warn(f'Translating {type(cosmo)} into {CosmoOutput}.')
                self.cosmo = self._import_cosmo(cosmo, CosmoOutput)
            else:
                self.cosmo = cosmo
        else:
            self.cosmo = CosmoOutput()

    def _import_cosmo(self, cosmo, CosmoOutput):
        r""" Translates a cosmology to the internal cosmology object

        Parameters
        ----------
        cosmo: clmm.Comology like
            CLMM Cosmology object
        CosmoOutput: clmm.modbackend Cosmology
            Cosmology Output for the output object

        Returns
        -------
        clmm.Cosmology like
            Cosmology object
        """
        params = {}
        for p in ('H0', 'Omega_dm0', 'Omega_b0', 'Omega_k0'):
            try:
                params[p] = cosmo[p]
            except:
                raise ValueError(f"Cosmology object {cosmo} missing parameter:{p}.")
        return CosmoOutput(**params)

    def set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        r""" Sets the definitios for the halo profile

        Parameters
        ----------
        halo_profile_model: str
            Halo mass profile, current options are 'nfw'
        massdef: str
            Mass definition, current options are 'mean'
        delta_mdef: int
            Overdensity number
        """
        raise NotImplementedError

    def set_concentration(self, cdelta):
        r""" Sets the concentration

        Parameters
        ----------
        cdelta: float
            Concentration
        """
        raise NotImplementedError

    def set_mass(self, mdelta):
        r""" Sets the value of the :math:`M_\Delta`

        Parameters
        ----------
        mdelta : float
            Galaxy cluster mass :math:`M_\Delta` in units of :math:`M_\odot`
        """
        raise NotImplementedError

    def eval_sigma_crit(self, z_len, z_src):
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
        raise NotImplementedError

    def eval_density(self, r3d, z_cl):
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

    def eval_sigma(self, r_proj, z_cl):
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

    def eval_sigma_mean(self, r_proj, z_cl):
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

    def eval_sigma_excess(self, r_proj, z_cl):
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

    def eval_shear(self, r_proj, z_cl, z_src):
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
        raise NotImplementedError
