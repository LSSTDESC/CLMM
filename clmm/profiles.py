import numpy as np
from scipy.optimize import fsolve
from scipy.special import gamma, gammainc

from .cosmology.parent_class import CLMMCosmology
from .utils import validate_argument

class HaloProfile:
    def __init__(self, mdelta, cdelta, z_cl, cosmo, validate_input=True):

        self.validate_input = validate_input

        self.set_cosmo(cosmo)
        self.set_mass(mdelta)
        self.set_concentration(cdelta)
        self.set_z(z_cl)


        self.massdef = ''
        self.delta_mdef = 0
        self.halo_profile_model = ''
        self.model = None

        self.model_dict = {'nfw': HaloProfileNFW,
                           'einasto': HaloProfileEinasto,
                           'hernquist': HaloProfileHernquist,
                          }

        self.mdef_dict = {'mean': self.cosmo.get_rho_m,
                          'critical': self.cosmo.get_rho_c,
                          'virial': self.cosmo.get_rho_c,
                         }

    def set_cosmo(self, cosmo):
        """ Sets the cosmology to the internal cosmology object

        Parameters
        ----------
        cosmo: clmm.Comology object, None
            CLMM Cosmology object. If is None, creates a new instance of CLMMCosmology.
        """
        if self.validate_input:
            validate_argument(locals(), 'cosmo', CLMMCosmology, none_ok=True)
        self._set_cosmo(cosmo)
        self.cosmo.validate_input = self.validate_input

    def _set_cosmo(self, cosmo):
        """ Sets the cosmology to the internal cosmology object"""
        self.cosmo = cosmo if cosmo is not None else CLMMCosmology()

    def set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        """ Sets the definitions for the halo profile

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
            if not halo_profile_model in self.model_dict:
                raise ValueError(
                    f"Halo density profile model {halo_profile_model} not currently supported")
        return self._set_halo_density_profile(halo_profile_model=halo_profile_model,
                                              massdef=massdef, delta_mdef=delta_mdef)

    def _set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        """set halo density profile"""
        # Check if we have already an instance of the required object, if not create one
        if not ((halo_profile_model==self.halo_profile_model)
                and (massdef == self.massdef)
                and (delta_mdef == self.delta_mdef)):
            self.halo_profile_model = halo_profile_model
            self.massdef = massdef
            self.delta_mdef = delta_mdef

    def set_mass(self, mdelta):
        """ Sets the value of the :math:`M_\Delta`

        Parameters
        ----------
        mdelta : float
            Galaxy cluster mass :math:`M_\Delta` in units of :math:`M_\odot`
        """
        if self.validate_input:
            validate_argument(locals(), 'mdelta', float, argmin=0)
        self._set_mass(mdelta)

    def _set_mass(self, mdelta):
        """ set mass"""
        self.mdelta = mdelta

    def set_concentration(self, cdelta):
        """ Sets the concentration

        Parameters
        ----------
        cdelta: float
            Concentration
        """
        if self.validate_input:
            validate_argument(locals(), 'cdelta', float, argmin=0)
        self._set_concentration(cdelta)

    def _set_concentration(self, cdelta):
        """" set concentration"""
        self.cdelta = cdelta

    def set_z(self, z_cl):
        """ Sets the redshift

        Parameters
        ----------
        z_cl: float
            Redshift
        """
        if self.validate_input:
            validate_argument(locals(), 'z_cl', float, argmin=0)
        self._set_z(z_cl)

    def _set_z(self, z_cl):
        self.z_cl = z_cl

    def set_einasto_alpha(self, alpha):
        r""" Sets the value of the :math:`\alpha` parameter for the Einasto profile

        Parameters
        ----------
        alpha : float
            Einasto :math:`\alpha` parameter
        """
        if self.halo_profile_model!='einasto':
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

        Returns
        -------
        alpha : float
            Einasto :math:`\alpha` parameter
        """
        if self.halo_profile_model!='einasto':
            raise NotImplementedError(f"Wrong profile model. Current profile = {self.halo_profile_model}")
        else:
            return self._get_einasto_alpha(z_cl)

    def rdelta(self):
        """
        Returns
        -------
        rdelta : float
            Spherical overdensity radius
        """
        rho = self.mdef_dict[self.massdef](self.z_cl)
        return ((self.mdelta * 3.) / (4. * np.pi * self.delta_mdef * rho)) ** (1./3.)

    def rscale(self):
        """
        Returns
        -------
        rscale : float
            Scale radius
        """
        return self.rdelta()/self.cdelta

    def M(self, r3d):
        """
        Parameters
        ----------
        r3d : array_like
            3D radius from the halo center

        Returns
        -------
        M : array
            Mass enclosed within a sphere of radius r3d
        """
        if self.validate_input:
            validate_argument(locals(), 'r3d', 'float_array', argmin=0)
        x = np.array(r3d)/self.rscale()
        M = self.mdelta * self._f_c(x) / self._f_c(self.cdelta)

        return M

    def to_def(self, massdef2, delta_mdef2):
        """
        Parameters
        ----------
        massdef2: float
            Background density definition to convert to (`critical`, `mean`)
        delta_mdef2: str
            Overdensity scale to convert to

        Returns
        -------
        HaloProfile:
            HaloProfile object
        """
        if self.validate_input:
            validate_argument(locals(), 'massdef2', str)
            validate_argument(locals(), 'delta_mdef2', int, argmin=0)
            if not massdef2 in self.mdef_dict:
                raise ValueError(
                    f"Halo density profile mass definition {massdef2} not currently supported")

        if not getattr(self, '_to_def', False):

            def f(params):
                m2, c2 = params
                def2 = self.model(m2, c2, self.z_cl, self.cosmo, massdef2, delta_mdef2)
                return self.mdelta - def2.M(self.rdelta()), def2.mdelta - self.M(def2.rdelta())

            mdelta2, cdelta2 = fsolve(func = f, x0 = [self.mdelta, self.cdelta])
            mdelta2, cdelta2 = fsolve(func = f, x0 = [mdelta2, cdelta2])

            return self.model(mdelta2, cdelta2, self.z_cl, self.cosmo, massdef2, delta_mdef2)

        else:
            return self._to_def(massdef2, delta_mdef2)

class HaloProfileNFW(HaloProfile):
    r"""
    Attributes
    ----------
    mdelta: float
        Halo mass for the given massdef :math:`M_\Delta` in units of :math:`M_\odot`
    cdelta: float
        Halo concentration
    z_cl: float
        Halo redshift
    cosmo: CLMMCosmology
        Cosmology object
    massdef: str
        Background density definition (`critical`, `mean`)
    delta_mdef: float
        Overdensity scale (200, 500, etc.)
    validate_input: bool
        Validade each input argument
    """
    def __init__(self, mdelta, cdelta, z_cl, cosmo,
                 massdef='mean', delta_mdef=200, validate_input=True):

        HaloProfile.__init__(self, mdelta=mdelta, cdelta=cdelta, z_cl=z_cl,
                             cosmo=cosmo, validate_input=validate_input)

        # Set halo profile and cosmology
        self.set_halo_density_profile('nfw', massdef, delta_mdef)
        self.model = self.model_dict[self.halo_profile_model]

    def _f_c(self, c):
        return np.log(1. + c) - c/(1. + c)

class HaloProfileEinasto(HaloProfile):
    r"""
    Attributes
    ----------
    mdelta: float
        Halo mass for the given massdef :math:`M_\Delta` in units of :math:`M_\odot`
    cdelta: float
        Halo concentration
    z_cl: float
        Halo redshift
    cosmo: CLMMCosmology
        Cosmology object
    alpha: float
        Einasto :math:`\alpha` parameter
    massdef: str
        Background density definition (`critical`, `mean`)
    delta_mdef: float
        Overdensity scale (200, 500, etc.)
    validate_input: bool
        Validade each input argument
    """
    def __init__(self, mdelta, cdelta, z_cl, cosmo, alpha,
                 massdef='mean', delta_mdef=200, validate_input=True):

        HaloProfile.__init__(self, mdelta=mdelta, cdelta=cdelta, z_cl=z_cl,
                             cosmo=cosmo, validate_input=validate_input)

        self.set_halo_density_profile('einasto', massdef, delta_mdef)
        self.set_einasto_alpha(alpha)
        self.model = self.model_dict[self.halo_profile_model]

    def _set_einasto_alpha(self, alpha):
        self.einasto_alpha = alpha

    def _get_einasto_alpha(self, z_cl=None):
        return self.einasto_alpha

    def _f_c(self, c):
        alpha = self.einasto_alpha
        return gamma(3./alpha)*gammainc(3./alpha, 2./alpha*c**alpha)

    def _to_def(self, massdef2, delta_mdef2):

        def f(params):
            m2, c2 = params
            def2 = self.model(m2, c2, self.z_cl, self.cosmo, self.einasto_alpha,
                              massdef2, delta_mdef2)
            return self.mdelta - def2.M(self.rdelta()), def2.mdelta - self.M(def2.rdelta())

        mdelta2, cdelta2 = fsolve(func = f, x0 = [self.mdelta, self.cdelta], maxfev = 1000)
        mdelta2, cdelta2 = fsolve(func = f, x0 = [mdelta2, cdelta2], maxfev = 100)

        return self.model(mdelta2, cdelta2, self.z_cl, self.cosmo,
                          self.einasto_alpha, massdef2, delta_mdef2)

class HaloProfileHernquist(HaloProfile):
    r"""
    Attributes
    ----------
    mdelta: float
        Halo mass for the given massdef :math:`M_\Delta` in units of :math:`M_\odot`
    cdelta: float
        Halo concentration
    z_cl: float
        Halo redshift
    cosmo: CLMMCosmology
        Cosmology object
    massdef: str
        Background density definition (`critical`, `mean`)
    delta_mdef: float
        Overdensity scale (200, 500, etc.)
    validate_input: bool
        Validade each input argument
    """
    def __init__(self, mdelta, cdelta, z_cl, cosmo,
                 massdef='mean', delta_mdef=200, validate_input=True):

        HaloProfile.__init__(self, mdelta=mdelta, cdelta=cdelta, z_cl=z_cl,
                             cosmo=cosmo, validate_input=validate_input)

        # Set halo profile and cosmology
        self.set_halo_density_profile('hernquist', massdef, delta_mdef)
        self.model = self.model_dict[self.halo_profile_model]

    def _f_c(self, c):
        return (c/(1. + c))**2.

