# CLMM Cosmology object abstract superclass and supported subclasses

class CLMMCosmology:
    """
    Cosmology object superclass for supporting multiple back-end cosmology objects
    """
    def __init__(self, cosmo, *args, **kwargs):
        self.cosmo = cosmo
        # self.name = name
        self.backend = None

    def _get_param(self, key):
        """
        Parameters
        ----------
        key: str
            CLMM-ified keyword
        """
        # return self.cosmo[key]
        raise NotImplementedError

    def get_h(self):
        return self._get_param('h')

    def get_omega_m(self):
        return self._get_param('Omega_m')

# Should self.name for each be consistent with clmm.modeling.__backends?

class AstroPyCosmology(CLMMCosmology):
    def __init__(self, cosmo):
        super(AstroPyCosmology, self).__init__(cosmo)
        self.name = 'ap'
        assert isinstance(cosmo, LambdaCDM)

    def get_omega_m(self):
        """
        from clmm.modbackend.generic.cclify_astropy_cosmo
        """
        return self.cosmo.Om0

    def get_h(self):
        """
        from clmm.modbackend.generic.cclify_astropy_cosmo
        """
        h = self.cosmo.H0 / 100.
        return h


class CCLCosmology(CLMMCosmology):
    def __init__(self, cosmo, name='ccl'):
        super(CCLCosmology, self).__init__(cosmo)
        self.name = 'ccl'

    def get_omega_m(self):
        """
        from clmm.modbackend.generic.astropyify_ccl_cosmo
        """
        Omega_m = self.cosmo.Omega_b + self.cosmo.Omega_c
        return Omega_m

    def get_h(self):
        """
        from clmm.modbackend.generic.astropyify_ccl_cosmo
        """
        return self.cosmo.h

class NumCosmoCosmology(CLMMCosmology):
    def __init__(self, cosmo):
        super(NumCosmoCosmology, self).__init__(cosmo)
        self.name = 'nc'

    def get_omega_m(self):
        """
        from clmm.modbackend.numcosmo
        """
        Omegam = self.cosmo.Omegab + self.cosmo.Omega_c
        return Omegam

    def get_h(self):
        """
        from clmm.modbackend.numcosmo
        """
        return self.cosmo.h
