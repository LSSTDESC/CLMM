# CLMM Cosmology object abstract superclass and supported subclasses

class CLMMCosmology:
    """
    Cosmology object superclass for supporting multiple back-end cosmology objects

    Attributes
    ----------
    get_par: dict
        Dictionary with functions to get each specific parameter
    set_par: dict
        Dictionary with functions to update each specific parameter
    """
    def __init__(self, cosmo, *args, **kwargs):
        self.cosmo = cosmo
        # self.name = name
        self.backend = None
        self.get_par = {'omega_m':lambda:None, 'h':lambda:None}
        self.set_par = {'omega_m':self._set_omega_m, 'h':self._set_h}

    def __getitem__(self, key):
        if isinstance(key, str):
            if key.lower() in self.get_par:
                return self.get_par[key.lower()]()
            else:
                raise ValueError(f'input({key}) must be in:{get_par.keys()}')
        else:
            raise TypeError(f'input must be str, not {type(key)}')
    def __setitem__(self, key, item):
        if isinstance(key, str):
            if key.lower() in self.get_par:
                return self.set_par[key.lower()](item)
            else:
                raise ValueError(f'key input({key}) must be in:{get_par.keys()}')
        else:
            raise TypeError(f'key input must be str, not {type(key)}')
    def _set_omega_m(self, value):
        """
        To be filled in child classes
        """
        raise NotImplementedError
    def _set_h(self, value):
        """
        To be filled in child classes
        """
        raise NotImplementedError
        

# Should self.name for each be consistent with clmm.modeling.__backends?

class AstroPyCosmology(CLMMCosmology):
    def __init__(self, cosmo):
        super(AstroPyCosmology, self).__init__(cosmo)
        self.name = 'ap'
        assert isinstance(cosmo, LambdaCDM)

        """
        from clmm.modbackend.generic.cclify_astropy_cosmo
        """
        self.get_par['omega_m'] = lambda: self.cosmo.Om0
        self.get_par['h'] = lambda: self.cosmo.H0 / 100.
    def _set_omega_m(self, value):
        self.cosmo.Om0 = value
        return
    def _set_h(self, value):
        self.cosmo.H0 = 100.*value
        return

class CCLCosmology(CLMMCosmology):
    def __init__(self, cosmo, name='ccl'):
        super(CCLCosmology, self).__init__(cosmo)
        self.name = 'ccl'

        """
        from clmm.modbackend.generic.astropyify_ccl_cosmo
        """
        self.get_par['omega_m'] = lambda: self.cosmo.Omega_b + self.cosmo.Omega_c
        self.get_par['h'] = lambda: self.cosmo.h

    def _set_omega_m(self, value):
        omega_m = self.cosmo.Omega_b + self.cosmo.Omega_c
        self.cosmo.Omega_b *= value/omega_m
        self.cosmo.Omega_c *= value/omega_m
        return
    def _set_h(self, value):
        self.cosmo.h = value
        return

class NumCosmoCosmology(CLMMCosmology):
    def __init__(self, cosmo):
        super(NumCosmoCosmology, self).__init__(cosmo)
        self.name = 'nc'

        """
        from clmm.modbackend.numcosmo
        """
        self.get_par['omega_m'] = lambda: self.cosmo.Omegab + self.cosmo.Omega_c
        self.get_par['h'] = lambda: self.cosmo.h
    def _set_omega_m(self, value):
        omega_m = self.cosmo.Omegab + self.cosmo.Omega_c
        self.cosmo.Omegab *= value/omega_m
        self.cosmo.Omega_c *= value/omega_m
        return
    def _set_h(self, value):
        self.cosmo.h = value
        return
