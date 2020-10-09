# CLMM Cosmology object abstract superclass and supported subclasses

class CLMMCosmology:
    """
    Cosmology object superclass for supporting multiple back-end cosmology objects

    Attributes
    ----------
    ger_par: dict
        Dictionary with functions to get each specific parameter
    """
    def __init__(self, cosmo, *args, **kwargs):
        self.cosmo = cosmo
        # self.name = name
        self.backend = None
        self.get_par = {'omega_m':lambda:None, 'h':lambda:None}

    def __getitem__(self, item):
        if isinstance(item, str):
            if item.lower() in self.get_par:
                return self.get_par[item.lower()]()
            else:
                raise ValueError(f'input({item}) must be in:{get_par.keys()}')
        else:
            raise TypeError(f'input must be str, not {type(item)}')

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


class CCLCosmology(CLMMCosmology):
    def __init__(self, cosmo, name='ccl'):
        super(CCLCosmology, self).__init__(cosmo)
        self.name = 'ccl'

        """
        from clmm.modbackend.generic.astropyify_ccl_cosmo
        """
        self.get_par['omega_m'] = lambda: self.cosmo.Omega_b + self.cosmo.Omega_c
        self.get_par['h'] = lambda: self.cosmo.h


class NumCosmoCosmology(CLMMCosmology):
    def __init__(self, cosmo):
        super(NumCosmoCosmology, self).__init__(cosmo)
        self.name = 'nc'

        """
        from clmm.modbackend.numcosmo
        """
        self.get_par['omega_m'] = lambda: self.cosmo.Omegab + self.cosmo.Omega_c
        self.get_par['h'] = lambda: self.cosmo.h
