"""@file ccl.py
Modeling using CCL
"""
# Functions to model halo profiles

import pyccl as ccl

import numpy as np
#import warnings
#from packaging import version

from . import func_layer
from . func_layer import *

from .parent_class import CLMModeling

from .. utils import _patch_rho_crit_to_cd2018

from .. cosmology.ccl import CCLCosmology
Cosmology = CCLCosmology

# functions for the 2h term
from scipy.integrate import simps 
from scipy.special import jv
from scipy.interpolate import interp1d

__all__ = ['CCLCLMModeling', 'Modeling', 'Cosmology']+func_layer.__all__


class CCLCLMModeling(CLMModeling):
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

    def __init__(self, massdef='mean', delta_mdef=200, halo_profile_model='nfw',
                 validate_input=True):
        CLMModeling.__init__(self, validate_input)
        # Update class attributes
        self.backend = 'ccl'
        self.mdef_dict = {
            'mean': 'matter',
            'critical': 'critical',
            'virial': 'critical'}
        self.hdpm_dict = {'nfw': ccl.halos.HaloProfileNFW}
        self.cosmo_class = CCLCosmology
        # Uncomment lines below when CCL einasto and hernquist profiles are stable
        # (also add version number)
        # if version.parse(ccl.__version__) >= version.parse('???'):
        #    self.hdpm_dict.update({
        #        'einasto': ccl.halos.HaloProfileEinasto,
        #        'hernquist': ccl.halos.HaloProfileHernquist})
        # Attributes exclusive to this class
        self.hdpm_opts = {'nfw': {'truncated': False,
                                  'projected_analytic': True,
                                  'cumul2d_analytic': True},
                          'einasto': {},
                          'hernquist': {}}
        self.mdelta = 0.0
        self.cor_factor = _patch_rho_crit_to_cd2018(
            ccl.physical_constants.RHO_CRITICAL)
        # Set halo profile and cosmology
        self.set_halo_density_profile(halo_profile_model, massdef, delta_mdef)
        self.set_cosmo(None)


    def _set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        """"set halo density profile"""
        # Check if we have already an instance of the required object, if not create one
        if not ((halo_profile_model==self.halo_profile_model)
                and (massdef == self.massdef)
                and (delta_mdef == self.delta_mdef)):
            self.halo_profile_model = halo_profile_model
            self.massdef = massdef

            cur_cdelta = 0.0
            cur_values = False
            if self.hdpm:
                cur_cdelta = self.conc.c
                cur_values = True

            self.mdef = ccl.halos.MassDef(delta_mdef, self.mdef_dict[massdef])
            self.conc = ccl.halos.ConcentrationConstant(c=4.0, mdef=self.mdef)
            self.mdef.concentration = self.conc
            self.hdpm = self.hdpm_dict[halo_profile_model](
                self.conc, **self.hdpm_opts[halo_profile_model])
            if cur_values:
                self.conc.c = cur_cdelta

    def _set_concentration(self, cdelta):
        """" set concentration"""
        self.conc.c = cdelta

    def _set_mass(self, mdelta):
        """" set mass"""
        self.mdelta = mdelta/self.cor_factor

    def _eval_3d_density(self, r3d, z_cl):
        """"eval 3d density"""
        a_cl = self.cosmo.get_a_from_z(z_cl)
        dens = self.hdpm.real(
            self.cosmo.be_cosmo, r3d/a_cl, self.mdelta, a_cl, self.mdef)
        return dens*self.cor_factor/a_cl**3

    def _eval_surface_density(self, r_proj, z_cl):
        """"eval surface density"""
        a_cl = self.cosmo.get_a_from_z(z_cl)
        dens = self.hdpm.projected(
            self.cosmo.be_cosmo, r_proj/a_cl, self.mdelta, a_cl, self.mdef)
        return dens*self.cor_factor/a_cl**2

    def _eval_mean_surface_density(self, r_proj, z_cl):
        """"eval mean surface density"""
        a_cl = self.cosmo.get_a_from_z(z_cl)
        dens = self.hdpm.cumul2d(
            self.cosmo.be_cosmo, r_proj/a_cl, self.mdelta,
            self.cosmo.get_a_from_z(z_cl), self.mdef)
        return dens*self.cor_factor/a_cl**2

    def _eval_excess_surface_density(self, r_proj, z_cl):
        """"eval excess surface density"""
        a_cl = self.cosmo.get_a_from_z(z_cl)
        args = (self.cosmo.be_cosmo, r_proj/a_cl, self.mdelta, a_cl, self.mdef)
        mean_dens = self.hdpm.cumul2d(*args)
        dens = self.hdpm.projected(*args)
        return (mean_dens-dens)*self.cor_factor/a_cl**2
    
    def _eval_excess_surface_density_2h(self, r_proj, z_cl , b , lsteps = 500 ):
        """"eval excess surface density 2-halo term"""
        
        Da = ccl.angular_diameter_distance( self.cosmo.be_cosmo, 1, 1./(1. + z_cl))  
        # Msun/Mpc**3
        rho_m = ccl.rho_x( self.cosmo.be_cosmo, 
                           1./(1. + z_cl), 
                           'matter', 
                           is_comoving = False )
        
        kk = np.logspace(-5.,5.,1000)

        pk = ccl.linear_matter_power( self.cosmo.be_cosmo , kk, 1./(1.+z_cl) )
        interp_pk = interp1d( kk, pk, kind='cubic' )

        theta = r_proj / Da

        # calculate integral, units [Mpc]**-3
        def __integrand__( l , theta ):

            k = l / ((1 + z_cl) * Da)      
            return l * jv( 2 , l * theta ) * interp_pk( k )
        
        ll = np.logspace( 0 , 6 , lsteps )

        val = np.array( [ simps( __integrand__( ll , t ) , ll ) for t in theta ] )

        return b * val * rho_m / ( 2 * np.pi  * ( 1 + z_cl )**3 * Da**2 )


Modeling = CCLCLMModeling
