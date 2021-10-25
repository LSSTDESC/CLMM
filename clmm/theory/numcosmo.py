"""@file numcosmo.py
NumCosmo implementation of CLMModeling
"""
import math
import numpy as np

import gi
from gi.repository import NumCosmoMath as Ncm
from gi.repository import NumCosmo as Nc
gi.require_version('NumCosmo', '1.0')
gi.require_version('NumCosmoMath', '1.0')

from . func_layer import *
from . import func_layer

from .parent_class import CLMModeling

from .. cosmology.numcosmo import NumCosmoCosmology
Cosmology = NumCosmoCosmology

__all__ = ['NumCosmoCLMModeling', 'Modeling', 'Cosmology']+func_layer.__all__


class NumCosmoCLMModeling(CLMModeling):
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

    def __init__(self, massdef='mean', delta_mdef=200, halo_profile_model='nfw'):
        CLMModeling.__init__(self)
        # Update class attributes
        Ncm.cfg_init()
        self.backend = 'nc'
        self.mdef_dict = {
            'mean': Nc.HaloDensityProfileMassDef.MEAN,
            'critical': Nc.HaloDensityProfileMassDef.CRITICAL,
            'virial': Nc.HaloDensityProfileMassDef.VIRIAL}
        self.hdpm_dict = {
            'nfw': Nc.HaloDensityProfileNFW.new,
            'einasto': Nc.HaloDensityProfileEinasto.new,
            'hernquist': Nc.HaloDensityProfileHernquist.new}
        # Set halo profile and cosmology
        self.set_halo_density_profile(halo_profile_model, massdef, delta_mdef)
        self.set_cosmo(None)

    def set_cosmo(self, cosmo):
        """"set cosmo"""
        self._set_cosmo(cosmo, NumCosmoCosmology)

        self.cosmo.smd = Nc.WLSurfaceMassDensity.new(self.cosmo.dist)
        self.cosmo.smd.prepare_if_needed(self.cosmo.be_cosmo)

    def set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        """"set halo density profile"""
        # Check if choices are supported and put in lower case
        massdef, halo_profile_model = self.validate_definitions(
            massdef, halo_profile_model)

        # Check if we have already an instance of the required object, if not create one
        if not((halo_profile_model==self.halo_profile_model)
                and (massdef==self.massdef)
                and (delta_mdef==self.delta_mdef)):
            self.halo_profile_model = halo_profile_model
            self.massdef = massdef

            cur_cdelta = 0.0
            cur_values = False
            if self.hdpm:
                cur_cdelta = self.hdpm.props.cDelta
                cur_log10_mdelta = self.hdpm.props.log10MDelta
                cur_values = True

            self.hdpm = self.hdpm_dict[halo_profile_model](
                self.mdef_dict[massdef], delta_mdef)
            if cur_values:
                self.hdpm.props.cDelta = cur_cdelta
                self.hdpm.props.log10MDelta = cur_log10_mdelta

    def get_mset(self):
        r"""
        Gets a mass set (NumCosmo internal use)
        """
        mset = Ncm.MSet.empty_new()
        mset.set(self.cosmo.be_cosmo)
        mset.set(self.hdpm)
        mset.set(self.cosmo.smd)
        return mset

    def set_mset(self, mset):
        r"""
        Sets a mass set (NumCosmo internal use)
        """
        self.cosmo.set_be_cosmo(mset.get(Nc.HICosmo.id()))

        self.hdpm = mset.get(Nc.HaloDensityProfile.id())
        self.cosmo.smd = mset.get(Nc.WLSurfaceMassDensity.id())

        self.cosmo.smd.prepare_if_needed(self.cosmo.be_cosmo)

    def _set_concentration(self, cdelta):
        """" set concentration"""
        self.hdpm.props.cDelta = cdelta

    def _set_mass(self, mdelta):
        """" set mass"""
        self.hdpm.props.log10MDelta = math.log10(mdelta)

    def eval_3d_density(self, r3d, z_cl, verbose=False):
        """"eval 3d density"""

        if self.halo_profile_model == 'einasto' and verbose:
            # print out the value of einasto 'alpha' parameter
            print(f"Einasto alpha = {self.hdpm.props.alpha}")

        func = lambda r3d, z_cl: self.hdpm.eval_density(
            self.cosmo.be_cosmo, r3d, z_cl)
        return np.vectorize(func)(r3d, z_cl)

    def eval_surface_density(self, r_proj, z_cl, verbose=False):
        """"eval surface density"""

        if self.halo_profile_model == 'einasto' and verbose:
                # print out the value of einasto 'alpha' parameter
                print(f"Einasto alpha = {self.hdpm.props.alpha}")

        self.cosmo.smd.prepare_if_needed(self.cosmo.be_cosmo)
        func = lambda r_proj, z_cl: self.cosmo.smd.sigma(
            self.hdpm, self.cosmo.be_cosmo, r_proj, z_cl)
        return np.vectorize(func)(r_proj, z_cl)

    def eval_mean_surface_density(self, r_proj, z_cl):
        """"eval mean surface density"""

        self.cosmo.smd.prepare_if_needed(self.cosmo.be_cosmo)
        func = lambda r_proj, z_cl: self.cosmo.smd.sigma_mean(
            self.hdpm, self.cosmo.be_cosmo, r_proj, z_cl)
        return np.vectorize(func)(r_proj, z_cl)

    def eval_excess_surface_density(self, r_proj, z_cl):
        """"eval excess surface density"""

        self.cosmo.smd.prepare_if_needed(self.cosmo.be_cosmo)
        func = lambda r_proj, z_cl: self.cosmo.smd.sigma_excess(
            self.hdpm, self.cosmo.be_cosmo, r_proj, z_cl)
        return np.vectorize(func)(r_proj, z_cl)

    def eval_tangential_shear(self, r_proj, z_cl, z_src):
        """"eval tangential shear"""

        self.cosmo.smd.prepare_if_needed(self.cosmo.be_cosmo)
        func = lambda r_proj, z_src, z_cl: self.cosmo.smd.shear(
            self.hdpm, self.cosmo.be_cosmo, r_proj, z_src, z_cl, z_cl)
        return np.vectorize(func)(r_proj, z_src, z_cl)

    def eval_convergence(self, r_proj, z_cl, z_src):
        """"eval convergence"""

        self.cosmo.smd.prepare_if_needed(self.cosmo.be_cosmo)
        func = lambda r_proj, z_src, z_cl: self.cosmo.smd.convergence(
            self.hdpm, self.cosmo.be_cosmo, r_proj, z_src, z_cl, z_cl)
        return np.vectorize(func)(r_proj, z_src, z_cl)

    def eval_reduced_tangential_shear(self, r_proj, z_cl, z_src):
        """"eval reduced tangential shear"""

        self.cosmo.smd.prepare_if_needed(self.cosmo.be_cosmo)
        if (isinstance(r_proj, (list, np.ndarray))
                and isinstance(z_src, (list, np.ndarray))
                and len(r_proj) == len(z_src)):
            func = self.cosmo.smd.reduced_shear_array_equal
        else:
            func = self.cosmo.smd.reduced_shear_array
        return func(self.hdpm, self.cosmo.be_cosmo, np.atleast_1d(r_proj), 1.0,
                    1.0, np.atleast_1d(z_src), z_cl, z_cl)

    def eval_magnification(self, r_proj, z_cl, z_src):
        """"eval magnification"""

        self.cosmo.smd.prepare_if_needed(self.cosmo.be_cosmo)

        func = lambda r_proj, z_src, z_cl: self.cosmo.smd.magnification(
            self.hdpm, self.cosmo.be_cosmo, r_proj, z_src, z_cl, z_cl)
        return np.vectorize(func)(r_proj, z_src, z_cl)

Modeling = NumCosmoCLMModeling
