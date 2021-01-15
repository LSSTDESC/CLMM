"""Tests for modeling.py"""

import sys
import json
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
import clmm.theory as theo
from clmm.theory.parent_class import CLMModeling


def test_unimplemented(modeling_data):
    """ Unit tests abstract class unimplemented methdods """

    m = CLMModeling()

    assert_raises(NotImplementedError, m.set_cosmo, None)
    assert_raises(NotImplementedError, m.set_halo_density_profile)
    assert_raises(NotImplementedError, m.set_concentration, 4.0)
    assert_raises(NotImplementedError, m.set_mass, 1.0e15)
    assert_raises(NotImplementedError, m.eval_3d_density, [0.3], 0.3)
    assert_raises(NotImplementedError, m.eval_surface_density, [0.3], 0.3)
    assert_raises(NotImplementedError, m.eval_mean_surface_density, [0.3], 0.3)
    assert_raises(NotImplementedError, m.eval_excess_surface_density, [0.3], 0.3)
    assert_raises(NotImplementedError, m.eval_tangential_shear, [0.3], 0.3, 0.5)
    assert_raises(NotImplementedError, m.eval_reduced_tangential_shear, [0.3], 0.3, 0.5)
    assert_raises(NotImplementedError, m.eval_convergence, [0.3], 0.3, 0.5)


def test_instantiate(modeling_data):
    """ Unit tests for modeling objects' instantiation """

    m = theo.Modeling()
    m.set_concentration(4.0)
    m.set_mass(1.0e15)
    assert m.backend == theo.be_nick

    assert_raises(ValueError, m.set_cosmo, 3.0)
    assert_raises(ValueError, m.set_halo_density_profile, halo_profile_model='bla')
    assert_raises(ValueError, m.set_halo_density_profile, massdef='blu')

    if theo.be_nick == 'nc':
        import gi
        gi.require_version("NumCosmoMath", "1.0")
        gi.require_version("NumCosmo", "1.0")

        import gi.repository.NumCosmoMath as Ncm
        import gi.repository.NumCosmo as Nc

        mset = m.get_mset()
        assert isinstance(mset, Ncm.MSet)

        m.set_mset(mset)
        assert_raises(AttributeError, m.set_mset, 3)

    r_proj = np.logspace(-2, 2, 100)
    z_cl = 0.5
    z_src = 0.85
    Sigma = m.eval_surface_density(r_proj, z_cl)
    Sigma_mean = m.eval_mean_surface_density(r_proj, z_cl)
    Sigma_excess = m.eval_excess_surface_density(r_proj, z_cl)

    assert_allclose(Sigma_excess, (Sigma_mean-Sigma), rtol=5.0e-15)

    shear = m.eval_tangential_shear(r_proj, z_cl, z_src)
    convergence = m.eval_convergence(r_proj, z_cl, z_src)
    reduced_shear = m.eval_reduced_tangential_shear(r_proj, z_cl, z_src)
    magnification = m.eval_magnification(r_proj, z_cl, z_src)

    assert_allclose(reduced_shear, shear/(1.0-convergence), rtol=8.0e-15)
    assert_allclose(magnification, 1.0/((1.0-convergence)**2-np.abs(shear)**2), rtol=5.0e-15)

    reduced_shear = m.eval_reduced_tangential_shear(r_proj, z_cl, np.repeat(z_src, len(r_proj)))
    assert_allclose(reduced_shear, shear/(1.0-convergence), rtol=8.0e-15)
