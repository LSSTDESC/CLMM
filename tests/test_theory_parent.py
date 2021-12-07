"""Tests for modeling.py"""

import numpy as np
from numpy.testing import assert_raises, assert_allclose
import clmm.theory as theo
from clmm.theory.parent_class import CLMModeling


def test_unimplemented(modeling_data):
    """ Unit tests abstract class unimplemented methdods """

    mod = CLMModeling()

    assert_raises(NotImplementedError, mod.set_cosmo, None)
    assert_raises(NotImplementedError, mod._set_halo_density_profile)
    assert_raises(NotImplementedError, mod.set_concentration, 4.0)
    assert_raises(NotImplementedError, mod.set_mass, 1.0e15)
    assert_raises(NotImplementedError, mod.eval_3d_density, [0.3], 0.3)
    assert_raises(NotImplementedError, mod.eval_surface_density, [0.3], 0.3)
    assert_raises(NotImplementedError, mod.eval_mean_surface_density, [0.3], 0.3)
    assert_raises(NotImplementedError, mod.eval_excess_surface_density, [0.3], 0.3)
    assert_raises(NotImplementedError, mod.eval_tangential_shear, [0.3], 0.3, 0.5)
    assert_raises(NotImplementedError, mod.eval_reduced_tangential_shear, [0.3], 0.3, 0.5)
    assert_raises(NotImplementedError, mod.eval_reduced_tangential_shear, [0.3], 0.3, 0.5, 'applegate14', 0.6, 0.4)
    assert_raises(NotImplementedError, mod.eval_convergence, [0.3], 0.3, 0.5)
    assert_raises(NotImplementedError, mod.eval_magnification, [0.3], 0.3, 0.5)


def test_instantiate(modeling_data):
    """ Unit tests for modeling objects' instantiation """

    mod = theo.Modeling()
    mod.set_concentration(4.0)
    mod.set_mass(1.0e15)
    assert mod.backend == theo.be_nick

    assert_raises(TypeError, mod.set_cosmo, 3.0)
    assert_raises(ValueError, mod.set_halo_density_profile, halo_profile_model='bla')
    assert_raises(ValueError, mod.set_halo_density_profile, massdef='blu')

    if theo.be_nick == 'nc':
        import gi
        gi.require_version("NumCosmoMath", "1.0")
        gi.require_version("NumCosmo", "1.0")

        import gi.repository.NumCosmoMath as Ncm
        import gi.repository.NumCosmo as Nc

        mset = mod.get_mset()
        assert isinstance(mset, Ncm.MSet)

        mod.set_mset(mset)
        assert_raises(AttributeError, mod.set_mset, 3)

    r_proj = np.logspace(-2, 2, 100)
    z_cl = 0.5
    z_src = 0.85
    sigma = mod.eval_surface_density(r_proj, z_cl)
    sigma_mean = mod.eval_mean_surface_density(r_proj, z_cl)
    sigma_excess = mod.eval_excess_surface_density(r_proj, z_cl)

    assert_allclose(sigma_excess, (sigma_mean-sigma), rtol=5.0e-15)

    shear = mod.eval_tangential_shear(r_proj, z_cl, z_src)
    convergence = mod.eval_convergence(r_proj, z_cl, z_src)
    reduced_shear = mod.eval_reduced_tangential_shear(r_proj, z_cl, z_src)
    magnification = mod.eval_magnification(r_proj, z_cl, z_src)

    assert_allclose(reduced_shear, shear/(1.0-convergence), rtol=1.0e-12)
    assert_allclose(magnification, 1.0/((1.0-convergence)**2-np.abs(shear)**2), rtol=1.0e-12)

    reduced_shear = mod.eval_reduced_tangential_shear(r_proj, z_cl, np.repeat(z_src, len(r_proj)))
    assert_allclose(reduced_shear, shear/(1.0-convergence), rtol=1.0e-12)

    beta_s_mean = 0.9
    beta_s_square_mean = 0.6
    source_redshift_inf = 1000. 
    shear_inf = mod.eval_tangential_shear(r_proj, z_cl, source_redshift_inf)
    convergence_inf = mod.eval_convergence(r_proj, z_cl, source_redshift_inf)
    reduced_shear = mod.eval_reduced_tangential_shear(r_proj, z_cl, z_src, 'applegate14', beta_s_mean, beta_s_square_mean)
    assert_allclose(reduced_shear, beta_s_mean * shear_inf/(1.0 - beta_s_square_mean / beta_s_mean * convergence_inf), rtol=1.0e-12)
    
    reduced_shear = mod.eval_reduced_tangential_shear(r_proj, z_cl, np.repeat(z_src, len(r_proj)), 'applegate14', beta_s_mean, beta_s_square_mean)
    assert_allclose(reduced_shear, beta_s_mean * shear_inf/(1.0 - beta_s_square_mean / beta_s_mean * convergence_inf), rtol=1.0e-12)
