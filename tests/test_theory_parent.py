"""Tests for modeling.py"""

import numpy as np
from numpy.testing import assert_raises, assert_allclose
import clmm.theory as theo
from clmm.theory.parent_class import CLMModeling
from clmm.utils import compute_beta_s_square_mean, compute_beta_s_mean

def test_unimplemented(modeling_data):
    """ Unit tests abstract class unimplemented methdods """

    mod = CLMModeling()

    assert_raises(NotImplementedError, mod.set_cosmo, None)
    assert_raises(NotImplementedError, mod._set_halo_density_profile)
    assert_raises(NotImplementedError, mod._set_einasto_alpha, 0.5)
    assert_raises(NotImplementedError, mod._get_einasto_alpha)
    assert_raises(NotImplementedError, mod._set_concentration, 4.0)
    assert_raises(NotImplementedError, mod._set_mass, 1.0e15)
    assert_raises(NotImplementedError, mod.set_concentration, 4.0)
    assert_raises(NotImplementedError, mod._get_mass)
    assert_raises(NotImplementedError, mod._get_concentration)
    assert_raises(NotImplementedError, mod.set_mass, 1.0e15)
    assert_raises(NotImplementedError, mod.eval_3d_density, [0.3], 0.3)
    assert_raises(NotImplementedError, mod.eval_surface_density, [0.3], 0.3)
    assert_raises(NotImplementedError, mod.eval_mean_surface_density, [0.3], 0.3)
    assert_raises(NotImplementedError, mod.eval_excess_surface_density, [0.3], 0.3)
    assert_raises(NotImplementedError, mod.eval_tangential_shear, [0.3], 0.3, 0.5)
    assert_raises(NotImplementedError, mod.eval_reduced_tangential_shear, [0.3], 0.3, 0.5)
    assert_raises(NotImplementedError, mod.eval_reduced_tangential_shear, [0.3], 0.3, 0.5, 'applegate14', 0.6, 0.4)
    assert_raises(NotImplementedError, mod.eval_reduced_tangential_shear, [0.3], 0.3, 0.5, 'schrabback18', 0.6, 0.4)
    assert_raises(NotImplementedError, mod.eval_convergence, [0.3], 0.3, 0.5)
    assert_raises(NotImplementedError, mod.eval_magnification, [0.3], 0.3, 0.5)


def test_instantiate(modeling_data):
    """ Unit tests for modeling objects' instantiation """

    mod = theo.Modeling()

    # test protected attributes
    def temp():
        mod.massdef = None
    assert_raises(AttributeError, temp)
    def temp():
        mod.delta_mdef = None
    assert_raises(AttributeError, temp)
    def temp():
        mod.halo_profile_model = None
    assert_raises(AttributeError, temp)

    # test set_x funcs and self.xdelta funcs are equivalent
    cdelta, mdelta = 3.0, 0.5e15
    mod.set_concentration(cdelta)
    mod.set_mass(mdelta)
    assert_allclose(mod.cdelta, cdelta, 1e-15)
    assert_allclose(mod.mdelta, mdelta, 1e-15)

    cdelta, mdelta = 4.0, 1.0e15
    mod.cdelta = cdelta
    mod.mdelta = mdelta
    assert_allclose(mod.cdelta, cdelta, 1e-15)
    assert_allclose(mod.mdelta, mdelta, 1e-15)

    # check backend
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

    #Tests with pre-fixed beta values
    reduced_shear = mod.eval_reduced_tangential_shear(r_proj, z_cl, z_src, 'applegate14', beta_s_mean, beta_s_square_mean)
    assert_allclose(reduced_shear, beta_s_mean * shear_inf/(1.0 - beta_s_square_mean / beta_s_mean * convergence_inf), rtol=1.0e-12)
    
    reduced_shear = mod.eval_reduced_tangential_shear(r_proj, z_cl, np.repeat(z_src, len(r_proj)), 'applegate14', beta_s_mean, beta_s_square_mean)
    assert_allclose(reduced_shear, beta_s_mean * shear_inf/(1.0 - beta_s_square_mean / beta_s_mean * convergence_inf), rtol=1.0e-12)
    
    reduced_shear = mod.eval_reduced_tangential_shear(r_proj, z_cl, z_src, 'schrabback18', beta_s_mean, beta_s_square_mean)
    assert_allclose(reduced_shear, (1. + (beta_s_square_mean / (beta_s_mean * beta_s_mean) - 1.) * beta_s_mean * convergence_inf) * (beta_s_mean * shear_inf / (1. - beta_s_mean * convergence_inf)), rtol=1.0e-12)

    reduced_shear = mod.eval_reduced_tangential_shear(r_proj, z_cl, np.repeat(z_src, len(r_proj)), 'schrabback18', beta_s_mean, beta_s_square_mean)
    assert_allclose(reduced_shear, (1. + (beta_s_square_mean / (beta_s_mean * beta_s_mean) - 1.) * beta_s_mean * convergence_inf) * (beta_s_mean * shear_inf / (1. - beta_s_mean * convergence_inf)), rtol=1.0e-12)
    
    #Tests where the function computes the beta values
    beta_s_mean = None
    beta_s_square_mean = None
    
    beta_s_square_test = compute_beta_s_square_mean(z_cl, 1000., mod.cosmo)
    beta_s_test = compute_beta_s_mean(z_cl, 1000., mod.cosmo)   
    reduced_shear = mod.eval_reduced_tangential_shear(r_proj, z_cl, z_src, 'applegate14', beta_s_mean, beta_s_square_mean)
    assert_allclose(reduced_shear, beta_s_test* shear_inf/(1.0 - beta_s_square_test / beta_s_test * convergence_inf), rtol=1.0e-12)
    
    reduced_shear = mod.eval_reduced_tangential_shear(r_proj, z_cl, np.repeat(z_src, len(r_proj)), 'applegate14', beta_s_mean, beta_s_square_mean)
    assert_allclose(reduced_shear, beta_s_test * shear_inf/(1.0 - beta_s_square_test / beta_s_test * convergence_inf), rtol=1.0e-12)
    
    reduced_shear = mod.eval_reduced_tangential_shear(r_proj, z_cl, z_src, 'schrabback18', beta_s_mean, beta_s_square_mean)
    assert_allclose(reduced_shear, (1. + (beta_s_square_test / (beta_s_test * beta_s_test) - 1.) * beta_s_test * convergence_inf) * (beta_s_test * shear_inf / (1. - beta_s_test * convergence_inf)), rtol=1.0e-12)
    
    reduced_shear = mod.eval_reduced_tangential_shear(r_proj, z_cl, np.repeat(z_src, len(r_proj)), 'schrabback18', beta_s_mean, beta_s_square_mean)
    assert_allclose(reduced_shear, (1. + (beta_s_square_test / (beta_s_test * beta_s_test) - 1.) * beta_s_test * convergence_inf) * (beta_s_test * shear_inf / (1. - beta_s_test * convergence_inf)), rtol=1.0e-12)

    assert_raises(ValueError, mod.eval_critical_surface_density, z_cl, use_pdz=True)

def test_einasto(modeling_data):
    """ Basic checks that verbose option for the Einasto profile runs """

    mod = theo.Modeling()
    mod.set_concentration(4.0)
    mod.set_mass(1.0e15)

    if theo.be_nick in ['ccl','nc']:
        mod.set_halo_density_profile('einasto')
        mod.eval_mean_surface_density(0.1,0.1, verbose=True)
        mod.eval_tangential_shear(0.1,0.1,0.5, verbose=True)
        mod.eval_convergence(0.1,0.1,0.5, verbose=True)
        mod.eval_reduced_tangential_shear(0.1,0.1,0.5, verbose=True)
        mod.eval_magnification(0.1,0.1,0.5, verbose=True)
#>>>>>>> main
