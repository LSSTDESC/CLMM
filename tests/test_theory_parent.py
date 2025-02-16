"""Tests for modeling.py"""

import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
import clmm.theory as theo
from clmm.theory.parent_class import CLMModeling


def test_unimplemented(modeling_data):
    """Unit tests abstract class unimplemented methdods"""

    mod = CLMModeling()

    assert_raises(NotImplementedError, mod.set_cosmo, None)
    assert_raises(NotImplementedError, mod._get_mass)
    assert_raises(NotImplementedError, mod._get_concentration)
    assert_raises(NotImplementedError, mod.set_mass, 1.0e15)
    assert_raises(NotImplementedError, mod._set_mass, 1.0e15)
    assert_raises(NotImplementedError, mod.set_concentration, 4.0)
    assert_raises(NotImplementedError, mod._set_concentration, 4.0)
    assert_raises(NotImplementedError, mod._update_halo_density_profile)
    assert_raises(NotImplementedError, mod._set_einasto_alpha, 0.5)
    assert_raises(NotImplementedError, mod._get_einasto_alpha)
    assert_raises(NotImplementedError, mod._get_delta_mdef_virial, 0.1)
    assert_raises(NotImplementedError, mod._set_projected_quad, True)
    assert_raises(NotImplementedError, mod.eval_3d_density, [0.3], 0.3)
    assert_raises(NotImplementedError, mod.eval_surface_density, [0.3], 0.3)
    assert_raises(NotImplementedError, mod.eval_mean_surface_density, [0.3], 0.3)
    assert_raises(NotImplementedError, mod.eval_excess_surface_density, [0.3], 0.3)
    assert_raises(NotImplementedError, mod.eval_tangential_shear, [0.3], 0.3, 0.5)
    assert_raises(NotImplementedError, mod.eval_convergence, [0.3], 0.3, 0.5)
    assert_raises(NotImplementedError, mod.eval_reduced_tangential_shear, [0.3], 0.3, 0.5)
    assert_raises(
        NotImplementedError,
        mod.eval_reduced_tangential_shear,
        [0.3],
        0.3,
        (0.6, 0.4),
        "beta",
        "order1",
    )
    assert_raises(NotImplementedError, mod.eval_magnification, [0.3], 0.3, 0.5)
    assert_raises(
        NotImplementedError, mod.eval_magnification, [0.3], 0.3, (0.6, 0.4), "beta", "order1"
    )
    assert_raises(NotImplementedError, mod.eval_magnification_bias, [0.3], 0.3, 0.5, 3.0)
    assert_raises(
        NotImplementedError,
        mod.eval_magnification_bias,
        [0.3],
        0.3,
        (0.6, 0.4),
        3.0,
        "beta",
        "order1",
    )


def test_instantiate(modeling_data):
    """Unit tests for modeling objects' instantiation"""

    mod = theo.Modeling()

    # test set_x funcs and self.xdelta funcs are equivalent
    cdelta, mdelta = 3.0, 0.5e15
    halo_profile_model, massdef, delta_mdef = "nfw", "mean", 300
    mod.set_concentration(cdelta)
    mod.set_mass(mdelta)
    mod.set_halo_density_profile(halo_profile_model, massdef, delta_mdef)
    assert_allclose(mod.cdelta, cdelta, 1e-15)
    assert_allclose(mod.mdelta, mdelta, 1e-15)
    assert_allclose(mod.delta_mdef, delta_mdef, 1e-15)
    assert_equal(mod.massdef, massdef)
    assert_equal(mod.halo_profile_model, halo_profile_model)

    cdelta, mdelta = 4.0, 1.0e15
    halo_profile_model, massdef, delta_mdef = "nfw", "mean", 200
    mod.cdelta = cdelta
    mod.mdelta = mdelta
    mod.massdef = massdef
    mod.delta_mdef = delta_mdef
    mod.halo_profile_model = halo_profile_model
    assert_allclose(mod.cdelta, cdelta, 1e-15)
    assert_allclose(mod.mdelta, mdelta, 1e-15)
    assert_allclose(mod.delta_mdef, delta_mdef, 1e-15)
    assert_equal(mod.massdef, massdef)
    assert_equal(mod.halo_profile_model, halo_profile_model)

    # check backend
    assert mod.backend == theo.be_nick

    assert_raises(TypeError, mod.set_cosmo, 3.0)
    assert_raises(ValueError, mod.set_halo_density_profile, halo_profile_model="bla")
    assert_raises(ValueError, mod.set_halo_density_profile, massdef="blu")

    if theo.be_nick in ["nc", "ccl"]:
        mod.set_halo_density_profile(massdef="virial")
        assert_equal(mod.massdef, "virial")

        # reset
        mod.massdef = "mean"

        mod.massdef = "virial"
        assert_equal(mod.massdef, "virial")

    if theo.be_nick == "nc":
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

    assert_allclose(sigma_excess, (sigma_mean - sigma), rtol=5.0e-15)

    sigma = mod.eval_surface_density(r_proj[0], z_cl)
    sigma_mean = mod.eval_mean_surface_density(r_proj[0], z_cl)
    sigma_excess = mod.eval_excess_surface_density(r_proj[0], z_cl)

    assert_allclose(sigma_excess, (sigma_mean - sigma), rtol=5.0e-15)

    shear = mod.eval_tangential_shear(r_proj, z_cl, z_src)
    convergence = mod.eval_convergence(r_proj, z_cl, z_src)
    reduced_shear = mod.eval_reduced_tangential_shear(r_proj, z_cl, z_src)
    magnification = mod.eval_magnification(r_proj, z_cl, z_src)

    assert_allclose(reduced_shear, shear / (1.0 - convergence), rtol=1.0e-12)
    assert_allclose(
        magnification, 1.0 / ((1.0 - convergence) ** 2 - np.abs(shear) ** 2), rtol=1.0e-12
    )

    reduced_shear = mod.eval_reduced_tangential_shear(r_proj, z_cl, np.full_like(r_proj, z_src))
    assert_allclose(reduced_shear, shear / (1.0 - convergence), rtol=1.0e-12)


def test_einasto(modeling_data):
    """Basic checks that verbose option for the Einasto profile runs"""

    mod = theo.Modeling()
    mod.set_concentration(4.0)
    mod.set_mass(1.0e15)

    if theo.be_nick in ["ccl", "nc"]:
        mod.set_halo_density_profile("einasto")
        mod.eval_mean_surface_density(0.1, 0.1, verbose=True)
        mod.eval_tangential_shear(0.1, 0.1, 0.5, verbose=True)
        mod.eval_convergence(0.1, 0.1, 0.5, verbose=True)
        mod.eval_reduced_tangential_shear(0.1, 0.1, 0.5, verbose=True)
        mod.eval_magnification(0.1, 0.1, 0.5, verbose=True)
        mod.eval_magnification_bias(0.1, 2, 0.1, 0.5, verbose=True)


def test_set_projected_quad(modeling_data):
    """Test set_projected_quad method"""
    mod = theo.Modeling()
    assert_raises(NotImplementedError, mod.set_projected_quad, True)

    if theo.be_nick == "ccl":
        assert_raises(NotImplementedError, mod.set_projected_quad, True)
        mod.set_halo_density_profile("hernquist")
        assert_raises(NotImplementedError, mod.set_projected_quad, True)
        mod.set_halo_density_profile("einasto")
        mod.set_projected_quad(True)
    else:
        assert_raises(NotImplementedError, mod.set_projected_quad, True)
