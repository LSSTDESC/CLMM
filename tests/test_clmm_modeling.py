"""Tests for modeling.py"""
import json
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
import clmm.modeling as md
from clmm.clmm_modeling import CLMModeling

def test_unimplemented (modeling_data):
    """ Unit tests abstract class unimplemented methdods """

    m = CLMModeling ()    

    assert_raises (NotImplementedError, m.set_cosmo,         None)
    assert_raises (NotImplementedError, m.set_halo_density_profile)
    assert_raises (NotImplementedError, m.set_concentration, 4.0)
    assert_raises (NotImplementedError, m.set_mass,          1.0e15)
    assert_raises (NotImplementedError, m.eval_da_z1z2,      0.0, 1.0)
    assert_raises (NotImplementedError, m.eval_sigma_crit,   0.4, 0.5)
    assert_raises (NotImplementedError, m.eval_density,      [0.3], 0.3)
    assert_raises (NotImplementedError, m.eval_sigma,        [0.3], 0.3)
    assert_raises (NotImplementedError, m.eval_sigma_mean,   [0.3], 0.3)
    assert_raises (NotImplementedError, m.eval_sigma_excess, [0.3], 0.3)
    assert_raises (NotImplementedError, m.eval_shear,        [0.3], 0.3, 0.5)

