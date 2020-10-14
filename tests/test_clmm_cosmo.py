"""Tests for modeling.py"""
import json
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
import clmm.modeling as md
from clmm.clmm_cosmo import CLMMCosmology

def test_unimplemented (modeling_data):
    """ Unit tests abstract class unimplemented methdods """

    cosmo = md.Cosmology ()

    assert_raises (NotImplementedError, CLMMCosmology._init_from_cosmo,  None, None)
    assert_raises (NotImplementedError, CLMMCosmology._init_from_params, None)
    assert_raises (NotImplementedError, CLMMCosmology._set_param,        None, None, None)
    assert_raises (NotImplementedError, CLMMCosmology._get_param,        None, None)
    assert_raises (AttributeError,      CLMMCosmology.set_be_cosmo,      None, None)
    assert_raises (NotImplementedError, CLMMCosmology.get_Omega_m,       None, None)
    assert_raises (NotImplementedError, CLMMCosmology.eval_da_z1z2,      None, None, None)
    assert_raises (AttributeError,      CLMMCosmology.eval_da,           None, None)

TOLERANCE = {'rtol': 1.0e-15}

def test_z_and_a (modeling_data):
    """ Unit tests abstract class z and a methdods """

    cosmo = md.Cosmology ()
    
    z = np.linspace (0.0, 10.0, 1000)
    
    assert_raises (ValueError, cosmo._get_a_from_z, z - 1.0)
    
    a = cosmo._get_a_from_z (z)
    
    assert_raises (ValueError, cosmo._get_z_from_a, a * 2.0)
    
    z_cpy = cosmo._get_z_from_a (a)
    
    assert_allclose (z_cpy, z, **TOLERANCE)

    a_cpy = cosmo._get_a_from_z (z_cpy)
    
    assert_allclose (a_cpy, a, **TOLERANCE)

    # Convert from a to z - scalar, list, ndarray
    assert_allclose(cosmo._get_a_from_z(0.5), 2./3., **TOLERANCE)
    assert_allclose(cosmo._get_a_from_z([0.1, 0.2, 0.3, 0.4]),
                    [10./11., 5./6., 10./13., 5./7.], **TOLERANCE)
    assert_allclose(cosmo._get_a_from_z(np.array([0.1, 0.2, 0.3, 0.4])),
                    np.array([10./11., 5./6., 10./13., 5./7.]), **TOLERANCE)

    # Convert from z to a - scalar, list, ndarray
    assert_allclose(cosmo._get_z_from_a(2./3.), 0.5, **TOLERANCE)
    assert_allclose(cosmo._get_z_from_a([10./11., 5./6., 10./13., 5./7.]),
                    [0.1, 0.2, 0.3, 0.4], **TOLERANCE)
    assert_allclose(cosmo._get_z_from_a(np.array([10./11., 5./6., 10./13., 5./7.])),
                    np.array([0.1, 0.2, 0.3, 0.4]), **TOLERANCE)

    # Some potential corner-cases for the two funcs
    assert_allclose(cosmo._get_a_from_z(np.array([0.0, 1300.])),
                    np.array ([1.0, 1./1301.]), **TOLERANCE)
    assert_allclose(cosmo._get_z_from_a(np.array([1.0, 1./1301.])),
                    np.array ([0.0, 1300.]), **TOLERANCE)

    # Test for exceptions when outside of domains
    assert_raises(ValueError, cosmo._get_a_from_z, -5.0)
    assert_raises(ValueError, cosmo._get_a_from_z, [-5.0, 5.0])
    assert_raises(ValueError, cosmo._get_a_from_z, np.array([-5.0, 5.0]))
    assert_raises(ValueError, cosmo._get_z_from_a, 5.0)
    assert_raises(ValueError, cosmo._get_z_from_a, [-5.0, 5.0])
    assert_raises(ValueError, cosmo._get_z_from_a, np.array([-5.0, 5.0]))

    # Convert from a to z to a (and vice versa)
    testval = 0.5
    assert_allclose (cosmo._get_a_from_z (cosmo._get_z_from_a (testval)), testval, **TOLERANCE)
    assert_allclose (cosmo._get_z_from_a (cosmo._get_a_from_z (testval)), testval, **TOLERANCE)
    