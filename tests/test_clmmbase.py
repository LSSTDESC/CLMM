"""Tests for the ultra super class"""
import numpy as np
import inspect, sys
# Tests require the entire package
import clmm

def test_constructor():
    """
    Verify that the constructor for __CLMMBase is working.
    """
    t0 = clmm.CLMMBase()
    assert t0.ask_type == None

    t1 = clmm.CLMMBase()
    t1.ask_type = ['something']
    np.testing.assert_array_equal(t1.ask_type, ['something'])

    t2 = clmm.CLMMBase()
    np.testing.assert_raises(TypeError, t2.ask_type, 3.14159)


