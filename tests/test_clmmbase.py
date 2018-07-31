"""Tests for the ultra super class"""
from clmm import __CLMMBase

def test_constructor():
    t0 = __CLMMBase()
    t1 = __CLMMBase()
    t1._ask_type = 'something'
    t2 = __CLMMBase()

    np.testing.assert_raises(TypeError, t2._ask_type, True)
    assert t0._ask_type == ''
    assert t1._ask_type == 'something'
