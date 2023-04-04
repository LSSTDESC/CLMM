"""Tests for examples/support/sampler.py"""
from numpy.testing import assert_allclose

from clmm.support.sampler import samplers, fitters


def test_samplers():
    """test samplers"""
    test_func = lambda x, a: (x + a) ** 2
    assert_allclose(samplers["minimize"](test_func, 0, args=[-1]), 1, 1e-3)
    assert_allclose(
        samplers["basinhopping"](test_func, 0, minimizer_kwargs={"args": [-1]}), 1, 1e-3
    )
    assert_allclose(fitters["curve_fit"](test_func, [0, 0], [1, 1], [0.01, 0.01])[0], 1, 1e-3)
