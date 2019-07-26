"""
Tests for polaraveraging.py
"""
import clmm
import clmm.polaraveraging as pa
from numpy import testing
import numpy as np


def test_theta_units_conversion():
    # tests for invaid input
    testing.assert_raises(ValueError, pa._theta_units_conversion, np.pi, 'crazy units')
    testing.assert_raises(ValueError, pa._theta_units_conversion, np.pi, 'Mpc')
    testing.assert_raises(ValueError, pa._theta_units_conversion, np.pi, 'Mpc', 0.5)

    # tests for angular conversion
    testing.assert_equal(pa._theta_units_conversion(np.pi, "rad"), np.pi)
    testing.assert_almost_equal(pa._theta_units_conversion(np.pi, "deg"), 180.)
    testing.assert_almost_equal(pa._theta_units_conversion(np.pi, "arcmin"), 180.*60)
    testing.assert_almost_equal(pa._theta_units_conversion(np.pi, "arcsec"), 180.*60*60)
