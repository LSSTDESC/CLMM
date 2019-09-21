"""
Tests for polaraveraging.py
"""
import clmm
import clmm.polaraveraging as pa
from numpy import testing
import numpy as np

import pytest

def test_theta_units_conversion():
    # tests for invaid input
    testing.assert_raises(ValueError, pa._theta_units_conversion, np.pi, 'radians', 'crazy units')
    testing.assert_raises(ValueError, pa._theta_units_conversion, np.pi, 'crazy units', 'radians')
    testing.assert_raises(ValueError, pa._theta_units_conversion, np.pi, 'radians', 'Mpc')
    testing.assert_raises(ValueError, pa._theta_units_conversion, np.pi, 'radians', 'Mpc', 0.5)

    # tests for angular conversion
    testing.assert_equal(pa._theta_units_conversion(np.pi, 'radians', 'radians'), np.pi)
    testing.assert_almost_equal(pa._theta_units_conversion(np.pi, 'radians', 'deg'), 180.)
    testing.assert_almost_equal(pa._theta_units_conversion(np.pi, 'radians', 'arcmin'), 180.*60)
    testing.assert_almost_equal(pa._theta_units_conversion(np.pi, 'radians', 'arcsec'), 180.*60*60)
    testing.assert_almost_equal(pa._theta_units_conversion(180., 'deg', 'radians'), np.pi)

    # Test for radians to kpc

    # Test for radians to Mpc

    # THIS WILL NOT SUPPORT PHYSICAL UNITS TO ANGULAR UNITS!!!
