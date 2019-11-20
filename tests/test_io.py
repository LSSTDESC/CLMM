"""
Tests for io.py
"""

from numpy import testing
import clmm

def test_types():
    testing.assert_raises(TypeError, clmm.lsst.load_from_dc2, 1.5, 'cosmoDC2_v1.1.4_small',
                          '.', _reader='test')
    testing.assert_raises(TypeError, clmm.lsst.load_from_dc2, 5, 'cosmoDC2_v1.1.4_small',
                          10, _reader='test')
    testing.assert_raises(TypeError, clmm.lsst.load_from_dc2, 5, 'cosmoDC2_v1.1.4_small', '.',
                          (-0.3, 0.3), (-0.3, 0.3), (0.1, 1.5), verbose=1.3, _reader='test')
    
def test_ranges():
    
    testing.assert_raises(ValueError, clmm.lsst.load_from_dc2, -1, 'cosmoDC2_v1.1.4_small', '.',
                          _reader='test')
    testing.assert_raises(ValueError, clmm.lsst.load_from_dc2, int(1e10), 'cosmoDC2_v1.1.4_small', '.',
                          _reader='test')
    testing.assert_raises(ValueError, clmm.lsst.load_from_dc2, 5, 'cosmoDC2_v1.1.4_small', '.',
                          (-0.3, 0.3), (-0.3, 0.3, 0.1), (0.1, 1.5), _reader='test')
    testing.assert_raises(ValueError, clmm.lsst.load_from_dc2, 5, 'cosmoDC2_v1.1.4_small', '.',
                          (0.3, -0.3), (-0.3, 0.3), (0.1, 1.5), _reader='test')