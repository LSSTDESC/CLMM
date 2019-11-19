"""
Tests for io.py
"""

from numpy import testing
import clmm

def test_types():
    testing.assert_raises(TypeError, clmm.lsst.load_from_dc2_rect, 1.5, 'cosmoDC2_v1.1.4_small', '.')
    testing.assert_raises(TypeError, clmm.lsst.load_from_dc2_rect, 5, 10, '.')
    testing.assert_raises(TypeError, clmm.lsst.load_from_dc2_rect, 5, 'cosmoDC2_v1.1.4_small', 10)
    
    testing.assert_raises(TypeError, clmm.lsst.load_from_dc2_rect, 5, 'cosmoDC2_v1.1.4_small', '.',
                          (-5, 0.3), (-0.3, 0.3), (0.1, 1.5))
    testing.assert_raises(TypeError, clmm.lsst.load_from_dc2_rect, 5, 'cosmoDC2_v1.1.4_small', '.',
                          (-0.3, 0.3), (-0.3, 0.3, 0.1), (0.1, 1.5))
    
    testing.assert_raises(TypeError, clmm.lsst.load_from_dc2_rect, 5, 'cosmoDC2_v1.1.4_small', '.',
                          (-0.3, 0.3), (-0.3, 0.3), (0.1, 1.5), verbose=1.3)
    
def test_ranges():
    
    testing.assert_raises(ValueError, clmm.lsst.load_from_dc2_rect, -1, 'cosmoDC2_v1.1.4_small', '.')
    
    testing.assert_raises(ValueError, clmm.lsst.load_from_dc2_rect, 5, 'cosmoDC2_v1.1.4_small', '.',
                          (0.3, -0.3), (-0.3, 0.3), (0.1, 1.5))
    testing.assert_raises(ValueError, clmm.lsst.load_from_dc2_rect, 5, 'cosmoDC2_v1.1.4_small', '.',
                          (0.3, -0.3), (-0.3, 0.3), (0.1, 1.5))
    testing.assert_raises(ValueError, clmm.lsst.load_from_dc2_rect, 5, 'cosmoDC2_v1.1.4_small', '.',
                          (-361., 0.3), (-0.3, 0.3), (0.1, 1.5))
    testing.assert_raises(ValueError, clmm.lsst.load_from_dc2_rect, 5, 'cosmoDC2_v1.1.4_small', '.',
                          (-0.3, 0.3), (-0.3, 95.), (0.1, 1.5))
    
    # can't check these because Travis doesn't have access to GCR
    # testing.assert_raises(ValueError, clmm.lsst.load_from_dc2_rect, 5, 'not_a_catalog', '.')
    # testing.assert_raises(ValueError, clmm.lsst.load_from_dc2_rect, int(1e20), 'cosmoDC2_v1.1.4_small', '.')

if __name__ == "__main__":
    test_types()
    test_ranges()