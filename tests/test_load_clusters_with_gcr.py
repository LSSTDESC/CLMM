"""
Tests for load_clusters_with_gcr.py
"""
import os
from numpy import testing
import clmm

def test_types():
    testing.assert_raises(TypeError, clmm.lsst.load_from_dc2, 1.5, 'cosmoDC2_v1.1.4_small',
                          '.', _reader='test')
    testing.assert_raises(TypeError, clmm.lsst.load_from_dc2, 5, 'cosmoDC2_v1.1.4_small',
                          10, _reader='test')

def test_ranges():
    testing.assert_raises(ValueError, clmm.lsst.load_from_dc2, -1, 'cosmoDC2_v1.1.4_small', '.',
                          _reader='test')
    testing.assert_raises(ValueError, clmm.lsst.load_from_dc2, int(1e10), 'cosmoDC2_v1.1.4_small', '.',
                          _reader='test')
    testing.assert_raises(ValueError, clmm.lsst.load_from_dc2, 5, 'cosmoDC2_v1.1.4_small', '.',
                          (-0.3, 0.3), (-0.3, 0.3, 0.1), (0.1, 1.5), _reader='test')
    testing.assert_raises(ValueError, clmm.lsst.load_from_dc2, 5, 'cosmoDC2_v1.1.4_small', '.',
                          (0.3, -0.3), (-0.3, 0.3), (0.1, 1.5), _reader='test')

def test_values():
    clmm.lsst.load_from_dc2(10, 'cosmoDC2_v1.1.4_small', '.', _reader='test')
    
    c = clmm.load_cluster('./3.p')
    testing.assert_equal(len(c.galcat), 10)
    testing.assert_equal(c.galcat.columns.keys(), 
                         ['galaxy_id', 'ra', 'dec', 'e1', 'e2', 'z', 'kappa'])
    testing.assert_equal(c.galcat[5]['e1'], 2.)
    testing.assert_equal(c.galcat[4]['z'], 0.4)
    
    for file in os.listdir('.'):
        if file[-2:]=='.p':
            os.remove(file)

