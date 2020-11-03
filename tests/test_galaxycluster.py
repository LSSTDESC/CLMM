"""
Tests for datatype and galaxycluster
"""
from numpy.testing import assert_raises, assert_equal
import clmm
from clmm import GCData
import os

def test_initialization():
    testdict1 = {'unique_id': '1', 'ra': 161.3, 'dec': 34., 'z': 0.3, 'galcat': GCData()}
    cl1 = clmm.GalaxyCluster(**testdict1)

    assert_equal(testdict1['unique_id'], cl1.unique_id)
    assert_equal(testdict1['ra'], cl1.ra)
    assert_equal(testdict1['dec'], cl1.dec)
    assert_equal(testdict1['z'], cl1.z)
    assert isinstance(cl1.galcat, GCData)


def test_integrity(): # Converge on name
    # Ensure we have all necessary values to make a GalaxyCluster
    assert_raises(TypeError, clmm.GalaxyCluster, ra=161.3, dec=34., z=0.3, galcat=GCData())
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, dec=34., z=0.3, galcat=GCData())
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=161.3, z=0.3, galcat=GCData())
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=34., galcat=GCData())

    # Test that we get errors when we pass in values outside of the domains
    assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1, ra=-360.3, dec=34., z=0.3, galcat=GCData())
    assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1, ra=360.3, dec=34., z=0.3, galcat=GCData())
    assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=95., z=0.3, galcat=GCData())
    assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=-95., z=0.3, galcat=GCData())
    assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=34., z=-0.3, galcat=GCData())

    # Test that inputs are the correct type
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=34., z=0.3, galcat=1)
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=34., z=0.3, galcat=[])
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=None, dec=34., z=0.3, galcat=GCData())
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=None, z=0.3, galcat=GCData())
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=34., z=None, galcat=GCData())

    # Test that id can support numbers and strings
    assert isinstance(clmm.GalaxyCluster(unique_id=1, ra=161.3, dec=34., z=0.3, galcat=GCData()).unique_id, str)
    # assert clmm.GalaxyCluster(unique_id=1.0, ra=161.3, dec=34., z=0.3, galcat=GCData()).unique_id == '1'
    assert isinstance(clmm.GalaxyCluster(unique_id='1', ra=161.3, dec=34., z=0.3, galcat=GCData()).unique_id, str)

    # Test that ra/dec/z can be converted from int/str to float if needed
    assert clmm.GalaxyCluster('1', '161.', '55.', '.3', GCData())
    assert clmm.GalaxyCluster('1', 161, 55, 1, GCData())

def test_save_load():
    cl1 = clmm.GalaxyCluster(unique_id='1', ra=161.3, dec=34., z=0.3, galcat=GCData())
    cl1.save('testcluster.pkl')
    cl2 = clmm.GalaxyCluster.load('testcluster.pkl')
    os.system('rm testcluster.pkl')

    assert_equal(cl2.unique_id, cl1.unique_id)
    assert_equal(cl2.ra, cl1.ra)
    assert_equal(cl2.dec, cl1.dec)
    assert_equal(cl2.z, cl1.z)

    # remeber to add tests for the tables of the cluster


# def test_find_data():
#     gc = GalaxyCluster('test_cluster', test_data)
#
#     tst.assert_equal([], gc.find_data(test_creator_diff, test_dict))
#
#     tst.assert_equal([test_data], gc.find_data(test_creator, test_dict))
#     tst.assert_equal([test_data], gc.find_data(test_creator, test_dict_sub))
#     tst.assert_equal([], gc.find_data(test_creator, test_dict_diff))
#
#     tst.assert_equal([test_data], gc.find_data(test_creator, test_dict, exact=True))
#     tst.assert_equal([], gc.find_data(test_creator, test_dict_sub, exact=True))
#     tst.assert_equal([], gc.find_data(test_creator, test_dict_diff, exact=True))

def test_print_gc():
    cl = clmm.GalaxyCluster(unique_id='1', ra=161.3, dec=34., z=0.3, galcat=GCData())
    print(cl)
    assert isinstance(cl.__str__(), str)

if __name__ == "__main__":
    test_initialization()
    test_integrity()
    test_print_cl()
