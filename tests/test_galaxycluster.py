"""
Tests for datatype and galaxycluster
"""
from numpy import testing
import clmm
from astropy.table import Table
import os

def test_initialization():
    testdict1 = {'unique_id': '1', 'ra': 161.3, 'dec': 34., 'z': 0.3, 'galcat': Table()}
    cl1 = clmm.GalaxyCluster(**testdict1)

    testing.assert_equal(testdict1['unique_id'], cl1.unique_id)
    testing.assert_equal(testdict1['ra'], cl1.ra)
    testing.assert_equal(testdict1['dec'], cl1.dec)
    testing.assert_equal(testdict1['z'], cl1.z)
    assert isinstance(cl1.galcat, Table)


def test_integrity(): # Converge on name
    # Ensure we have all necessary values to make a GalaxyCluster
    testing.assert_raises(TypeError, clmm.GalaxyCluster, ra=161.3, dec=34., z=0.3, galcat=Table())
    testing.assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, dec=34., z=0.3, galcat=Table())
    testing.assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=161.3, z=0.3, galcat=Table())
    testing.assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=34., galcat=Table())

    # Test that we get errors when we pass in values outside of the domains
    testing.assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1, ra=-360.3, dec=34., z=0.3, galcat=Table())
    testing.assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1, ra=360.3, dec=34., z=0.3, galcat=Table())
    testing.assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=95., z=0.3, galcat=Table())
    testing.assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=-95., z=0.3, galcat=Table())
    testing.assert_raises(ValueError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=34., z=-0.3, galcat=Table())

    # Test that inputs are the correct type
    testing.assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=34., z=0.3, galcat=1)
    testing.assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=34., z=0.3, galcat=[])
    testing.assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=None, dec=34., z=0.3, galcat=Table())
    testing.assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=None, z=0.3, galcat=Table())
    testing.assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=34., z=None, galcat=Table())

    # Test that id can support numbers and strings
    assert isinstance(clmm.GalaxyCluster(unique_id=1, ra=161.3, dec=34., z=0.3, galcat=Table()).unique_id, str)
    # assert clmm.GalaxyCluster(unique_id=1.0, ra=161.3, dec=34., z=0.3, galcat=Table()).unique_id == '1'
    assert isinstance(clmm.GalaxyCluster(unique_id='1', ra=161.3, dec=34., z=0.3, galcat=Table()).unique_id, str)

    # Test that ra/dec/z can be converted from int/str to float if needed
    assert clmm.GalaxyCluster('1', '161.', '55.', '.3', Table())
    assert clmm.GalaxyCluster('1', 161, 55, 1, Table())

def test_save_load():
    cl1 = clmm.GalaxyCluster(unique_id='1', ra=161.3, dec=34., z=0.3, galcat=Table())
    cl1.save('testcluster.pkl')
    cl2 = clmm.load_cluster('testcluster.pkl')
    os.system('rm testcluster.pkl')

    testing.assert_equal(cl2.unique_id, cl1.unique_id)
    testing.assert_equal(cl2.ra, cl1.ra)
    testing.assert_equal(cl2.dec, cl1.dec)
    testing.assert_equal(cl2.z, cl1.z)

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
    cl = clmm.GalaxyCluster(unique_id='1', ra=161.3, dec=34., z=0.3, galcat=Table())
    print(cl)
    assert isinstance(cl.__str__(), str)

if __name__ == "__main__":
    test_initialization()
    test_integrity()
    test_print_cl()
