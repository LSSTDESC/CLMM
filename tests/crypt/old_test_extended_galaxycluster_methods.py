"""
Tests for datatype and galaxycluster
"""
from numpy import testing as tst
import clmm

# test_creator = 'Mitch'
# test_creator_diff = 'Witch'

# test_dict = {'test%d'%i:True for i in range(3)}
# test_dict_diff = {'test%d'%i:False for i in range(3)}
# test_dict_sub = {'test%d'%i:True for i in range(2)}

# test_table = []

# test_data = GCData(test_creator, test_dict, test_table)
# test_data_diff = GCData(test_creator, test_dict_diff, test_table)


# def test_check_subdict():
#
#     assert check_subdict(test_dict_sub, test_dict)
#     assert not check_subdict(test_dict, test_dict_sub)
#     assert not check_subdict(test_dict_sub, test_dict_diff)
#
# def test_find_in_datalist():
#
#     tst.assert_equal([test_data], find_in_datalist(test_dict, [test_data]))
#     tst.assert_equal([test_data], find_in_datalist(test_dict_sub, [test_data]))
#     tst.assert_equal([], find_in_datalist(test_dict_diff, [test_data]))
#
#     tst.assert_equal([test_data], find_in_datalist(test_dict, [test_data], exact=True))
#     tst.assert_equal([], find_in_datalist(test_dict_sub, [test_data], exact=True))
    # tst.assert_equal([], find_in_datalist(test_dict_diff, [test_data], exact=True))

# def test_find_data():
#
#     gc = GalaxyCluster('test_cluster', test_data)
#
#     tst.assert_equal([], gc.find_data(test_creator_diff, test_dict))
#
#     tst.assert_equal([test_data], gc.find_data(test_creator, test_dict))
#     tst.assert_equal([test_data], gc.find_data(test_creator, test_dict_sub))
#     tst.assert_equal([], gc.find_data(test_creator, test_dict_diff))
#
#     tst.assert_equal([test_data], gc.find_data(test_creator, test_dict, exact=True))
    # tst.assert_equal([], gc.find_data(test_creator, test_dict_sub, exact=True))
    # tst.assert_equal([], gc.find_data(test_creator, test_dict_diff, exact=True))

# def test_add_data():

    # gc = GalaxyCluster('test_cluster')
    # tst.assert_raises(TypeError, gc.add_data, '')
    # tst.assert_raises(TypeError, gc.add_data, '', force=True)
    # tst.assert_equal(None, gc.add_data(test_data, force=True))

    # gc = GalaxyCluster('test_cluster')
    # tst.assert_equal(None, gc.add_data(test_data))
    # tst.assert_equal(None, gc.add_data(test_data_diff))
    # tst.assert_raises(ValueError, gc.add_data, test_data)
    # tst.assert_equal(None, gc.add_data(test_data, force=True))
    #
# def test_remove_data():
#
#     gc = GalaxyCluster('test_cluster', test_data)
#     tst.assert_raises(ValueError, gc.remove_data, test_creator_diff, test_dict)
#     tst.assert_raises(ValueError, gc.remove_data, test_creator, test_dict_sub)
#     tst.assert_raises(ValueError, gc.remove_data, test_creator, test_dict_diff)
#     tst.assert_equal(None, gc.remove_data(test_creator, test_dict))
#     tst.assert_raises(ValueError, gc.remove_data, test_creator, test_dict)
#
# def test_read_GC():
#     pass

# def test_write_GC():
#     pass
