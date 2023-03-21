"""
Tests for datatype and galaxycluster
"""
from numpy.testing import assert_raises, assert_equal

from clmm import GCData
from clmm import Cosmology


def test_init():
    """test init"""
    gcdata = GCData()
    assert_equal(None, gcdata.meta['cosmo'])


def test_update_cosmo():
    """test update cosmo"""
    # Define inputs
    cosmo1 = Cosmology(H0=70.0, Omega_dm0=0.3-0.045, Omega_b0=0.045)
    desc1 = cosmo1.get_desc()
    gcdata = GCData()
    # check it has __str__ and __repr__
    assert isinstance(gcdata.__str__(), str)
    assert isinstance(gcdata.__repr__(), str)
    assert isinstance(gcdata._repr_html_(), str)
    # manual update
    gcdata.update_cosmo_ext_valid(gcdata, cosmo1, overwrite=False)
    assert_equal(desc1, gcdata.meta['cosmo'])
    # check that adding cosmo metadata manually is forbidden
    assert_raises(ValueError, gcdata.meta.__setitem__, 'cosmo', None)
    assert_raises(ValueError, gcdata.meta.__setitem__, 'cosmo', cosmo1)
    # update_cosmo funcs
    # input_cosmo=None, data_cosmo=None
    gcdata = GCData()
    gcdata.update_cosmo_ext_valid(gcdata, None, overwrite=False)
    assert_equal(None, gcdata.meta['cosmo'])

    gcdata = GCData()
    gcdata.update_cosmo_ext_valid(gcdata, None, overwrite=True)
    assert_equal(None, gcdata.meta['cosmo'])

    gcdata = GCData()
    gcdata.update_cosmo(None, overwrite=False)
    assert_equal(None, gcdata.meta['cosmo'])

    gcdata = GCData()
    gcdata.update_cosmo(None, overwrite=True)
    assert_equal(None, gcdata.meta['cosmo'])

    # input_cosmo!=None, data_cosmo=None
    gcdata = GCData()
    gcdata.update_cosmo_ext_valid(gcdata, cosmo1, overwrite=True)
    assert_equal(desc1, gcdata.meta['cosmo'])

    for overwrite in (True, False):
        gcdata = GCData()
        gcdata.update_cosmo(cosmo1, overwrite=overwrite)
        assert_equal(desc1, gcdata.meta['cosmo'])

    # input_cosmo=data_cosmo!=None
    for overwrite in (True, False):
        gcdata = GCData()
        gcdata.update_cosmo(cosmo1)
        gcdata.update_cosmo_ext_valid(gcdata, cosmo1, overwrite=overwrite)
        assert_equal(desc1, gcdata.meta['cosmo'])

        gcdata = GCData()
        gcdata.update_cosmo(cosmo1)
        gcdata.update_cosmo(cosmo1, overwrite=overwrite)
        assert_equal(desc1, gcdata.meta['cosmo'])

    # input_cosmo(!=None) != data_cosmo(!=None)
    cosmo2 = Cosmology(H0=60.0, Omega_dm0=0.3-0.045, Omega_b0=0.045)
    desc2 = cosmo2.get_desc()

    gcdata = GCData()
    gcdata.update_cosmo(cosmo1)
    assert_raises(ValueError, gcdata.update_cosmo_ext_valid,
                  gcdata, cosmo2, overwrite=False)
    assert_raises(ValueError, gcdata.update_cosmo_ext_valid, gcdata, cosmo2)

    gcdata = GCData()
    gcdata.update_cosmo(cosmo1)
    gcdata.update_cosmo_ext_valid(gcdata, cosmo2, overwrite=True)
    assert_equal(desc2, gcdata.meta['cosmo'])

    gcdata = GCData()
    gcdata.update_cosmo(cosmo1)
    gcdata.update_cosmo(cosmo1, overwrite=False)
    assert_equal(desc1, gcdata.meta['cosmo'])

    gcdata = GCData()
    gcdata.update_cosmo(cosmo1)
    assert_raises(ValueError, gcdata.update_cosmo, cosmo2, overwrite=False)
    assert_raises(ValueError, gcdata.update_cosmo, cosmo2)

    # Test casing for colnames and meta
    gcdata = GCData()
    gcdata.update_cosmo(cosmo1)
    for key in ('cosmo', 'COSMO', 'Cosmo'):
        assert_equal(desc1, gcdata.meta[key])

    gcdata['Ra'] = [1]
    for key in ('Ra', 'ra', 'RA',):
        assert_equal(1, gcdata[key][0])

def test_pzfuncs():

    ngals = 5
    zbins = [.3, .5, .8]
    pzpdf = [[.1, .5, .1]]*ngals

    # no pdf
    gcdata = GCData()
    assert not gcdata.has_pzpdfs()
    assert_raises(ValueError, gcdata.get_pzpdfs)

    # shared bins
    gcdata = GCData()
    gcdata.pzpdf_info['type'] = 'shared_bins'
    assert not gcdata.has_pzpdfs()
    gcdata.pzpdf_info['zbins'] = zbins
    gcdata['pzpdf'] = pzpdf
    assert gcdata.has_pzpdfs()

    # unique bins
    gcdata = GCData()
    gcdata.pzpdf_info['type'] = 'individual_bins'
    assert not gcdata.has_pzpdfs()
    gcdata['pzbins'] = [zbins]*ngals
    gcdata['pzpdf'] = pzpdf
    assert gcdata.has_pzpdfs()

    # not implemented
    gcdata = GCData()
    gcdata.pzpdf_info['type'] = 'other'
    assert_raises(NotImplementedError, gcdata.has_pzpdfs)
    assert_raises(NotImplementedError, gcdata.get_pzpdfs)

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
    # tst.assert_raises(ValueError, gc.add_data, '')
    # tst.assert_raises(ValueError, gc.add_data, '', force=True)
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
