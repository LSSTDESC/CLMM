"""
Tests for datatype and galaxycluster
"""

from numpy.testing import assert_raises, assert_equal, assert_warns, assert_no_warnings

from clmm import GCData
from clmm import Cosmology


def test_init():
    """test init"""
    gcdata = GCData()
    assert_equal(None, gcdata.meta["cosmo"])
    assert_equal("euclidean", gcdata.meta["coordinate_system"])

    # assert that default coordinate_system warning is raised
    assert_warns(UserWarning, GCData)

    # assert that no warning is raised when coordinate_system is set
    assert_no_warnings(GCData, meta={"coordinate_system": "celestial"})
    assert_no_warnings(GCData, copy_indices=False)
    assert_no_warnings(GCData, masked=False)
    assert_no_warnings(GCData, copy=False)

    # assert errros are raised when coordinate_system is not valid
    assert_raises(ValueError, GCData, meta={"coordinate_system": "other"})
    assert_raises(TypeError, GCData, meta={"coordinate_system": 2})
    assert_raises(TypeError, GCData, meta={"coordinate_system": None})
    assert_raises(TypeError, GCData, meta={"coordinate_system": ["celestial", 2]})


def test_update_cosmo():
    """test update cosmo"""
    # Define inputs
    cosmo1 = Cosmology(H0=70.0, Omega_dm0=0.3 - 0.045, Omega_b0=0.045)
    desc1 = cosmo1.get_desc()
    gcdata = GCData()
    # check it has __str__ and __repr__
    assert isinstance(gcdata.__str__(), str)
    assert isinstance(gcdata.__repr__(), str)
    assert isinstance(gcdata._repr_html_(), str)
    assert gcdata._str_pzpdf_info() is None
    # manual update
    gcdata.update_cosmo_ext_valid(gcdata, cosmo1, overwrite=False)
    assert_equal(desc1, gcdata.meta["cosmo"])
    # check that adding cosmo metadata manually is forbidden
    assert_raises(ValueError, gcdata.meta.__setitem__, "cosmo", None)
    assert_raises(ValueError, gcdata.meta.__setitem__, "cosmo", cosmo1)
    # update_cosmo funcs
    # input_cosmo=None, data_cosmo=None
    gcdata = GCData()
    gcdata.update_cosmo_ext_valid(gcdata, None, overwrite=False)
    assert_equal(None, gcdata.meta["cosmo"])

    gcdata = GCData()
    gcdata.update_cosmo_ext_valid(gcdata, None, overwrite=True)
    assert_equal(None, gcdata.meta["cosmo"])

    gcdata = GCData()
    gcdata.update_cosmo(None, overwrite=False)
    assert_equal(None, gcdata.meta["cosmo"])

    gcdata = GCData()
    gcdata.update_cosmo(None, overwrite=True)
    assert_equal(None, gcdata.meta["cosmo"])

    # input_cosmo!=None, data_cosmo=None
    gcdata = GCData()
    gcdata.update_cosmo_ext_valid(gcdata, cosmo1, overwrite=True)
    assert_equal(desc1, gcdata.meta["cosmo"])

    for overwrite in (True, False):
        gcdata = GCData()
        gcdata.update_cosmo(cosmo1, overwrite=overwrite)
        assert_equal(desc1, gcdata.meta["cosmo"])

    # input_cosmo=data_cosmo!=None
    for overwrite in (True, False):
        gcdata = GCData()
        gcdata.update_cosmo(cosmo1)
        gcdata.update_cosmo_ext_valid(gcdata, cosmo1, overwrite=overwrite)
        assert_equal(desc1, gcdata.meta["cosmo"])

        gcdata = GCData()
        gcdata.update_cosmo(cosmo1)
        gcdata.update_cosmo(cosmo1, overwrite=overwrite)
        assert_equal(desc1, gcdata.meta["cosmo"])

    # input_cosmo(!=None) != data_cosmo(!=None)
    cosmo2 = Cosmology(H0=60.0, Omega_dm0=0.3 - 0.045, Omega_b0=0.045)
    desc2 = cosmo2.get_desc()

    gcdata = GCData()
    gcdata.update_cosmo(cosmo1)
    assert_raises(ValueError, gcdata.update_cosmo_ext_valid, gcdata, cosmo2, overwrite=False)
    assert_raises(ValueError, gcdata.update_cosmo_ext_valid, gcdata, cosmo2)

    gcdata = GCData()
    gcdata.update_cosmo(cosmo1)
    gcdata.update_cosmo_ext_valid(gcdata, cosmo2, overwrite=True)
    assert_equal(desc2, gcdata.meta["cosmo"])

    gcdata = GCData()
    gcdata.update_cosmo(cosmo1)
    gcdata.update_cosmo(cosmo1, overwrite=False)
    assert_equal(desc1, gcdata.meta["cosmo"])

    gcdata = GCData()
    gcdata.update_cosmo(cosmo1)
    assert_raises(ValueError, gcdata.update_cosmo, cosmo2, overwrite=False)
    assert_raises(ValueError, gcdata.update_cosmo, cosmo2)

    # Test casing for colnames and meta
    gcdata = GCData()
    gcdata.update_cosmo(cosmo1)
    for key in ("cosmo", "COSMO", "Cosmo"):
        assert_equal(desc1, gcdata.meta[key])

    gcdata["Ra"] = [1]
    for key in ("Ra", "ra", "RA"):
        assert_equal(1, gcdata[key][0])


def test_pzfuncs():
    ngals = 5
    zbins = [0.3, 0.5, 0.8]
    pzpdf = [[0.1, 0.5, 0.1]] * ngals

    # no pdf
    gcdata = GCData()
    assert not gcdata.has_pzpdfs()
    assert_raises(ValueError, gcdata.get_pzpdfs)
    assert gcdata._str_pzpdf_info() is None
    assert isinstance(gcdata.__str__(), str)
    assert isinstance(gcdata.__repr__(), str)
    assert isinstance(gcdata._repr_html_(), str)

    # shared bins
    gcdata = GCData()
    gcdata.pzpdf_info["type"] = "shared_bins"
    assert not gcdata.has_pzpdfs()
    gcdata.pzpdf_info["zbins"] = zbins
    gcdata["pzpdf"] = pzpdf
    assert gcdata.has_pzpdfs()
    assert isinstance(gcdata._str_pzpdf_info(), str)
    assert isinstance(gcdata.__str__(), str)
    assert isinstance(gcdata.__repr__(), str)
    assert isinstance(gcdata._repr_html_(), str)

    # unique bins
    gcdata = GCData()
    gcdata.pzpdf_info["type"] = "individual_bins"
    assert not gcdata.has_pzpdfs()
    gcdata["pzbins"] = [zbins] * ngals
    gcdata["pzpdf"] = pzpdf
    assert gcdata.has_pzpdfs()
    assert isinstance(gcdata._str_pzpdf_info(), str)
    assert isinstance(gcdata.__str__(), str)
    assert isinstance(gcdata.__repr__(), str)
    assert isinstance(gcdata._repr_html_(), str)

    # quantiles
    gcdata = GCData()
    gcdata.pzpdf_info["type"] = "quantiles"
    assert not gcdata.has_pzpdfs()
    gcdata.pzpdf_info["quantiles"] = (0.16, 0.5, 0.84)
    gcdata["pzquantiles"] = [[0.1, 0.2, 0.3]] * ngals
    assert gcdata.has_pzpdfs()
    assert isinstance(gcdata._str_pzpdf_info(), str)
    assert isinstance(gcdata.__str__(), str)
    assert isinstance(gcdata.__repr__(), str)
    assert isinstance(gcdata._repr_html_(), str)

    # not implemented
    gcdata = GCData()
    gcdata.pzpdf_info["type"] = "other"
    assert isinstance(gcdata._str_pzpdf_info(), str)
    assert isinstance(gcdata.__str__(), str)
    assert isinstance(gcdata.__repr__(), str)
    assert isinstance(gcdata._repr_html_(), str)
    assert_raises(NotImplementedError, gcdata.has_pzpdfs)
    assert_raises(NotImplementedError, gcdata.get_pzpdfs)


def test_update_coordinate_system():
    ngals = 5
    e1 = [0.1] * ngals
    e2 = [0.2] * ngals
    ex = [0.3] * ngals

    # raise error if not updated from update_coordinate_system
    gcdata = GCData(meta={"coordinate_system": "celestial"})
    gcdata["e1"] = e1
    gcdata["e2"] = e2
    gcdata["ex"] = ex
    assert_raises(ValueError, gcdata.meta.__setitem__, "coordinate_system", "euclidean")

    # raise error if column is not present
    gcdata = GCData(meta={"coordinate_system": "euclidean"})
    gcdata["e1"] = e1
    gcdata["e2"] = e2
    gcdata["ex"] = ex
    assert_raises(ValueError, gcdata.update_coordinate_system, "celestial", "et")

    # update coordinate system with bogus system
    gcdata = GCData(meta={"coordinate_system": "celestial"})
    gcdata["e1"] = e1
    gcdata["e2"] = e2
    gcdata["ex"] = ex
    assert_raises(ValueError, gcdata.update_coordinate_system, "other")
    assert_raises(TypeError, gcdata.update_coordinate_system, 2)
    assert_raises(TypeError, gcdata.update_coordinate_system, None)
    assert_raises(TypeError, gcdata.update_coordinate_system, ["celestial", 2])

    # update coordinate system without conversion
    gcdata = GCData(meta={"coordinate_system": "euclidean"})
    gcdata["e1"] = e1
    gcdata["e2"] = e2
    gcdata["ex"] = ex
    assert_warns(UserWarning, gcdata.update_coordinate_system, "celestial")
    assert_equal("celestial", gcdata.meta["coordinate_system"])
    assert_equal(e1, gcdata["e1"])
    assert_equal(e2, gcdata["e2"])
    assert_equal(ex, gcdata["ex"])

    # update coordinate system to the same system
    gcdata = GCData(meta={"coordinate_system": "celestial"})
    gcdata["e1"] = e1
    gcdata["e2"] = e2
    gcdata["ex"] = ex
    assert_no_warnings(gcdata.update_coordinate_system, "celestial")
    assert_equal("celestial", gcdata.meta["coordinate_system"])
    assert_no_warnings(gcdata.update_coordinate_system, "celestial", "e2", "ex")
    assert_equal(e1, gcdata["e1"])
    assert_equal(e2, gcdata["e2"])
    assert_equal(ex, gcdata["ex"])

    # update coordinate system with conversion
    gcdata = GCData(meta={"coordinate_system": "celestial"})
    gcdata["e1"] = e1
    gcdata["e2"] = e2
    gcdata["ex"] = ex
    assert_warns(UserWarning, gcdata.update_coordinate_system, "euclidean", "e2", "ex")
    assert_equal("euclidean", gcdata.meta["coordinate_system"])
    assert_equal(e1, gcdata["e1"])
    assert_equal(e2, -gcdata["e2"])
    assert_equal(ex, -gcdata["ex"])
