"""
Tests for datatype and galaxycluster
"""

import os
import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_allclose, assert_warns
import clmm
from clmm import GCData
from scipy.stats import multivariate_normal

TOLERANCE = {"rtol": 1.0e-7, "atol": 1.0e-7}


def test_initialization():
    """test initialization"""
    testdict1 = {
        "unique_id": "1",
        "ra": 161.3,
        "dec": 34.0,
        "z": 0.3,
        "galcat": GCData(),
        "coordinate_system": "pixel",
    }
    cl1 = clmm.GalaxyCluster(**testdict1)

    assert_equal(testdict1["unique_id"], cl1.unique_id)
    assert_equal(testdict1["ra"], cl1.ra)
    assert_equal(testdict1["dec"], cl1.dec)
    assert_equal(testdict1["z"], cl1.z)
    assert isinstance(cl1.galcat, GCData)
    assert_equal(testdict1["coordinate_system"], cl1.coordinate_system)


def test_integrity():  # Converge on name
    """test integrity"""  # Converge on name
    # Ensure we have all necessary values to make a GalaxyCluster
    assert_raises(TypeError, clmm.GalaxyCluster, ra=161.3, dec=34.0, z=0.3, galcat=GCData())
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, dec=34.0, z=0.3, galcat=GCData())
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=161.3, z=0.3, galcat=GCData())
    assert_raises(TypeError, clmm.GalaxyCluster, unique_id=1, ra=161.3, dec=34.0, galcat=GCData())

    # Test that we get errors when we pass in values outside of the domains
    assert_raises(
        ValueError,
        clmm.GalaxyCluster,
        unique_id=1,
        ra=-360.3,
        dec=34.0,
        z=0.3,
        galcat=GCData(),
        coordinate_system="pixel",
    )
    assert_raises(
        ValueError,
        clmm.GalaxyCluster,
        unique_id=1,
        ra=360.3,
        dec=34.0,
        z=0.3,
        galcat=GCData(),
        coordinate_system="pixel",
    )
    assert_raises(
        ValueError,
        clmm.GalaxyCluster,
        unique_id=1,
        ra=161.3,
        dec=95.0,
        z=0.3,
        galcat=GCData(),
        coordinate_system="pixel",
    )
    assert_raises(
        ValueError,
        clmm.GalaxyCluster,
        unique_id=1,
        ra=161.3,
        dec=-95.0,
        z=0.3,
        galcat=GCData(),
        coordinate_system="pixel",
    )
    assert_raises(
        ValueError,
        clmm.GalaxyCluster,
        unique_id=1,
        ra=161.3,
        dec=34.0,
        z=-0.3,
        galcat=GCData(),
        coordinate_system="pixel",
    )
    assert_raises(
        ValueError,
        clmm.GalaxyCluster,
        unique_id=1,
        ra=161.3,
        dec=34.0,
        z=0.3,
        galcat=GCData(),
        coordinate_system="blah",
    )

    # Test that inputs are the correct type
    assert_raises(
        TypeError,
        clmm.GalaxyCluster,
        unique_id=None,
        ra=161.3,
        dec=34.0,
        z=0.3,
        galcat=GCData(),
        coordinate_system="pixel",
    )
    assert_raises(
        TypeError,
        clmm.GalaxyCluster,
        unique_id=1,
        ra=161.3,
        dec=34.0,
        z=0.3,
        galcat=1,
        coordinate_system="pixel",
    )
    assert_raises(
        TypeError,
        clmm.GalaxyCluster,
        unique_id=1,
        ra=161.3,
        dec=34.0,
        z=0.3,
        galcat=[],
        coordinate_system="pixel",
    )
    assert_raises(
        TypeError,
        clmm.GalaxyCluster,
        unique_id=1,
        ra=None,
        dec=34.0,
        z=0.3,
        galcat=GCData(),
        coordinate_system="pixel",
    )
    assert_raises(
        TypeError,
        clmm.GalaxyCluster,
        unique_id=1,
        ra=161.3,
        dec=None,
        z=0.3,
        galcat=GCData(),
        coordinate_system="pixel",
    )
    assert_raises(
        TypeError,
        clmm.GalaxyCluster,
        unique_id=1,
        ra=161.3,
        dec=34.0,
        z=None,
        galcat=GCData(),
        coordinate_system="pixel",
    )
    assert_raises(
        TypeError,
        clmm.GalaxyCluster,
        unique_id=1,
        ra=161.3,
        dec=34.0,
        z=0.3,
        galcat=None,
        coordinate_system=2,
    )

    # Test that id can support numbers and strings
    assert isinstance(
        clmm.GalaxyCluster(
            unique_id=1, ra=161.3, dec=34.0, z=0.3, galcat=GCData(), coordinate_system="pixel"
        ).unique_id,
        str,
    )
    assert isinstance(
        clmm.GalaxyCluster(
            unique_id="1", ra=161.3, dec=34.0, z=0.3, galcat=GCData(), coordinate_system="pixel"
        ).unique_id,
        str,
    )

    # Test that ra/dec/z can be converted from int/str to float if needed
    assert clmm.GalaxyCluster("1", "161.", "55.", ".3", GCData())
    assert clmm.GalaxyCluster("1", 161, 55, 1, GCData())

    # Test default ra_min=0
    cl = clmm.GalaxyCluster("1", -10, 55, 1, GCData({"ra": [-10.0], "dec": [0.0]}))
    assert_equal(cl.ra, 350)
    assert_equal(cl.galcat["ra"], [350])

    # Test set_galcat_ra_lower
    assert_raises(ValueError, cl.set_ra_lower, 7)
    cl.set_ra_lower(-180)
    assert_equal(cl.ra, -10)
    assert_equal(cl.galcat["ra"], [-10])


def test_save_load():
    """test save load"""
    cl1 = clmm.GalaxyCluster(unique_id="1", ra=161.3, dec=34.0, z=0.3, galcat=GCData())
    cl1.save("testcluster.pkl")
    cl2 = clmm.GalaxyCluster.load("testcluster.pkl")
    os.system("rm testcluster.pkl")

    assert_equal(cl2.unique_id, cl1.unique_id)
    assert_equal(cl2.ra, cl1.ra)
    assert_equal(cl2.dec, cl1.dec)
    assert_equal(cl2.z, cl1.z)

    # remeber to add tests for the tables of the cluster


# def test_find_data():
#     """test find data"""
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
    """test print gc"""
    # Cluster with empty galcat
    cluster = clmm.GalaxyCluster(unique_id="1", ra=161.3, dec=34.0, z=0.3, galcat=GCData())
    print(cluster)
    assert isinstance(cluster.__str__(), str)
    assert isinstance(cluster.__repr__(), str)
    assert isinstance(cluster._repr_html_(), str)
    # Cluster with galcat
    galcat = GCData(
        [[120.1, 119.9, 119.9], [41.9, 42.2, 42.2], [1, 1, 1], [1, 2, 3]],
        names=("ra", "dec", "z", "id"),
    )
    cluster = clmm.GalaxyCluster(unique_id="1", ra=161.3, dec=34.0, z=0.3, galcat=galcat)
    print(cluster)
    assert isinstance(cluster.__str__(), str)
    assert isinstance(cluster.__repr__(), str)


def test_integrity_of_lensfuncs():
    """test integrity of lensfuncs"""
    ra_source, dec_source = [120.1, 119.9, 119.9], [41.9, 42.2, 42.2]
    id_source, z_src = [1, 2, 3], [1, 1, 1]
    shape_component1 = np.array([0.143, 0.063, -0.171])
    shape_component2 = np.array([-0.011, 0.012, -0.250])

    galcat = GCData(
        [ra_source, dec_source, z_src, id_source, shape_component1, shape_component2],
        names=("ra", "dec", "z", "id", "e1", "e2"),
    )
    galcat_noz = GCData([ra_source, dec_source, id_source], names=("ra", "dec", "id"))
    cosmo = clmm.Cosmology(H0=70.0, Omega_dm0=0.275, Omega_b0=0.025)

    # Missing cosmo
    cluster = clmm.GalaxyCluster(unique_id="1", ra=161.3, dec=34.0, z=0.3, galcat=galcat)
    assert_raises(TypeError, cluster.add_critical_surface_density, None)
    # Missing cl redshift
    cluster = clmm.GalaxyCluster(unique_id="1", ra=161.3, dec=34.0, z=0.3, galcat=galcat)
    cluster.z = None
    assert_raises(TypeError, cluster.add_critical_surface_density, cosmo)
    # Missing galaxy redshift
    cluster = clmm.GalaxyCluster(unique_id="1", ra=161.3, dec=34.0, z=0.3, galcat=galcat_noz)
    assert_raises(TypeError, cluster.add_critical_surface_density, cosmo)
    # Missing galaxy pdf if use_pdz is true
    cluster = clmm.GalaxyCluster(unique_id="1", ra=161.3, dec=34.0, z=0.3, galcat=galcat_noz)
    assert_raises(TypeError, cluster.add_critical_surface_density, cosmo, use_pdz=True)
    # Check metadata addition
    pzbins = np.linspace(0.0001, 5, 100)
    cluster = clmm.GalaxyCluster(unique_id="1", ra=161.3, dec=34.0, z=0.3, galcat=galcat)
    cluster.galcat.pzpdf_info["zbins"] = pzbins
    cluster.galcat["pzbins"] = [pzbins for i in range(len(z_src))]
    cluster.galcat["pzpdf"] = [multivariate_normal.pdf(pzbins, mean=z, cov=0.3) for z in z_src]

    for pztype in ("individual_bins", "shared_bins"):
        cluster.galcat.pzpdf_info["type"] = pztype

        cluster.compute_tangential_and_cross_components(
            is_deltasigma=True, use_pdz=True, cosmo=cosmo, add=True
        )
        for comp_name in ("et", "ex"):
            assert_equal(cluster.galcat.meta[f"{comp_name}_sigmac_type"], "effective")


def test_integrity_of_probfuncs():
    """test integrity of prob funcs"""
    ra_source, dec_source = [120.1, 119.9, 119.9], [41.9, 42.2, 42.2]
    id_source, z_srcs = [1, 2, 3], [1, 1, 1]
    cluster = clmm.GalaxyCluster(
        unique_id="1",
        ra=161.3,
        dec=34.0,
        z=0.3,
        galcat=GCData([ra_source, dec_source, z_srcs, id_source], names=("ra", "dec", "z", "id")),
    )
    # true redshift
    cluster.compute_background_probability(use_pdz=False, p_background_name="p_bkg_true")
    expected = np.array([1.0, 1.0, 1.0])
    assert_allclose(cluster.galcat["p_bkg_true"], expected, **TOLERANCE)

    # photoz + deltasigma
    assert_raises(TypeError, cluster.compute_background_probability, use_photoz=True)
    pzbins = np.linspace(0.0001, 5, 1000)
    cluster.galcat.pzpdf_info["zbins"] = pzbins
    cluster.galcat["pzbins"] = [pzbins for i in range(len(z_srcs))]
    cluster.galcat["pzpdf"] = [multivariate_normal.pdf(pzbins, mean=z, cov=0.01) for z in z_srcs]
    for pztype in ("individual_bins", "shared_bins"):
        cluster.galcat.pzpdf_info["type"] = pztype
        cluster.compute_background_probability(use_pdz=True, p_background_name="p_bkg_pz")
        assert_allclose(cluster.galcat["p_bkg_pz"], expected, **TOLERANCE)


def test_integrity_of_weightfuncs():
    """test integrity of weight funcs"""
    cosmo = clmm.Cosmology(H0=71.0, Omega_dm0=0.265 - 0.0448, Omega_b0=0.0448, Omega_k0=0.0)
    z_lens = 0.1
    z_src = [0.22, 0.35, 1.7]
    shape_component1 = np.array([0.143, 0.063, -0.171])
    shape_component2 = np.array([-0.011, 0.012, -0.250])
    shape_component1_err = np.array([0.11, 0.01, 0.2])
    shape_component2_err = np.array([0.14, 0.16, 0.21])
    p_background = np.array([1.0, 1.0, 1.0])
    cluster = clmm.GalaxyCluster(
        unique_id="1",
        ra=161.3,
        dec=34.0,
        z=z_lens,
        galcat=GCData(
            [
                shape_component1,
                shape_component2,
                shape_component1_err,
                shape_component2_err,
                z_src,
            ],
            names=("e1", "e2", "e1_err", "e2_err", "z"),
        ),
    )

    # true redshift + deltasigma
    cluster.compute_galaxy_weights(cosmo=cosmo, use_shape_noise=False, is_deltasigma=True)
    expected = np.array([4.58644320e-31, 9.68145632e-31, 5.07260777e-31])
    assert_allclose(cluster.galcat["w_ls"] * 1e20, expected * 1e20, **TOLERANCE)

    # photoz + deltasigma
    pzbins = np.linspace(0.0001, 5, 100)
    cluster.galcat.pzpdf_info["zbins"] = pzbins
    cluster.galcat["pzbins"] = [pzbins for i in range(len(z_src))]
    cluster.galcat["pzpdf"] = [multivariate_normal.pdf(pzbins, mean=z, cov=0.3) for z in z_src]
    for pztype in ("individual_bins", "shared_bins"):
        cluster.galcat.pzpdf_info["type"] = pztype
        cluster.compute_galaxy_weights(
            cosmo=cosmo, use_shape_noise=False, use_pdz=True, is_deltasigma=True
        )
        expected = np.array([9.07709345e-33, 1.28167582e-32, 4.16870389e-32])
        assert_allclose(cluster.galcat["w_ls"] * 1e20, expected * 1e20, **TOLERANCE)

        # test with noise
        cluster.compute_galaxy_weights(
            cosmo=cosmo,
            use_shape_noise=True,
            use_pdz=True,
            use_shape_error=True,
            is_deltasigma=True,
        )

        expected = np.array([9.07709345e-33, 1.28167582e-32, 4.16870389e-32])
        assert_allclose(cluster.galcat["w_ls"] * 1e20, expected * 1e20, **TOLERANCE)


def test_pzpdf_random_draw():
    """test draw_gal_z_from_pdz"""
    z_lens = 0.1
    z_src = [0.22, 0.35, 1.7]
    shape_component1 = np.array([0.143, 0.063, -0.171])
    shape_component2 = np.array([-0.011, 0.012, -0.250])
    cluster_kwargs = dict(unique_id="1", ra=161.3, dec=34.0, z=z_lens)
    gcat_args = [shape_component1, shape_component2, z_src]
    gcat_kwargs = {"names": ("e1", "e2", "z")}

    # set up photoz
    pzbins = np.linspace(0.0001, 5, 100)
    # test raising TypeError when required column is no available
    for pztype in ("individual_bins", "shared_bins"):
        cluster = clmm.GalaxyCluster(**cluster_kwargs, galcat=GCData(gcat_args, **gcat_kwargs))
        cluster.galcat.pzpdf_info["type"] = pztype

        assert_raises(TypeError, cluster.draw_gal_z_from_pdz)

        cluster.galcat.pzpdf_info["zbins"] = pzbins
        cluster.galcat["pzbins"] = [pzbins for i in range(len(z_src))]
        assert_raises(TypeError, cluster.draw_gal_z_from_pdz)

        cluster.galcat.pzpdf_info.pop("zbins")
        cluster.galcat.remove_column("pzbins")
        cluster.galcat["pzpdf"] = [multivariate_normal.pdf(pzbins, mean=z, cov=0.3) for z in z_src]
        assert_raises(TypeError, cluster.draw_gal_z_from_pdz)

        # add pzbins back to galcat
        cluster.galcat.pzpdf_info["zbins"] = pzbins
        cluster.galcat["pzbins"] = [pzbins for i in range(len(z_src))]
        # test raising TypeError when the name of the new column is already in cluster.galcat
        # also test default overwrite=False and zcol_out='z'
        assert_raises(TypeError, cluster.draw_gal_z_from_pdz)
        # Test raising warnings when xmin<min(pzbins) and xmax>max(pzbins)
        assert_warns(
            UserWarning, cluster.draw_gal_z_from_pdz, zcol_out="z_test", xmin=pzbins.min() / 10
        )
        cluster.galcat.remove_column("z_test")
        assert_warns(
            UserWarning, cluster.draw_gal_z_from_pdz, zcol_out="z_test", xmax=pzbins.max() * 10
        )
        # test drawing 1 object from the whole range of pzpdf
        np.random.seed(0)
        cluster.draw_gal_z_from_pdz(zcol_out="z_random")
        assert_allclose(
            cluster.galcat["z_random"].data,
            [[0.514074], [0.791846], [1.843482]],
            rtol=1e-6,
            atol=1e-6,
        )

        # test drawing nobj objects and specifying xmin and xmax
        # also test overwrite=True
        np.random.seed(0)
        cluster.draw_gal_z_from_pdz(
            zcol_out="z_random", overwrite=True, nobj=2, xmin=pzbins[50], xmax=pzbins[51]
        )
        assert_allclose(
            cluster.galcat["z_random"].data,
            [[2.553019, 2.561422], [2.555744, 2.552821], [2.546698, 2.557922]],
            rtol=1e-6,
            atol=1e-6,
        )

    # test raise errors with unkown pdf type
    cluster.galcat.pzpdf_info["type"] = None
    cluster.galcat.remove_column("z_random")
    assert_raises(TypeError, cluster.draw_gal_z_from_pdz, zcol_out="z_random", nobj=2)
    cluster.galcat.pzpdf_info["type"] = "quantile"
    assert_raises(NotImplementedError, cluster.draw_gal_z_from_pdz, zcol_out="z_random", nobj=2)


def test_plot_profiles():
    """test plot profiles"""
    # Input values
    ra_lens, dec_lens, z_lens = 120.0, 42.0, 0.5
    ra_source = [120.1, 119.9]
    dec_source = [41.9, 42.2]
    z_src = [1.0, 2.0]
    shear1 = [0.2, 0.4]
    shear2 = [0.3, 0.5]
    # Set up radial values
    bins_radians = [0.002, 0.003, 0.004]
    bin_units = "radians"
    # create cluster
    cluster = clmm.GalaxyCluster(
        unique_id="test",
        ra=ra_lens,
        dec=dec_lens,
        z=z_lens,
        galcat=GCData(
            [ra_source, dec_source, shear1, shear2, z_src], names=("ra", "dec", "e1", "e2", "z")
        ),
    )
    cluster.compute_tangential_and_cross_components()
    cluster.make_radial_profile(bin_units, bins=bins_radians, include_empty_bins=True)
    # missing profile name
    assert_raises(ValueError, cluster.plot_profiles, table_name="made_up_table")
    # missing shear component
    assert_raises(ValueError, cluster.plot_profiles, cross_component="made_up_component")
    # check basic plot is working
    cluster.plot_profiles()
    # check it passes missing a component error
    cluster.plot_profiles(cross_component_error="made_up_component")


def test_coordinate_system():
    """test coordinate system"""
    # Input values
    ra_lens, dec_lens, z_lens = 120.0, 42.0, 0.5
    ra_source = [120.1, 119.9]
    dec_source = [41.9, 42.2]
    z_src = [1.0, 2.0]
    shear1 = [0.2, 0.4]
    shear2_pixel = [0.3, 0.5]
    shear2_sky = [-0.3, -0.5]
    # Set up radial values
    bins_radians = [0.002, 0.003, 0.004]
    bin_units = "radians"
    # create cluster
    cl_pixel = clmm.GalaxyCluster(
        unique_id="test",
        ra=ra_lens,
        dec=dec_lens,
        z=z_lens,
        galcat=GCData(
            [ra_source, dec_source, shear1, shear2_pixel, z_src],
            names=("ra", "dec", "e1", "e2", "z"),
        ),
        coordinate_system="pixel",
    )
    cl_sky = clmm.GalaxyCluster(
        unique_id="test",
        ra=ra_lens,
        dec=dec_lens,
        z=z_lens,
        galcat=GCData(
            [ra_source, dec_source, shear1, shear2_sky, z_src], names=("ra", "dec", "e1", "e2", "z")
        ),
        coordinate_system="sky",
    )

    cl_pixel.compute_tangential_and_cross_components()
    cl_sky.compute_tangential_and_cross_components()

    assert_allclose(
        cl_pixel.galcat["et"],
        cl_sky.galcat["et"],
        **TOLERANCE,
        err_msg="Tangential component conversion between ellipticity coordinate systems failed",
    )
    assert_allclose(
        cl_pixel.galcat["ex"],
        -cl_sky.galcat["ex"],
        **TOLERANCE,
        err_msg="Cross component conversion between ellipticity coordinate systems failed",
    )
