"""Tests for dataops.py"""

import numpy as np
from numpy.testing import assert_allclose, assert_raises, assert_array_equal, assert_warns

import clmm
from clmm import GCData
from scipy.stats import multivariate_normal
import clmm.dataops as da
from clmm.theory import compute_critical_surface_density_eff

TOLERANCE = {"rtol": 1.0e-7, "atol": 1.0e-7}


def test_compute_cross_shear():
    """test compute cross shear"""
    shear1, shear2, phi = 0.15, 0.08, 0.52
    expected_cross_shear = 0.08886301350787848
    cross_shear = da._compute_cross_shear(shear1, shear2, phi)
    assert_allclose(cross_shear, expected_cross_shear)

    shear1 = np.array([0.15, 0.40])
    shear2 = np.array([0.08, 0.30])
    phi = np.array([0.52, 1.23])
    expected_cross_shear = [0.08886301350787848, 0.48498333705834484]
    cross_shear = da._compute_cross_shear(shear1, shear2, phi)
    assert_allclose(cross_shear, expected_cross_shear)

    # Edge case tests
    assert_allclose(da._compute_cross_shear(100.0, 0.0, 0.0), 0.0, **TOLERANCE)
    assert_allclose(da._compute_cross_shear(100.0, 0.0, np.pi / 2), 0.0, **TOLERANCE)
    assert_allclose(da._compute_cross_shear(0.0, 100.0, 0.0), -100.0, **TOLERANCE)
    assert_allclose(da._compute_cross_shear(0.0, 100.0, np.pi / 2), 100.0, **TOLERANCE)
    assert_allclose(da._compute_cross_shear(0.0, 100.0, np.pi / 4.0), 0.0, **TOLERANCE)
    assert_allclose(da._compute_cross_shear(0.0, 0.0, 0.3), 0.0, **TOLERANCE)


def test_compute_tangential_shear():
    """test compute tangential shear"""
    shear1, shear2, phi = 0.15, 0.08, 0.52
    expected_tangential_shear = -0.14492537676438383
    tangential_shear = da._compute_tangential_shear(shear1, shear2, phi)
    assert_allclose(tangential_shear, expected_tangential_shear)

    shear1 = np.array([0.15, 0.40])
    shear2 = np.array([0.08, 0.30])
    phi = np.array([0.52, 1.23])
    expected_tangential_shear = [-0.14492537676438383, 0.1216189244145496]
    tangential_shear = da._compute_tangential_shear(shear1, shear2, phi)
    assert_allclose(tangential_shear, expected_tangential_shear)

    # test for reasonable values
    assert_allclose(da._compute_tangential_shear(100.0, 0.0, 0.0), -100.0, **TOLERANCE)
    assert_allclose(da._compute_tangential_shear(0.0, 100.0, np.pi / 4.0), -100.0, **TOLERANCE)
    assert_allclose(da._compute_tangential_shear(0.0, 0.0, 0.3), 0.0, **TOLERANCE)


def test_compute_4theta_shear():
    """test compute quadrupole 4theta shear"""
    shear1, shear2, phi = 0.15, 0.08, 0.52
    expected_4theta_shear = -0.003271676989552594
    four_theta_shear = da._compute_4theta_shear(shear1, shear2, phi)
    assert_allclose(four_theta_shear, expected_4theta_shear)

    shear1 = np.array([0.15, 0.40])
    shear2 = np.array([0.08, 0.30])
    phi = np.array([0.52, 1.23])
    expected_4theta_shear = [-0.003271676989552594, -0.2111087147687582]
    four_theta_shear = da._compute_4theta_shear(shear1, shear2, phi)
    assert_allclose(four_theta_shear, expected_4theta_shear)

    # test for reasonable values
    assert_allclose(da._compute_4theta_shear(100.0, 0.0, 0.0), 100.0, **TOLERANCE)
    assert_allclose(da._compute_4theta_shear(0.0, 100.0, np.pi / 8.0), 100.0, **TOLERANCE)
    assert_allclose(da._compute_4theta_shear(0.0, 0.0, 0.3), 0.0, **TOLERANCE)


def test_calculate_major_axis():
    """test calculate major axis"""
    ra_lens, dec_lens = 180.0, 0.0

    ra_mem, dec_mem, weight_mem = [180.0, 180.0], [-0.5, 0.5], [1.0, 1.0]
    expected_major_axis = np.pi / 2.0
    assert_allclose(
        da._calculate_major_axis(ra_lens, dec_lens, ra_mem, dec_mem, weight_mem),
        expected_major_axis,
        **TOLERANCE,
    )

    ra_mem, dec_mem, weight_mem = [179.5, 180.5], [0.0, 0.0], [1.0, 1.0]
    expected_major_axis = 0.0
    assert_allclose(
        da._calculate_major_axis(ra_lens, dec_lens, ra_mem, dec_mem, weight_mem),
        expected_major_axis,
        **TOLERANCE,
    )

    ra_mem, dec_mem, weight_mem = [179.99, 180.01], [-0.01, 0.01], [1.0, 1.0]
    expected_major_axis = -np.pi / 4.0
    assert_allclose(
        da._calculate_major_axis(ra_lens, dec_lens, ra_mem, dec_mem, weight_mem),
        expected_major_axis,
        **TOLERANCE,
    )

    ra_mem, dec_mem, weight_mem = [179.99, 180.01], [0.01, -0.01], [1.0, 1.0]
    expected_major_axis = np.pi / 4.0
    assert_allclose(
        da._calculate_major_axis(ra_lens, dec_lens, ra_mem, dec_mem, weight_mem),
        expected_major_axis,
        **TOLERANCE,
    )


def test_rotate_shear():
    """test rotate shear components"""
    shear1, shear2, phi = 0.15, 0.08, 0.52
    phi_major_45 = np.pi / 4.0
    phi_major_90 = np.pi / 2.0
    phi_major_180 = np.pi
    expected_shear1_45, expected_shear2_45 = 0.08, -0.15
    expected_shear1_90, expected_shear2_90 = -0.15, -0.08
    expected_shear1_180, expected_shear2_180 = 0.15, 0.08

    shear1_45, shear2_45 = da._rotate_shear(shear1, shear2, phi_major_45)
    shear1_90, shear2_90 = da._rotate_shear(shear1, shear2, phi_major_90)
    shear1_180, shear2_180 = da._rotate_shear(shear1, shear2, phi_major_180)

    assert_allclose([shear1_45, shear2_45], [expected_shear1_45, expected_shear2_45], **TOLERANCE)
    assert_allclose([shear1_90, shear2_90], [expected_shear1_90, expected_shear2_90], **TOLERANCE)
    assert_allclose(
        [shear1_180, shear2_180], [expected_shear1_180, expected_shear2_180], **TOLERANCE
    )


def test_compute_lensing_angles_flatsky():
    """test compute lensing angles flatsky"""
    ra_l, dec_l = 161.0, 65.0
    ra_s, dec_s = np.array([-355.0, 355.0]), np.array([-85.0, 85.0])

    # Ensure that we throw a warning with >1 deg separation
    assert_warns(
        UserWarning,
        da._compute_lensing_angles_flatsky,
        ra_l,
        dec_l,
        np.array([151.32, 161.34]),
        np.array([41.49, 51.55]),
    )

    # Test outputs for reasonable values
    ra_l, dec_l = 161.32, 51.49
    ra_s, dec_s = np.array([161.29, 161.34]), np.array([51.45, 51.55])
    thetas, phis = da._compute_lensing_angles_flatsky(ra_l, dec_l, ra_s, dec_s)

    assert_allclose(
        thetas,
        np.array([0.00077050407583119666, 0.00106951489719733675]),
        **TOLERANCE,
        err_msg="Reasonable values with flat sky not matching to precision for theta",
    )

    assert_allclose(
        phis,
        np.array([-1.13390499136495481736, 1.77544123918164542530]),
        **TOLERANCE,
        err_msg="Reasonable values with flat sky not matching to precision for phi",
    )

    # lens and source at the same ra
    assert_allclose(
        da._compute_lensing_angles_flatsky(ra_l, dec_l, np.array([161.32, 161.34]), dec_s),
        [
            [0.00069813170079771690, 0.00106951489719733675],
            [-1.57079632679489655800, 1.77544123918164542530],
        ],
        **TOLERANCE,
        err_msg="Failure when lens and a source share an RA",
    )

    # lens and source at the same dec
    assert_allclose(
        da._compute_lensing_angles_flatsky(ra_l, dec_l, ra_s, np.array([51.49, 51.55])),
        [
            [0.00032601941539388962, 0.00106951489719733675],
            [0.00000000000000000000, 1.77544123918164542530],
        ],
        **TOLERANCE,
        err_msg="Failure when lens and a source share a DEC",
    )

    # lens and source at the same ra and dec
    assert_allclose(
        da._compute_lensing_angles_flatsky(
            ra_l, dec_l, np.array([ra_l, 161.34]), np.array([dec_l, 51.55])
        ),
        [
            [0.00000000000000000000, 0.00106951489719733675],
            [0.00000000000000000000, 1.77544123918164542530],
        ],
        TOLERANCE["rtol"],
        err_msg="Failure when lens and a source share an RA and a DEC",
    )

    # angles over the branch cut between 0 and 360
    assert_allclose(
        da._compute_lensing_angles_flatsky(0.1, dec_l, np.array([359.9, 359.5]), dec_s),
        [
            [0.0022828333888309108, 0.006603944760273219],
            [-0.31079754672938664, 0.15924369771830643],
        ],
        TOLERANCE["rtol"],
        err_msg="Failure when ra_l and ra_s are close but on the opposite sides of the 0 axis",
    )

    # coordinate_system conversion
    ra_l, dec_l = 161.32, 51.49
    ra_s, dec_s = np.array([161.29, 161.34]), np.array([51.45, 51.55])
    thetas_euclidean, phis_euclidean = da._compute_lensing_angles_flatsky(
        ra_l, dec_l, ra_s, dec_s, coordinate_system="euclidean"
    )
    thetas_celestial, phis_celestial = da._compute_lensing_angles_flatsky(
        ra_l, dec_l, ra_s, dec_s, coordinate_system="celestial"
    )

    assert_allclose(
        da._compute_lensing_angles_flatsky(-180, dec_l, np.array([180.1, 179.7]), dec_s),
        [[0.0012916551296819666, 0.003424250083245557], [-2.570568636904587, 0.31079754672944354]],
        TOLERANCE["rtol"],
        err_msg="Failure when ra_l and ra_s are the same but one is defined negative",
    )

    assert_allclose(
        thetas_celestial,
        thetas_euclidean,
        **TOLERANCE,
        err_msg="Conversion from euclidean to celestial coordinate system for theta failed",
    )

    assert_allclose(
        phis_celestial,
        np.pi - phis_euclidean,
        **TOLERANCE,
        err_msg="Conversion from euclidean to celestial coordinate system for phi failed",
    )


def test_compute_lensing_angles_astropy():
    """test compute lensing angles astropy"""

    # coordinate_system conversion
    ra_l, dec_l = 161.32, 51.49
    ra_s, dec_s = np.array([161.29, 161.34]), np.array([51.45, 51.55])
    thetas_euclidean, phis_euclidean = da._compute_lensing_angles_astropy(
        ra_l, dec_l, ra_s, dec_s, coordinate_system="euclidean"
    )
    thetas_celestial, phis_celestial = da._compute_lensing_angles_astropy(
        ra_l, dec_l, ra_s, dec_s, coordinate_system="celestial"
    )

    assert_allclose(
        thetas_celestial,
        thetas_euclidean,
        **TOLERANCE,
        err_msg="Conversion from euclidean to celestial coordinate system for theta failed",
    )

    assert_allclose(
        phis_celestial,
        np.pi - phis_euclidean,
        **TOLERANCE,
        err_msg="Conversion from euclidean to celestial coordinate system for phi failed",
    )


def test_compute_tangential_and_cross_components(modeling_data):
    """test compute tangential and cross components"""
    # Input values
    reltol = modeling_data["dataops_reltol"]
    ra_lens, dec_lens, z_lens = 120.0, 42.0, 0.5
    phi_major = 0.0
    gals = GCData(
        {
            "ra": np.array([120.1, 119.9]),
            "dec": np.array([41.9, 42.2]),
            "id": np.array([1, 2]),
            "e1": np.array([0.2, 0.4]),
            "e2": np.array([0.3, 0.5]),
            "z": np.array([1.0, 2.0]),
        }
    )
    # Correct values
    expected_flat = {
        "angsep": np.array([0.0021745039090962414, 0.0037238407383072053]),
        "cross_shear": np.array([0.2780316984090899, 0.6398792901134982]),
        "tangential_shear": np.array([-0.22956126563459447, -0.02354769805831558]),
        "four_theta_shear": np.array([-0.3324295, -0.43566851]),
        "const_shear": np.array([0.2, 0.4]),
        # DeltaSigma expected values for clmm.Cosmology(H0=70.0, Omega_dm0=0.275, Omega_b0=0.025)
        "cross_DS": np.array([8.58093068e14, 1.33131522e15]),
        # [1224.3326297393244, 1899.6061989365176])*0.7*1.0e12*1.0002565513832675
        "tangential_DS": np.array([-7.08498103e14, -4.89926917e13]),
        # [-1010.889584349285, -69.9059242788237])*0.7*1.0e12*1.0002565513832675
        "four_theta_DS": np.array([-1.02598173e15, -9.06439876e14]),
        "const_DS": np.array([6.17262745e14, 8.32228959e14]),
    }

    expected_curve = {
        "angsep": np.array([0.002175111279323424171, 0.003723129781247932167]),
        "cross_shear": np.array([0.277590689496438781, 0.639929479722048944]),
        "tangential_shear": np.array([-0.23009434826803484841, -0.02214183783401518779]),
        "four_theta_shear": np.array([-0.33189127, -0.43360245]),
        "const_shear": np.array([0.2, 0.4]),
        # DeltaSigma expected values for clmm.Cosmology(H0=70.0, Omega_dm0=0.275, Omega_b0=0.025)
        "cross_DS": np.array([8.56731976e14, 1.33141964e15]),
        "tangential_DS": np.array([-7.10143363e14, -4.60676976e13]),
        "four_theta_DS": np.array([-1.02432059e15, -9.02141297e14]),
        "const_DS": np.array([6.17262745e14, 8.32228959e14]),
    }

    # Geometries to test
    geo_tests = [("flat", expected_flat), ("curve", expected_curve)]
    # Test domains on inputs
    ra_l, dec_l = 161.0, 65.0
    ra_s, dec_s = np.array([-355.0, 355.0]), np.array([-85.0, 85.0])
    shear1, shear2 = gals["e1"][:2], gals["e2"][:2]
    assert_raises(
        ValueError,
        da.compute_tangential_and_cross_components,
        -365.0,
        dec_l,
        ra_s,
        dec_s,
        shear1,
        shear2,
    )
    assert_raises(
        ValueError,
        da.compute_tangential_and_cross_components,
        365.0,
        dec_l,
        ra_s,
        dec_s,
        shear1,
        shear2,
    )
    assert_raises(
        ValueError,
        da.compute_tangential_and_cross_components,
        ra_l,
        95.0,
        ra_s,
        dec_s,
        shear1,
        shear2,
    )
    assert_raises(
        ValueError,
        da.compute_tangential_and_cross_components,
        ra_l,
        -95.0,
        ra_s,
        dec_s,
        shear1,
        shear2,
    )
    assert_raises(
        ValueError,
        da.compute_tangential_and_cross_components,
        ra_l,
        dec_l,
        ra_s - 10.0,
        dec_s,
        shear1,
        shear2,
    )
    assert_raises(
        ValueError,
        da.compute_tangential_and_cross_components,
        ra_l,
        dec_l,
        ra_s + 10.0,
        dec_s,
        shear1,
        shear2,
    )
    assert_raises(
        ValueError,
        da.compute_tangential_and_cross_components,
        ra_l,
        dec_l,
        ra_s,
        dec_s - 10.0,
        shear1,
        shear2,
    )
    assert_raises(
        ValueError,
        da.compute_tangential_and_cross_components,
        ra_l,
        dec_l,
        ra_s,
        dec_s + 10.0,
        shear1,
        shear2,
    )
    # test incosnsitent data
    assert_raises(
        TypeError,
        da.compute_tangential_and_cross_components,
        ra_lens=ra_lens,
        dec_lens=dec_lens,
        ra_source=gals["ra"][0],
        dec_source=gals["dec"],
        shear1=gals["e1"],
        shear2=gals["e2"],
    )
    assert_raises(
        TypeError,
        da.compute_tangential_and_cross_components,
        ra_lens=ra_lens,
        dec_lens=dec_lens,
        ra_source=gals["ra"][:1],
        dec_source=gals["dec"],
        shear1=gals["e1"],
        shear2=gals["e2"],
    )
    # test not implemented geometry
    assert_raises(
        NotImplementedError,
        da.compute_tangential_and_cross_components,
        ra_lens=ra_lens,
        dec_lens=dec_lens,
        ra_source=gals["ra"],
        dec_source=gals["dec"],
        shear1=gals["e1"],
        shear2=gals["e2"],
        geometry="something crazy",
    )
    for geometry, expected in geo_tests:
        # Pass arrays directly into function
        angsep, tshear, xshear = da.compute_tangential_and_cross_components(
            ra_lens=ra_lens,
            dec_lens=dec_lens,
            ra_source=gals["ra"],
            dec_source=gals["dec"],
            shear1=gals["e1"],
            shear2=gals["e2"],
            geometry=geometry,
        )
        assert_allclose(
            angsep,
            expected["angsep"],
            **TOLERANCE,
            err_msg="Angular Separation not correct when passing arrays",
        )
        assert_allclose(
            tshear,
            expected["tangential_shear"],
            **TOLERANCE,
            err_msg="Tangential Shear not correct when passing arrays",
        )
        assert_allclose(
            xshear,
            expected["cross_shear"],
            **TOLERANCE,
            err_msg="Cross Shear not correct when passing arrays",
        )

        ## Turn on quadrupole option
        angsep, _, _, ftshear, cnshear = da.compute_tangential_and_cross_components(
            ra_lens=ra_lens,
            dec_lens=dec_lens,
            ra_source=gals["ra"],
            dec_source=gals["dec"],
            shear1=gals["e1"],
            shear2=gals["e2"],
            geometry=geometry,
            include_quadrupole=True,
            phi_major=0.0,
        )
        assert_allclose(
            ftshear,
            expected["four_theta_shear"],
            **TOLERANCE,
            err_msg="4theta Shear not correct when passing arrays and phi_major",
        )
        assert_allclose(
            cnshear,
            expected["const_shear"],
            **TOLERANCE,
            err_msg="Constant Shear not correct when passing arrays and phi_major",
        )

        ## Pass members info instead of major axis directly
        angsep, _, _, ftshear, cnshear = da.compute_tangential_and_cross_components(
            ra_lens=ra_lens,
            dec_lens=dec_lens,
            ra_source=gals["ra"],
            dec_source=gals["dec"],
            shear1=gals["e1"],
            shear2=gals["e2"],
            geometry=geometry,
            include_quadrupole=True,
            info_mem=[np.array([119.99, 120.01]), np.array([42.0, 42.0]), np.array([1.0, 1.0])],
        )
        assert_allclose(
            ftshear,
            expected["four_theta_shear"],
            **TOLERANCE,
            err_msg="4theta Shear not correct when passing arrays and mem_info",
        )
        assert_allclose(
            cnshear,
            expected["const_shear"],
            **TOLERANCE,
            err_msg="Constant Shear not correct when passing arrays and mem_info",
        )

        # Pass LISTS into function
        angsep, tshear, xshear = da.compute_tangential_and_cross_components(
            ra_lens=ra_lens,
            dec_lens=dec_lens,
            ra_source=list(gals["ra"]),
            dec_source=list(gals["dec"]),
            shear1=list(gals["e1"]),
            shear2=list(gals["e2"]),
            geometry=geometry,
        )
        assert_allclose(
            angsep,
            expected["angsep"],
            **TOLERANCE,
            err_msg="Angular Separation not correct when passing lists",
        )
        assert_allclose(
            tshear,
            expected["tangential_shear"],
            **TOLERANCE,
            err_msg="Tangential Shear not correct when passing lists",
        )
        assert_allclose(
            xshear,
            expected["cross_shear"],
            **TOLERANCE,
            err_msg="Cross Shear not correct when passing lists",
        )

        ## Turn on quadrupole option
        angsep, _, _, ftshear, cnshear = da.compute_tangential_and_cross_components(
            ra_lens=ra_lens,
            dec_lens=dec_lens,
            ra_source=list(gals["ra"]),
            dec_source=list(gals["dec"]),
            shear1=list(gals["e1"]),
            shear2=list(gals["e2"]),
            geometry=geometry,
            include_quadrupole=True,
            phi_major=0.0,
        )
        assert_allclose(
            ftshear,
            expected["four_theta_shear"],
            **TOLERANCE,
            err_msg="4theta Shear not correct when passing lists and phi_major",
        )
        assert_allclose(
            cnshear,
            expected["const_shear"],
            **TOLERANCE,
            err_msg="Constant Shear not correct when passing listss and phi_major",
        )

        ## Pass members info instead of major axis directly
        angsep, _, _, ftshear, cnshear = da.compute_tangential_and_cross_components(
            ra_lens=ra_lens,
            dec_lens=dec_lens,
            ra_source=list(gals["ra"]),
            dec_source=list(gals["dec"]),
            shear1=list(gals["e1"]),
            shear2=list(gals["e2"]),
            geometry=geometry,
            include_quadrupole=True,
            info_mem=[np.array([119.99, 120.01]), np.array([42.0, 42.0]), np.array([1.0, 1.0])],
        )
        assert_allclose(
            ftshear,
            expected["four_theta_shear"],
            **TOLERANCE,
            err_msg="4theta Shear not correct when passing lists and mem_info",
        )
        assert_allclose(
            cnshear,
            expected["const_shear"],
            **TOLERANCE,
            err_msg="Constant Shear not correct when passing lists and mem_info",
        )

        # Test without validation
        angsep, tshear, xshear = da.compute_tangential_and_cross_components(
            ra_lens=ra_lens,
            dec_lens=dec_lens,
            ra_source=list(gals["ra"]),
            dec_source=list(gals["dec"]),
            shear1=list(gals["e1"]),
            shear2=list(gals["e2"]),
            geometry=geometry,
            validate_input=False,
        )
        assert_allclose(
            angsep,
            expected["angsep"],
            **TOLERANCE,
            err_msg="Angular Separation not correct when passing lists",
        )
        assert_allclose(
            tshear,
            expected["tangential_shear"],
            **TOLERANCE,
            err_msg="Tangential Shear not correct when passing lists",
        )
        assert_allclose(
            xshear,
            expected["cross_shear"],
            **TOLERANCE,
            err_msg="Cross Shear not correct when passing lists",
        )

        ## Turn on quadrupole option
        angsep, _, _, ftshear, cnshear = da.compute_tangential_and_cross_components(
            ra_lens=ra_lens,
            dec_lens=dec_lens,
            ra_source=list(gals["ra"]),
            dec_source=list(gals["dec"]),
            shear1=list(gals["e1"]),
            shear2=list(gals["e2"]),
            geometry=geometry,
            include_quadrupole=True,
            phi_major=0.0,
            validate_input=False,
        )
        assert_allclose(
            ftshear,
            expected["four_theta_shear"],
            **TOLERANCE,
            err_msg="4theta Shear not correct when passing lists and phi_major",
        )
        assert_allclose(
            cnshear,
            expected["const_shear"],
            **TOLERANCE,
            err_msg="Constant Shear not correct when passing listss and phi_major",
        )

        ## Pass members info instead of major axis directly
        angsep, _, _, ftshear, cnshear = da.compute_tangential_and_cross_components(
            ra_lens=ra_lens,
            dec_lens=dec_lens,
            ra_source=list(gals["ra"]),
            dec_source=list(gals["dec"]),
            shear1=list(gals["e1"]),
            shear2=list(gals["e2"]),
            geometry=geometry,
            include_quadrupole=True,
            info_mem=[np.array([119.99, 120.01]), np.array([42.0, 42.0]), np.array([1.0, 1.0])],
            validate_input=False,
        )
        assert_allclose(
            ftshear,
            expected["four_theta_shear"],
            **TOLERANCE,
            err_msg="4theta Shear not correct when passing lists and mem_info",
        )
        assert_allclose(
            cnshear,
            expected["const_shear"],
            **TOLERANCE,
            err_msg="Constant Shear not correct when passing lists and mem_info",
        )

        # Test for ValueError if neither phi_major or mem positions are given with quadrupole
        assert_raises(
                ValueError,
                da.compute_tangential_and_cross_components,
                ra_lens=ra_lens,
                dec_lens=dec_lens,
                ra_source=list(gals["ra"]),
                dec_source=list(gals["dec"]),
                shear1=list(gals["e1"]),
                shear2=list(gals["e2"]),
                geometry=geometry,
                include_quadrupole=True,
                validate_input=False,
                )

        # Test without validation and float arguments
        angsep, tshear, xshear = da.compute_tangential_and_cross_components(
            ra_lens=ra_lens,
            dec_lens=dec_lens,
            ra_source=gals["ra"][0],
            dec_source=gals["dec"][0],
            shear1=gals["e1"][0],
            shear2=gals["e2"][0],
            geometry=geometry,
            validate_input=False,
        )
        assert_allclose(
            angsep,
            expected["angsep"][0],
            **TOLERANCE,
            err_msg="Angular Separation not correct when passing lists",
        )
        assert_allclose(
            tshear,
            expected["tangential_shear"][0],
            **TOLERANCE,
            err_msg="Tangential Shear not correct when passing lists",
        )
        assert_allclose(
            xshear,
            expected["cross_shear"][0],
            **TOLERANCE,
            err_msg="Cross Shear not correct when passing lists",
        )
    # Test invalid coordinate system name
    assert_raises(
        ValueError,
        da.compute_tangential_and_cross_components,
        ra_l,
        dec_l,
        ra_s,
        dec_s + 10.0,
        shear1,
        shear2,
        coordinate_system="crazy",
    )
    # Test invalid coordinate system type
    assert_raises(
        ValueError,
        da.compute_tangential_and_cross_components,
        ra_l,
        dec_l,
        ra_s,
        dec_s + 10.0,
        shear1,
        shear2,
        coordinate_system=1,
    )
    # Use the cluster method
    cluster = clmm.GalaxyCluster(
        unique_id="blah", ra=ra_lens, dec=dec_lens, z=z_lens, galcat=gals["ra", "dec", "e1", "e2"]
    )
    cluster_quad = clmm.GalaxyCluster(
        unique_id="blah",
        ra=ra_lens,
        dec=dec_lens,
        z=z_lens,
        galcat=gals["ra", "dec", "e1", "e2"],
        include_quadrupole=True,
    )
    # Test error with bad name/missing column
    assert_raises(
        TypeError, cluster.compute_tangential_and_cross_components, shape_component1="crazy name"
    )
    # Test output
    for geometry, expected in geo_tests:
        angsep3, tshear3, xshear3 = cluster.compute_tangential_and_cross_components(
            geometry=geometry
        )
        assert_allclose(
            angsep3,
            expected["angsep"],
            **TOLERANCE,
            err_msg="Angular Separation not correct when using cluster method",
        )
        assert_allclose(
            tshear3,
            expected["tangential_shear"],
            **TOLERANCE,
            err_msg="Tangential Shear not correct when using cluster method",
        )
        assert_allclose(
            xshear3,
            expected["cross_shear"],
            **TOLERANCE,
            err_msg="Cross Shear not correct when using cluster method",
        )
        # include_quadrupole=True, with phi_major input
        _, _, _, ftshear2, cnshear2 = cluster_quad.compute_tangential_and_cross_components(
            geometry=geometry, phi_major=0.0
        )
        assert_allclose(
            ftshear2,
            expected["four_theta_shear"],
            **TOLERANCE,
            err_msg="4theta Shear not correct when using cluster method w/ phi_major",
        )
        assert_allclose(
            cnshear2,
            expected["const_shear"],
            **TOLERANCE,
            err_msg="Constant Shear not correct when using cluster method w/ phi_major",
        )
        # include_quadrupole=True, with info_mem instead of phi_major input
        _, _, _, ftshear3, cnshear3 = cluster_quad.compute_tangential_and_cross_components(
            geometry=geometry,
            info_mem=[np.array([119.99, 120.01]), np.array([42.0, 42.0]), np.array([1.0, 1.0])],
        )
        assert_allclose(
            ftshear3,
            expected["four_theta_shear"],
            **TOLERANCE,
            err_msg="4theta Shear not correct when using cluster method w/ info_mem",
        )
        assert_allclose(
            cnshear3,
            expected["const_shear"],
            **TOLERANCE,
            err_msg="Constant Shear not correct when using cluster method w/ info_mem",
        )
    # Check behaviour for the deltasigma option.
    cosmo = clmm.Cosmology(H0=70.0, Omega_dm0=0.275, Omega_b0=0.025)

    # check values for DeltaSigma
    sigma_c = cosmo.eval_sigma_crit(z_lens, gals["z"])
    # check validation between is_deltasigma and sigma_c
    assert_raises(
        TypeError,
        da.compute_tangential_and_cross_components,
        ra_lens=ra_lens,
        dec_lens=dec_lens,
        ra_source=gals["ra"],
        dec_source=gals["dec"],
        shear1=gals["e1"],
        shear2=gals["e2"],
        is_deltasigma=False,
        sigma_c=sigma_c,
    )
    assert_raises(
        TypeError,
        da.compute_tangential_and_cross_components,
        ra_lens=ra_lens,
        dec_lens=dec_lens,
        ra_source=gals["ra"],
        dec_source=gals["dec"],
        shear1=gals["e1"],
        shear2=gals["e2"],
        is_deltasigma=True,
        sigma_c=None,
    )
    # check validation between include_quadrupole and {phi_major|info_mem}
    assert_raises(
        TypeError,
        da.compute_tangential_and_cross_components,
        ra_lens=ra_lens,
        dec_lens=dec_lens,
        ra_source=gals["ra"],
        dec_source=gals["dec"],
        shear1=gals["e1"],
        shear2=gals["e2"],
        include_quadrupole=True,
    )
    # test values
    for geometry, expected in geo_tests:
        angsep_DS, tDS, xDS = da.compute_tangential_and_cross_components(
            ra_lens=ra_lens,
            dec_lens=dec_lens,
            ra_source=gals["ra"],
            dec_source=gals["dec"],
            shear1=gals["e1"],
            shear2=gals["e2"],
            is_deltasigma=True,
            sigma_c=sigma_c,
            geometry=geometry,
        )
        assert_allclose(
            angsep_DS, expected["angsep"], reltol, err_msg="Angular Separation not correct"
        )
        assert_allclose(
            tDS, expected["tangential_DS"], reltol, err_msg="Tangential Shear not correct"
        )
        assert_allclose(xDS, expected["cross_DS"], reltol, err_msg="Cross Shear not correct")
        ## Turn on include_quadrupole w/ phi_major
        _, _, _, ftDS, cnDS = da.compute_tangential_and_cross_components(
            ra_lens=ra_lens,
            dec_lens=dec_lens,
            ra_source=gals["ra"],
            dec_source=gals["dec"],
            shear1=gals["e1"],
            shear2=gals["e2"],
            is_deltasigma=True,
            sigma_c=sigma_c,
            geometry=geometry,
            include_quadrupole=True,
            phi_major=0.0,
        )
        assert_allclose(ftDS, expected["four_theta_DS"], reltol, err_msg="4theta shear not correct")
        assert_allclose(cnDS, expected["const_DS"], reltol, err_msg="constant shear not correct")
        ## Turn on include_quadrupole w/ info_mem
        _, _, _, ftDS, cnDS = da.compute_tangential_and_cross_components(
            ra_lens=ra_lens,
            dec_lens=dec_lens,
            ra_source=gals["ra"],
            dec_source=gals["dec"],
            shear1=gals["e1"],
            shear2=gals["e2"],
            is_deltasigma=True,
            sigma_c=sigma_c,
            geometry=geometry,
            include_quadrupole=True,
            info_mem=[np.array([119.99, 120.01]), np.array([42.0, 42.0]), np.array([1.0, 1.0])],
        )
        assert_allclose(ftDS, expected["four_theta_DS"], reltol, err_msg="4theta shear not correct")
        assert_allclose(cnDS, expected["const_DS"], reltol, err_msg="constant shear not correct")
    # Tests with the cluster object
    # cluster object missing source redshift, and function call missing cosmology
    cluster = clmm.GalaxyCluster(
        unique_id="blah", ra=ra_lens, dec=dec_lens, z=z_lens, galcat=gals["ra", "dec", "e1", "e2"]
    )
    assert_raises(TypeError, cluster.compute_tangential_and_cross_components, is_deltasigma=True)
    # cluster object OK but function call missing cosmology
    cluster = clmm.GalaxyCluster(
        unique_id="blah",
        ra=ra_lens,
        dec=dec_lens,
        z=z_lens,
        galcat=gals["ra", "dec", "e1", "e2", "z"],
    )
    cluster_quad = clmm.GalaxyCluster(
        unique_id="blah",
        ra=ra_lens,
        dec=dec_lens,
        z=z_lens,
        galcat=gals["ra", "dec", "e1", "e2", "z"],
        include_quadrupole=True,
    )
    assert_raises(TypeError, cluster.compute_tangential_and_cross_components, is_deltasigma=True)
    # check values for DeltaSigma
    for geometry, expected in geo_tests:
        angsep_DS, tDS, xDS = cluster.compute_tangential_and_cross_components(
            cosmo=cosmo, is_deltasigma=True, geometry=geometry
        )
        assert_allclose(
            angsep_DS,
            expected["angsep"],
            reltol,
            err_msg="Angular Separation not correct when using cluster method",
        )
        assert_allclose(
            tDS,
            expected["tangential_DS"],
            reltol,
            err_msg="Tangential Shear not correct when using cluster method",
        )
        assert_allclose(
            xDS,
            expected["cross_DS"],
            reltol,
            err_msg="Cross Shear not correct when using cluster method",
        )
        # Turn on include_quadrupole w/ phi_major
        _, _, _, ftDS, cnDS = cluster_quad.compute_tangential_and_cross_components(
            cosmo=cosmo, is_deltasigma=True, geometry=geometry, phi_major=0.0
        )
        assert_allclose(
            ftDS,
            expected["four_theta_DS"],
            reltol,
            err_msg="4theta Shear not correct when using cluster method w/ phi_major",
        )
        assert_allclose(
            cnDS,
            expected["const_DS"],
            reltol,
            err_msg="Constant Shear not correct when using cluster method w/ phi_major",
        )
        # Turn on include_quadrupole w/ info_mem
        _, _, _, ftDS, cnDS = cluster_quad.compute_tangential_and_cross_components(
            cosmo=cosmo,
            is_deltasigma=True,
            geometry=geometry,
            info_mem=[np.array([119.99, 120.01]), np.array([42.0, 42.0]), np.array([1.0, 1.0])],
        )
        assert_allclose(
            ftDS,
            expected["four_theta_DS"],
            reltol,
            err_msg="4theta Shear not correct when using cluster method w/ info_mem",
        )
        assert_allclose(
            cnDS,
            expected["const_DS"],
            reltol,
            err_msg="Constant Shear not correct when using cluster method w/ info_mem",
        )

    # test basic weights functionality
    cluster.compute_galaxy_weights()
    expected = np.array([1.0, 1.0])
    assert_allclose(cluster.galcat["w_ls"], expected, **TOLERANCE)


def test_compute_background_probability():
    """test for compute background probability"""
    z_lens = 0.1
    z_src = np.array([0.22, 0.35, 1.7])

    # true redshift
    p_bkg = da.compute_background_probability(
        z_lens, z_src=z_src, use_pdz=False, pzpdf=None, pzbins=None, validate_input=True
    )
    expected = np.array([1.0, 1.0, 1.0])
    assert_allclose(p_bkg, expected, **TOLERANCE)
    assert_raises(
        ValueError,
        da.compute_background_probability,
        z_lens,
        z_src=None,
        use_pdz=False,
        pzpdf=None,
        pzbins=None,
        validate_input=True,
    )

    # photoz + deltasigma
    pzbin = np.linspace(0.0001, 5, 100)
    pzbins = [pzbin for i in range(z_src.size)]
    pzpdf = [multivariate_normal.pdf(pzbin, mean=z, cov=0.3) for z in z_src]
    assert_raises(
        ValueError,
        da.compute_background_probability,
        z_lens,
        z_src=z_src,
        use_pdz=True,
        pzpdf=None,
        pzbins=pzbins,
        validate_input=True,
    )


def test_compute_galaxy_weights():
    """test for compute galaxy weights"""
    cosmo = clmm.Cosmology(H0=71.0, Omega_dm0=0.265 - 0.0448, Omega_b0=0.0448, Omega_k0=0.0)
    z_lens = 0.1
    z_src = [0.22, 0.35, 1.7]
    shape_component1 = np.array([0.143, 0.063, -0.171])
    shape_component2 = np.array([-0.011, 0.012, -0.250])
    shape_component1_err = np.array([0.11, 0.01, 0.2])
    shape_component2_err = np.array([0.14, 0.16, 0.21])

    # true redshift + deltasigma
    sigma_c = cosmo.eval_sigma_crit(z_lens, z_src)
    weights = da.compute_galaxy_weights(
        is_deltasigma=True,
        sigma_c=sigma_c,
        use_shape_noise=False,
        shape_component1=shape_component1,
        shape_component2=shape_component2,
        use_shape_error=False,
        shape_component1_err=shape_component1_err,
        shape_component2_err=shape_component2_err,
        validate_input=True,
    )
    expected = np.array([4.58644320e-31, 9.68145632e-31, 5.07260777e-31])
    assert_allclose(weights * 1e20, expected * 1e20, **TOLERANCE)

    # photoz + deltasigma
    pzbin = np.linspace(0.0001, 5, 100)
    pzbins = [pzbin for i in range(len(z_src))]
    pzpdf = [multivariate_normal.pdf(pzbin, mean=z, cov=0.3) for z in z_src]
    sigma_c_eff = compute_critical_surface_density_eff(
        cosmo=cosmo,
        z_cluster=z_lens,
        pzbins=pzbins,
        pzpdf=pzpdf,
    )
    weights = da.compute_galaxy_weights(
        is_deltasigma=True,
        sigma_c=sigma_c_eff,
        use_shape_noise=False,
        shape_component1=shape_component1,
        shape_component2=shape_component2,
        use_shape_error=False,
        shape_component1_err=None,
        shape_component2_err=None,
        validate_input=True,
    )

    expected = np.array([9.07709345e-33, 1.28167582e-32, 4.16870389e-32])
    assert_allclose(weights * 1e20, expected * 1e20, **TOLERANCE)

    # photoz + deltasigma - shared bins
    pzbin = np.linspace(0.0001, 5, 100)
    pzbins = pzbin
    pzpdf = [multivariate_normal.pdf(pzbin, mean=z, cov=0.3) for z in z_src]
    weights = da.compute_galaxy_weights(
        is_deltasigma=True,
        sigma_c=sigma_c_eff,
        use_shape_noise=False,
        shape_component1=shape_component1,
        shape_component2=shape_component2,
        use_shape_error=False,
        shape_component1_err=None,
        shape_component2_err=None,
        validate_input=True,
    )

    expected = np.array([9.07709345e-33, 1.28167582e-32, 4.16870389e-32])
    assert_allclose(weights * 1e20, expected * 1e20, **TOLERANCE)

    # test with noise
    weights = da.compute_galaxy_weights(
        is_deltasigma=True,
        sigma_c=sigma_c_eff,
        use_shape_noise=True,
        shape_component1=shape_component1,
        shape_component2=shape_component2,
        use_shape_error=False,
        shape_component1_err=None,
        shape_component2_err=None,
        validate_input=True,
    )

    expected = np.array([9.07709345e-33, 1.28167582e-32, 4.16870389e-32])
    assert_allclose(weights * 1e20, expected * 1e20, **TOLERANCE)

    # test with is_deltasigma=False and geometric weights only
    weights = da.compute_galaxy_weights(
        is_deltasigma=False,
        sigma_c=None,
        use_shape_noise=False,
        shape_component1=shape_component1,
        shape_component2=shape_component2,
        use_shape_error=False,
        shape_component1_err=None,
        shape_component2_err=None,
        validate_input=True,
    )

    expected = np.array([1.0, 1.0, 1.0])
    assert_allclose(weights, expected, **TOLERANCE)

    # # test error when missing information
    assert_raises(
        ValueError,
        da.compute_galaxy_weights,
        is_deltasigma=True,
        sigma_c=sigma_c_eff,
        use_shape_noise=True,
        shape_component1=None,
        shape_component2=None,
        use_shape_error=False,
        shape_component1_err=None,
        shape_component2_err=None,
        validate_input=True,
    )

    # test error when missing information
    assert_raises(
        ValueError,
        da.compute_galaxy_weights,
        is_deltasigma=True,
        sigma_c=sigma_c_eff,
        use_shape_noise=False,
        use_shape_error=True,
        shape_component1_err=None,
        shape_component2_err=None,
        validate_input=True,
    )


def _test_profile_table_output(
    profile,
    expected_rmin,
    expected_radius,
    expected_rmax,
    expected_p0,
    expected_p1,
    expected_nsrc,
    expected_gal_id=None,
    p0="p_0",
    p1="p_1",
):
    """Func to make the validation of the table with the expected values"""
    assert_allclose(
        profile["radius_min"],
        expected_rmin,
        **TOLERANCE,
        err_msg="Minimum radius in bin not expected.",
    )
    assert_allclose(
        profile["radius"], expected_radius, **TOLERANCE, err_msg="Mean radius in bin not expected."
    )
    assert_allclose(
        profile["radius_max"],
        expected_rmax,
        **TOLERANCE,
        err_msg="Maximum radius in bin not expected.",
    )
    assert_allclose(
        profile[p0], expected_p0, **TOLERANCE, err_msg="Tangential shear in bin not expected"
    )
    assert_allclose(
        profile[p1], expected_p1, **TOLERANCE, err_msg="Cross shear in bin not expected"
    )
    assert_array_equal(profile["n_src"], expected_nsrc)
    if expected_gal_id is not None:
        assert_array_equal(profile["gal_id"], np.array(expected_gal_id, dtype=object))


def test_make_radial_profiles():
    """test make radial profiles"""
    # Set up a cluster object and compute cross and tangential shears
    ra_lens, dec_lens, z_lens = 120.0, 42.0, 0.5
    gals = GCData(
        {
            "ra": np.array([120.1, 119.9, 119.9]),
            "dec": np.array([41.9, 42.2, 42.2]),
            "id": np.array([1, 2, 3]),
            "e1": np.array([0.2, 0.4, 0.4]),
            "e2": np.array([0.3, 0.5, 0.5]),
            "z": np.ones(3),
        }
    )
    angsep_units, bin_units = "radians", "radians"
    # Set up radial values
    bins_radians = np.array([0.002, 0.003, 0.004])
    expected_radius_flat = [0.0021745039090962414, 0.0037238407383072053]
    expected_radius_curve = [0.002175111279323424171, 0.003723129781247932167]
    expected_flat = {
        "angsep": np.array([0.0021745039090962414, 0.0037238407383072053, 0.0037238407383072053]),
        "cross_shear": np.array([0.2780316984090899, 0.6398792901134982, 0.6398792901134982]),
        "tan_shear": np.array([-0.22956126563459447, -0.02354769805831558, -0.02354769805831558]),
    }
    expected_curve = {
        "angsep": np.array(
            [0.002175111279323424171, 0.003723129781247932167, 0.003723129781247932167]
        ),
        "cross_shear": np.array([0.277590689496438781, 0.639929479722048944, 0.639929479722048944]),
        "tan_shear": np.array(
            [-0.23009434826803484841, -0.02214183783401518779, -0.02214183783401518779]
        ),
    }
    # Geometries to test
    geo_tests = [("flat", expected_flat), ("curve", expected_curve)]
    for geometry, expected in geo_tests:
        #######################################
        ### Use without cluster object ########
        #######################################
        if geometry == "flat":
            expected_radius = expected_radius_flat
        elif geometry == "curve":
            expected_radius = expected_radius_curve
        angsep, tshear, xshear = da.compute_tangential_and_cross_components(
            ra_lens=ra_lens,
            dec_lens=dec_lens,
            ra_source=gals["ra"],
            dec_source=gals["dec"],
            shear1=gals["e1"],
            shear2=gals["e2"],
            geometry=geometry,
        )
        # Tests passing int as bins arg makes the correct bins
        bins = 2
        vec_bins = clmm.utils.make_bins(np.min(angsep), np.max(angsep), bins)
        assert_array_equal(
            da.make_radial_profile(
                [tshear, xshear, gals["z"]], angsep, angsep_units, bin_units, bins=bins
            )[0],
            da.make_radial_profile(
                [tshear, xshear, gals["z"]], angsep, angsep_units, bin_units, bins=vec_bins
            )[0],
        )
        # Test the outputs of compute_tangential_and_cross_components just to be safe
        assert_allclose(
            angsep,
            expected["angsep"],
            **TOLERANCE,
            err_msg="Angular Separation not correct when testing shear profiles",
        )
        assert_allclose(
            tshear,
            expected["tan_shear"],
            **TOLERANCE,
            err_msg="Tangential Shear not correct when testing shear profiles",
        )
        assert_allclose(
            xshear,
            expected["cross_shear"],
            **TOLERANCE,
            err_msg="Cross Shear not correct when testing shear profiles",
        )
        # Test default behavior, remember that include_empty_bins=False excludes all bins with N>=1
        profile = da.make_radial_profile(
            [tshear, xshear, gals["z"]],
            angsep,
            angsep_units,
            bin_units,
            bins=bins_radians,
            include_empty_bins=False,
        )
        _test_profile_table_output(
            profile,
            bins_radians[:2],
            expected_radius[:2],
            bins_radians[1:3],
            expected["tan_shear"][:2],
            expected["cross_shear"][:2],
            [1, 2],
        )
        # Test metadata
        assert_array_equal(profile.meta["bin_units"], bin_units)
        assert_array_equal(profile.meta["cosmo"], None)
        # Test simple unit convesion
        profile = da.make_radial_profile(
            [tshear, xshear, gals["z"]],
            angsep * 180.0 / np.pi,
            "degrees",
            bin_units,
            bins=bins_radians,
            include_empty_bins=False,
        )
        _test_profile_table_output(
            profile,
            bins_radians[:2],
            expected_radius[:2],
            bins_radians[1:3],
            expected["tan_shear"][:2],
            expected["cross_shear"][:2],
            [1, 2],
        )
        # including empty bins
        profile = da.make_radial_profile(
            [tshear, xshear, gals["z"]],
            angsep,
            angsep_units,
            bin_units,
            bins=bins_radians,
            include_empty_bins=True,
        )
        _test_profile_table_output(
            profile,
            bins_radians[:-1],
            expected_radius,
            bins_radians[1:],
            expected["tan_shear"][:-1],
            expected["cross_shear"][:-1],
            [1, 2],
        )
        # test with return_binnumber
        profile, binnumber = da.make_radial_profile(
            [tshear, xshear, gals["z"]],
            angsep,
            angsep_units,
            bin_units,
            bins=bins_radians,
            include_empty_bins=True,
            return_binnumber=True,
        )
        _test_profile_table_output(
            profile,
            bins_radians[:-1],
            expected_radius,
            bins_radians[1:],
            expected["tan_shear"][:-1],
            expected["cross_shear"][:-1],
            [1, 2],
        )
        assert_array_equal(binnumber, [1, 2, 2])
        ###################################
        ### Test with cluster object ######
        ###################################
        cluster = clmm.GalaxyCluster(
            unique_id="blah",
            ra=ra_lens,
            dec=dec_lens,
            z=z_lens,
            galcat=gals["ra", "dec", "e1", "e2", "z", "id"],
        )
        cluster.compute_tangential_and_cross_components(geometry=geometry)
        cluster.make_radial_profile(bin_units, bins=bins_radians, include_empty_bins=False)
        # Test default behavior, remember that include_empty_bins=False excludes all bins with N>=1
        _test_profile_table_output(
            cluster.profile,
            bins_radians[:2],
            expected_radius[:2],
            bins_radians[1:3],
            expected["tan_shear"][:2],
            expected["cross_shear"][:2],
            [1, 2],
            p0="gt",
            p1="gx",
        )
        # including empty bins
        cluster.make_radial_profile(
            bin_units, bins=bins_radians, include_empty_bins=True, table_name="profile2"
        )
        _test_profile_table_output(
            cluster.profile2,
            bins_radians[:-1],
            expected_radius,
            bins_radians[1:],
            expected["tan_shear"][:-1],
            expected["cross_shear"][:-1],
            [1, 2],
            p0="gt",
            p1="gx",
        )
        # Test with galaxy id's
        cluster.make_radial_profile(
            bin_units,
            bins=bins_radians,
            include_empty_bins=True,
            gal_ids_in_bins=True,
            table_name="profile3",
        )
        _test_profile_table_output(
            cluster.profile3,
            bins_radians[:-1],
            expected_radius,
            bins_radians[1:],
            expected["tan_shear"][:-1],
            expected["cross_shear"][:-1],
            [1, 2],
            [[1], [2, 3]],
            p0="gt",
            p1="gx",
        )
        # Test it runs with galaxy id's and int bins
        cluster.make_radial_profile(
            bin_units, bins=5, include_empty_bins=True, gal_ids_in_bins=True, table_name="profile3"
        )
        # And overwriting table
        cluster.make_radial_profile(
            bin_units,
            bins=bins_radians,
            include_empty_bins=True,
            gal_ids_in_bins=True,
            table_name="profile3",
        )
        _test_profile_table_output(
            cluster.profile3,
            bins_radians[:-1],
            expected_radius,
            bins_radians[1:],
            expected["tan_shear"][:-1],
            expected["cross_shear"][:-1],
            [1, 2],
            [[1], [2, 3]],
            p0="gt",
            p1="gx",
        )
        # Test it runs with galaxy id's and int bins and no empty bins
        cluster.make_radial_profile(
            bin_units,
            bins=bins_radians,
            include_empty_bins=False,
            gal_ids_in_bins=True,
            table_name="profile3",
        )
        _test_profile_table_output(
            cluster.profile3,
            bins_radians[:2],
            expected_radius[:2],
            bins_radians[1:3],
            expected["tan_shear"][:2],
            expected["cross_shear"][:2],
            [1, 2],
            p0="gt",
            p1="gx",
        )
        # Test passing zeror errors
        cluster_err = clmm.GalaxyCluster(
            unique_id="blah",
            ra=ra_lens,
            dec=dec_lens,
            z=z_lens,
            galcat=gals["ra", "dec", "e1", "e2", "z", "id"],
        )
        cluster_err.compute_tangential_and_cross_components(geometry=geometry)
        cluster_err.galcat["et_err"] = 0
        cluster_err.galcat["ex_err"] = 0
        cluster_err.make_radial_profile(
            bin_units,
            bins=bins_radians,
            include_empty_bins=False,
            tan_component_in_err="et_err",
            cross_component_in_err="ex_err",
        )
        for c in cluster_err.profile.colnames:
            assert_allclose(
                cluster.profile[c],
                cluster_err.profile[c],
                **TOLERANCE,
                err_msg=f"Value for {c} in bin not expected.",
            )

    ########################################
    ### Basic tests of cluster object ######
    ########################################
    # Test error of missing redshift
    cluster = clmm.GalaxyCluster(
        unique_id="blah", ra=ra_lens, dec=dec_lens, z=z_lens, galcat=gals["ra", "dec", "e1", "e2"]
    )
    cluster.compute_tangential_and_cross_components()
    assert_raises(TypeError, cluster.make_radial_profile, bin_units)
    # Test error of missing shear
    cluster = clmm.GalaxyCluster(
        unique_id="blah",
        ra=ra_lens,
        dec=dec_lens,
        z=z_lens,
        galcat=gals["ra", "dec", "e1", "e2", "z", "id"],
    )
    assert_raises(TypeError, cluster.make_radial_profile, bin_units)
    # Test error with overwrite=False
    cluster.compute_tangential_and_cross_components()
    cluster.make_radial_profile(bin_units, bins=bins_radians, table_name="profile3")
    assert_raises(
        AttributeError,
        cluster.make_radial_profile,
        bin_units,
        bins=bins_radians,
        table_name="profile3",
        overwrite=False,
    )
    # Check error of missing id's
    cluster_noid = clmm.GalaxyCluster(
        unique_id="blah",
        ra=ra_lens,
        dec=dec_lens,
        z=z_lens,
        galcat=gals["ra", "dec", "e1", "e2", "z"],
    )
    cluster_noid.compute_tangential_and_cross_components()
    assert_raises(TypeError, cluster_noid.make_radial_profile, bin_units, gal_ids_in_bins=True)
