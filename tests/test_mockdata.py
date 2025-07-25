"""Tests for examples/support/mock_data.py"""

import warnings
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal
import clmm
import clmm.dataops as da
from clmm.support import mock_data as mock
from clmm.support.sampler import fitters
from clmm.utils import redshift_distributions as zdist

TOLERANCE = {"rtol": 5.0e-4, "atol": 1.0e-4}


def test_mock_data():
    """Run generate_galaxy_catalog 1000 times and assert that retrieved mass is always consistent
    with input
    """
    cosmo = clmm.Cosmology(H0=70.0, Omega_dm0=0.27 - 0.045, Omega_b0=0.045, Omega_k0=0.0)
    # Basic raise tests
    assert_raises(ValueError, mock.generate_galaxy_catalog, 1e15, 0.3, 4, cosmo, 0.8, ngals=None)
    assert_raises(
        ValueError, mock.generate_galaxy_catalog, 1e15, 0.3, 4, cosmo, 0.8, ngals=1, ngal_density=1
    )
    assert_raises(
        ValueError, mock.generate_galaxy_catalog, 1e15, 0.3, 4, cosmo, "unknown_src", ngals=10
    )
    assert_raises(
        ValueError, mock.generate_galaxy_catalog, 1e15, 0.3, 4, cosmo, "unknown_src", ngal_density=1
    )
    # Test warning if bad gals
    with warnings.catch_warnings(record=True) as warn:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        np.random.seed(314)
        mock.generate_galaxy_catalog(1e15, 0.3, 4, cosmo, 0.30001, ngals=1000, nretry=0)
        # Verify some things
        assert len(warn) == 1
    # Simple test to check if option with ngal_density is working
    # A proper test should be implemented
    mock.generate_galaxy_catalog(1e15, 0.3, 4, cosmo, 0.8, ngals=None, ngal_density=1)

    # Simple test to check if option with zsrc=chang13 is working
    # A proper test should be implemented
    mock.generate_galaxy_catalog(1e15, 0.3, 4, cosmo, "chang13", ngals=100)
    mock.generate_galaxy_catalog(1e15, 0.3, 4, cosmo, "chang13", ngal_density=1)

    # Simple test to check if option with zsrc=desc_src is working
    # A proper test should be implemented
    mock.generate_galaxy_catalog(1e15, 0.3, 4, cosmo, "desc_srd", ngals=100)
    mock.generate_galaxy_catalog(1e15, 0.3, 4, cosmo, "desc_srd", ngal_density=1)

    # Test to check unknown pdz
    assert_raises(
        ValueError,
        mock.generate_galaxy_catalog,
        1e15,
        0.3,
        4,
        cosmo,
        0.8,
        ngals=100,
        photoz_sigma_unscaled=0.1,
        pzpdf_type="xxx",
    )

    # Test pdz with bad arguments
    assert_raises(
        TypeError,
        mock.generate_galaxy_catalog,
        1e15,
        0.3,
        4,
        cosmo,
        0.8,
        ngals=100,
        photoz_sigma_unscaled="xxx",
    )
    assert_raises(
        TypeError,
        mock.generate_galaxy_catalog,
        1e15,
        0.3,
        4,
        cosmo,
        0.8,
        ngals=100,
        photoz_sigma_unscaled=0.1,
        pz_bins="xxx",
    )

    # Simple test to check if option with pdz is working
    # A proper test should be implemented
    for kwargs in (
        {"pzpdf_type": None},
        {"pzpdf_type": "individual_bins"},
        {"pzpdf_type": "shared_bins"},
        {"pzpdf_type": "quantiles"},
        {},
    ):
        for shapenoise in (None, 0.5):
            for pz_bins in (21, np.linspace(0, 2, 21)):
                cat = mock.generate_galaxy_catalog(
                    1e15,
                    0.3,
                    4,
                    cosmo,
                    0.8,
                    ngals=20,
                    mean_e_err=0.01,
                    pz_bins=pz_bins,
                    photoz_sigma_unscaled=0.1,
                    shapenoise=shapenoise,
                    **kwargs
                )
                if kwargs.get("pzpdf_type", "shared_bins") == "shared_bins":
                    assert "zbins" in cat.pzpdf_info
                    assert "pzpdf" in cat.colnames
                elif kwargs.get("pzpdf_type", "shared_bins") == "individual_bins":
                    assert "pzbins" in cat.colnames
                    assert "pzpdf" in cat.colnames
                elif kwargs.get("pzpdf_type", "shared_bins") == "quantiles":
                    assert "quantiles" in cat.pzpdf_info
                    assert "pzquantiles" in cat.colnames
                elif kwargs.get("pzpdf_type", "shared_bins") is None:
                    assert "zbins" not in cat.pzpdf_info
                    assert "pzbins" not in cat.colnames
                    assert "pzpdf" not in cat.colnames

    def nfw_shear_profile(r, logm, z_src):
        return clmm.compute_reduced_tangential_shear(
            r, 10.0**logm, 4, 0.3, z_src, cosmo, delta_mdef=200, halo_profile_model="nfw"
        )

    def mass_mock_cluster(mass=15.0, guess=15.0):
        # Set up mock cluster
        ngals = 5000
        cluster_ra = 20.0
        cluster_dec = -23.2
        cluster_z = 0.3

        data = mock.generate_galaxy_catalog(
            10**mass,
            cluster_z,
            4,
            cosmo,
            0.8,
            ngals=ngals,
            cluster_ra=cluster_ra,
            cluster_dec=cluster_dec,
        )

        # Check whether the given ngals is the retrieved ngals
        assert_equal(len(data["ra"]), ngals)

        # Check that there are no galaxies with |e|>1
        assert_equal(np.count_nonzero((data["e1"] > 1) | (data["e1"] < -1)), 0)
        assert_equal(np.count_nonzero((data["e2"] > 1) | (data["e2"] < -1)), 0)

        # Create shear profile
        cl = clmm.GalaxyCluster("test_cluster", cluster_ra, cluster_dec, cluster_z, data)
        theta, g_t, g_x = cl.compute_tangential_and_cross_components(geometry="flat")
        binned = cl.make_radial_profile(
            "Mpc", bins=da.make_bins(0.5, 5.0, 100), cosmo=cosmo, include_empty_bins=False
        )

        popt, pcov = fitters["curve_fit"](
            lambda r, logm: nfw_shear_profile(r, logm, 0.8),
            binned["radius"],
            binned["gt"],
            np.ones_like(binned["gt"]) * 1.0e-5,
            bounds=[13.0, 17.0],
            p0=guess,
        )

        return popt[0]

    # input_masses = np.random.uniform(13.5, 16., 25)
    # logm_0 = np.random.uniform(13.5, 16., 25)
    input_masses = np.array([13.82, 15.94, 13.97, 14.17, 14.95, 15.49, 14.93, 15.59])
    guess_masses = np.array([15.70, 15.10, 15.08, 14.31, 13.61, 13.90, 15.07, 13.82])
    meas_masses = [mass_mock_cluster(m, g) for m, g in zip(input_masses, guess_masses)]
    assert_allclose(meas_masses, input_masses, **TOLERANCE)


def test_z_distr():
    """
    Test the redshift distribution options: single plan, uniform, Chang13, DESC SRD
    """
    cosmo = clmm.Cosmology(H0=70.0, Omega_dm0=0.27 - 0.045, Omega_b0=0.045, Omega_k0=0.0)

    np.random.seed(256429)

    # Set up mock cluster
    ngals = 50000
    mass = 15.0
    zmin = 0.4
    zmax = 3.0
    bins = np.arange(zmin, zmax + 0.1, 0.1)

    data = mock.generate_galaxy_catalog(10**mass, 0.3, 4, cosmo, 0.8, ngals=ngals)
    # Check that all galaxies are at z=0.8
    assert_equal(np.count_nonzero(data["z"] != 0.8), 0)

    data = mock.generate_galaxy_catalog(
        10**mass, 0.3, 4, cosmo, "uniform", ngals=260000, zsrc_min=zmin, zsrc_max=zmax
    )

    # Check that all galaxies are within the given limits
    assert_equal(np.count_nonzero((data["z"] < zmin) | (data["z"] > zmax)), 0)
    # Check that the z distribution is uniform
    hist = np.histogram(data["z"], bins=bins)
    assert_allclose(hist[0], 10000 * np.ones(len(hist[0])), atol=200, rtol=0.02)

    # Check the Chang et al. (2013) redshift distribution
    data = mock.generate_galaxy_catalog(
        10**mass, 0.3, 4, cosmo, "chang13", ngals=ngals, zsrc_min=zmin, zsrc_max=zmax
    )
    # Check that there all galaxies are within the given limits
    assert_equal(np.count_nonzero((data["z"] < zmin) | (data["z"] > zmax)), 0)
    # Check that the z distribution follows Chang13 distribution
    hist = np.histogram(data["z"], bins=bins)
    norm = zdist.chang2013(zmax, is_cdf=True) - zdist.chang2013(zmin, is_cdf=True)
    # Expected number of galaxies in bin i = Ntot*distrib(z_{bin center})*binsize/norm
    chang = np.array([zdist.chang2013(z) * ngals * 0.1 / norm for z in bins[:-1] + 0.05])
    assert_allclose(hist[0], chang, atol=100, rtol=0.1)

    # Check the DESC SRD (2018) redshift distribution
    data = mock.generate_galaxy_catalog(
        10**mass, 0.3, 4, cosmo, "desc_srd", ngals=ngals, zsrc_min=zmin, zsrc_max=zmax
    )
    # Check that there all galaxies are within the given limits
    assert_equal(np.count_nonzero((data["z"] < zmin) | (data["z"] > zmax)), 0)
    # Check that the z distribution follows Chang13 distribution
    hist = np.histogram(data["z"], bins=bins)
    norm = zdist.desc_srd(zmax, is_cdf=True) - zdist.desc_srd(zmin, is_cdf=True)
    # Expected number of galaxies in bin i = Ntot*distrib(z_{bin center})*binsize/norm
    srd = np.array([zdist.desc_srd(z) * ngals * 0.1 / norm for z in bins[:-1] + 0.05])
    assert_allclose(hist[0], srd, atol=100, rtol=0.1)


def test_shapenoise():
    """
    Test that the shape noise distribution is Gaussian around the shear and does not produce
    unphysical ellipticities.
    """
    cosmo = clmm.Cosmology(H0=70.0, Omega_dm0=0.27 - 0.045, Omega_b0=0.045, Omega_k0=0.0)

    np.random.seed(285713)

    # Verify that shape noise does not produce unrealistic ellipticities
    data = mock.generate_galaxy_catalog(10**15.0, 0.3, 4, cosmo, 0.8, ngals=50000, shapenoise=0.5)
    # Check that there are no galaxies with |e|>1
    assert_equal(np.count_nonzero((data["e1"] > 1) | (data["e1"] < -1)), 0)
    assert_equal(np.count_nonzero((data["e2"] > 1) | (data["e2"] < -1)), 0)

    # Verify that the shape noise is Gaussian around 0 (for the very small shear here)
    sigma = 0.25
    data = mock.generate_galaxy_catalog(
        10**12.0, 0.3, 4, cosmo, 0.8, ngals=50000, shapenoise=sigma
    )
    # Check that there are no galaxies with |e|>1
    assert_equal(np.count_nonzero((data["e1"] > 1) | (data["e1"] < -1)), 0)
    assert_equal(np.count_nonzero((data["e2"] > 1) | (data["e2"] < -1)), 0)
    # Check that shape noise is Guassian with correct std dev
    bins = np.arange(-1, 1.1, 0.1)
    gauss = (
        5000 * np.exp(-0.5 * (bins[:-1] + 0.05) ** 2 / sigma**2) / (sigma * np.sqrt(2 * np.pi))
    )
    assert_allclose(np.histogram(data["e1"], bins=bins)[0], gauss, atol=50, rtol=0.05)
    assert_allclose(np.histogram(data["e2"], bins=bins)[0], gauss, atol=50, rtol=0.05)


def test_coordinate_system():
    """
    Test that the coordinate system is correctly set up and that the galaxies are in the correct
    position.
    """
    cosmo = clmm.Cosmology(H0=70.0, Omega_dm0=0.27 - 0.045, Omega_b0=0.045, Omega_k0=0.0)

    # Verify that the coordinate system is correctly set up
    np.random.seed(285713)
    euclidean_data = mock.generate_galaxy_catalog(
        10**15.0,
        0.3,
        4,
        cosmo,
        0.8,
        ngals=50000,
        shapenoise=0.05,
        photoz_sigma_unscaled=0.05,
        coordinate_system="euclidean",
    )
    np.random.seed(285713)
    celestial_data = mock.generate_galaxy_catalog(
        10**15.0,
        0.3,
        4,
        cosmo,
        0.8,
        ngals=50000,
        shapenoise=0.05,
        photoz_sigma_unscaled=0.05,
        coordinate_system="celestial",
    )

    assert_equal(euclidean_data["ra"], celestial_data["ra"])
    assert_equal(euclidean_data["dec"], celestial_data["dec"])
    assert_allclose(
        euclidean_data["e1"],
        celestial_data["e1"],
        **TOLERANCE,
        err_msg="Conversion from euclidean to celestial coordinate system for theta failed"
    )
    assert_allclose(
        euclidean_data["e2"],
        -celestial_data["e2"],
        **TOLERANCE,
        err_msg="Conversion from euclidean to celestial coordinate system for theta failed"
    )

    assert_raises(
        ValueError,
        mock.generate_galaxy_catalog,
        10**15.0,
        0.3,
        4,
        cosmo,
        0.8,
        ngals=50000,
        coordinate_system="blah",
    )
