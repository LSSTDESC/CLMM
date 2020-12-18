"""Tests for examples/support/mock_data.py"""
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import clmm
import clmm.dataops as da
import sys
from clmm.support import mock_data as mock
from clmm.support.sampler import fitters
from clmm import Cosmology


TOLERANCE = {'rtol': 5.0e-4, 'atol': 1.e-4}
cosmo = Cosmology(H0 = 70.0, Omega_dm0 = 0.27 - 0.045, Omega_b0 = 0.045, Omega_k0 = 0.0)


def test_mock_data():
    """ Run generate_galaxy_catalog 1000 times and assert that retrieved mass is always consistent with input
    """
    
    
    def nfw_shear_profile(r, logm, z_src):
        m = 10.**logm
        gt_model = clmm.compute_reduced_tangential_shear(r,
                                                         m, 4, 0.3, z_src, cosmo,
                                                         delta_mdef=200,
                                                         halo_profile_model='nfw')
        return gt_model

    def mass_mock_cluster(mass=15.,guess=15.):
        
        # Set up mock cluster
        ngals=50000
        data = mock.generate_galaxy_catalog(10**mass, 0.3, 4, cosmo, 0.8, ngals=ngals)
        
        # Check whether the given ngals is the retrieved ngals
        assert_equal(len(data['ra']),ngals)
        
        # Check that there are no galaxies with |e|>1
        assert_equal(np.count_nonzero((data['e1']>1) | (data['e1']<-1)),0)
        assert_equal(np.count_nonzero((data['e2']>1) | (data['e2']<-1)),0)
        
        # Create shear profile
        cl = clmm.GalaxyCluster("test_cluster", 0.0, 0.0, 0.3, data)
        theta, g_t, g_x = cl.compute_tangential_and_cross_components(geometry="flat")
        binned = cl.make_radial_profile("Mpc", bins=da.make_bins(0.5, 5.0, 100), 
                                  cosmo=cosmo, include_empty_bins=False)

        popt,pcov = fitters['curve_fit'](lambda r, logm: nfw_shear_profile(r, logm, 0.8), 
                            binned['radius'], 
                            binned['gt'], 
                            np.ones_like(binned['gt'])*1.e-5, 
                            bounds=[13.,17.],p0=guess)

        return popt[0]
    
    
    #input_masses = np.random.uniform(13.5, 16., 25)
    #logm_0 = np.random.uniform(13.5, 16., 25)
    input_masses = np.array([13.82,15.94,13.97,14.17,14.95,15.49,14.93,15.59])
    guess_masses = np.array([15.70,15.10,15.08,14.31,13.61,13.90,15.07,13.82])
    meas_masses = [mass_mock_cluster(m,g) for m,g in zip(input_masses,guess_masses)]
    assert_allclose(meas_masses,input_masses, **TOLERANCE)

    
def test_z_distr():
    """
    Test the redshift distribution options: single plan, uniform, Chang13.
    """    
    
    np.random.seed(256429)
    
    # Set up mock cluster
    ngals=50000; mass=15.
    zmin=0.4; zmax=3.0
    bins = np.arange(zmin,zmax+0.1,0.1)
    
    data = mock.generate_galaxy_catalog(10**mass, 0.3, 4, cosmo, 0.8, ngals=ngals)
    # Check that all galaxies are at z=0.8
    assert_equal(np.count_nonzero(data['z']!=0.8),0)
    
    data = mock.generate_galaxy_catalog(10**mass, 0.3, 4, cosmo, 'uniform', ngals=260000,
                                        zsrc_min=zmin, zsrc_max=zmax)
    
    # Check that all galaxies are within the given limits
    assert_equal(np.count_nonzero((data['z']<zmin)|(data['z']>zmax)),0)
    # Check that the z distribution is uniform
    hist = np.histogram(data['z'],bins=bins)
    assert_allclose(hist[0],10000*np.ones(len(hist[0])),atol=200,rtol=0.02)
    
    
    data = mock.generate_galaxy_catalog(10**mass, 0.3, 4, cosmo, 'chang13', ngals=ngals,
                                        zsrc_min=zmin, zsrc_max=zmax)
    # Check that there all galaxies are within the given limits
    assert_equal(np.count_nonzero((data['z']<zmin)|(data['z']>zmax)),0)
    # Check that the z distribution follows Chang13 distribution
    hist = np.histogram(data['z'],bins=bins)
    chang = np.array([mock._chang_z_distrib(z) for z in bins[:-1]+0.05])
    assert_allclose(hist[0],chang*ngals/2,atol=100,rtol=0.1)
    
    
    
    
def test_shapenoise():
    """
    Test that the shape noise distribution is Gaussian around the shear and does not produce unphysical ellipticities.
    """    
    
    np.random.seed(285713)

    # Verify that shape noise does not produce unrealistic ellipticities
    data = mock.generate_galaxy_catalog(10**15., 0.3, 4, cosmo, 0.8, ngals=50000,shapenoise=0.5)
    # Check that there are no galaxies with |e|>1
    assert_equal(np.count_nonzero((data['e1']>1) | (data['e1']<-1)),0)
    assert_equal(np.count_nonzero((data['e2']>1) | (data['e2']<-1)),0)
    
    
    # Verify that the shape noise is Gaussian around 0 (for the very small shear here)
    sigma=0.25
    data = mock.generate_galaxy_catalog(10**12., 0.3, 4, cosmo, 0.8, ngals=50000,shapenoise=sigma)
    # Check that there are no galaxies with |e|>1
    assert_equal(np.count_nonzero((data['e1']>1) | (data['e1']<-1)),0)
    assert_equal(np.count_nonzero((data['e2']>1) | (data['e2']<-1)),0)
    # Check that shape noise is Guassian with correct std dev
    bins=np.arange(-1,1.1,0.1)
    gauss = 5000*np.exp(-0.5*(bins[:-1]+0.05)**2/sigma**2)/(sigma*np.sqrt(2*np.pi))
    assert_allclose(np.histogram(data['e1'],bins=bins)[0],gauss,atol=50,rtol=0.05)
    assert_allclose(np.histogram(data['e2'],bins=bins)[0],gauss,atol=50,rtol=0.05)
