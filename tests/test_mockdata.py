"""Tests for examples/support/mock_data.py"""
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.cosmology import FlatLambdaCDM
import clmm
import clmm.polaraveraging as pa
import sys
sys.path.append('examples/support')
import mock_data as mock
from sampler import fitters

TOLERANCE = {'rtol': 5.0e-4, 'atol': 1.e-4}


def test_mock_data():
    """ Run generate_galaxy_catalog 1000 times and assert that retrieved mass is always consistent with input
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.27, Ob0=0.045)
    
    
    def nfw_shear_profile(r, logm, z_src):
        m = 10.**logm
        gt_model = clmm.predict_reduced_tangential_shear(r*cosmo.h,
                                                         m, 4, 0.3, z_src, cosmo,
                                                         delta_mdef=200,
                                                         halo_profile_model='nfw')
        return gt_model

    def mass_mock_cluster(mass=15.,guess=15.):
        
        # Set up mock cluster
        ngals=50000
        data = mock.generate_galaxy_catalog(10**mass, 0.3, 4, cosmo, 200, 0.8, ngals=ngals)
        
        # Check whether the given ngals is the retrieved ngals
        assert_equal(len(data['ra']),ngals)
        
        # Check that there are no galaxies with |e|>1
        assert_equal(np.count_nonzero((data['e1']>1) | (data['e1']<-1)),0)
        assert_equal(np.count_nonzero((data['e2']>1) | (data['e2']<-1)),0)
        
        # Create shear profile
        cl = clmm.GalaxyCluster("test_cluster", 0.0, 0.0, 0.3, data)
        theta, g_t, g_x = pa.compute_tangential_and_cross_components(cl, geometry="flat")
        binned = pa.make_binned_profile(cl, "radians", "Mpc", bins=pa.make_bins(0.5, 5.0, 100), 
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

    
# Test the Chang distribution
#def test_chang_z_distr():
    
    