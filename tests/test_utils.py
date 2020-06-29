# pylint: disable=no-member, protected-access
""" Tests for utils.py """
import numpy as np
from numpy.testing import assert_raises, assert_allclose
from astropy.cosmology import FlatLambdaCDM

import clmm.utils as utils
from clmm.utils import compute_radial_averages, make_bins, convert_shapes_to_epsilon


TOLERANCE = {'rtol': 1.0e-6, 'atol': 0}


def test_compute_radial_averages():
    """ Tests compute_radial_averages, a function that computes several binned statistics """
    # Make some test data
    binvals = np.array([2., 3., 6., 8., 4., 9.])
    xbins1 = [0., 10.]
    xbins2 = [0., 5., 10.]

    # Test requesting an unsupported error model
    assert_raises(ValueError, compute_radial_averages, binvals, binvals, [0., 10.], 'glue')

    # Check the default error model
    assert_allclose(compute_radial_averages(binvals, binvals, xbins1)[:4],
                    [[np.mean(binvals)], [np.mean(binvals)], [np.std(binvals)/np.sqrt(len(binvals))],
                    [6]],
                    **TOLERANCE)


    # Test 3 objects in one bin with various error models
    assert_allclose(compute_radial_averages(binvals, binvals, xbins1, error_model='std/sqrt_n')[:4],
                    [[np.mean(binvals)], [np.mean(binvals)], [np.std(binvals)/np.sqrt(len(binvals))], [6]],
                    **TOLERANCE)
    assert_allclose(compute_radial_averages(binvals, binvals, xbins1, error_model='std')[:4],
                    [[np.mean(binvals)], [np.mean(binvals)], [np.std(binvals)], 
                    [6]], **TOLERANCE)

    # A slightly more complicated case with two bins
    inbin1 = binvals[(binvals > xbins2[0]) & (binvals < xbins2[1])]
    inbin2 = binvals[(binvals > xbins2[1]) & (binvals < xbins2[2])]
    assert_allclose(compute_radial_averages(binvals, binvals, xbins2, error_model='std/sqrt_n')[:4],
                    [[np.mean(inbin1), np.mean(inbin2)], [np.mean(inbin1), np.mean(inbin2)],
                     [np.std(inbin1)/np.sqrt(len(inbin1)), np.std(inbin2)/np.sqrt(len(inbin2))],
                     [3,3]], **TOLERANCE)
    assert_allclose(compute_radial_averages(binvals, binvals, xbins2, error_model='std')[:4],
                    [[np.mean(inbin1), np.mean(inbin2)], [np.mean(inbin1), np.mean(inbin2)],
                     [np.std(inbin1), np.std(inbin2)], 
                     [3,3]], **TOLERANCE)

    # Test a much larger, random sample with unevenly spaced bins
    binvals = np.loadtxt('tests/data/radial_average_test_array.txt')
    xbins2 = [0.0, 3.33, 6.66, 10.0]
    inbin1 = binvals[(binvals > xbins2[0]) & (binvals < xbins2[1])]
    inbin2 = binvals[(binvals > xbins2[1]) & (binvals < xbins2[2])]
    inbin3 = binvals[(binvals > xbins2[2]) & (binvals < xbins2[3])]
    assert_allclose(compute_radial_averages(binvals, binvals, xbins2, error_model='std/sqrt_n')[:4],
                    [[np.mean(inbin1), np.mean(inbin2), np.mean(inbin3)],
                     [np.mean(inbin1), np.mean(inbin2), np.mean(inbin3)],
                     [np.std(inbin1)/np.sqrt(len(inbin1)), np.std(inbin2)/np.sqrt(len(inbin2)),
                      np.std(inbin3)/np.sqrt(len(inbin3))],
                     [inbin1.size, inbin2.size, inbin3.size]], **TOLERANCE)
    assert_allclose(compute_radial_averages(binvals, binvals, xbins2, error_model='std')[:4],
                    [[np.mean(inbin1), np.mean(inbin2), np.mean(inbin3)],
                     [np.mean(inbin1), np.mean(inbin2), np.mean(inbin3)],
                     [np.std(inbin1), np.std(inbin2), np.std(inbin3)],
                     [inbin1.size, inbin2.size, inbin3.size]], **TOLERANCE)



def test_make_bins():
    """ Test the make_bins function. Right now this function is pretty simplistic and the
    tests are pretty circular. As more functionality is added here the tests will
    become more substantial.
    """
    # Test various combinations of rmin and rmax with default values
    assert_allclose(make_bins(0.0, 10.), np.linspace(0.0, 10., 11), **TOLERANCE)
    assert_raises(ValueError, make_bins, 0.0, -10.)
    assert_raises(ValueError, make_bins, -10., 10.)
    assert_raises(ValueError, make_bins, -10., -5.)
    assert_raises(ValueError, make_bins, 10., 0.0)

    # Test various nbins
    assert_allclose(make_bins(0.0, 10., nbins=3), np.linspace(0.0, 10., 4), **TOLERANCE)
    assert_allclose(make_bins(0.0, 10., nbins=13), np.linspace(0.0, 10., 14), **TOLERANCE)
    assert_raises(ValueError, make_bins, 0.0, 10., -10)
    assert_raises(ValueError, make_bins, 0.0, 10., 0)

    # Test default method
    assert_allclose(make_bins(0.0, 10., nbins=10),
                    make_bins(0.0, 10., nbins=10, method='evenwidth'),
                    **TOLERANCE)

    # Test the different binning methods
    assert_allclose(make_bins(0.0, 10., nbins=10, method='evenwidth'),
                    np.linspace(0.0, 10., 11), **TOLERANCE)
    assert_allclose(make_bins(1.0, 10., nbins=10, method='evenlog10width'),
                    np.logspace(np.log10(1.0), np.log10(10.), 11), **TOLERANCE)
    
    # Test equaloccupation method. It needs a source_seps array, so create one
    test_array = np.sqrt(np.random.uniform(-10,10,1361)**2 + np.random.uniform(-10,10,1361)**2)
    test_bins = make_bins(1.0, 10., nbins=10, method='equaloccupation', source_seps=test_array)
    # Check that all bins have roughly equal occupation. 
    # Assert needs atol=2, because len(source_seps)/nbins may not be an integer, 
    # and for some random arrays atol=1 is not enough. 
    assert_allclose(np.diff(np.histogram(test_array,bins=test_bins)[0]),
                    np.zeros(9), atol=2)
    test_bins = make_bins(0.51396, 6.78, nbins=23, method='equaloccupation', source_seps=test_array)
    assert_allclose(np.diff(np.histogram(test_array,bins=test_bins)[0]),
                    np.zeros(22), atol=2)
                    

def _rad_to_mpc_helper(dist, redshift, cosmo, do_inverse):
    """ Helper function to clean up test_convert_rad_to_mpc. Truth is computed using
    astropy so this test is very circular. Once we swap to CCL very soon this will be
    a good source of truth. """
    d_a = cosmo.angular_diameter_distance(redshift).to('Mpc').value
    if do_inverse:
        truth = dist / d_a
    else:
        truth = dist * d_a
    assert_allclose(utils._convert_rad_to_mpc(dist, redshift, cosmo, do_inverse),
                    truth, **TOLERANCE)


def test_convert_rad_to_mpc():
    """ Test conversion between physical and angular units and vice-versa. """
    # Set some default values if I want them
    redshift = 0.25
    cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

    # Test the default behavior
    assert_allclose(utils._convert_rad_to_mpc(0.33, redshift, cosmo),
                    utils._convert_rad_to_mpc(0.33, redshift, cosmo, False), **TOLERANCE)

    # Test basic conversions each way
    _rad_to_mpc_helper(0.003, redshift, cosmo, do_inverse=False)
    _rad_to_mpc_helper(1.0, redshift, cosmo, do_inverse=True)

    # Convert back and forth and make sure I get the same answer
    midtest = utils._convert_rad_to_mpc(0.003, redshift, cosmo)
    assert_allclose(utils._convert_rad_to_mpc(midtest, redshift, cosmo, do_inverse=True),
                    0.003, **TOLERANCE)

    # Test some different redshifts
    for onez in [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]:
        _rad_to_mpc_helper(0.33, onez, cosmo, do_inverse=False)
        _rad_to_mpc_helper(1.0, onez, cosmo, do_inverse=True)

    # Test some different H0
    for oneh0 in [30., 50., 67.3, 74.7, 100.]:
        _rad_to_mpc_helper(0.33, 0.5, FlatLambdaCDM(H0=oneh0, Om0=0.3), do_inverse=False)
        _rad_to_mpc_helper(1.0, 0.5, FlatLambdaCDM(H0=oneh0, Om0=0.3), do_inverse=True)

    # Test some different Omega_M
    for oneomm in [0.1, 0.3, 0.5, 1.0]:
        _rad_to_mpc_helper(0.33, 0.5, FlatLambdaCDM(H0=70., Om0=oneomm), do_inverse=False)
        _rad_to_mpc_helper(1.0, 0.5, FlatLambdaCDM(H0=70., Om0=oneomm), do_inverse=True)


def test_convert_units():
    """ Test the wrapper function to convert units. Corner cases should be tested in the
    individual functions. This function should test one case for all supported conversions
    and the error handling.
    """
    # Make an astropy cosmology object for testing
    cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

    # Test that each unit is supported
    utils.convert_units(1.0, 'radians', 'degrees')
    utils.convert_units(1.0, 'arcmin', 'arcsec')
    utils.convert_units(1.0, 'Mpc', 'kpc')
    utils.convert_units(1.0, 'Mpc', 'kpc')

    # Error checking
    assert_raises(ValueError, utils.convert_units, 1.0, 'radians', 'CRAZY')
    assert_raises(ValueError, utils.convert_units, 1.0, 'CRAZY', 'radians')
    assert_raises(TypeError, utils.convert_units, 1.0, 'arcsec', 'Mpc')
    assert_raises(TypeError, utils.convert_units, 1.0, 'arcsec', 'Mpc', None, cosmo)
    assert_raises(TypeError, utils.convert_units, 1.0, 'arcsec', 'Mpc', 0.5, None)

    # Test cases to make sure angular -> angular is fitting together
    assert_allclose(utils.convert_units(np.pi, 'radians', 'degrees'), 180., **TOLERANCE)
    assert_allclose(utils.convert_units(180.0, 'degrees', 'radians'), np.pi, **TOLERANCE)
    assert_allclose(utils.convert_units(1.0, 'degrees', 'arcmin'), 60., **TOLERANCE)
    assert_allclose(utils.convert_units(1.0, 'degrees', 'arcsec'), 3600., **TOLERANCE)

    # Test cases to make sure physical -> physical is fitting together
    assert_allclose(utils.convert_units(1.0, 'Mpc', 'kpc'), 1.0e3, **TOLERANCE)
    assert_allclose(utils.convert_units(1000., 'kpc', 'Mpc'), 1.0, **TOLERANCE)
    assert_allclose(utils.convert_units(1.0, 'Mpc', 'pc'), 1.0e6, **TOLERANCE)

    # Test conversion from angular to physical
    # Using astropy, circular now but this will be fine since we are going to be
    # swapping to CCL soon and then its kosher
    r_arcmin, redshift = 20.0, 0.5
    d_a = cosmo.angular_diameter_distance(redshift).to('kpc').value
    truth = r_arcmin * (1.0 / 60.0) * (np.pi / 180.0) * d_a
    assert_allclose(utils.convert_units(r_arcmin, 'arcmin', 'kpc', redshift, cosmo),
                    truth, **TOLERANCE)

    # Test conversion both ways between angular and physical units
    # Using astropy, circular now but this will be fine since we are going to be
    # swapping to CCL soon and then its kosher
    r_kpc, redshift = 20.0, 0.5
    d_a = cosmo.angular_diameter_distance(redshift).to('kpc').value
    truth = r_kpc * (1.0 / d_a) * (180. / np.pi) * 60.
    assert_allclose(utils.convert_units(r_kpc, 'kpc', 'arcmin', redshift, cosmo),
                    truth, **TOLERANCE)

def test_build_ellipticities():
    
    # second moments are floats
    q11 = 0.5
    q22 = 0.3
    q12 = 0.02
    
    assert_allclose(utils.build_ellipticities(q11,q22,q12),(0.25, 0.05, 0.12710007580505459, 
                                                            0.025420015161010917), **TOLERANCE)
    
    # second moments are numpy array
    q11 = np.array([0.5,0.3])
    q22 = np.array([0.8,0.2])
    q12 = np.array([0.01,0.01])

    assert_allclose(utils.build_ellipticities(q11,q22,q12),([-0.23076923,  0.2],
                                                            [0.01538462, 0.04],
                                                            [-0.11697033,  0.10106221],
                                                            [0.00779802, 0.02021244]), **TOLERANCE)
    
def test_shape_conversion():
    """ Test the helper function that convert user defined shapes into
    epsilon ellipticities or reduced shear. Both can be used for the galcat in 
    the GalaxyCluster object"""
    
    
    # Number of random ellipticities to check
    niter=25

    # Create random second moments and from that random ellipticities
    q11,q22 = np.random.randint(0,20,(2,niter))
    # Q11 seperate to avoid a whole bunch of nans
    q12 = np.random.uniform(-1,1,niter)*np.sqrt(q11*q22)
    x1,x2,e1,e2 = utils.build_ellipticities(q11,q22,q12)
    
    # Test conversion from 'chi' to epsilon
    e1_2,e2_2 = convert_shapes_to_epsilon(x1,x2,shape_definition='chi')
    assert_allclose(e1,e1_2, **TOLERANCE)
    assert_allclose(e2,e2_2, **TOLERANCE)
    
    # Test that 'epsilon' just returns the same values
    e1_2,e2_2 = convert_shapes_to_epsilon(e1,e2,shape_definition='epsilon')
    assert_allclose(e1,e1_2, **TOLERANCE)
    assert_allclose(e2,e2_2, **TOLERANCE)
    
    # Test that 'reduced_shear' just returns the same values
    e1_2,e2_2 = convert_shapes_to_epsilon(e1,e2,shape_definition='reduced_shear')
    assert_allclose(e1,e1_2, **TOLERANCE)
    assert_allclose(e2,e2_2, **TOLERANCE)
    
    # Test that 'shear' just returns the right values for reduced shear
    e1_2,e2_2 = convert_shapes_to_epsilon(e1,e2,shape_definition='shear',kappa=0.2)
    assert_allclose(e1/0.8,e1_2, **TOLERANCE)
    assert_allclose(e2/0.8,e2_2, **TOLERANCE)

def test_compute_lensed_ellipticities():
    
    # Validation test with floats
    es1 = 0
    es2 = 0
    gamma1 = 0.2
    gamma2 = 0.2
    kappa = 0.5   
    assert_allclose(utils.compute_lensed_ellipticity(es1, es2, gamma1, gamma2, kappa),(0.4,0.4), **TOLERANCE)
    
    # Validation test with array
    es1 = np.array([0,0.5])
    es2 = np.array([0,0.1])
    gamma1 = np.array([0.2,0.])
    gamma2 = np.array([0.2,0.3])
    kappa = np.array([0.5,0.2])   
    
    assert_allclose(utils.compute_lensed_ellipticity(es1, es2, gamma1, gamma2, kappa),
                    ([0.4, 0.38656171],[0.4, 0.52769188]), **TOLERANCE)
