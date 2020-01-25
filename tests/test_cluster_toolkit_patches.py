""" Tests for the cluster_toolkit patches"""
import numpy as np
from numpy.testing import assert_allclose
from clmm.cluster_toolkit_patches import _patch_zevolution_cluster_toolkit_rho_m as patch1
from clmm.constants import Constants as c


TOLERANCE = {'rtol': 1.0e-6, 'atol': 1.0e-6}


def test_evolve_omega_m_flatlcdm():
    # Test for several values of omega_m and redshift
    omega_m0_list = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    zlist = [0.1, 0.2, 0.3, 0.5, 0.6418, 1.0, 1.3, 30., 119.328]

    rhocrit_mks = 3.*100.*100./(8.*np.pi*c.GNEWT.value)
    rhocrit_cosmo = rhocrit_mks * 1000. * 1000. * c.PC_TO_METER.value * 1.e6 / c.SOLAR_MASS.value
    rhocrit_cltk = 2.77533742639e+11
    ratio = rhocrit_cosmo/rhocrit_cltk

    for i in range(len(omega_m0_list)):
        for j in range(len(zlist)):
            assert_allclose(patch1(omega_m0_list[i], zlist[j]),
                            omega_m0_list[i] * (1.0 + zlist[j])**3 * ratio, **TOLERANCE)

    # Check that at redshift zero nothing changes
    assert_allclose(patch1(0.3, 0.0), 0.3*ratio, **TOLERANCE)
