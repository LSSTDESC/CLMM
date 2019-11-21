import numpy as np
from numpy import testing
import clmm

def test_get_reduced_shear():
    testing.assert_raises(ValueError, clmm.get_reduced_shear_from_convergence, np.array([-1,2,1]),
                          np.array([0.4,0.2]))
    
    testing.assert_equal(clmm.get_reduced_shear_from_convergence(1,0.5), 2)
    testing.assert_equal(clmm.get_reduced_shear_from_convergence(np.array([0.5, 0.75, 1.25]),
                                                                 np.array([0.75, -0.2, 0])),
                                                                 np.array([2, 0.625, 1.25]))
    