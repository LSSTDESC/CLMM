"""General utility functions that are used in multiple modules"""
import numpy as np


def convert_shapes_to_epsilon(shape_1, shape_2, shape_definition="epsilon", kappa=0):
    r"""Convert shape components 1 and 2 appropriately to make them estimators of the
    reduced shear once averaged.  The shape 1 and 2 components may correspond to ellipticities
    according the :math:`\epsilon`- or :math:`\chi`-definition, but also to the 1 and 2 components
    of the shear. See Bartelmann & Schneider 2001 for details
    (https://arxiv.org/pdf/astro-ph/9912508.pdf).

    The :math:`\epsilon`-ellipticity is a direct estimator of
    the reduced shear. The shear :math:`\gamma` may be converted to reduced shear :math:`g` if the
    convergence :math:`\kappa` is known. The conversions are given below.

    .. math::
     \epsilon = \frac{\chi}{1+(1-|\chi|^2)^{1/2}}

    .. math::
     g=\frac{\gamma}{1-\kappa}

    - If `shape_definition = 'chi'`, this function returns the corresponding `epsilon`
      ellipticities

    - If `shape_definition = 'shear'`, it returns the corresponding reduced shear, given the
      convergence `kappa`

    - If `shape_definition = 'epsilon'` or `'reduced_shear'`, it returns them as is as no
      conversion is needed.

    Parameters
    ----------
    shape_1 : float, numpy.ndarray
        Input shapes or shears along principal axis (g1 or e1)
    shape_2 : float, numpy.ndarray
        Input shapes or shears along secondary axis (g2 or e2)
    shape_definition : str, optional
        Definition of the input shapes, can be ellipticities 'epsilon' or 'chi' or shears 'shear'
        or 'reduced_shear'. Defaut: 'epsilon'
    kappa : float, numpy.ndarray, optional
        Convergence for transforming to a reduced shear. Default is 0

    Returns
    -------
    epsilon_1 : float, numpy.ndarray
        Epsilon ellipticity (or reduced shear) along principal axis (epsilon1)
    epsilon_2 : float, numpy.ndarray
        Epsilon ellipticity (or reduced shear) along secondary axis (epsilon2)
    """

    if shape_definition in ("epsilon", "reduced_shear"):
        epsilon_1, epsilon_2 = shape_1, shape_2
    elif shape_definition == "chi":
        chi_to_eps_conversion = 1.0 / (1.0 + (1 - (shape_1**2 + shape_2**2)) ** 0.5)
        epsilon_1, epsilon_2 = (
            shape_1 * chi_to_eps_conversion,
            shape_2 * chi_to_eps_conversion,
        )
    elif shape_definition == "shear":
        epsilon_1, epsilon_2 = shape_1 / (1.0 - kappa), shape_2 / (1.0 - kappa)
    else:
        raise TypeError("Please choose epsilon, chi, shear, reduced_shear")
    return epsilon_1, epsilon_2


def build_ellipticities(q11, q22, q12):
    """Build ellipticties from second moments. See, e.g., Schneider et al. (2006)

    Parameters
    ----------
    q11 : float, numpy.ndarray
        Second brightness moment tensor, component (1,1)
    q22 : float, numpy.ndarray
        Second brightness moment tensor, component (2,2)
    q12 :  float, numpy.ndarray
        Second brightness moment tensor, component (1,2)

    Returns
    -------
    chi1, chi2 : float, numpy.ndarray
        Ellipticities using the "chi definition"
    epsilon1, epsilon2 : float, numpy.ndarray
        Ellipticities using the "epsilon definition"
    """
    norm_x, norm_e = q11 + q22, q11 + q22 + 2 * np.sqrt(q11 * q22 - q12 * q12)
    chi1, chi2 = (q11 - q22) / norm_x, 2 * q12 / norm_x
    epsilon1, epsilon2 = (q11 - q22) / norm_e, 2 * q12 / norm_e
    return chi1, chi2, epsilon1, epsilon2


def compute_lensed_ellipticity(ellipticity1_true, ellipticity2_true, shear1, shear2, convergence):
    r"""Compute lensed ellipticities from the intrinsic ellipticities, shear and convergence.
    Following Schneider et al. (2006)

    .. math::
        \epsilon^{\text{lensed}}=\epsilon^{\text{lensed}}_1+i\epsilon^{\text{lensed}}_2=
        \frac{\epsilon^{\text{true}}+g}{1+g^\ast\epsilon^{\text{true}}},

    where, the complex reduced shear :math:`g` is obtained from the shear
    :math:`\gamma=\gamma_1+i\gamma_2` and convergence :math:`\kappa` as :math:`g =
    \gamma/(1-\kappa)`, and the complex intrinsic ellipticity is :math:`\epsilon^{\text{
    true}}=\epsilon^{\text{true}}_1+i\epsilon^{\text{true}}_2`

    Parameters
    ----------
    ellipticity1_true : float, numpy.ndarray
        Intrinsic ellipticity of the sources along the principal axis
    ellipticity2_true : float, numpy.ndarray
        Intrinsic ellipticity of the sources along the second axis
    shear1 :  float, numpy.ndarray
        Shear component (not reduced shear) along the principal axis at the source location
    shear2 :  float, numpy.ndarray
        Shear component (not reduced shear) along the 45-degree axis at the source location
    convergence :  float, numpy.ndarray
        Convergence at the source location
    Returns
    -------
    e1, e2 : float, numpy.ndarray
        Lensed ellipicity along both reference axes.
    """
    # shear (as a complex number)
    shear = shear1 + shear2 * 1j
    # intrinsic ellipticity (as a complex number)
    ellipticity_true = ellipticity1_true + ellipticity2_true * 1j
    # reduced shear
    reduced_shear = shear / (1.0 - convergence)
    # lensed ellipticity
    lensed_ellipticity = (ellipticity_true + reduced_shear) / (
        1.0 + reduced_shear.conjugate() * ellipticity_true
    )
    return np.real(lensed_ellipticity), np.imag(lensed_ellipticity)
