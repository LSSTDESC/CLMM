"""@file generic.py
Model independent theory functions
"""
# Functions to model halo profiles

import warnings
import numpy as np
from scipy.optimize import fsolve
from scipy.special import gamma, gammainc

# functions that are general to all backends


def compute_reduced_shear_from_convergence(shear, convergence):
    """Calculates reduced shear from shear and convergence

    Parameters
    ----------
    shear : array_like, float
        Shear
    convergence : array_like, float
        Convergence

    Returns
    -------
    g : array_like, float
        Reduced shear
    """
    reduced_shear = np.array(shear) / (1.0 - np.array(convergence))
    return reduced_shear


def compute_magnification_bias_from_magnification(magnification, alpha):
    r"""Computes magnification bias from magnification :math:`\mu` and slope parameter
    :math:`\alpha` as :

    .. math::
        \mu^{\alpha - 1}

    The alpha parameter depends on the source sample and is computed as the slope of the
    cummulative numer counts at a given magnitude:

    .. math::
        \alpha \equiv \alpha(f) = - \frac{\mathrm d}{\mathrm d \log{f}} \log{n_0(>f)}

    or,

    .. math::
        \alpha \equiv \alpha(m) = 2.5 \frac{\mathrm d}{\mathrm d m} \log{n_0(<m)}

    see e.g.  Bartelmann & Schneider 2001; Umetsu 2020

    Parameters
    ----------
    magnification : array_like
        Magnification
    alpha : array like
        Source cummulative number density slope

    Returns
    -------
    magnification bias : array_like
        magnification bias
    """
    if np.any(np.array(magnification) < 0):
        warnings.warn(
            "Magnification is negative for certain radii, \
                    returning nan for magnification bias in this case."
        )
    return np.array(magnification) ** (np.array([alpha]).T - 1)


def compute_rdelta(mdelta, redshift, cosmo, massdef="mean", delta_mdef=200):
    r"""Computes the radius for mdelta

    .. math::
        r_\Delta=\left(\frac{3 M_\Delta}{4 \pi \Delta \rho_{bkg}(z)}\right)^{1/3}

    Parameters
    ----------
    mdelta : float
        Mass in units of :math:`M_\odot`
    redshift : float
        Redshift of the cluster
    cosmo : clmm.Cosmology
        Cosmology object
    massdef : str, None
        Profile mass definition ("mean", "critical", "virial").
    delta_mdef : int, None
        Mass overdensity definition.

    Returns
    -------
    float
        Radius in :math:`M\!pc`.
    """
    # Check massdef
    if massdef == "mean":
        rho = cosmo.get_rho_m
    elif massdef in ("critical", "virial"):
        rho = cosmo.get_rho_c
    else:
        raise ValueError(f"massdef(={massdef}) must be mean, critical or virial")
    return ((3.0 * mdelta) / (4.0 * np.pi * delta_mdef * rho(redshift))) ** (1.0 / 3.0)


def compute_profile_mass_in_radius(
    r3d,
    redshift,
    cosmo,
    mdelta,
    cdelta,
    massdef="mean",
    delta_mdef=200,
    halo_profile_model="nfw",
    alpha=None,
):
    r"""Computes the mass inside a given radius of the profile.
    The mass is calculated as

    .. math::
        M(<\text{r3d}) = M_{\Delta}\;
        \frac{f\left(\frac{\text{r3d}}{r_{\Delta}/c_{\Delta}}\right)}{f(c_{\Delta})},

    where :math:`f(x)` for the different models are

    NFW:

    .. math::
        \quad \ln(1+x)-\frac{x}{1+x}

    Einasto: (:math:`\gamma` is the lower incomplete gamma function)

    .. math::
        \gamma(\frac{3}{\alpha}, \frac{2}{\alpha}x^{\alpha})

    Hernquist:

    .. math::
        \left(\frac{x}{1+x}\right)^2


    Parameters
    ----------
    r3d : array_like, float
        Radial position from the cluster center in :math:`M\!pc`.
    refshift : float
        Redshift of the cluster
    mdelta : float
        Mass of the profile in units of :math:`M_\odot`
    cdelta : float
        Concentration of the profile.
    massdef : str, None
        Profile mass definition ("mean", "critical", "virial").
    delta_mdef : int, None
        Mass overdensity definition.
    halo_profile_model : str
        Profile model parameterization ("nfw", "einasto", "hernquist").
    alpha : float, None
        Einasto slope, required when `halo_profile_model='einasto'`.

    Returns
    -------
    array_like, float
        Mass in units of :math:`M_\odot`
    """
    # pylint: disable=unnecessary-lambda-assignment
    rdelta = compute_rdelta(mdelta, redshift, cosmo, massdef, delta_mdef)
    # Check halo_profile_model
    if halo_profile_model == "nfw":
        prof_integ = lambda c: np.log(1.0 + c) - c / (1.0 + c)
    elif halo_profile_model == "einasto":
        if alpha is None:
            raise ValueError("alpha must be provided when Einasto profile is selected!")
        prof_integ = lambda c: gamma(3.0 / alpha) * gammainc(3.0 / alpha, 2.0 / alpha * c**alpha)
    elif halo_profile_model == "hernquist":
        prof_integ = lambda c: (c / (1.0 + c)) ** 2.0
    else:
        raise ValueError(
            f"halo_profile_model=(={halo_profile_model=}) must be " "nfw, einasto, or hernquist!"
        )
    r3d_norm = np.array(r3d) / (rdelta / cdelta)
    return mdelta * prof_integ(r3d_norm) / prof_integ(cdelta)


def convert_profile_mass_concentration(
    mdelta,
    cdelta,
    redshift,
    cosmo,
    massdef,
    delta_mdef,
    halo_profile_model,
    massdef2=None,
    delta_mdef2=None,
    halo_profile_model2=None,
    alpha=None,
    alpha2=None,
):
    r"""
    Parameters
    ----------
    mdelta : float
        Mass of the profile in units of :math:`M_\odot`
    cdelta : float
        Concentration of the profile.
    refshift : float
        Redshift of the cluster
    cosmo : clmm.Cosmology
        Cosmology object
    massdef : str, None
        Input profile mass definition ("mean", "critical", "virial").
    delta_mdef : int, None
        Input mass overdensity definition.
    halo_profile_model : str, None
        Input profile model parameterization ("nfw", "einasto", "hernquist").
    massdef2 : str, None
        Profile mass definition to convert to ("mean", "critical", "virial").
        If None, `massdef2=massdef`.
    delta_mdef2 : int, None
        Mass overdensity definition to convert to.
        If None, `delta_mdef2=delta_mdef`.
    halo_profile_model2 : str, None
        Profile model parameterization to convert to ("nfw", "einasto", "hernquist").
        If None, `halo_profile_model2=halo_profile_model`.
    alpha : float, None
        Input Einasto slope when `halo_profile_model='einasto'`.
    alpha2 : float, None
        Einasto slope to convert to when `halo_profile_model='einasto'`.
        If None, `alpha2=alpha`.

    Returns
    -------
    HaloProfile:
        HaloProfile object
    """
    # pylint: disable=unused-argument
    rdelta = compute_rdelta(mdelta, redshift, cosmo, massdef, delta_mdef)
    # Prep other args
    loc, keys = locals(), ("massdef", "delta_mdef", "halo_profile_model", "alpha")
    kwargs = {key: loc[key] for key in keys}
    kwargs2 = {key: (loc[key] if loc[f"{key}2"] is None else loc[f"{key}2"]) for key in keys}

    # Eq. to solve
    def delta_mass(params):
        mdelta2, cdelta2 = params
        rdelta2 = compute_rdelta(
            mdelta2, redshift, cosmo, kwargs2["massdef"], kwargs2["delta_mdef"]
        )
        mdelta2_rad1 = compute_profile_mass_in_radius(
            rdelta, redshift, cosmo, mdelta2, cdelta2, **kwargs2
        )
        mdelta1_rad2 = compute_profile_mass_in_radius(
            rdelta2, redshift, cosmo, mdelta, cdelta, **kwargs
        )
        return mdelta - mdelta2_rad1, mdelta2 - mdelta1_rad2

    # Iterate 2 times:
    mdelta2, cdelta2 = fsolve(func=delta_mass, x0=[mdelta, cdelta])
    mdelta2, cdelta2 = fsolve(func=delta_mass, x0=[mdelta2, cdelta2])

    return mdelta2, cdelta2
