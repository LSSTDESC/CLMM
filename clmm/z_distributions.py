from .utils import _srd_z_distrib, _chang_z_distrib

def chang2013(redshift, is_cdf=False):
    """
    Chang et al (2013) unnormalized galaxy redshift distribution function, with the fiducial
    set of parameters.

    Parameters
    ----------
    redshift : float
        Galaxy redshift
    is_cdf : bool
        If True, returns cumulative distribution function.

    Returns
    -------
    The value of the distribution at z
    """
    return _chang_z_distrib(redshift, is_cdf)

def desc_srd(redshift, is_cdf=False):
    """
    Unnormalized galaxy redshift distribution function used in
    the LSST/DESC Science Requirement Document (arxiv:1809.01669).

    Parameters
    ----------
    redshift : float
        Galaxy redshift
    is_cdf : bool
        If True, returns cumulative distribution function.

    Returns
    -------
    The value of the distribution at z
    """
    return _srd_z_distrib(redshift, is_cdf)
