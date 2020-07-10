"""Functions to compute polar/azimuthal averages of data in radial bins (formerly known as polaraveraging.py)"""
# try: # 7481794
#     import pyccl as ccl
# except ImportError:
#     pass
import numpy as np
import warnings

from .galaxycluster import GalaxyCluster
from .gcdata import GCData
from .hybrid import make_binned_profile
from .utils import _compute_tangential_shear, _compute_cross_shear, _compute_lensing_angles_flatsky


def compute_tangential_and_cross_components(cluster=None, shape_component1='e1', shape_component2='e2', tan_component='et', cross_component='ex', ra_lens=None, dec_lens=None, ra_source_list=None, dec_source_list=None, shear1=None, shear2=None, geometry='flat', add_to_cluster=True):
    r"""Computes tangential- and cross- components for shear or ellipticity

    To compute the shear, we need the right ascension and declination of the lens and of
    all of the sources. We also need the two shear components of all of the sources.

    These quantities can be handed to `compute_shear` in three ways

    1. Pass in each as parameters::

        compute_shear(ra_lens, dec_lens, ra_source_list, dec_source_list, shape_component1, shape_component2)

    2. Given a `GalaxyCluster` object::

        compute_shear(cluster)
_compute_cross_shear
    3. As a method of `GalaxyCluster`::

        cluster.compute_shear()

    The angular separation between the source and the lens, :math:`\theta`, and the azimuthal
    position of the source relative to the lens, :math:`\phi`, are computed within the function
    and the angular separation is returned.

    In the flat sky approximation, these angles are calculated using (_lens: lens, _source: source,
    RA is from right to left)

    .. math::

        \theta^2 = & \left(\delta_s - \delta_l\right)^2 +
        \left(\alpha_l-\alpha_s\right)^2\cos^2(\delta_l)\\
        \tan\phi = & \frac{\delta_s - \delta_l}{\left(\alpha_l - \alpha_s\right)\cos(\delta_l)}

    The tangential, :math:`g_t`, and cross, :math:`g_x`, ellipticity/shear components are calculated using the two
    ellipticity/shear components :math:`g_1` and :math:`g_2` of the source galaxies, following Eq.7 and Eq.8
    in Schrabback et al. (2018), arXiv:1611:03866
    which is consistent with arXiv:0509252


    .. math::

        g_t =& -\left( g_1\cos\left(2\phi\right) - g_2\sin\left(2\phi\right)\right)\\
        g_x =& g_1 \sin\left(2\phi\right) - g_2\cos\left(2\phi\right)


    Parameters
    ----------
    cluster: GalaxyCluster, optional
        Instance of `GalaxyCluster()` and must contain right ascension and declinations of both
        the lens and sources and the two shear components all of the sources. If this
        object is specified, right ascension, declination, and shear or ellipticity inputs are ignored.
    shape_component1: string, optional
        Name of the column in the `galcat` astropy table of the cluster object that contains
        the shape or shear measurement along the first axis. Default: `e1`
    shape_component1: string, optional
        Name of the column in the `galcat` astropy table of the cluster object that contains
        the shape or shear measurement along the second axis. Default: `e2`
    tan_component: string, optional
        Name of the column to be added to the `galcat` astropy table that will contain the
        tangential component computed from columns `shape_component1` and `shape_component2`.
        Default: `et`_compute_cross_shear
    cross_component: string, optional
        Name of the column to be added to the `galcat` astropy table that will contain the
        cross component computed from columns `shape_component1` and `shape_component2`.
        Default: `ex`
    ra_lens: float, optional
        Right ascension of the lensing cluster
    dec_lens: float, optional
        Declination of the lensing cluster
    ra_source_list: array_like, optional
        Right ascensions of each source galaxy
    dec_source_list: array_like, optional
        Declinations of each source galaxy
    shear1: array_like, optional
        The measured shear (or reduced shear or ellipticity) of the source galaxies
    shear2: array_like, optional
        The measured shear (or reduced shear or ellipticity) of the source galaxies
    geometry: str, optional
        Sky geometry to compute angular separation.
        Flat is currently the only supported option.
    add_to_cluster: bool
        If `True` and a cluster was input, add the computed shears to the `GalaxyCluster` object

    Returns
    -------
    angsep: array_like
        Angular separation between lens and each source galaxy in radians
    tangential_shear: array_like
        Tangential shear for each source galaxy
    cross_shear: array_like
        Cross shear for each source galaxy
    """
    if cluster is not None:
        required_cols = ['ra', 'dec', shape_component1, shape_component2]
        if not all([t_ in cluster.galcat.columns for t_ in required_cols]):
            raise TypeError('GalaxyCluster\'s galaxy catalog missing required columns.' +\
                            'Do you mean to first convert column names?')

        ra_lens, dec_lens = cluster.ra, cluster.dec
        ra_source_list, dec_source_list = cluster.galcat['ra'], cluster.galcat['dec']
        shear1, shear2 = cluster.galcat[shape_component1], cluster.galcat[shape_component2]

    # If a cluster object is not specified, we require all of these inputs
    elif any(t_ is None for t_ in (ra_lens, dec_lens, ra_source_list, dec_source_list,
                                   shear1, shear2)):
        raise TypeError('To compute tangential- and cross- shape components, please provide a GalaxyCluster object or ra and dec' +\
                        'of lens and sources and shears or ellipticities of the sources.')

    # If there is only 1 source, make sure everything is a scalar
    if all(not hasattr(t_, '__len__') for t_ in [ra_source_list, dec_source_list, shear1, shear2]):
        pass
    # Otherwise, check that the length of all of the inputs match
    elif not all(len(t_) == len(ra_source_list) for t_ in [dec_source_list, shear1, shear2]):
        raise TypeError('To compute the tangential- and cross- components of shear or ellipticity you should supply the same number of source' +\
                        'positions and shear or ellipticity.')

    # Compute the lensing angles
    if geometry == 'flat':
        angsep, phi = _compute_lensing_angles_flatsky(ra_lens, dec_lens, ra_source_list,
                                                      dec_source_list)
    else:
        raise NotImplementedError(f"Sky geometry {geometry} is not currently supported")

    # Compute the tangential and cross shears
    tangential_shear = _compute_tangential_shear(shear1, shear2, phi)
    cross_shear = _compute_cross_shear(shear1, shear2, phi)

    if add_to_cluster:
        cluster.galcat['theta'] = angsep
        cluster.galcat[tan_component] = tangential_shear
        cluster.galcat[cross_component] = cross_shear

    return angsep, tangential_shear, cross_shear


# Monkey patch functions onto Galaxy Cluster object
GalaxyCluster.compute_tangential_and_cross_components = compute_tangential_and_cross_components
GalaxyCluster.make_binned_profile = make_binned_profile
