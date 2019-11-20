"""A collection of scripts that can be used to plot the various quantities that CLMM models."""
import matplotlib.pyplot as plt
from .galaxycluster import GalaxyCluster


def plot_profiles(cluster=None, rbins=None, tangential_shear=None, tangential_shear_error=None,
                  cross_shear=None, cross_shear_error=None, r_units=None):
    """Plot shear profiles

    This function can be called by either passing in an instance of `GalaxyCluster` or as an
    attribute of an instance of a `GalaxyCluster` object assuming that that instance has had
    a shear profile computed and saved as a `.profile` attribute. This function can also be
    called by passing in `rbins` along with the respective shears.

    We require at least `rbins` information and `tangential_shear` information.

    Parameters
    ----------
    cluster: GalaxyCluster, optional
        Instance of `GalaxyCluster()` that contains a `.profile` attribute.
    rbins: array_like, optional
        The centers of the radial bins that was used to compute the shears.
    tangential_shear: array_like, optional
        The tangential shear at the radii of `rbins`
    tangential_shear_error: array_like, optional
        The uncertainty on the tangential shear
    cross_shear: array_like, optional
        The cross shear at the radii of `rbins`
    cross_shear_error: array_like, optional
        The uncertainty on the cross shear
    r_units: str, optional
        Units of `rbins` for x-axis label

    Returns
    -------
    fig:
        The matplotlib figure object that has been plotted to.
    """
    # If a cluster object was passed, use these arrays
    if cluster is not None and hasattr(cluster, 'profile'):
        rbins = cluster.profile['radius']
        r_units = cluster.profile_bin_units
        tangential_shear = cluster.profile['gt']
        try:
            tangential_shear_error = cluster.profile['gt_err']
        except:
            pass
        try:
            cross_shear = cluster.profile['gx']
        except:
            pass
        try:
            cross_shear_error = cluster.profile['gx_err']
        except:
            pass

    # Plot the tangential shears
    fig, axes = plt.subplots()
    axes.errorbar(rbins, tangential_shear,
                  yerr=tangential_shear_error,
                  fmt='bo-', label="Tangential Shear")

    # Plot the cross shears
    try:
        plt.plot(rbins, cross_shear,
                 yerr=cross_shear_error,
                 fmt='ro-', label="Cross Shear")
        plt.errorbar(r, gx, gxerr, label=None)
    except:
        pass

    axes.legend()
    axes.set_xlabel(f'Radius [{r_units}]')
    axes.set_ylabel(r'$\gamma$')

    return fig, axes

GalaxyCluster.plot_profiles = plot_profiles
