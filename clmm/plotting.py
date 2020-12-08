"""A collection of scripts that can be used to plot the various quantities that CLMM models."""
import matplotlib.pyplot as plt
from .galaxycluster import GalaxyCluster
import warnings


def plot_profiles(cluster=None, rbins=None, tangential_component='gt', tangential_component_error='gt_err',
                  cross_component='gx', cross_component_error='gx_err', r_units=None, table_name='profile',
                  xscale='linear',yscale='linear'):
    """Plot shear profiles

    This function can be called by either passing in an instance of `GalaxyCluster` or as an
    attribute of an instance of a `GalaxyCluster` object assuming that that instance has had
    a shear profile computed and saved as a `.profile` attribute. This function can also be
    called by passing in `rbins` along with the respective shears.

    We require at least `rbins` information and `tangential_component` information.

    Parameters
    ----------
    cluster: GalaxyCluster, optional
        Instance of `GalaxyCluster()` that contains a `.profile` attribute.
    rbins: array_like, optional
        The centers of the radial bins that was used to compute the shears.
    tangential_component: array_like, optional
        The tangential component at the radii of `rbins`, or the name of the column in the galcat Table corresponding to the tangential component of the shear or reduced shear (Delta Sigma not yet implemented). Default: 'gt'
    tangential_component_error: array_like, optional
        The uncertainty in the tangential component or the name of the column in the galcat Table corresponding to the uncertainty in tangential component of the shear or reduced shear. Default: 'gt_err'
    cross_component: array_like, optional
        The cross component at the radii of `rbins` or the name of the column in the galcat Table corresponding to the cross component of the shear or reduced shear. Default: 'gx'
    cross_component_error: array_like, optional
        The uncertainty in the cross component or the name of the column in the galcat Table corresponding to the uncertainty in the cross component of the shear or reduced shear. Default: 'gx_err'
    r_units: str, optional
        Units of `rbins` for x-axis label
    table_name: str, optional
        Name of the GalaxyCluster() `.profile` attribute. Default: 'profile'
    xscale:
        matplotlib.pyplot.xscale parameter to set x-axis scale (e.g. to logarithmic axis)
    yscale:
        matplotlib.pyplot.yscale parameter to set y-axis scale (e.g. to logarithmic axis)

    Returns
    -------
    fig:
        The matplotlib figure object that has been plotted to.
    axes:
        The matplotlib axes object that has been plotted to.
    """

    # If a cluster object was passed, use these arrays
    if cluster is not None:
        if not hasattr(cluster, table_name):
            ValueError(f"GalaxyClusters does not have a {table_name} table.")
        cluster_profile = getattr(cluster, table_name)
        for col in (tangential_component, tangential_component_error,
            cross_component, cross_component_error):
            if col not in cluster_profile.colnames:
                warnings.warn(f"Column for plotting {col} does not exist.")
        return _plot_profiles(
            rbins=cluster_profile['radius'],
            r_units=cluster_profile.meta['bin_units'],
            tangential_component=(cluster_profile[tangential_component] if
                tangential_component in cluster_profile.colnames else None),
            tangential_component_error=(cluster_profile[tangential_component_error] if
                tangential_component_error in cluster_profile.colnames else None),
            cross_component=(cluster_profile[cross_component] if
                cross_component in cluster_profile.colnames else None),
            cross_component_error=(cluster_profile[cross_component_error] if
                cross_component_error in cluster_profile.colnames else None),
            xscale=xscale, yscale=yscale)
    else:
        return _plot_profiles(rbins, tangential_component, tangential_component_error,
                  cross_component, cross_component_error, r_units,
                  xscale=xscale, yscale=yscale)

def _plot_profiles(rbins, tangential_component, tangential_component_error,
                  cross_component, cross_component_error, r_units,
                  xscale='linear', yscale='linear'):
    """Plot shear profiles

    This function can be called by either passing in an instance of `GalaxyCluster` or as an
    attribute of an instance of a `GalaxyCluster` object assuming that that instance has had
    a shear profile computed and saved as a `.profile` attribute. This function can also be
    called by passing in `rbins` along with the respective shears.

    We require at least `rbins` information and `tangential_component` information.

    Parameters
    ----------
    rbins: array_like
        The centers of the radial bins that was used to compute the shears.
    tangential_component: array_like
        The tangential component at the radii of `rbins`
    tangential_component_error: array_like
        The uncertainty in the tangential component
    cross_component: array_like
        The cross component at the radii of `rbins`
    cross_component_error: array_like
        The uncertainty in the cross component
    r_units: str
        Units of `rbins` for x-axis label
    xscale:
        matplotlib.pyplot.xscale parameter to set x-axis scale (e.g. to logarithmic axis)
    yscale:
        matplotlib.pyplot.yscale parameter to set y-axis scale (e.g. to logarithmic axis)

    Returns
    -------
    fig:
        The matplotlib figure object that has been plotted to.
    axes:
        The matplotlib axes object that has been plotted to.
    """
    # Plot the tangential shears
    fig, axes = plt.subplots()
    axes.errorbar(rbins, tangential_component,
                  yerr=tangential_component_error,
                  fmt='bo-', label="Tangential component")
    # Plot the cross shears
    axes.errorbar(rbins, cross_component,
                 yerr=cross_component_error,
                 fmt='ro-', label="Cross component")
    # format
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.legend()
    axes.set_xlabel(f'Radius [{r_units}]')
    axes.set_ylabel(r'$\gamma$')

    return fig, axes

GalaxyCluster.plot_profiles = plot_profiles
