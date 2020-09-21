"""A collection of scripts that can be used to plot the various quantities that CLMM models."""
import matplotlib.pyplot as plt
from .galaxycluster import GalaxyCluster


def plot_profiles(cluster=None, rbins=None, tangential_component=None, tangential_component_error=None,
                  cross_component=None, cross_component_error=None, r_units=None, table_name='profile',
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
    if cluster is not None and hasattr(cluster, table_name):
        cluster_profile = getattr(cluster,table_name)
        rbins = cluster_profile['radius']
        r_units = cluster_profile.meta['bin_units']
        if tangential_component != 'gt':
            ValueError("The function requires a column called 'gt' to run.")
        if cross_component != 'gx':
            ValueError("The function requires a column called 'gx' to run.")
        if 'gt' not in cluster_profile.colnames:
            ValueError("The function requires a column called 'gt' to run.")
        if 'gx' not in cluster_profile.colnames:
            ValueError("The function requires a column called 'gx' to run.")
        if type(tangential_component)==str:
            tangential_component = cluster_profile[tangential_component]
        else:
            tangential_component = cluster_profile['gt']
        try:
            if type(tangential_component_error)==str:
                tangential_component_error = cluster_profile[tangential_component_error]
            else:
                tangential_component_error = cluster_profile['gt_err']
        except:
            pass
        try:
            if type(cross_component)==str:
                cross_component = cluster_profile[cross_component]
            else:
                cross_component = cluster_profile['gx']
        except:
            pass
        try:
            if type(cross_component_error)==str:
                cross_component_error = cluster_profile[cross_component_error]
            else:
                cross_component_error = cluster_profile['gx_err']
        except:
            pass

    # Plot the tangential shears
    fig, axes = plt.subplots()
    axes.errorbar(rbins, tangential_component,
                  yerr=tangential_component_error,
                  fmt='bo-', label="Tangential component")

    # Plot the cross shears
    try:
        axes.errorbar(rbins, cross_component,
                 yerr=cross_component_error,
                 fmt='ro-', label="Cross component")
    except:
        pass
    
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.legend()
    axes.set_xlabel(f'Radius [{r_units}]')
    axes.set_ylabel(r'$\gamma$')
    
    return fig, axes

GalaxyCluster.plot_profiles = plot_profiles
