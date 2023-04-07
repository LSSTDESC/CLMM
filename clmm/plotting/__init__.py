"""A collection of scripts that can be used to plot the various quantities that CLMM models."""
# Set mpl backend run plots on github actions
import os
import matplotlib as mpl

# pylint: disable=wrong-import-position
if os.environ.get("DISPLAY", "") == "test":
    print("no display found. Using non-interactive Agg backend")
    mpl.use("Agg")
import matplotlib.pyplot as plt


def plot_profiles(
    rbins,
    tangential_component,
    tangential_component_error,
    cross_component,
    cross_component_error,
    r_units=None,
    xscale="linear",
    yscale="linear",
    tangential_component_label="Tangential component",
    cross_component_label="Cross component",
):
    """Plot shear profiles

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
    axes.errorbar(
        rbins,
        tangential_component,
        yerr=tangential_component_error,
        fmt="bo-",
        label=tangential_component_label,
        lw=0.2,
        markersize=1.5,
        elinewidth=0.5,
        capthick=3,
    )
    # Plot the cross shears
    axes.errorbar(
        rbins,
        cross_component,
        yerr=cross_component_error,
        fmt="ro-",
        label=cross_component_label,
        lw=0.2,
        markersize=1.5,
        elinewidth=0.5,
        capthick=3,
    )
    # format
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.legend()
    axes.set_xlabel(f"Radius [{r_units}]")
    axes.set_ylabel(rf"{tangential_component_label} | {cross_component_label}")

    return fig, axes
