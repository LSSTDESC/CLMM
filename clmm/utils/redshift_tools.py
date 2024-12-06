"""General utility functions that are used in multiple modules"""
import warnings
import numpy as np
from scipy.integrate import simpson

import qp


def _integ_pzfuncs(pzpdf, pzbins, zmin=0.0, zmax=5, kernel=None, ngrid=1000):
    r"""
    Integrates the product of a photo-z pdf with a given kernel.
    This function was created to allow for data with different photo-z binnings.

    Parameters
    ----------
    pzpdf : list of arrays
        Photometric probablility density functions of the source galaxies.
    pzbins : list of arrays
        Redshift axis on which the individual photoz pdf is tabulated.
    zmin : float, optional
        Minimum redshift for integration. Default: 0
    zmax : float, optional
        Maximum redshift for integration. Default: 5
    kernel : function, optional
        Function to be integrated with the pdf, must be f(z_array) format.
        Default: kernel(z)=1
    ngrid : int, optional
        Number of points for the interpolation of the redshift pdf.

    Returns
    -------
    numpy.ndarray
        Kernel integrated with the pdf of each galaxy.
    """

    if len(np.array(pzbins).shape) > 1:
        # Each galaxy as a diiferent zbin
        _qp_type = qp.interp_irregular
    else:
        _qp_type = qp.interp

    qp_ensemble = qp.Ensemble(
        _qp_type,
        data={
            "xvals": np.array(pzbins),
            "yvals": np.array(pzpdf),
        },
    )

    if kernel is None:

        def kernel(z):
            # pylint: disable=unused-argument
            return 1.0

    z_grid = np.linspace(zmin, zmax, ngrid)
    pz_matrix = qp_ensemble.pdf(z_grid)
    kernel_matrix = kernel(z_grid)

    return simpson(pz_matrix * kernel_matrix, x=z_grid, axis=1)


def compute_for_good_redshifts(
    function,
    z1,
    z2,
    bad_value,
    warning_message,
    z1_arg_name="z1",
    z2_arg_name="z2",
    r_proj=None,
    **kwargs,
):
    """Computes function only for `z1` < `z2`, the rest is filled with `bad_value`

    Parameters
    ----------
    function: function
        Function to be executed
    z1: float, array_like
        Redshift lower
    z2: float, array_like
        Redshift higher
    bad_value: any
        Value to fill when `z1` >= `z2`
    warning_message: str
        Warning message to be displayed when `z1` >= `z2`
    z1_arg_name: str, optional
        Name of the keyword argument that `z1` is passed to. Default: 'z1'
    z2_arg_name: str, optional
        Name of the keyword argument that `z2` is passed to. Default: 'z2'
    r_proj: float, array_like, optional
        Value to be passed to keyword argument `r_proj` of `function`. Default: None

    Returns
    -------
    Return type of `function`
        Output of `function` with value for `z1` >= `z2` replaced by `bad_value`
    """
    kwargs = {z1_arg_name: locals()["z1"], z2_arg_name: locals()["z2"], **kwargs}

    z_good = np.less(z1, z2)
    if r_proj is not None:
        r_proj = np.array(r_proj) * np.full_like(z_good, True)
        z_good = z_good * r_proj.astype(bool)
        kwargs.update({"r_proj": r_proj[z_good] if np.iterable(r_proj) else r_proj})

    if not np.all(z_good):
        warnings.warn(warning_message, stacklevel=2)
        if np.iterable(z_good):
            res = np.full(z_good.shape, bad_value)
            if np.any(z_good):
                kwargs[z1_arg_name] = np.array(z1)[z_good] if np.iterable(z1) else z1
                kwargs[z2_arg_name] = np.array(z2)[z_good] if np.iterable(z2) else z2
                res[z_good] = function(**kwargs)
        else:
            res = bad_value
    else:
        res = function(**kwargs)
    return res
