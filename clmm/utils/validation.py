"""General utility functions that are used in multiple modules"""

import numpy as np
from ..constants import Constants as const


def arguments_consistency(arguments, names=None, prefix=""):
    r"""Make sure all arguments have the same length (or are scalars)

    Parameters
    ----------
    arguments: list, arrays, tuple
        Group of arguments to be checked
    names: list, tuple, None, optional
        Names for each array. Default: None
    prefix: str, optional
        Customized prefix for error message. Default: ''

    Returns
    -------
    list, arrays, tuple
        Group of arguments, converted to numpy arrays if they have length
    """
    sizes = [len(arg) if hasattr(arg, "__len__") else None for arg in arguments]
    # check there is a name for each argument
    if names:
        if len(names) != len(arguments):
            raise TypeError(
                f"names (len={len(names)}) must have same length "
                f"as arguments (len={len(arguments)})"
            )
        msg = ", ".join([f"{n}({s})" for n, s in zip(names, sizes)])
    else:
        msg = ", ".join([f"{s}" for s in sizes])
    # check consistency
    if any(sizes):
        # Check that all of the inputs have length and they match
        if not all(sizes) or any(s != sizes[0] for s in sizes[1:]):
            # make error message
            raise TypeError(f"{prefix} inconsistent sizes: {msg}")
        return tuple(np.array(arg) for arg in arguments)
    return arguments


def _patch_rho_crit_to_cd2018(rho_crit_external):
    r"""Convertion factor for rho_crit of any external modult to
    CODATA 2018+IAU 2015

    rho_crit_external: float
        Critical density of the Universe in units of :math:`M_\odot\ Mpc^{-3}`
    """

    rhocrit_mks = 3.0 * 100.0 * 100.0 / (8.0 * np.pi * const.GNEWT.value)
    rhocrit_cd2018 = (
        rhocrit_mks * 1000.0 * 1000.0 * const.PC_TO_METER.value * 1.0e6 / const.SOLAR_MASS.value
    )

    return rhocrit_cd2018 / rho_crit_external


_valid_types = {
    float: (float, int, np.floating, np.integer),
    int: (int, np.integer),
    "float_array": (float, int, np.floating, np.integer),
    "int_array": (int, np.integer),
    "array": (list, tuple, np.ndarray),
}


def _is_valid(arg, valid_type):
    r"""Check if argument is of valid type, supports arrays.

    Parameters
    ----------
    arg: any
        Argument to be tested.
    valid_type: str, type
        Valid types for argument, options are object types, list/tuple of types, or:

            * 'int_array' - interger, interger array
            * 'float_array' - float, float array

    Returns
    -------
    valid: bool
        Is argument valid
    """
    if valid_type == "function":
        return callable(arg)
    return (
        isinstance(arg[0], _valid_types[valid_type])
        if (valid_type in ("int_array", "float_array") and np.iterable(arg))
        else isinstance(arg, _valid_types.get(valid_type, valid_type))
    )


def validate_argument(
    loc,
    argname,
    valid_type,
    none_ok=False,
    argmin=None,
    argmax=None,
    eqmin=False,
    eqmax=False,
    shape=None,
):
    r"""Validate argument type and raise errors.

    Parameters
    ----------
    loc: dict
        Dictionary with all input arguments. Should be locals().
    argname: str
        Name of argument to be tested.
    valid_type: str, type
        Valid types for argument, options are object types, list/tuple of types, or:

            * 'int_array' - interger, interger array
            * 'float_array' - float, float array

    none_ok: bool, optional
        If True, accepts None as a valid type. Default: False
    argmin : int, float, None, optional
        Minimum value allowed. Default: None
    argmax : int, float, None, optional
        Maximum value allowed. Default: None
    eqmin : bool, optional
        If True, accepts min(arg)==argmin. Default: False
    eqmax : bool, optional
        If True, accepts max(arg)==argmax. Default: False
    shape : tuple of ints, None, optional
        Shape of object allowed. Default: None
    """
    var = loc[argname]
    # Check for None
    if none_ok and (var is None):
        return
    # Check for type
    valid = (
        any(_is_valid(var, types) for types in valid_type)
        if isinstance(valid_type, (list, tuple))
        else _is_valid(var, valid_type)
    )
    if not valid:
        err = f"{argname} must be {valid_type}, received {type(var).__name__}"
        raise TypeError(err)
    # Check min/max
    if any(t is not None for t in (argmin, argmax)):
        try:
            var_array = np.array(var, dtype=float)
        except Exception as exc:
            err = (
                f"{argname} ({type(var).__name__}) cannot be converted to number"
                " for min/max validation."
            )
            raise TypeError(err) from exc
        if argmin is not None:
            if var_array.min() < argmin if eqmin else var_array.min() <= argmin:
                err = (
                    f"{argname} must be greater than {argmin},"
                    f" received min({argname}): {var_array.min()}"
                )
                raise ValueError(err)
        if argmax is not None:
            if var_array.max() > argmax if eqmax else var_array.max() >= argmax:
                err = (
                    f"{argname} must be lesser than {argmax},"
                    f" received max({argname}): {var_array.max()}"
                )
                raise ValueError(err)
    # Check for shape
    if shape is not None:
        if np.shape(var) != shape:
            err = (
                f"{argname} must be of shape {shape}," f"received shape({argname}): {np.shape(var)}"
            )
            raise ValueError(err)


def _validate_ra(loc, ra_name, is_array):
    r"""Validate RA type and raise errors.

    Parameters
    ----------
    loc: dict
        Dictionary with all input arguments. Should be locals().
    ra_name: str
        Name of RA in args.
    is_array: bool
        Accepts array as input.
    """
    v_type = "float_array" if is_array else (float, str)
    validate_argument(loc, ra_name, v_type, argmin=-360, eqmin=True, argmax=360, eqmax=True)


def _validate_dec(loc, dec_name, is_array):
    r"""Validate DEC type and raise errors.

    Parameters
    ----------
    loc: dict
        Dictionary with all input arguments. Should be locals().
    dec_name: str
        Name of DEC in args.
    is_array: bool
        Accepts array as input.
    """
    v_type = "float_array" if is_array else (float, str)
    validate_argument(loc, dec_name, v_type, argmin=-90, eqmin=True, argmax=90, eqmax=True)


def _validate_is_deltasigma_sigma_c(is_deltasigma, sigma_c):
    r""" "Validate the compatibility between is_deltasigma and sigma_c arguments.


    Parameters
    ----------
    is_deltasigma: bool
        If `False`, values are computed for shear, else they are computed for :math:`\Delta \Sigma`.
    sigma_c : None, array_like
        Critical (effective) surface density in units of :math:`M_\odot\ Mpc^{-2}`.
    """
    if is_deltasigma and sigma_c is None:
        raise TypeError("sigma_c (=None) must be provided when is_deltasigma=True")
    if not is_deltasigma and sigma_c is not None:
        raise TypeError(f"sigma_c (={sigma_c}) must be None when is_deltasigma=False")

def _validate_coordinate_system(loc, coordinate_system, valid_type):
    r"""Validate the coordinate system.

    Parameters
    ----------
    loc: dict
        Dictionary with all input arguments. Should be locals().
    coordinate_system: str
        Coordinate system of the ellipticity components. Must be either 'celestial' or 'euclidean'.
    valid_type: str, type
        Valid types for argument, options are object types, list/tuple of types, or:

            * 'int_array' - interger, interger array
            * 'float_array' - float, float array
    """
    validate_argument(loc, coordinate_system, valid_type)
    if loc[coordinate_system] not in ["celestial", "euclidean"]:
        raise ValueError(f"{coordinate_system} must be 'celestial' or 'euclidean'.")

class DiffArray:
    """Array where arr1==arr2 is actually all(arr1==arr)"""

    def __init__(self, array):
        self.value = np.array(array)

    def __eq__(self, other):
        # pylint: disable=unidiomatic-typecheck
        if type(other) != type(self):
            return False
        if self.value.size != other.value.size:
            return False
        return (self.value == other.value).all()

    def __repr__(self):
        out = str(self.value)
        if self.value.size > 4:
            out = self._get_lim_str(out) + " ... " + self._get_lim_str(out[::-1])[::-1]
        return out

    def _get_lim_str(self, out):
        # pylint: disable=undefined-loop-variable
        # get count starting point
        for init_index, char in enumerate(out):
            if all(char != _char for _char in "[]() "):
                break
        # get str
        sep = 0
        for i, char in enumerate(out[init_index + 1 :]):
            sep += int(char == " " and out[i + init_index] != " ")
            if sep == 2:
                break
        return out[: i + init_index + 1]
