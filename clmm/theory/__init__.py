"""@file __init__.py
Theory package
"""
# ------------------------------------------------------------------------------
# Modeling backend loader
import importlib
import warnings
import os
from . import func_layer
from . import generic

from .generic import (
    compute_reduced_shear_from_convergence,
    compute_magnification_bias_from_magnification,
    compute_rdelta,
    compute_profile_mass_in_radius,
    convert_profile_mass_concentration,
)
from .func_layer import (
    compute_3d_density,
    compute_surface_density,
    compute_mean_surface_density,
    compute_excess_surface_density,
    compute_excess_surface_density_2h,
    compute_surface_density_2h,
    compute_critical_surface_density_eff,
    compute_tangential_shear,
    compute_convergence,
    compute_reduced_tangential_shear,
    compute_magnification,
    compute_magnification_bias,
)

# Functions that do the loading of different backends


def backend_is_available(be_key):
    """Check if required backend is available"""
    conf = __backends.get(be_key)
    if not conf:
        return False
    try:
        if "preload" in conf:
            conf["preload"]()
        for module in conf["prereqs"]:
            importlib.import_module(module)
        return True
    except ImportError:
        return False


def _load_backend(be_module):
    """Loads one backend module

    Parameters
    ----------
    be_module : str
        Name of backend module
    """
    backend = importlib.import_module(f"clmm.theory.{be_module}")

    # pylint: disable=protected-access
    if all(obj in vars(backend) for obj in ("Modeling", "Cosmology")):
        # pylint: disable=global-statement
        global Modeling
        global Cosmology
        Modeling = backend.Modeling
        Cosmology = backend.Cosmology
        func_layer._modeling_object = backend.Modeling()
    else:
        func_layer._modeling_object = None


def _load_backend_fallback(be_key):
    """Loads the backend of choice if available or send a warning and try to load
    the backends in the order of the __backends dictionary.


    Parameters
    ----------
    be_key : str
        Key for the selected backend in the __backends dictionary.
    """
    first_msg = True
    for key in (be_key, *filter(lambda k: k != be_key, __backends.keys())):
        conf = __backends[key]
        if conf["available"]:
            _load_backend(conf["module"])
            return key
        if first_msg:
            warnings.warn(
                f"CLMM Backend requested '{conf['name']}' is not available, trying others..."
            )
            first_msg = False
        else:
            warnings.warn(f"* {conf['name']} BACKEND also not available")
    raise ImportError("No modeling backend available.")


def load_backend_env():
    """Loads the backend of choice if available or send a warning and try to load
    the backends in the order of the __backends dictionary.


    Parameters
    ----------
    be_key : str
        Key for the selected backend in the __backends dictionary.
    """
    #  Backend check:
    #    Checks all backends and set available to True for those that can be
    #    corretly loaded.
    for nick, be_conf in __backends.items():
        be_conf["available"] = backend_is_available(nick)

    #  Backend nick:
    #    If the environment variable CLMM_MODELING_BACKEND is set it gets its value,
    #    falls back to 'ccl' => CCL if CLMM_MODELING_BACKEND is not set.
    be_nick_env = os.environ.get("CLMM_MODELING_BACKEND", "ccl")
    if be_nick_env not in __backends:
        raise ValueError(f"CLMM Backend {be_nick_env}'' is not supported")
    #  Backend load:
    return _load_backend_fallback(be_nick_env)


##########################
# Prepare the code backend
##########################

# Global variables

# pylint: disable=invalid-name
Modeling = None
Cosmology = None

#  Preload functions:
#    Some backends depend on more complicated modules and thus on a preload
#    function.


def __numcosmo_preload():
    # pylint: disable=import-outside-toplevel
    import gi

    gi.require_version("NumCosmoMath", "1.0")
    gi.require_version("NumCosmo", "1.0")


#  Backend dictionary __backends:
#    Dictonary controling the backends, it must test if the backend is available
#    and loadable.
#    - name: The backend name;
#    - module: The actual module name, must be a .py file inside the modbackend
#      directory;
#    - prereqs: modules that need to be loadable to allow the backend to work;
#    - preload: an optional function that must be called before the modules in
#      prereqs are tested;
#    - available: must always starts False;

__backends = {
    "ccl": {"name": "ccl", "available": False, "module": "ccl", "prereqs": ["pyccl"]},
    "nc": {
        "name": "NumCosmo",
        "available": False,
        "module": "numcosmo",
        "prereqs": ["gi.repository.NumCosmoMath", "gi.repository.NumCosmo"],
        "preload": __numcosmo_preload,
    },
    "ct": {
        "name": "cluster_toolkit+astropy",
        "available": False,
        "module": "cluster_toolkit",
        "prereqs": ["cluster_toolkit", "astropy"],
    },
    "notabackend": {
        "name": "notaname",
        "available": False,
        "module": "notamodule",
        "prereqs": ["notaprereq"],
    },
}

be_nick = load_backend_env()
