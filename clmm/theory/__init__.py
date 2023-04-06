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

globals().update({k: getattr(generic, k) for k in generic.__all__})

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

    #  Imports all backend symbols:
    globals().update({k: getattr(backend, k) for k in backend.__all__})
    globals().update({k: getattr(func_layer, k) for k in func_layer.__all__})

    try:
        # pylint: disable=undefined-variable
        func_layer.gcm = Modeling()
    except NotImplementedError:
        func_layer.gcm = None


def _load_backend_fallback(be_key, backends_config):
    """Loads the backend of choice if available or send a warning and try to load
    the backends in the order of the __backends dictionary.


    Parameters
    ----------
    be_key : str
        Key for the selected backend in the __backends dictionary.
    backends_config : dict
        Dictionary with the configuration to load backends.
    """
    conf = backends_config[be_key]
    if conf["available"]:
        _load_backend(conf["module"])
        return be_key
    warnings.warn(f"CLMM Backend requested '{conf['name']}' is not available, trying others...")
    print(f"CLMM Backend requested '{conf['name']}' is not available, trying others...")
    for be_key2, conf in backends_config.items():
        if conf["available"]:
            _load_backend(conf["module"])
            warnings.warn(f"* USING {conf['name']} BACKEND")
            return be_key2
        if be_key2 != be_key:
            warnings.warn(f"* {conf['name']} BACKEND also not available")
    raise ImportError("No modeling backend available.")


def load_backend_env(backends_config=None):
    """Loads the backend of choice if available or send a warning and try to load
    the backends in the order of the __backends dictionary.


    Parameters
    ----------
    be_key : str
        Key for the selected backend in the __backends dictionary.
    """
    if backends_config is None:
        backends_config = __backends

    #  Backend check:
    #    Checks all backends and set available to True for those that can be
    #    corretly loaded.
    for nick, be_conf in backends_config.items():
        be_conf["available"] = backend_is_available(nick)

    #  Backend nick:
    #    If the environment variable CLMM_MODELING_BACKEND is set it gets its value,
    #    falls back to 'ccl' => CCL if CLMM_MODELING_BACKEND is not set.
    be_nick_env = os.environ.get("CLMM_MODELING_BACKEND", "ccl")
    if be_nick_env not in backends_config:
        raise ValueError(f"CLMM Backend {be_nick_env}'' is not supported")

    #  Backend load:
    return _load_backend_fallback(be_nick_env, backends_config)


##########################
# Prepare the code backend
##########################

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
