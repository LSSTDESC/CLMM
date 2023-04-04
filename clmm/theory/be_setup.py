"""@file be_setup.py
Setup config for the different backends
"""
# ------------------------------------------------------------------------------
# Modeling backend setups
__all__ = []

#  Preload functions:
#    Some backends depend on more complicated modules and thus on a preload
#    function.


def __numcosmo_preload():
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
