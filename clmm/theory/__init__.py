#------------------------------------------------------------------------------
# Modeling backend loader
import importlib
import warnings
import os
from clmm.theory import be_setup

#  Preload functions:
#    Some backends depend on more complicated modules and thus on a preload
#    function.


def __numcosmo_preload():
    import gi
    gi.require_version("NumCosmoMath", "1.0")
    gi.require_version("NumCosmo", "1.0")



#  Backend check:
#    Checks all backends and set available to True for those that can be
#    corretly loaded.
for _, be in be_setup.__backends.items():
    try:
        if 'preload' in be:
            be['preload']()
        for module in be['prereqs']:
            importlib.import_module(module)
        be['available'] = True
    except:
        pass

#  Backend nick:
#    If the environment variable CLMM_MODELING_BACKEND is set it gets its value,
#    falls back to 'ct' => cluster_toolkit if CLMM_MODELING_BACKEND is not set.
be_nick = os.environ.get('CLMM_MODELING_BACKEND', 'ct')
if not be_nick in be_setup.__backends:
    raise ValueError("CLMM Backend `%s' is not supported" %(be_nick))

#  Backend load:
#  Loads the backend of choice if available or send a warning and try to load
#  the backends in the order of the dictionary above.

if not be_setup.__backends[be_nick]['available']:
    warnings.warn("CLMM Backend requested `%s' is not available, trying others..." %(be_setup.__backends[be_nick]['name']))
    loaded = False
    for be1 in be_setup.__backends:
        if be_setup.__backends[be1]['available']:
            backend = importlib.import_module("clmm.theory."+be_setup.__backends[be1]['module'])
            loaded = True
            be_nick = be1
            break
    if not loaded:
        raise ImportError("No modeling backend available.")
else:
    backend = importlib.import_module("clmm.theory."+be_setup.__backends[be_nick]['module'])

#  Import all backend symbols:
#    Updates __all__ with the exported symbols from the backend and
#    import all symbols in the current namespace.

__all__ = backend.__all__
globals().update({k: getattr(backend, k) for k in backend.__all__})

from . import func_layer

try:
    func_layer.gcm = Modeling()
except NotImplementedError:
    func_layer.gcm = None


def backend_is_available(be1):
    if not be1 in be_setup.__backends:
        raise ValueError("CLMM Backend `%s' is not supported" %(be1))
    else:
        return be_setup.__backends[be1]['available']
