#------------------------------------------------------------------------------
# Modeling backend loader
import importlib
import warnings
import os
from clmm.theory import be_setup

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
be_nick = os.environ.get('CLMM_MODELING_BACKEND', 'ccl')
if not be_nick in be_setup.__backends:
    raise ValueError("CLMM Backend `%s' is not supported" %(be_nick))
be_conf = be_setup.__backends[be_nick]

#  Backend load:
#  Loads the backend of choice if available or send a warning and try to load
#  the backends in the order of the dictionary above.
if be_conf['available']:
    backend = importlib.import_module("clmm.theory."+be_conf['module'])
else:
    warnings.warn(f"CLMM Backend requested `{be_conf['name']}' is not available, trying others...")
    loaded = False
    be_nick0 = be_nick
    for be_nick, be_conf in be_setup.__backends.items():
        if be_conf['available']:
            backend = importlib.import_module("clmm.theory."+be_conf['module'])
            loaded = True
            warnings.warn(f"* USING {be_conf['name']} BACKEND")
            break
        if be_nick!=be_nick0:
            warnings.warn(f"* {be_conf['name']} BACKEND also not available")
    if not loaded:
        raise ImportError("No modeling backend available.")

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
