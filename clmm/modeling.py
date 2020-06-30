#------------------------------------------------------------------------------
# Modeling backend loader  
import importlib
import warnings
import os

#  Preload functions: 
#    Some backends depend on more complicated modules and thus on a preload
#    function.
def __numcosmo_preload ():
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
__backends = {'ct':  {'name': 'cluster_toolkit+astropy', 'available': False,
                      'module': 'cluster_toolkit',
                      'prereqs': ['cluster_toolkit', 'astropy']},
              'nc':  {'name': 'NumCosmo', 'available': False,
                      'module': 'numcosmo',
                      'prereqs': ['gi.repository.NumCosmoMath', 'gi.repository.NumCosmo'], 
                      'preload': __numcosmo_preload},
              'ccl': {'name': 'ccl', 'available': False,
                      'module': 'ccl',
                      'prereqs': ['pyccl']}}

#  Backend check:
#    Checks all backends and set available to True for those that can be 
#    corretly loaded.
for _, be in __backends.items ():
    try:
        if 'preload' in be:
            be['preload'] ()
        for module in be['prereqs']:
            importlib.import_module (module)
        be['available'] = True
    except:
        pass

#  Backend nick:
#    If the environment variable CLMM_MODELING_BACKEND is set it gets its value, 
#    falls back to 'ct' => cluster_toolkit if CLMM_MODELING_BACKEND is not set.
be_nick = os.environ.get ('CLMM_MODELING_BACKEND', 'ct')
if not be_nick in __backends:
    raise ValueError ("CLMM Backend `%s' is not supported" % (be_nick))

#  Backend load:
#  Loads the backend of choice if available or send a warning and try to load
#  the backends in the order of the dictionary above.
if not __backends[be_nick]['available']:
    warnings.warn ("CLMM Backend requested `%s' is not available, trying others..." % (__backends[be_nick]['name']))
    loaded = False
    for be1 in __backends:
        if __backends[be1]['available']:
            backend = importlib.import_module (".modbackend." + __backends[be1]['module'], package = __package__)
            loaded = True
            be_nick = be1
            break
    if not loaded:
        raise ImportError ("No modeling backend available.") 
else:
    backend = importlib.import_module (".modbackend." + __backends[be_nick]['module'], package = __package__)

#  Import all backend symbols:
#    Updates __all__ with the exported symbols from the backend and
#    import all symbols in the current namespace.

__all__ = backend.__all__
globals().update({k: getattr (backend, k) for k in backend.__all__})

def backend_is_available (be1):
    if not be1 in __backends:
        raise ValueError ("CLMM Backend `%s' is not supported" % (be1))
        return False
    else:
        return __backends[be1]['available']
