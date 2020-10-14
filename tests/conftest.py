import pytest
import importlib
import clmm
import os

@pytest.fixture (scope = "module", params = ["ct", "nc", "ccl", "testnotabackend"])
def modeling_data (request):
    param = request.param
    
    try:
        avail = clmm.modeling.backend_is_available (param)
    except ValueError:
        avail = False
    
    if (not avail):
        pytest.skip (f"Unsupported backend `{param}'.")

    os.environ['CLMM_MODELING_BACKEND'] = param
    importlib.reload (clmm.modeling)
    importlib.reload (clmm.polaraveraging)
    importlib.reload (clmm)
    
    print (clmm.modeling)
    
    return param




