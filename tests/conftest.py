import pytest
import importlib
import clmm
import os

@pytest.fixture (scope = "module", params = ["ct", "nc"]) #, "ccl"]) # Removed CCL for now
def modeling_data (request):
    param = request.param

    if (not clmm.modeling.backend_is_available (param)):
        pytest.skip (f"Unsupported backend `{param}'.")

    os.environ['CLMM_MODELING_BACKEND'] = param
    importlib.reload (clmm.modeling)    
    return param




