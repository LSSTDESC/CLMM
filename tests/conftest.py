import sys
import pytest
import importlib
import clmm
import os

@pytest.fixture(scope="module", params=["ct", "nc", "ccl", "notabackend", "testnotabackend"])


def modeling_data(request):
    param = request.param
    nonexist = False

    try:
        avail = clmm.theory.backend_is_available(param)
    except ValueError:
        avail = False
        nonexist = True

    if (nonexist):
        pytest.skip(f"Unsupported backend `{param}'.")

    os.environ['CLMM_MODELING_BACKEND'] = param
    importlib.reload(clmm.theory)
    importlib.reload(clmm.dataops)
    importlib.reload(clmm)

    return param

@pytest.fixture(scope="module", params=[{}, {"H0": 67.0, "Omega_b0": 0.06, "Omega_dm0": 0.22, "Omega_k0": 0.0}, {"H0": 67.0, "Omega_b0": 0.06, "Omega_dm0": 0.22, "Omega_k0": 0.01}])


def cosmo_init(request):
    param = request.param

    return param


