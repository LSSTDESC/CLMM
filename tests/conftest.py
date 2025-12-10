import os

os.environ["MPLBACKEND"] = "template"
import sys
import pytest
import importlib
import clmm


# @pytest.fixture(scope="module", params=["ccl", "nc", "ct", "notabackend", "testnotabackend"])
@pytest.fixture(
    scope="module",
    params=[
        {
            "nick": "ccl",
            "cosmo_reltol": 8.0e-8,
            "dataops_reltol": 3.0e-8,
            "theory_reltol": 2.0e-6,
            "theory_reltol_num": 5.0e-5,
            "ps_reltol": 5.0e-3,
        },
        {
            "nick": "nc",
            "cosmo_reltol": 1.0e-8,
            "dataops_reltol": 1.0e-8,
            "theory_reltol": 1.0e-8,
            "theory_reltol_num": 1.0e-8,
            "ps_reltol": 1.0e-5,
        },
        {
            "nick": "ct",
            "cosmo_reltol": 1.0e-5,
            "dataops_reltol": 5.0e-6,
            "theory_reltol": 3.5e-3,
            "theory_reltol_num": 3.5e-3,
            "ps_reltol": 5.0e-3,
        },
        {
            "nick": "notabackend",
            "cosmo_reltol": 8.0e-8,
            "dataops_reltol": 3.0e-8,
            "theory_reltol": 2.0e-6,
            "ps_reltol": 5.0e-3,
        },
    ],
)
def modeling_data(request):
    param = request.param

    try:
        avail = clmm.theory.backend_is_available(param["nick"])
    except ValueError:
        pytest.fail(f"Unsupported backend '{param}'.")

    if avail or param["nick"] == "notabackend":
        os.environ["CLMM_MODELING_BACKEND"] = param["nick"]
        importlib.reload(clmm.theory)
        importlib.reload(clmm.dataops)
        importlib.reload(clmm)
        return param
    else:
        pytest.fail(f"Backend not available '{param}'.")


@pytest.fixture(
    scope="module",
    params=[
        {},
        {"H0": 67.0, "Omega_b0": 0.06, "Omega_dm0": 0.22, "Omega_k0": 0.0},
        {"H0": 67.0, "Omega_b0": 0.06, "Omega_dm0": 0.22, "Omega_k0": 0.01},
    ],
)
def cosmo_init(request):
    param = request.param

    return param


@pytest.fixture(scope="module", params=["nfw", "einasto", "hernquist"])
# @pytest.fixture(scope="module", params=['nfw'])
def profile_init(request):
    param = request.param

    return param
