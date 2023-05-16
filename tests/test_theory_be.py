"""Tests for backend of theory.py"""
import importlib
from numpy.testing import assert_raises


def test_base(monkeypatch):
    """Unit tests back end fails"""
    import clmm

    # safekeep original code that will be monkeypatched in these tests
    Modeling_safe = clmm.theory.Modeling
    backends_safe = clmm.theory.__backends
    # Unknown backend required
    monkeypatch.setenv("CLMM_MODELING_BACKEND", "not_available_be")
    assert_raises(ValueError, importlib.reload, clmm.theory)
    # no backends available
    monkeypatch.setenv("CLMM_MODELING_BACKEND", "notabackend")
    clmm.theory.__backends = {
        "notabackend": {
            "name": "notaname",
            "available": False,
            "module": "parent_class",
            "prereqs": ["notaprereq"],
        },
        # This calls the warning "BACKEND also not available"
        "notabackend2": {
            "name": "notaname",
            "available": False,
            "module": "parent_class",
            "prereqs": ["notaprereq"],
        },
    }
    assert_raises(ImportError, clmm.theory.load_backend_env)
    # broken backend
    clmm.theory.__backends["notabackend"]["prereqs"] = []

    monkeypatch.setenv("CLMM_MODELING_BACKEND", "notabackend2")
    clmm.theory.load_backend_env()

    monkeypatch.setenv("CLMM_MODELING_BACKEND", "notabackend")

    def nie():
        raise NotImplementedError

    clmm.theory.Modeling = nie
    clmm.theory.load_backend_env()
    assert clmm.theory.func_layer._modeling_object is None
    # restore original code that will be monkeypatched here
    clmm.theory.Modeling = Modeling_safe
    clmm.theory.__backends = backends_safe
