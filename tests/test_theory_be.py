"""Tests for backend of theory.py"""
import importlib
from numpy.testing import assert_raises

def test_base(monkeypatch):
    """ Unit tests back end fails """
    import clmm
    # safekeep original code that will be monkeypatched in these tests
    Modeling_safe = clmm.theory.Modeling
    backends_safe = clmm.theory.be_setup.__backends
    # Unknown backend required
    monkeypatch.setenv("CLMM_MODELING_BACKEND", "not_available_be")
    assert_raises(ValueError, importlib.reload, clmm.theory)
    # no backends available
    monkeypatch.setenv("CLMM_MODELING_BACKEND", "notabackend")
    clmm.theory.be_setup.__backends = {
              'notabackend': {'name': 'notaname', 'available': False,
                              'module': 'be_setup',
                              'prereqs': ['notaprerq']},
              # This calls the warning "BACKEND also not available"
              'notabackend2': {'name': 'notaname', 'available': False,
                              'module': 'be_setup',
                              'prereqs': ['notaprerq']}}
    assert_raises(ImportError, importlib.reload, clmm.theory)
    # broken backend
    clmm.theory.be_setup.__backends['notabackend']['available'] = True

    monkeypatch.setenv("CLMM_MODELING_BACKEND", "notabackend2")
    importlib.reload(clmm.theory)

    monkeypatch.setenv("CLMM_MODELING_BACKEND", "notabackend")
    def nie():
        raise NotImplementedError
    clmm.theory.Modeling = nie
    importlib.reload(clmm.theory)
    assert clmm.theory.func_layer.gcm is None
    # restore original code that will be monkeypatched here
    clmm.theory.Modeling = Modeling_safe
    clmm.theory.be_setup.__backends = backends_safe
