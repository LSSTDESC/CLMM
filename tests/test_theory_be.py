"""Tests for backend of theory.py"""
import os
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
                              'module': 'notamodule',
                              'prereqs': ['notaprerq']}}
    assert_raises(ImportError, importlib.reload, clmm.theory)
    # broken backend
    clmm.theory.be_setup.__backends['notabackend']['available'] = True
    del clmm.theory.Modeling
    try:
        importlib.reload(clmm.theory)
        module_problem = False
    except:
        module_problem = True
    assert(module_problem)
    # restore original code that will be monkeypatched here
    clmm.theory.Modeling = Modeling_safe
    clmm.theory.be_setup.__backends = backends_safe
