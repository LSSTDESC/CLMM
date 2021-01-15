"""Tests for backend of theory.py"""
import importlib
from numpy.testing import assert_raises

def test_base(monkeypatch):
    """ Unit tests back end fails """
    monkeypatch.setenv("CLMM_MODELING_BACKEND", "not_available_be")
    import clmm
    assert_raises(ValueError, importlib.reload, clmm.theory)
