"""Tests for backend of theory.py"""
import importlib
import pytest
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


def get_ccl_versions_from_md(filename):
    lines = [l for l in open(filename).readlines() if "CCL" in l and "versions" in l]
    print(lines)
    if len(lines) != 1:
        raise SyntaxError(f"Number of lines with CCL version (={len(lines)}) is not 1.")
    if "between" in lines:
        vmin = lines[0].split("versions between ")[1].split(" and ")[0]
        vmax = lines[0].split(" and ")[1].split(")")[0]
    else:
        vmin = lines[0].split("versions ")[1].split(" or later")[0]
        vmax = None
    return vmin, vmax


def test_documented_ccl_versions_consistency():
    from clmm.theory import _ccl_supported_versions

    # Compare to readme
    vmin, vmax = get_ccl_versions_from_md("README.md")
    assert vmin == _ccl_supported_versions.VMIN
    assert vmax == _ccl_supported_versions.VMAX

    # Compare to install
    vmin, vmax = get_ccl_versions_from_md("INSTALL.md")
    assert vmin == _ccl_supported_versions.VMIN
    assert vmax == _ccl_supported_versions.VMAX


def test_clmm_versioning_error():
    import os

    try:
        import clmm

        avail = clmm.theory.backend_is_available("ccl")
    except ValueError:
        pytest.skip(f"CCL backend not available'.")

    if avail:
        from clmm.theory import _ccl_supported_versions

        vmin_safe = _ccl_supported_versions.VMIN
        vmax_safe = _ccl_supported_versions.VMAX

        _ccl_supported_versions.VMIN = "999"
        _ccl_supported_versions.VMAX = "999"

        os.environ["CLMM_MODELING_BACKEND"] = "ccl"

        assert_raises(EnvironmentError, importlib.reload, clmm.theory.ccl)

        _ccl_supported_versions.VMIN = vmin_safe
        _ccl_supported_versions.VMAX = vmax_safe
