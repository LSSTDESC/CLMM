"""Tests for backend of theory.py"""
import importlib
import pytest
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

def get_ccl_versions_from_md(filename):
    lines = [l for l in open(filename).readlines()
            if 'CCL' and 'versions between' in l]
    if len(lines)!=1:
        raise SyntaxError(f"Number of lines with CCL version (={len(lines)}) is not 1.")
    vmin = lines[0].split('versions between ')[1].split(' and ')[0]
    vmax = lines[0].split(' and ')[1].split(')')[0]
    return vmin, vmax

def test_documented_ccl_versions_consistency():
    from clmm.theory import _ccl_supported_versions

    # Compare to readme
    vmin, vmax = get_ccl_versions_from_md('README.md')
    assert vmin == _ccl_supported_versions.vmin
    assert vmax == _ccl_supported_versions.vmax

    # Compare to install
    vmin, vmax = get_ccl_versions_from_md('INSTALL.md')
    assert vmin == _ccl_supported_versions.vmin
    assert vmax == _ccl_supported_versions.vmax

def test_clmm_versioning_error():
    import os
    try:
        import clmm
        avail = clmm.theory.backend_is_available('ccl')
    except ValueError:
        pytest.skip(f"CCL backend not available'.")

    if avail:
        from clmm.theory import _ccl_supported_versions

        vmin_safe = _ccl_supported_versions.vmin
        vmax_safe = _ccl_supported_versions.vmax

        _ccl_supported_versions.vmin = '999'
        _ccl_supported_versions.vmax = '999'

        os.environ['CLMM_MODELING_BACKEND'] = 'ccl'

        assert_raises(EnvironmentError, importlib.reload, clmm.theory.ccl)

        _ccl_supported_versions.vmin = vmin_safe
        _ccl_supported_versions.vmax = vmax_safe
