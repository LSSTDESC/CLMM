"""Tests for the ultra super class"""
import numpy as np
import inspect, sys
# Tests require the entire package
import clmm

def test_constructor():
    """
    Verify that the constructor for __CLMMBase is working.
    """
    t0 = clmm.CLMMBase()
    assert t0.ask_type == None

    t1 = clmm.CLMMBase()
    t1.ask_type = ['something']
    np.testing.assert_array_equal(t1.ask_type, ['something'])

    t2 = clmm.CLMMBase()
    np.testing.assert_raises(TypeError, t2.ask_type, 3.14159)


def test_new_classes():
    """
    Test that every class in CLMM is inheriting __CLMMBase and
    is setting ask_type appropriately.

    Notes
    -----
    - skip_classes holds all CLMM classes that either are not bottom level
      or should not inherit CLMMBase. Every bottom level class that is not
      in core should inherit CLMMBase
    """
    # CLMM classes to skip that should not inherit CLMMBase or is not
    # a lowest level child class
    skip_classes = ['GCData_', 'GCData', 'Parameter', 'CLMMBase', 'Model']

    # Load all of the classes in the clmm module and remove skippable things
    class_list = inspect.getmembers(sys.modules[clmm.__name__], inspect.isclass)
    obj_list = [thing[0] for thing in class_list]
    pkg_list = [str(thing[1]) for thing in class_list]

    # Drop all non-clmm
    pkg_list = [element.split('.')[0][-4:] for element in pkg_list]
    obj_list = [obj for obj, pkg in zip(obj_list, pkg_list) if pkg == 'clmm']

    # Remove objets that should not inherit CLMMBase
    obj_list = list(set(obj_list) - set(skip_classes))

    # Instantiate each object and check that its attirbute has been set
    for obj in obj_list:
        try:
            class_instance = eval('clmm.'+obj)()
            assert class_instance.ask_type is not None
        except TypeError:
            print("All attributes for {} should be optional".format(obj))



