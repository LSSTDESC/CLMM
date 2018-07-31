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
    print(t0.ask_type)
    np.testing.assert_array_equal(t0.ask_type, [])

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
    - If the asserts in this function fail, uncomment the print line
      to see the classes that are being checked. By adding classes
      to either extern_classes or skip_classes will have them be
      skipped by the analysis. If you are unsure of if you should
      be adding a class to skip, ask.
    - extern_classes is defined to be classes loaded by CLMM that are
      external. This includes astropy/colossus/pyccl classes.
    - skip_classes is defined to be classes within CLMM that do not
      do work on data and should NOT inherit the class.
    """
    # External classes
    extern_classes = ['FlatLambdaCDM', 'Table']
    # Internal classes to skip
    skip_classes = ['GCData_', 'GCData', 'Parameter', 'CLMMBase']

    # Load all of the classes in the clmm module and remove skippable things
    class_list = inspect.getmembers(sys.modules[clmm.__name__], inspect.isclass)
    class_list = [thing[0] for thing in class_list]
    class_list = list(set(class_list) - set(skip_classes) - set(extern_classes))

    # Print the list of classes being analyzed. You will only see output
    # on assertion failures
    print(class_list)

    # Assert that we have not added a new class to the api. If this fails, we
    # need to instantiate an example of these objects below and test that
    # ask_type has been set to a reasonable value
    assert len(class_list) == 2


def test_model_super():
    """Test the super properties of Model"""
    tmodel = clmm.Model(lambda x: x)
    np.testing.assert_array_equal(tmodel.ask_type, [])


def test_shearazimuthalaverager_super():
    """Test the super properties of ShearAzimuthalAverager"""
    tsaa = clmm.ShearAzimuthalAverager(None, None)
    assert tsaa.ask_type

