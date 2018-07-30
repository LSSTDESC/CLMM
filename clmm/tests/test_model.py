"""Tests for the models.models modules"""
from __future__ import absolute_import, print_function

import numpy as np
from numpy.testing import assert_raises
import six

from models import models
from models.parameter import Parameter


def assert_block(test_model):
    """Block of asserts for type checks in models.Model

    Parameters
    ----------
    test_model : Model instance
        Instance of the Model class to run asserts on
    """

    assert callable(test_model.func)

    assert (np.iterable(test_model.independent_vars) \
            and not isinstance(test_model.independent_vars, dict)) \
            or test_model.independent_vars is None
    if test_model.independent_vars is not None:
        for element in test_model.independent_vars:
            assert isinstance(element, six.string_types)

    assert (np.iterable(test_model.params) \
            and not isinstance(test_model.params, dict)) \
            or (test_model.params is None)
    if test_model.params is not None:
        for element in test_model.params:
            assert isinstance(element, Parameter)


def test_model_superclass():
    """Test the Models superclass. """

    assert_raises(TypeError, models.Model, lambda x: x, [1])
    assert_raises(TypeError, models.Model, lambda x: x, 'r')
    assert_raises(TypeError, models.Model, 'x*x', ['r'])

    test_model = models.Model(lambda x: x)
    assert_block(test_model)

    test_model = models.Model(lambda x: x, ['r'])
    assert_block(test_model)

    # Everything below should fail until Parameter is defined
    test_model = models.Model(lambda x: x*x, ['r'], [Parameter()])
    assert_block(test_model)

    assert_raises(TypeError, models.Model, lambda x: x, ['r'], ['param1'])
