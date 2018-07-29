import models.models as models
from numpy.testing import assert_raises


def assert_block(test_model):
    """Block of asserts for type checks in models.Model """
    assert(callable(test_model.func))

    assert(isinstance(test_model.independent_vars, list) or (test_model.independent_vars is None))
    if test_model.independent_vars is not None:
        for element in test_model.independent_vars:
            assert(isinstance(element, str))

    assert(isinstance(test_model.params, list) or (test_model.params is None))
    if test_model.params is not None:
        for element in test_model.params:
            assert(isinstance(element, Parameter))


def test_model_superclass() :
    # model.Models (callable [func], list of str, [independent_vars], list of
    # Parameter obj [params])

    assert_raises(TypeError, models.Model, lambda x:x, [1])
    assert_raises(TypeError, models.Model, lambda x:x, 'r')
    assert_raises(TypeError, models.Model, 'x*x', ['r'])

    test_model = models.Model(lambda x:x)
    assert_block(test_model)
   
    test_model = models.Model(lambda x:x, ['r'])
    assert_block(test_model)

    # Everything below should fail until Parameter is defined
    test_model = models.Model(lambda x: x*x, ['r'], [Parameter()])
    assert_block(test_model)

    assert_raises(TypeError, models.Model, lambda x:x, ['r'], Parameter())
    assert_raises(TypeError, models.Model, lambda x:x, ['r'], ['param1'])
