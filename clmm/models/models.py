"""Model class"""
import numpy as np
import six


class Parameter():
    """Temporary dummy class so this file can pass tests"""
    def __init__(self):
        return None


class Model():
    """A generalized superclass of what a model is. A model has parameters
    and a functional form.  The parameters may vary or be fixed.

    Attributes
    ----------
    func : callable
        functional form of the model, should be wrapped by the class

    independent_vars : array-like of str, optional
        arguments to func that are independent variables (default to None)

    params : array-like, optional
        list of Parameter objects (default to None)

    """


    def __init__(self, func, independent_vars=None, params=None):
        """
        Parameters
        ----------
        func : callable
            functional form of the model, should be wrapped by the class

        independent_vars : array-like of str
            arguments to func that are independent variables (default to None)

        params : array-like
            list of Parameter objects (default to None)

        """
        self._func = func
        self._independent_vars = independent_vars
        self._params = params

    @property
    def func(self):
        if not callable(self._func):
            raise TypeError('func should be a callable')
        return self._func

    @property
    def independent_vars(self):
        if (np.iterable(self._independent_vars) \
                    and not isinstance(self._independent_vars, dict)) \
                or all(isinstance(var, string_types)
                        for var in self._independent_vars) \
                or self._independent_vars is None:
            return self._independent_vars
        raise TypeError('independent_vars should be a list of str or None')

    @property
    def params(self):
        if (np.iterable(self._params) \
                    and not isinstance(self._params, dict)) \
                and all(isinstance(param, Parameter) for param in self._params) \
                or self._params is None:
            return self._params
        raise TypeError('params should be a list of type Parameter')

