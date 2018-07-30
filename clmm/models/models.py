"""Model class"""
import numpy as np
import six

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

        if not callable(func):
            raise TypeError('func should be a callable')
        else :
            self.func = func

        if (np.iterable(independent_vars) and not isinstance(independent_vars, dict) 
            and all(isinstance(var,str) for var in independent_vars)
            and not isinstance(independent_vars, six.string_types) ) \
            or (independent_vars is None) :
            self.independent_vars = independent_vars
        else :
             raise TypeError('independent_vars should be a list of str or None')


        if (np.iterable(params) and all(isinstance(param, Parameter) for param in params)) \
           or (params is None) :
            self.params = params
        else :
            raise TypeError('params should be a list of type Parameter')
        
