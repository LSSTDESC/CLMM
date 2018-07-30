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
        self.func = func
        self.independent_vars = independent_vars
        self.params = params

        if not callable(self.func):
            raise TypeError('func should be a callable')

        if (np.iterable(self.independent_vars) and not isinstance(self.independent_vars, dict) 
            and all(isinstance(var,str) for var in independent_vars)
            and not isinstance(self.independent_vars, six.string_types) ) \
            or (independent_vars is None) :
            self.independent_vars = independent_vars
        else :
             raise TypeError('independent_vars should be a list of str or None')




        # if not self.independent_vars is None:
        #     raise TypeError('independent_vars should be a list of str or None')
        # elif all(ind_var_bools_and):
        #     raise TypeError('independent_vars should be a list of str or None')

        # if (np.iterable(self.params) \
        #             and not isinstance(self.params, dict)) \
        #         and all(isinstance(param, Parameter) for param in self.params) \
        #         or self.params is None:
        #     pass
        # else :
        #     raise TypeError('params should be a list of type Parameter')
