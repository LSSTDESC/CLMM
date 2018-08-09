"""Define a superclass that all classes in the CLMM package inherit from"""
import numpy as np

class CLMMBase():
    """
    Highest level super class for the project. Every class outside of core
    should have this as its highest level ancestor.

    Attributes
    ----------
    _ask_type : list or str
        List of data types that each class will accept as inputs.
        Queried by the manager class.
    """
    _ask_type = None

    @property
    def ask_type(self):
        return self._ask_type

    @ask_type.setter
    def ask_type(self, ask_type_list):
        if not np.iterable(ask_type_list):
            raise TypeError('ask_type should be a list')
        self._ask_type = ask_type_list
