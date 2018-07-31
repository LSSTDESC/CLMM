"""Define a superclass that all classes in the CLMM package inherit from"""
import numpy as np

class CLMMBase():
    _ask_type = []

    @property
    def ask_type(self):
        return self._ask_type

    @ask_type.setter
    def ask_type(self, ask_type_list):
        if not np.iterable(ask_type_list):
            raise TypeError()
        self._ask_type = ask_type_list
