from __future__ import absolute_import, division, print_function

import numpy as np


class Prior():
    '''
    This defines a min/max value and some function form to a parameter.

    Attributes
    ----------
    funcform : string
        Mathematical expression for the prior

    priormin : float, optional
        minimum value for parameter in fit

    priormax : float, optional
        maximum value for parameter in fit

    '''

    def __init__(self, funcform, priormin=None, priormax=None):
        '''
        Parameters
        ----------
        funcform : string
            Mathematical expression for the prior

        priormin : float, optional
            minimum value for parameter in fit

        priormax : float, optional
            maximum value for parameter in fit

        '''
        self._funcform = funcform
        self._priormin = priormin
        self._priormax = priormax

    @property
    def funcform(self):
        if not callable(self._funcform):
            raise TypeError('funcform must be a callable')
        return self._funcform

    @property
    def priormin(self):
        if self._priormin is None:
            self._priormin = -np.inf
        # do a float division to account for ints
        if not isinstance(self._priormin/2, float):
            raise TypeError('priormin must be a float')
        return self._priormin

    @property
    def priormax(self):
        if self._priormax is None:
            self._priormax = np.inf
        # do a float division to account for ints
        if not isinstance(self._priormax/2, float):
            raise TypeError('priormax must be a float')
        return self._priormax

