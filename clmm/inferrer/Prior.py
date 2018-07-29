'''Class for priors'''        
        

class Prior() :
    ''' 
    This defines a min/max value and some function form to a parameter.

    Attributes
    ----------                                                                                                                                                                                                 
    priormin : float
        minimum value for parameter in fit

    priormax : float
        maximum value for parameter in fit

    funcform : string
        Mathematical expression for the prior

    '''

    def __init__(self, priormin, priormax, funcform) :
        '''
        Parameters                                                                                                                                                                                                 
        ----------                                                                                                                                                                                                 
        priormin : float
            minimum value for parameter in fit
        
        priormax : float
            maximum value for parameter in fit

        funcform : string
            Mathematical expression for the prior

        '''
        
        self.priormin = priormin
        self.priormax = priormax
        self.funcform = funcform
