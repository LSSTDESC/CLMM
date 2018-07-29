'''Model class'''

class Model() :
    '''A generalized superclass of what a model is. A model has parameters
    and a functional form.  The parameters may vary or be fixed.

    Attributes
    ----------
    func : callable
        functional form of the model, should be wrapped by the class

    independent_vars : list of str
        arguments to func that are independent variables (default to None)

    params : list 
        list of Parameter objects (default to None)

    '''


    def __init__(self, func, independent_vars=None, params=None) :
        '''
        Parameters
        ----------
        func : callable
            functional form of the model, should be wrapped by the class
        
        independent_vars : list of str
            arguments to func that are independent variables (default to None)
        
        params : list 
            list of Parameter objects (default to None)

        '''
        
        self.func = func
        self.independent_vars = independent_vars
        self.params = params


