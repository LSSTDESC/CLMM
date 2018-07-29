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
        
        if isinstance(func, callable) :
            self.func = func
        else :
            raise TypeError('func should be a callable')

        if ( isinstance(independent_vars, list) and all(isinstance(var,str) for var in independent_vars) ) or (independent_vars is None) :
            self.independent_vars = independent_vars
        else :
            raise TypeError('independent_vars should be a list of str or None')

        if ( isinstance(params,list) and all(isinstance(param, Parameter) for param in params) ) or (params is None) :
            self.params = params
        else :
            raise TypeError('params should be a list of type Parameter')
        
            
