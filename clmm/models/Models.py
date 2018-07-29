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


class Parameter() :
    '''This is an object that may or may not vary, and can be constrained
with a prior.  This is one of the key aspects that define a model.

    Attributes                                                                                                                                                                                                         ----------                                                                                                                                                                                               
    name : str, optional                                                                                                                                                                                       
        Name of the Parameter.                                                                                                                                                                                 

    value : float, optional
        Numerical value of the parameter.  If value is given, vary is False

    vary : bool, optional
        If we vary this Parameter in the analysis (default True)

    prior : object that defines a prior

    Methods
    -------

    '''

    def __init__(self, name, value=None, prior=None) :

        '''
        Parameters                                                                                                                                                                                                 
        ----------                                                                                                                                                                                                 
        name : str
            Name of the Parameter.                                                                                                                                                                                 
        
        value : float, optional
            Numerical value of the parameter.  If value is given, we do not vary this parameter

        prior : object of Prior type, optional
            object that defines a prior, must be defined if value is not given to vary the parameter.
 
        '''

        self.name = name
        self.value = value
        self.prior = prior

        self.__set_parameter_variation()

    def __set_parameter_variation(self) :
        '''Checks if value is given (then set vary to false) and if value
is not given, check that prior is not None.
        '''
        
        if self.value is not None : 
            self.vary = False
        else :
            self.vary = True

        if self.value == None : 
            if self.prior is not None : 
                RaiseAssertionError("Parameter.prior for {} needs to be set if Parameter.value is not given.".format(self.name))
        
