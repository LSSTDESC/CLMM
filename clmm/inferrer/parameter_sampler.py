'''
ParameterSampler object which has methods to run chains per cluster
object (or on a stacked cluster object).
'''

class ParameterSampler():

    '''
    Object to run chains per cluster

    Attributes
    ----------
    model: Model object
        Model object on which we base the inference/parameter fitting
        (e.g. model profile used)
    config_parms: dict
        Dicitionary with key word arguments for making the inference
        including the inference method and sampling tool.
    available_samplers: dictionary
        Dictionary of implemented samplers with sampler names as keys 
        and samplers as values
    sampler: ???
        Defines which sampler will be used (e.g. emcee)
    '''
    def __init__(self, model, config_params, sampler_type='emcee'):
        '''
        Parameters
        ----------
        model: Model object
            Model object on which we base the inference/parameter fitting
            (e.g. model profile used)
        config_parms: dict
            Dicitionary with key word arguments for making the inference
            including the inference method and sampling tool.
        sampler_type: string
            Defines which sampler will be used (e.g. emcee)
        '''
        self.available_samplers = {'emcee':self._emcee}
        self.model = model
        self.config_params = config_params

        if sampler_type in self.available_samplers:
            self.sampler = self.available_samplers[sampler_type]
        else:
            ValueError('sample_type (%s) not in available_samplers:%s'%(
                sample_type, str(self.available_samplers.keys())))

    def run(self, input_data):
        '''
        Runs chains for constraining parameters of each cluster individially

        Parameters
        ----------
        input_data: dict
            Dictionary with input data for the model

        Returns
        -------
        astropy.table
            Table with the chain of parameters sampled
        '''
        return self.sampler(input_data)

    def _emcee(self, data):
        '''
        Add emcee sampler here

        Parameters
        ----------
        data: dict
            Dictionary with input data for the model

        Returns
        -------
        astropy.table
            Table with the chain of parameters sampled
        '''
        #return emcee(self.data, self.model, **self.config_params)
        pass
