'''
Inferrer object which has methods to both run chains per cluster
object (or on a stacked cluster object), read in existing chains per
cluster object, and conduct importance sampling on a collection of
chains.
'''

class ImportanceSampler():
    '''
    Object to conduct importance sampling

    Attributes
    ----------
    chains_in_collection: array-like
        chains from individual cluster fit in this collection
    collection_info: list
    collection_output_chain: output from importance sampling
    '''
    def __init__(self, input_chains, **config_params) :
        '''
        Parameters
        ----------
        chains: dictionary
            Dictionary with chain information and information on how collection was selected
        config_parms: key word arguments for making the inference including the inference method and sampling tool.
            
        '''

        self.chains_in_collection = self.get_collection_chains(input_chains)
        self.collection_info = self.get_collection_info(input_chains)
        self.importance_sampling_params = **config_params
        self.collection_output_chain = None


    def run_importance_sampling_on_chains(self):
        '''
        Runs Important Sampling on this collection of chains. 
        '''
        collection_chains = [obj['chain'] for obj in self.gc_objects.values() \
                                    if obj['bin'] == bin_name]
        #self.is_chains[bin_name] = some_function(collection_chains)

    def _name_bin(self, bin_spec):
        '''
        Creates a name for a bin, given the specification.

        Parameters
        ----------
        bin_spec: dict
            Dicitionary with specifications for a certain bin, keys are
            names of property in gc_object and values are touples with
            inferior, superior limits
        
        Retuns
        ------
        name: string
            Name of the bin
        '''
        name = ''
        for b, l in bin_spec.items():
            name += b+str(l)
        return name#create string with bin name 



        

    def get_collection_chains(self, input_chains) :
        '''
        Parameters
        ----------
        input_chains : dictionary
            dictionary containing collection chains and collection information
        '''
        return input_chains['chains']

    def get_collection_info(self, input_chains) :
        '''
        Parameters
        ----------
        input_chains : dictionary
            dictionary containing collection chains and collection information
        '''
        return input_chains['info']
        
    def get_importance_sampling_method(self, **params) :
        '''
        Select the method for importance sampling (to define elsewhere)
        '''

        method_key =  params.pop['importance_sampling_method']

        #return importance_sampling_methods[method_key]
        pass

    def run_importance_sampling(self):
        '''
        Runs Importance Sampling on collection.

        Parameters
        ----------
        bins_specs: list
            List with specifications for each bin
        '''
        importance_sampling_method = self.get_importance_sampling_method(**self.importance_sampling_params)

        self.collection_output_chain = importance_sampling_method(self.chains_in_collection)
        
