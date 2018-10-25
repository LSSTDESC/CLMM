'''
Object to collect chains.  Perhaps this can just be a function(?)
'''

class CollectChains():

    '''
    Object to create collection of chains from galaxy cluster objects

    Attributes
    ----------
    gc_objects: dictionary
        Dictionary with cluster id as keys and a dictionary containing
        a list of relevant information for constraining parameters and 
        individual chains, and bin names (added here) as values
    
    output_collected_chains: dictionary
        Dictionary {'chains': tuple of collated chains, 'info': list of collection information}
    
    '''
    def __init__(self, gc_objects, **config_params):
        '''
        Parameters
        ----------
        gc_objects: dictionary
            Dictionary with cluster id as keys and a list of relevant
            information for constraining parameters as values
        config_parms: key word arguments for defining collections of chains
            
        '''

        self.gc_objects = gc_objects
        self.config_params = config_params
        
        self.bin_specs = self.define_bin_specs()

        self.output_collected_chains = None

    def run_collect_chains(self) :
        self.add_bin_info()
        self.collate_chains_in_collections()
        self.populate_output()
        
    def define_bin_specs(self) :
        '''
        Define the bin specs according to the config params
        '''

        #return get_bin_specs_method(**config_params)
        pass

    def populate_output(self) :
        '''
        Pack the chain collections and each collection information into the output dictionary.

        '''

        self.output_collected_chains = {'chains': self.collection_chains, 'info': self.bin_spec}
        
    def add_bin_info(self):
        '''
        Adds information about which bin each cluster belongs to
        self.gc_objects.  Each gc object now "knows" what bin it belongs to.
        '''

        for name, obj in self.gc_objects.items():
            for bin_spec in bins_specs:
                in_bin = True
                for col, lims in bin_spec.items():
                    if col in obj:
                        in_bin *= (obj[col]>=lims[0])*(obj[col]<lims[1])
                if in_bin:
                    self.gc_objects[name]['bin'] = \
                            self._name_bin(bin_spec)
                    break

    def collate_chains_in_collections(self, bin_spec) :
        '''
        Collate chains from all galaxy cluster objects into collections 
        that are determined by the bin they are assigned.

        '''

        bin_name = self._name_bin(bin_spec)
        self.collection_chains = {bin_name: [obj['chain'] \
                                    for obj in self.gc_objects.values() \
                                    if obj['bin'] == bin_name]}
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

