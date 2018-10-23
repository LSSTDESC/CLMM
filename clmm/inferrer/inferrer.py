'''
Inferrer object which has methods to both run chains per cluster
object (or on a stacked cluster object), read in existing chains per
cluster object, and conduct importance sampling on a collection of
chains.
'''

class Inferrer():
    '''
    Object to both run chains per cluster and conduct importance sampling

    Attributes
    ----------
    gc_objects: dictionary
        Dictionary with cluster id as keys and a dictionary containing
        a list of relevant information for constraining parameters,
        bin names (added later), and chains (add later) as values
    is_chains: dictionary
        Dictionary with creators as keys and a list of clmm.GCData objects as values
    '''

    def __init__(self, gc_input):
        '''
        Parameters
        ----------
        gc_input: dictionay
            Dictionary with cluster id as keys and a list of relevant
            information for constraining parameters as values
            
        '''
        self.gc_objects = {n:{'data':d} for n, d in gc_input.items()}
        self.is_chains = {}


    def run_gc_chains(self):
        '''
        Runs chains for constraining parameters of each cluster
        individially
        '''
        #self.gc_chains[gc.name] = some_function(self.gc_objects)
        pass

    def run_is_chains(self, bins_specs):
        '''
        Runs Important Sampling on all given bins. It is required that 
        all self.gc_objects in all bins be filled with individual chains.

        Parameters
        ----------
        bins_specs: list
            List with specifications for each bin
        '''
        self._add_bin_info(bins_specs)
        for bin_spec in bins_specs:
            self._run_is_chain(bin_spec)

    def _run_is_chain(self, bin_spec):
        '''
        Runs Important Sampling on one specific bin. It is required that 
        all self.gc_objects in this bin be filled with individual chains.

        Parameters
        ----------
        bin_spec: dict
            Dicitionary with specifications for a certain bin, keys are
            names of property in gc_object and values are touples with
            inferior, superior limits
        '''
        bin_name = self._name_bin(bin_spec)
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

    def _add_bin_info(self, bins_specs):
        '''
        Adds information about which bin each cluster belongs to
        self.gc_objects

        Parameters
        ----------
        bins_specs: list
            List with specifications for each bin
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
