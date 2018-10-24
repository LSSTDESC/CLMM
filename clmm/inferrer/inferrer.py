'''
Inferrer object which has methods to both run chains per cluster
object (or on a stacked cluster object).
'''

class Inferrer():

    '''
    Object to both run chains per cluster

    Attributes
    ----------
    gc_objects: dictionary
        Dictionary with cluster id as keys and a dictionary containing
        a list of relevant information for constraining parameters,
        bin names (added later), and chains (add later) as values
    is_chains: dictionary
        Dictionary with creators as keys and a list of clmm.GCData objects as values
    out_data: dictionary
        Dictionary with cluster id as keys and dictionaries containing data
        to be exported (mean and std of fitted parameters) as values
    '''
    def __init__(self, gc_objects, model, **config_params):
        '''
        Parameters
        ----------
        gc_input: dictionary
            Dictionary with cluster id as keys and a list of relevant
            information for constraining parameters as values
        model: Model object
            Model object on which we base the inference/parameter fitting (e.g. model profile used)
        config_parms: key word arguments for making the inference including the inference method and sampling tool.
            
        '''
        self.gc_objects = {n:{'data':d} for n, d in gc_objects.items()}
        self.out_data = {}

    def run_gc_chains(self):
        '''
        Runs chains for constraining parameters of each cluster
        individially
        '''
        #self.gc_chains[gc.name] = some_function(self.gc_objects)
        pass

    def compute_out_data(self, statistics):
        '''
        Computers the statistical properties to be exported

        Parameters
        ----------
        statistics: dict
            Dictionary with names of statistical properties as keys
            and the functions as values
        '''
        is_chain_props = {n:{sn:sf(c) for sn, sf in statistics.items()}
             for n,c in self.is_chains.items()}
        self.out_data = {name:in_chain_props[obj['bin']] \
                            for name, obj in self.gc_objects.item()}

