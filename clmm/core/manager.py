'''
Manager makes the interactions between GalaxyCluster and the other clmm objects
The Manager class talks to galaxy_cluster.GalaxyCluster objects (which can have several types, observed, simulated, different datasets, etc.) and talks with other functions/methods etc that will use the GalaxyCluster objects, e.g. inferring the mass, measuring profiles.

'''
from clmm.core.datatypes import GCData, find_in_datalist

class Manager():

    '''
    Makes the interactions between GalaxyCluster and the other clmm objects
    '''

    def __init__(self):
        '''
        '''
        pass

    def add_data(self, cluster, incoming_creator, incoming_specs, incoming_values):
        '''
        Adds data to GalaxyCluster object 

        Parameters
        ----------
        cluster: GalaxyCluster object
            Object where the data will be added
        incoming_creator: string
            Type of object (i.e. model, summarizer, inferrer) that made this data
        incoming_specs: dict
            Specifications of how the data was created  
        incoming_values: astropy.Table
            Data with units            
            Data to be added to cluster (e.g. assumed source redshift distribution, cluster redshift, cluster mass if calculated), this needs to be compatible as other attributes of the GCData object
        Main functionality of this method is to convert incoming_data to a GCData type of object to be added to the cluster, then add to the cluster object. 
        '''
        incoming_data = GCData(incoming_creator, incoming_specs, incoming_values)
        
        cluster.add_data(incoming_data)
        return

    def apply(self, cluster, func, func_spec):
        '''
        Applies function to and writes output into GalaxyCluster object
    
        Parameters
        ----------
        cluster: GalaxyCluster object
            Object input and output for function
        function: clmm function
            clmm function to be applied to cluster
        func_spec: dict
            Inputs of func
        '''

 
        cluster_output = self.prepare(cluster)
        # Run the inference/function thing on cluster
        func_out = func(cluster_output, **func_spec)
        incoming_creator = func.__name__
        incoming_specs = func_specs
        incoming_values = func_out
        self.add_data(cluster, incoming_creator, incoming_specs, incoming_values)
        
    def prepare(self, cluster):
        '''
        Prepare data from GalaxyCluster objects to be used in inference methods.  
        Inference methods are generally agnostic to what GalaxyCluster looks like.
        ## Note: Leave as pass below until we decide on inferrer contents ##
        '''
        pass

    def deliver(self, cluster, func_out, func, func_specs):
        '''
        Put results from inference into GalaxyCluster objects
        '''
        pass

    def _ask(self):
        '''
        '''
        pass

    def _pack(self):
        '''
        '''
        pass

    def _unpack(self):
        '''
        '''
        pass

    def _signcreator(self):
        '''
        '''
        pass
