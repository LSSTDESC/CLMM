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

        # Run the inference/function thing on cluster
        unpacked_cluster = self._unpack(cluster)
        incoming_values = func(unpacked_cluster, **func_spec)

        packed_data = self._pack(func, func_specs, incoming_values)
        cluster.add_data(packed_data)
        
    def prepare(self):
        '''
        Prepare data from GalaxyCluster objects to be used in inference methods.  
        Inference methods are generally agnostic to what GalaxyCluster looks like.
        ## Note: Leave as pass below until we decide on inferrer contents ##
        '''
        pass

    def deliver(self):
        '''
        Put results from inference into GalaxyCluster objects
        '''
        pass

    def _ask(self):
        '''
        '''
        pass

    def _pack(self, func, func_specs, incoming_values):
        '''
        Create GCData with the correct arguments based on a function that was run

        Parameters
        ----------
        incoming_values: astropy.Table
            Data with units            
            Data to be added to cluster (e.g. assumed source redshift distribution, cluster redshift, cluster mass if calculated), this needs to be compatible as other attributes of the GCData object
        Main functionality of this method is to convert incoming_data to a GCData type of object to be added to the cluster, then add to the cluster object. 
        '''
        incoming_creator = func.__name__ # may want to simplify it
        incoming_specs = func_specs # may add more info?
        return GCData(incoming_creator, incoming_specs, incoming_values)

    def _unpack(self):
        '''
        Extracts the correct data from GalaxyCluster to be used in a certain function
        '''
        pass

    def _signcreator(self):
        '''
        '''
        pass
