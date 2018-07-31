'''
Manager makes the interactions between GalaxyCluster and the other clmm objects
'''

class Manager():

    '''
    Makes the interactions between GalaxyCluster and the other clmm objects
    '''

    def __init__(self):
        '''
        '''
        pass

    def add_data(self, cluster, incoming_data):
        '''
        Adds data to GalaxyCluster object 

        Parameters
        ----------
        cluster: GalaxyCluster object
            Object where the data will be added
        incoming_data: ???
            Data to be added to cluster
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
        pass

    def prepare(self):
        '''
        Prepare data from GalaxyCluster objects to be used in inference methods
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
