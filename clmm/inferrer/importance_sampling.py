'''
Inferrer object which has methods to both run chains per cluster
object (or on a stacked cluster object), read in existing chains per
cluster object, and conduct importance sampling on a collection of
chains.
'''

class ImportanceSampler():
    '''
    Object to conduct importance sampling for a given collection.
    This reads in the hash to find out which clusters belong to which
    collection.  Then, for any given collection, goes through the
    galaxy_cluster_chains, checks if that will be needed for
    importance sampling as it conducts the importance sampling.

    Attributes
    ----------
    collection_id : int
        identifier of the collection this importance sampler instance is 
        responsible for  
    collection_hash_table: dictionary
        keys are collection id's
            values are dictionaries
                'collection_cluster_list' : list of clusters in collection
                'collection_info: : dictionary of relevant collection information
    collection_output_chain: output from importance sampling

    '''
    def __init__(self) :
        '''
        
        '''
        self.collection_id = None
        self._collection_output_chains = {}
        return

    def populate_collection_hash_table(self, collection_hash_table) :
        '''
        '''
        # self.collection_hash_table = reader(collection_hash_table)


    def _get_collection_name_of_galaxy_cluster(self, galaxy_cluster) :
        '''
        Cross reference cluster collection name from the hash table
        '''
        # do stuff with cluster_chain_id and the self.collection_hash_table
        return cluster_collection_name

    def _cluster_is_in_collection(self, galaxy_cluster) :
        
        cluster_collection_name = self._get_collection_name_of_galaxy_cluster(galaxy_cluster)
        return self.collection_id == cluster_collection_name

    def _get_sampler_method(self, **config_params) :
        #  Need to link this to methods
        return sampler_method
    
    def set_collection_id(self, collection_id) :
        ''' Needs to be a first step when looping over collection ids '''
        self.collection_id = collection_id


    def adjust_collection_output_chain(self, sampler, galaxy_cluster) :
        '''
        '''
        # Do something with the chaini in galaxy_cluster
        # modify self.collection_output_chain
        pass
        
    def importance_sample(self, galaxy_cluster, **config_params) :
        ''' 
        This gets run as we loop thorough galaxy clusters. 
        '''
        if self._cluster_is_in_collection(galaxy_cluster) :
            sampler = self._get_sampler_method(**config_params)
            self.adjust_collection_output_chain(sampler, galaxy_cluster)

        
