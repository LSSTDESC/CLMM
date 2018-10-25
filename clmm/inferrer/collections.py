'''
Object to make collections based on definition of bins and list of clusters
'''

class Collections():

    '''
    Object to create collection of chains from galaxy cluster objects

    Attributes
    ----------
    bins: dictionary
        Dictionary with keys defined by _bin_names and values by bin
        specifications in dictionary form:
            property name (ex: Mass): property limits (ex: (1e13, 1e14))
    '''
    def __init__(self, bin_specs):
        '''
        Parameters
        ----------
        bins_specs: list
            List with specifications for each bin in dictionary form:
                property name (ex: Mass): property limits (ex: (1e13, 1e14))
            
        '''
        self.bins = {self._name_bin(bin_spec):bin_spec
                        for bin_spec in bin_specs}

    def get_cl_in_bins(self, gc_objects):
        '''
        Adds information about which bin each cluster belongs to
        self.gc_objects.  Each gc object now "knows" what bin it belongs to.

        Parameters
        ----------
        gc_objects: dictionary
            Dictionary with cluster id as keys and a dictionary containing
            a relevant information for binning
            Ex: {'cl1':{'mass':2e13, z:0.8}, 'cl2':...}


        Returns
        -------
        bin_collections: dictionary
            Dictionary with bin_names as keys and list of cluster ids inside
            each bin as values
        '''

        bin_collections = {name:[] for name in self.bins}
        for gc_name, gc_obj in self.gc_objects.items():
            for bin_name, bin_spec in bins.items():
                in_bin = True
                for col, lims in bin_spec.items():
                    if col in gc_obj:
                        in_bin *= lims[0] <= gc_obj[col] < lims[1]
                    else:
                        ValueError('%s not found in cluster'%col)
                if in_bin:
                    bin_collections.append(name)
                    break
        return bin_collections

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

