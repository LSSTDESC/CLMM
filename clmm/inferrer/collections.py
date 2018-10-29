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

