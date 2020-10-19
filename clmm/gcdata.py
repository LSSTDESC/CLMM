"""
Define the custom data type
"""
from astropy.table import Table as APtable
import pickle

class GCData(APtable):
    """
    GCData: A data objetc for gcdata. Right now it behaves as an astropy table.

    Parameters
    ----------
    meta: dict
        Dictionary with metadata for this object

    Same as astropy tables
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs: Same used for astropy tables
        """
        APtable.__init__(self, *args, **kwargs)
    def add_meta(self, name, value):
        """
        Add metadata to GCData

        Parameters
        ----------
        name: str
            Name of metadata
        value:
            Value of metadata

        Returns
        -------
        None
        """
        self.meta[name] = value
        return
    def add_metas(self, names, values):
        """
        Add metadatas to GCData

        Parameters
        ----------
        names: list
            List of metadata names
        values: list
            List of metadata values

        Returns
        -------
        None
        """
        for name, vale in zip(names, values):
            self.add_meta(name, value)
        return
    def __repr__(self):
        """Generates string for repr(GCData)"""
        output = f'{self.__class__.__name__}('
        output+= ', '.join([f'{key}={value!r}'
            for key, value in self.meta.items()]
                +['columns: '+', '.join(self.colnames)])
        output+= ')'
        return output
    def __str__(self):
        """Generates string for print(GCData)"""
        output = f'self.__class__.__name__\n> defined by:'
        output+= ', '.join([f'{key}={str(value)}'
            for key, value in self.meta.items()])
        output+= f'\n> with columns: '
        output+= ', '.join(self.colnames)
        return output
    def __getitem__(self, item):
        """
        Makes sure GCData keeps its properties after [] operations are used

        Returns
        -------
        GCData
            Data with [] operations applied
        """
        out = APtable.__getitem__(self, item)
        return out

"""
Additional functions specific to clmm.GCData
Note: Not being used anymore
"""
#
#def confirm_GCData(data):
#    """
#    Typechecks GCData elements
#    """
#    # if not (type(data.creator) == str and type(data.specs) == dict and type(data.values) == astropy.Table):
#    #     raise TypeError('GCData creator should be string, specs should be dict, values should be astropy table')
#    pass
#
#
#def read_GCData(filename):
#    """
#    Reads GCData from file
#    """
#    pass
#
#
#def write_GCData(data, filename):
#    """
#    Writes GCData to file
#    """
#    pass
#
#
#def check_subdict(lookup_dict, reference_dict):
#    '''
#    Checks if a dictionary is a subset of another dictionary, with the same keys and values
#
#    Parameters
#    ----------
#    lookup_dict: dict
#        Subset dictionary
#    reference_dict: dict
#        Main dictionary
#
#    Returns
#    -------
#    bool
#        If a lookup_dict is a subset of reference_dict
#    '''
#    return lookup_dict.items() <= reference_dict.items()
#
#
#def find_in_datalist(lookup_specs, datalist, exact=False):
#    '''
#    Finds data with given specs in a datalist, allows for partial match
#
#    Parameters
#    ----------
#    datalist: list
#        List of clmm.GCData objects to search for lookup_specs
#    lookup_specs: dict
#        Specs required
#    exact: boolean, optional
#        Does the match have to be symmetric?
#
#    Returns
#    -------
#    found: list
#        List of clmm.GCData object data with required creator and set of specs
#    '''
#    found = []
#    for data in datalist:
#        if check_subdict(lookup_specs, data.specs) :
#            found.append(data)
#    if exact:
#        for match in found:
#            if check_subdict(match.specs, lookup_specs):
#                return [match]
#        found = []
#    return found
