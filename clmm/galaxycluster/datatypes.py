"""
Define the custom data type
"""

from astropy import table
from collections import namedtuple

"""
GCData: A namedtuple tying values with units to the metadata of where the values came from and thus how the values are used

Parameters
----------
creator: string, type of object (i.e. model, summarizer, inferrer) that made this data
specs: specifications of how the data was created
values: astropy table with column names and units

Notes
-----

"""
GCData = namedtuple('GCData', ['creator', 'specs', 'values'])

"""
Additional functions specific to clmm.GCData
"""
def check_subdict(lookup_dict, reference_dict):
    '''
    Checks if a dictionary is a subset of another dictionary, with the same keys and values

    Parameters
    ----------
    lookup_dict: dict
        Subset dictionary
    reference_dict: dict
        Main dictionary

    Returns
    -------
    bool
        If a lookup_dict is a subset of reference_dict
    '''
    return lookup_dict.items() <= reference_dict.items()

def find_in_datalist(lookup_specs, datalist, exact=False):
    '''
    Finds data with given specs in a datalist, allows for partial match

    Parameters
    ----------
    datalist: list
        List of clmm.GCData objects to search for lookup_specs
    lookup_specs: dict
        Specs required
    exact: boolean, optional
        Does the match have to be symmetric?

    Returns
    -------
    list
        List of clmm.GCData object data with required creator and set of specs
    None
        If no objects are found
    '''
    found = []
    for data in datalist:
        if check_subdict(lookup_specs, data.specs) :
            found.append(data)
    if len(found) == 0:
        raise ValueError('no data found with these lookup_specs')
    else:
        if exact:
            for match in found:
                reverse = check_subdict(match.specs, lookup_specs)
                if reverse:
                    return reverse
        return found
