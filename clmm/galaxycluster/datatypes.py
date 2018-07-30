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

def find_in_datalist(lookup_specs, datalist):
    '''
    Finds data with given specs in a datalist, allows for partial match

    Parameters
    ----------
    datalist: list
        List of clmm.GCData objects to search for lookup_specs
    lookup_specs: dict
        Specs required

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
        print('*** WARNING *** no data found with these specification!')
        return False
    elif len(found) = 1:
        print('*** SUCCESS *** one data found with these specification')
    else:
        print('*** WARNING *** multiple data found with these specification!')
    return found

def find_in_datalist_exact(datalist, lookup_specs):
    '''
    Finds data with given specs in a datalist, requires exact match

    Parameters
    ----------
    datalist: list
        list of clmm.GCData objects to search for lookup_specs
    lookup_specs: dict
        specs required

    Returns
    -------
    reverse
        list of clmm.GCData object data with required creator and set of specs
        if not found, returns None

    '''
    matches = find_in_datalist(lookup_specs, datalist)
    if matches:
        for match in matches:
            reverse = check_subdict(match.specs, lookup_specs)
            if reverse:
                return reverse
